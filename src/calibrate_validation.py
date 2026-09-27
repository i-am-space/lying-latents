"""Stage 1, validate step — false-certification rate of search-then-validate.

The search half (naive per-latent scan) was measured in calibrate_pipeline.py. Its
validate half (zero ONE latent, need a 2% held-out accuracy drop) has no power, so its
FWER says nothing. This script replaces it with a validate step that can detect an effect
(config/preregistration.yaml stage1_amendment_2, fixed before this ran):

  search   : score all latents on the training split, take the top k = 10   (unchanged)
  validate : ablate the candidates, paired one-sided test that the probe's log-loss rises,
             certify if p <= alpha. Primary variant: all k ablated jointly (one test).
             Secondary: each latent alone, any-of-k uncorrected / Bonferroni.
  regimes  : same-data (validate on the rows the probe was fit on = conventional
             workflow) and held-out (sample-splitting).

FWER = fraction of B label permutations with >= 1 certification. A planted-signal power
gate (rule B1) says whether those numbers can be interpreted at all; nothing is tuned to
pass it (rule B2).

Usage: python src/calibrate_validation.py --config config/default.yaml --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from calibrate_pipeline import (fit_probe_gpu, score_auroc, score_mean_diff,
                                score_probe_weight, to_gpu)
from common import cache_path, load_config, results_dir, rng_for
from planted_fdr import generate_planted_labels

REGIMES = ("same_data", "held_out")


# ---------------------------------------------------------------------------
# the validate step
# ---------------------------------------------------------------------------

def ablated_logits(Xs: torch.Tensor, logits: torch.Tensor, w: torch.Tensor,
                   mean: torch.Tensor, std: torch.Tensor, idx: torch.Tensor):
    """Probe logits after setting the raw activation of latents `idx` to 0.

    Xs is the standardised matrix. Zeroing raw latent j moves its standardised value to
    -mean_j/std_j, changing the logit by w_j * (that - Xs_j); no matrix is cloned.
    Returns (per-latent (n, k), all-k-jointly (n,)).
    """
    z0 = -mean[idx] / std[idx]
    dj = w[idx].unsqueeze(0) * (z0.unsqueeze(0) - Xs[:, idx])        # (n, k)
    return logits.unsqueeze(1) + dj, logits + dj.sum(dim=1)


def paired_loss_test(logits_full: torch.Tensor, logits_abl: torch.Tensor,
                     y: torch.Tensor):
    """One-sided paired test that ablation RAISES the probe's log-loss.

    d_i = logloss_i(ablated) - logloss_i(full); t = mean(d) / (sd(d) / sqrt(n)); with
    n in the thousands the normal tail gives p = P(Z >= t). Works for logits_abl of shape
    (n,) or (n, k). Returns (mean d, p), same trailing shape.
    """
    l0 = F.binary_cross_entropy_with_logits(logits_full, y, reduction="none")
    if logits_abl.dim() == 2:
        l1 = F.binary_cross_entropy_with_logits(
            logits_abl, y.unsqueeze(1).expand_as(logits_abl), reduction="none")
        d = l1 - l0.unsqueeze(1)
    else:
        d = F.binary_cross_entropy_with_logits(logits_abl, y, reduction="none") - l0
    n = d.shape[0]
    mean_d = d.mean(dim=0)
    sd = d.std(dim=0).clamp(min=1e-12)
    t = (mean_d / (sd / math.sqrt(n))).double()
    return mean_d, 0.5 * torch.special.erfc(t / math.sqrt(2.0))


def run_one(X: torch.Tensor, y: torch.Tensor, cfg: dict, rng: np.random.Generator) -> dict:
    """Search then validate on one labelling (real, permuted or planted).

    Fresh 70/30 split, fit the probe on the 70%, score latents on the same 70%, take the
    top k, then run the paired test in both regimes. RNG use: split, then probe seed.
    """
    s1, sv = cfg["stage1"], cfg["stage1_validate"]
    n = len(y)
    ntr = int(n * (1 - s1["probe_holdout_fraction"]))
    split = rng.permutation(n)
    tr, te = split[:ntr], split[ntr:]
    X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]

    w, b, mean, std = fit_probe_gpu(
        X_tr, y_tr, C=s1["probe_C"], lr=s1.get("probe_lr", 0.1),
        max_iter=s1["probe_max_iter"], seed=int(rng.integers(2**31)))

    Xs = {"same_data": (X_tr - mean) / std, "held_out": (X_te - mean) / std}
    ys = {"same_data": y_tr, "held_out": y_te}
    logits = {r: Xs[r] @ w + b for r in REGIMES}

    out = {}
    for method in s1["scoring_methods"]:
        if method == "mean_diff":
            scores = score_mean_diff(X_tr, y_tr)
        elif method == "auroc":
            scores = score_auroc(X_tr, y_tr)
        elif method == "probe_weight":
            scores = score_probe_weight(w)
        else:
            raise ValueError(f"unknown scoring method: {method!r}")
        top = torch.argsort(scores, descending=True)[:sv["top_k"]]

        rec = {"top": top.cpu().tolist()}
        for regime in REGIMES:
            l_single, l_joint = ablated_logits(Xs[regime], logits[regime], w, mean, std, top)
            mean_j, p_j = paired_loss_test(logits[regime], l_joint, ys[regime])
            _, p_s = paired_loss_test(logits[regime], l_single, ys[regime])
            vals = torch.stack([p_j.float(), mean_j, p_s.min().float()]).tolist()
            rec[regime] = {"joint_p": vals[0], "joint_mean_d": vals[1], "single_p_min": vals[2]}
        out[method] = rec
    return out


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def certified(rec: dict, alpha: float, k: int) -> dict:
    return {"joint": rec["joint_p"] <= alpha,
            "any_uncorrected": rec["single_p_min"] <= alpha,
            "any_bonferroni": rec["single_p_min"] <= alpha / k}


VARIANTS = ("joint", "any_uncorrected", "any_bonferroni")


def planted_power(X: torch.Tensor, X_np: np.ndarray, cfg: dict, device: torch.device) -> dict:
    """Same workflow on labels generated from k KNOWN latents. Fresh stream, so the
    permutations are unaffected."""
    sv = cfg["stage1_validate"]
    pc = sv["planted_check"]
    alpha, k = sv["alpha"], sv["top_k"]
    methods = cfg["stage1"]["scoring_methods"]
    rng = rng_for(cfg, "validate_planted")
    sd = X_np.std(axis=0)
    Z = ((X_np - X_np.mean(axis=0)) / np.where(sd > 1e-8, sd, 1.0)).astype(np.float32)
    p = X_np.shape[1]

    result = {}
    for amp in pc["amplitudes"]:
        rows = []
        for r in range(pc["replicates"]):
            S = np.sort(rng.choice(p, size=pc["k"], replace=False))
            y_np, _ = generate_planted_labels(Z, S, pc["form"], amp, rng)
            res = run_one(X, to_gpu(y_np, device), cfg, rng)
            S_set = set(S.tolist())
            rows.append({m: {"recall": len(S_set & set(res[m]["top"])) / len(S_set),
                             **{reg: certified(res[m][reg], alpha, k) for reg in REGIMES}}
                         for m in methods})
        per = {}
        for m in methods:
            per[m] = {"mean_recall_at_k": float(np.mean([r[m]["recall"] for r in rows]))}
            for reg in REGIMES:
                per[m][reg] = {v: float(np.mean([r[m][reg][v] for r in rows])) for v in VARIANTS}
        result[str(amp)] = per
        print(f"  planted amplitude {amp}: " + "  ".join(
            f"{m}: recall={per[m]['mean_recall_at_k']:.2f} joint fires same/held="
            f"{per[m]['same_data']['joint']:.0%}/{per[m]['held_out']['joint']:.0%}" for m in methods),
            flush=True)
    del Z

    gate = str(pc["gate_amplitude"])
    has_power = {reg: bool(any(result[gate][m][reg]["joint"] >= pc["min_detect_fraction"]
                               for m in methods)) for reg in REGIMES}
    return {"replicates": pc["replicates"], "k": pc["k"], "gate_amplitude": pc["gate_amplitude"],
            "min_detect_fraction": pc["min_detect_fraction"], "by_amplitude": result,
            "criterion_has_power": has_power, "criterion_has_power_both": all(has_power.values())}


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def make_figure(out: dict, methods: list[str], sv: dict, rd: Path) -> None:
    fig, axes = plt.subplots(1, 2 if "planted" in out else 1,
                             figsize=(12 if "planted" in out else 6, 4.2), squeeze=False)
    ax = axes[0, 0]
    x = np.arange(len(methods))
    for j, (reg, col) in enumerate((("same_data", "#c0392b"), ("held_out", "#3b6ea5"))):
        vals, lo, hi = [], [], []
        for m in methods:
            f = out["null"][m][reg]["joint"]
            vals.append(f["fwer"]); lo.append(max(0.0, f["fwer"] - f["ci95"][0])); hi.append(max(0.0, f["ci95"][1] - f["fwer"]))
        ax.bar(x + (j - 0.5) * 0.36, vals, 0.34, yerr=[lo, hi], color=col, capsize=3,
               label=reg.replace("_", "-"))
    ax.axhline(sv["alpha"], color="k", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("FWER (>= 1 certification), primary variant")
    ax.set_title(f"Search-then-validate under the global null\n(B={out['n_permutations']}, "
                 f"top-{sv['top_k']} ablated jointly, alpha={sv['alpha']})")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8)

    if "planted" in out:
        ax = axes[0, 1]
        pc = out["planted"]
        cols = {"3.0": ("#c0392b", "#e6a19b"), "1.0": ("#3b6ea5", "#a9c1dc")}
        combos = [(a, reg) for a in pc["by_amplitude"] for reg in REGIMES]
        w = 0.8 / len(combos)
        for j, (a, reg) in enumerate(combos):
            vals = [pc["by_amplitude"][a][m][reg]["joint"] for m in methods]
            ax.bar(x + (j - (len(combos) - 1) / 2) * w, vals, w * 0.95,
                   color=cols.get(a, ("gray", "lightgray"))[REGIMES.index(reg)],
                   label=f"amp {a}, {reg.replace('_', '-')}")
        ax.axhline(pc["min_detect_fraction"], color="k", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(methods)
        ax.set_ylabel("fraction of planted replicates certified")
        ax.set_title("Power gate (dashed = 80%)")
        ax.set_ylim(0, 1.05); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(rd / "fig12_stage1_validation.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1 — validate-step FWER calibration")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--limit-perms", type=int, default=None, help="debug: cap permutations")
    ap.add_argument("--skip-planted", action="store_true", help="debug: skip the power gate")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")
    cfg = load_config(args.config)
    rd = results_dir(cfg)
    cp = Path(args.cache) if args.cache else cache_path(cfg)
    if not cp.exists():
        raise SystemExit(f"Cache not found at {cp}. Run src/cache_activations.py first.")
    d = np.load(cp, allow_pickle=True)
    prim = cfg["aggregation"]["primary"]
    X_np = d[f"X_{prim}"].astype(np.float32, copy=False)
    y_np = d["labels"].astype(np.float32)
    n, p = X_np.shape
    print(f"loaded: n={n} p={p} aggregator={prim} hash={d['config_hash']}")

    sv = cfg["stage1_validate"]
    alpha, k = sv["alpha"], sv["top_k"]
    B = args.limit_perms or sv["n_permutations"]
    methods = cfg["stage1"]["scoring_methods"]
    X, y = to_gpu(X_np, device), to_gpu(y_np, device)
    rng = rng_for(cfg, "validate_permutation")

    print(f"\n=== validate step: B={B} permutations, top-{k}, alpha={alpha} ===")
    real = run_one(X, y, cfg, rng)
    for m in methods:
        print(f"  [real labels, {m}] joint p same/held = "
              f"{real[m]['same_data']['joint_p']:.3g}/{real[m]['held_out']['joint_p']:.3g}")

    t0 = time.time()
    null = {m: {r: {"joint_p": [], "joint_mean_d": [], "single_p_min": []} for r in REGIMES}
            for m in methods}
    for b in range(B):
        res = run_one(X, to_gpu(rng.permutation(y_np), device), cfg, rng)
        for m in methods:
            for r in REGIMES:
                for key in ("joint_p", "joint_mean_d", "single_p_min"):
                    null[m][r][key].append(res[m][r][key])
        if (b + 1) % 50 == 0 or b == 0:
            el = time.time() - t0
            print(f"  perm {b+1:>4}/{B}  elapsed={el/60:.1f}min  ETA={el/(b+1)*(B-b-1)/60:.1f}min", flush=True)
    print(f"completed {B} permutations in {(time.time()-t0)/60:.1f} min")

    out = {"config_hash": str(d["config_hash"]), "master_seed": cfg["master_seed"],
           "n": n, "p": p, "n_permutations": B, "top_k": k, "alpha": alpha,
           "scoring_methods": methods, "null": {}, "real": {}}
    arrays = {}
    for m in methods:
        out["null"][m], out["real"][m] = {}, {}
        for r in REGIMES:
            pj = np.array(null[m][r]["joint_p"]); ps = np.array(null[m][r]["single_p_min"])
            arrays[f"{m}__{r}__joint_p"] = pj
            arrays[f"{m}__{r}__single_p_min"] = ps
            arrays[f"{m}__{r}__joint_mean_d"] = np.array(null[m][r]["joint_mean_d"])
            out["null"][m][r] = {}
            for v, hits in (("joint", pj <= alpha), ("any_uncorrected", ps <= alpha),
                            ("any_bonferroni", ps <= alpha / k)):
                out["null"][m][r][v] = {"fwer": float(hits.mean()), "n_certified": int(hits.sum()),
                                        "ci95": list(wilson(int(hits.sum()), B))}
            out["null"][m][r]["joint_mean_d_null_mean"] = float(np.mean(null[m][r]["joint_mean_d"]))
            out["real"][m][r] = {**real[m][r], **{v: bool(c) for v, c in certified(real[m][r], alpha, k).items()}}
        c = out["null"][m]
        print(f"\n[{m}] primary (joint) FWER  same-data {c['same_data']['joint']['fwer']:.3f} "
              f"{tuple(round(x, 3) for x in c['same_data']['joint']['ci95'])}  |  held-out "
              f"{c['held_out']['joint']['fwer']:.3f} {tuple(round(x, 3) for x in c['held_out']['joint']['ci95'])}")
        print(f"   any-of-{k} uncorrected  same {c['same_data']['any_uncorrected']['fwer']:.3f} "
              f"held {c['held_out']['any_uncorrected']['fwer']:.3f}   Bonferroni  same "
              f"{c['same_data']['any_bonferroni']['fwer']:.3f} held {c['held_out']['any_bonferroni']['fwer']:.3f}")

    if not args.skip_planted:
        print("\n=== power gate (planted signals) ===")
        out["planted"] = planted_power(X, X_np, cfg, device)
        pw = out["planted"]["criterion_has_power"]
        print(f"  criterion_has_power: same-data={pw['same_data']}  held-out={pw['held_out']}")
        if not out["planted"]["criterion_has_power_both"]:
            print("  POWER GATE FAILED in at least one regime: per rule B1 that regime's FWERs are UNINFORMATIVE.")

    (rd / "stage1_validation.json").write_text(json.dumps(out, indent=2, default=float))
    np.savez(rd / "stage1_validation_null.npz", **arrays)
    make_figure(out, methods, sv, rd)
    print(f"\nwrote {rd}/stage1_validation.json, stage1_validation_null.npz, fig12_stage1_validation.png")


if __name__ == "__main__":
    main()
