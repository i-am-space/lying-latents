"""Stage 4 — hurdle vs Gaussian knockoffs, paired, in the Stage 3 p = 2048 setting.

Stage 3 found that Gaussian knockoffs keep FDR at or below q but lose power at p = 2048, and
that part of the loss is due to the zero atom. This asks whether the Gaussian-copula hurdle
sampler (src/hurdle_sampler.py) recovers that part without inflating FDR
(config/preregistration.yaml stage4_amendment_1, fixed before this ran).

Three arms on the same planted-signal benchmark, with identical draws:
  gauss_real     real latents, Gaussian knockoffs              (baseline)
  hurdle_real    real latents, hurdle knockoffs                (the repair)
  gauss_ceiling  Gaussian data, Gaussian knockoffs              (no zero atom: the ceiling)
gauss_real and hurdle_real share data and labels, so their contrast is paired. Swap two-sample
tests (Stage 2's classifier and MMD) check exchangeability for every arm; the Gaussian arms are
the positive and negative controls for those tests (rule E0).

Usage: python src/stage4_benchmark.py --config config/default.yaml --device cuda --stage diagnose
       python src/stage4_benchmark.py --config config/default.yaml --device cuda --stage full
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

from common import cache_path, load_config, results_dir, rng_for
from hurdle_sampler import GaussianCopulaHurdleSampler, evaluate_hurdle_diagnostics
from knockoff_audit import standardise, swap_sweep
from planted_fdr import generate_planted_labels
from planted_fdr_controls import (METRICS, build_datasets_p2048, cell_rng, fit_lasso_checked,
                                  mean_se, score_fit)
from stage4_repairs import mean_corr          # zero-variance-safe mean corr(X_j, X~_j)

ARM_COLOR = {"gauss_real": "#c0392b", "hurdle_real": "#2e8b57", "gauss_ceiling": "#3b6ea5"}


# ---------------------------------------------------------------------------
# data and knockoff generators
# ---------------------------------------------------------------------------

def build(cfg: dict, X_all: np.ndarray) -> tuple[dict, dict]:
    """Stage 3 p2048 datasets (same rows, covariance and S), plus the hurdle sampler fitted on
    the raw activations of the same rows."""
    base = cfg[cfg["stage4"]["base"]]
    ds, info = build_datasets_p2048(cfg, base, X_all)
    ds.pop("real_first")

    # The builder's first draw from s3p2048_data picks the random rows; repeat it to get the raw
    # activations, and check they are exactly the rows the Gaussian arms use.
    rows = np.sort(rng_for(cfg, "s3p2048_data").choice(X_all.shape[0], base["n_rows"], replace=False))
    X = X_all[rows].astype(np.float64)
    Z, mu, sd = standardise(X)
    if not np.array_equal(Z, ds["real_random"]["Z"]):
        raise SystemExit("hurdle rows differ from the Gaussian arm's rows; refusing to run unpaired")
    del Z

    hc = cfg["stage4"]["hurdle"]
    t = time.time()
    h = GaussianCopulaHurdleSampler(s_method=hc["s_method"], covariance_estimator=hc["covariance"],
                                    positive_tail=hc["positive_tail"])
    h.fit(X, rng=rng_for(cfg, "s4_hurdle_fit"))
    s = np.diag(h.S)
    info["hurdle"] = {"fit_seconds": time.time() - t, "latent_mean_s": float(s.mean()),
                      "latent_min_s": float(s.min()), "latent_max_s": float(s.max())}
    print(f"  hurdle fitted in {time.time() - t:.0f}s  latent mean s={s.mean():.4f}", flush=True)
    return ds, {"X": X, "mu": mu, "sd": sd, "sampler": h, "info": info}


def draw(kind: str, dkey: str, ds: dict, hur: dict, seed: int, raw: bool = False):
    """One knockoff draw on the standardised scale of its data. With raw=True also return the
    raw-scale draw: mapping standardised values back loses exact zeros to rounding, so the
    zero-mass diagnostic must use the sampler's own output."""
    if kind == "gaussian":
        np.random.seed(seed)
        Zk = ds[dkey]["sampler"].sample_knockoffs()
        return (Zk, Zk * hur["sd"] + hur["mu"]) if raw else Zk
    if kind == "hurdle":
        Xk = hur["sampler"].sample_knockoffs(rng=np.random.default_rng(seed))
        Zk = (Xk - hur["mu"]) / hur["sd"]              # the same affine map as the real data
        return (Zk, Xk) if raw else Zk
    raise ValueError(f"unknown knockoff kind {kind!r}")


# ---------------------------------------------------------------------------
# exchangeability diagnostics
# ---------------------------------------------------------------------------

def exchangeability(cfg: dict, ds: dict, hur: dict, arms) -> dict:
    ex = cfg["stage4"]["exchangeability"]
    rng = rng_for(cfg, "s4_diagnostics")
    out = {}
    for ai, (name, dkey, kind) in enumerate(arms):
        seed = int(cell_rng(cfg, "s4_diagnostics", ai).integers(2**31))
        Zk, Xk_raw = draw(kind, dkey, ds, hur, seed, raw=True)
        Z = ds[dkey]["Z"]
        print(f"  [{name}]", flush=True)
        rows = swap_sweep(Z, Zk, cfg, rng, ex["swap_sizes"], ex["swap_replicates"])
        res = {"swap": rows}
        res["mean_corr_X_Xk"] = mean_corr(Z, Zk)
        if dkey == "real_random":
            res["zero_mass"] = evaluate_hurdle_diagnostics(hur["X"], Xk_raw)
        out[name] = res
    p = ds["real_random"]["Z"].shape[1]
    full = {n: next(r["auc"] for r in out[n]["swap"] if r["swap_size"] == p) for n in out}
    mmdp = {n: next(r["p_value"] for r in out[n]["swap"] if r["swap_size"] == p) for n in out}
    e0 = full["gauss_real"] >= 0.90 and 0.45 <= full["gauss_ceiling"] <= 0.55
    e1 = full["hurdle_real"] <= 0.55 and mmdp["hurdle_real"] >= 0.05
    return {"arms": out, "full_swap_auc": full, "full_swap_mmd_p": mmdp,
            "E0_diagnostics_valid": bool(e0), "E1_hurdle_not_detected_as_violated": bool(e1)}


# ---------------------------------------------------------------------------
# benchmark grid (draw bank, as in the Stage 3 p2048 run)
# ---------------------------------------------------------------------------

def run_grid(cfg: dict, ds: dict, hur: dict, device, R: int, amps, forms, ks, arms):
    s4 = cfg["stage4"]
    qs, lam = s4["nominal_fdr_targets"], s4["lasso"]["lambda"]
    names = [a[0] for a in arms]
    p = ds["real_random"]["Z"].shape[1]
    shape = (len(amps), len(forms), len(ks), R)
    M = {m: {a: np.zeros(shape + (len(qs),), dtype=np.float32) for a in names} for m in METRICS}
    conv = {a: np.zeros(shape, dtype=bool) for a in names}
    corr = {a: [] for a in names}
    no_r, no_thr = np.zeros(p), {q: float("inf") for q in qs}
    t_draw = {a: 0.0 for a in names}
    t_fit, n_fit, t0 = 0.0, 0, time.time()

    for rep in range(R):
        for ai, (name, dkey, kind) in enumerate(arms):
            Z = ds[dkey]["Z"]
            seed = int(cell_rng(cfg, "s4_knockoff", ai, rep).integers(2**31))
            t = time.time()
            Zk = draw(kind, dkey, ds, hur, seed)
            t_draw[name] += time.time() - t
            corr[name].append(mean_corr(Z, Zk))
            Phi = torch.from_numpy(np.hstack([Z, Zk]).astype(np.float32)).to(device)
            del Zk
            for ia, amp in enumerate(amps):
                for jf, form in enumerate(forms):
                    for kk, k in enumerate(ks):
                        rng = cell_rng(cfg, "s4_planted", ia, jf, kk, rep)
                        S_idx = np.sort(rng.choice(p, size=k, replace=False))
                        y_np, _ = generate_planted_labels(Z, S_idx, form, amp, rng)
                        y_t = torch.from_numpy(y_np.astype(np.float32)).to(device)
                        t = time.time()
                        w, cv, _, _ = fit_lasso_checked(Phi, y_t, lam, s4["lasso"]["max_iter"], s4["lasso"]["tol"])
                        w_np = w.cpu().numpy()
                        sc = score_fit(np.abs(w_np[:p]) - np.abs(w_np[p:]), no_r, no_thr, qs, set(S_idx.tolist()))
                        t_fit += time.time() - t
                        n_fit += 1
                        for m in METRICS:
                            M[m][name][ia, jf, kk, rep] = sc[m]
                        conv[name][ia, jf, kk, rep] = cv
            del Phi
        el = time.time() - t0
        print(f"  rep {rep + 1:>2}/{R}  elapsed {el / 60:.1f}m  ETA {el / (rep + 1) * (R - rep - 1) / 60:.1f}m", flush=True)
    timing = {"seconds_per_draw": {a: v / R for a, v in t_draw.items()},
              "seconds_per_fit": t_fit / max(1, n_fit)}
    return M, conv, corr, timing


def aggregate(cfg: dict, M, conv, amps, forms, ks, arms):
    s4 = cfg["stage4"]
    qs = s4["nominal_fdr_targets"]
    names = [a[0] for a in arms]
    conds = []
    for a in names:
        for ia, amp in enumerate(amps):
            for jf, form in enumerate(forms):
                for kk, k in enumerate(ks):
                    for iq, q in enumerate(qs):
                        row = {"arm": a, "amplitude": amp, "form": form, "k": k, "q": q,
                               "power_capped_k_lt_1_over_q": bool(k < math.ceil(1 / q)),
                               "fraction_converged": float(conv[a][ia, jf, kk].mean())}
                        for m in ("ko_fdr", "ko_pow", "ko_nd", "k0_pow"):
                            row[m], row[m + "_se"] = mean_se(M[m][a][ia, jf, kk, :, iq])
                        row["fdr_inflated"] = bool(row["ko_fdr"] - 1.96 * row["ko_fdr_se"] > q)
                        conds.append(row)
    contrasts = []
    for cname, a, b in s4["contrasts"]:
        for ia, amp in enumerate(amps):
            for jf, form in enumerate(forms):
                for kk, k in enumerate(ks):
                    for iq, q in enumerate(qs):
                        row = {"contrast": cname, "a": a, "b": b, "amplitude": amp, "form": form, "k": k, "q": q}
                        for m in ("ko_pow", "ko_fdr"):
                            mu, se = mean_se(M[m][a][ia, jf, kk, :, iq] - M[m][b][ia, jf, kk, :, iq])
                            row[m + "_diff"], row[m + "_diff_se"] = mu, se
                            row[m + "_significant"] = bool(se > 0 and abs(mu) > 1.96 * se)
                        contrasts.append(row)
    summary = {"arms": {}, "contrasts": {}}
    for a in names:
        cs = [c for c in conds if c["arm"] == a]
        summary["arms"][a] = {"mean_power": float(np.mean([c["ko_pow"] for c in cs])),
                              "mean_fdr": float(np.mean([c["ko_fdr"] for c in cs])),
                              "max_fdr": float(max(c["ko_fdr"] for c in cs)),
                              "n_fdr_inflated": int(sum(c["fdr_inflated"] for c in cs)),
                              "fraction_converged": float(conv[a].mean())}
    for cname, a, b in s4["contrasts"]:
        cs = [c for c in contrasts if c["contrast"] == cname]
        summary["contrasts"][cname] = {
            "a": a, "b": b, "n_cells": len(cs),
            "mean_power_diff": float(np.mean([c["ko_pow_diff"] for c in cs])),
            "n_power_sig_positive": int(sum(c["ko_pow_significant"] and c["ko_pow_diff"] > 0 for c in cs)),
            "n_power_sig_negative": int(sum(c["ko_pow_significant"] and c["ko_pow_diff"] < 0 for c in cs)),
            "mean_fdr_diff": float(np.mean([c["ko_fdr_diff"] for c in cs])),
            "n_fdr_sig_positive": int(sum(c["ko_fdr_significant"] and c["ko_fdr_diff"] > 0 for c in cs))}
    gain = summary["contrasts"]["hurdle_gain"]["mean_power_diff"]
    gap = summary["contrasts"]["atom_gap"]["mean_power_diff"]
    summary["gap_closed_fraction"] = float(gain / gap) if abs(gap) > 1e-9 else None
    summary["E2_hurdle_fdr_valid"] = summary["arms"]["hurdle_real"]["n_fdr_inflated"] == 0
    return conds, contrasts, summary


def make_figure(conds, amps, forms, ks, arms, q_ref, rd: Path) -> None:
    fig, axes = plt.subplots(2, len(forms), figsize=(6 * len(forms), 8), squeeze=False)
    for j, form in enumerate(forms):
        for i, (met, lab) in enumerate((("ko_pow", "power"), ("ko_fdr", "realised FDR"))):
            ax = axes[i, j]
            for ci, (name, _, _) in enumerate(arms):
                for kk, k in enumerate(ks):
                    rows = sorted([c for c in conds if c["arm"] == name and c["form"] == form
                                   and c["k"] == k and c["q"] == q_ref], key=lambda c: c["amplitude"])
                    ax.errorbar([c["amplitude"] * (1 + 0.03 * (ci - 1)) for c in rows], [c[met] for c in rows],
                                yerr=[1.96 * c[met + "_se"] for c in rows], color=ARM_COLOR.get(name, "gray"),
                                marker=["o", "s", "^"][kk % 3], ms=4, capsize=2, lw=1,
                                label=f"{name}, k={k}" if (i, j) == (0, 0) else None)
            if met == "ko_fdr":
                ax.axhline(q_ref, color="k", ls="--", lw=1)
            ax.set_xscale("log"); ax.set_xlabel("signal amplitude"); ax.set_ylabel(lab)
            ax.set_title(f"{form}: {lab} at q = {q_ref}", fontsize=10)
    axes[0, 0].legend(fontsize=6, ncol=3)
    fig.suptitle("Stage 4, p = 2048: hurdle vs Gaussian knockoffs (95% intervals)", y=1.0)
    fig.tight_layout()
    fig.savefig(rd / "fig17_stage4_hurdle.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 4 — hurdle vs Gaussian knockoffs")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--stage", default="diagnose", choices=["diagnose", "full"])
    ap.add_argument("--limit-reps", type=int, default=None)
    ap.add_argument("--skip-exchangeability", action="store_true", help="debug: skip the swap tests")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = load_config(args.config)
    rd = results_dir(cfg)
    s4 = cfg["stage4"]
    cp = Path(args.cache) if args.cache else cache_path(cfg)
    if not cp.exists():
        raise SystemExit(f"Cache not found at {cp}. Run src/cache_activations.py first.")
    d = np.load(cp, allow_pickle=True)
    X_all = d[f"X_{cfg['aggregation']['primary']}"].astype(np.float32, copy=False)
    arms = [tuple(a) for a in s4["arms"]]
    print(f"device: {device} | cache {d['config_hash']} | stage: {args.stage}")
    t0 = time.time()

    print("\n=== data, Gaussian samplers and hurdle sampler ===")
    ds, hur = build(cfg, X_all)
    info = hur["info"]
    del X_all

    exch = None
    if not args.skip_exchangeability:
        print("\n=== exchangeability swap tests (E0 controls, E1 hurdle) ===")
        exch = exchangeability(cfg, ds, hur, arms)
        for n in exch["full_swap_auc"]:
            print(f"  {n:13s} full-swap AUC {exch['full_swap_auc'][n]:.3f}  MMD p {exch['full_swap_mmd_p'][n]:.3f}  "
                  f"mean corr(X,Xk) {exch['arms'][n]['mean_corr_X_Xk']:.3f}"
                  + (f"  zero-indicator acc {exch['arms'][n]['zero_mass']['median_classifier_accuracy']:.3f}"
                     if "zero_mass" in exch["arms"][n] else ""))
        print(f"  E0 diagnostics valid: {exch['E0_diagnostics_valid']}   "
              f"E1 hurdle not detected as violated: {exch['E1_hurdle_not_detected_as_violated']}")

    if args.stage == "diagnose":
        amp, form, k = s4["diagnose"]["pilot_cell"]
        R = args.limit_reps or s4["diagnose"]["pilot_replicates"]
        print(f"\n=== pilot: amplitude {amp}, {form}, k = {k}, {R} replicates ===")
        M, conv, corr, timing = run_grid(cfg, ds, hur, device, R, [amp], [form], [k], arms)
        iq = s4["nominal_fdr_targets"].index(0.10)
        pilot = {a[0]: {"power_q0.10": float(M["ko_pow"][a[0]][0, 0, 0, :, iq].mean()),
                        "fdr_q0.10": float(M["ko_fdr"][a[0]][0, 0, 0, :, iq].mean()),
                        "mean_corr_X_Xk": float(np.mean(corr[a[0]]))} for a in arms}
        for a, v in pilot.items():
            print(f"  {a:13s} power {v['power_q0.10']:.3f}  FDR {v['fdr_q0.10']:.3f}  corr(X,Xk) {v['mean_corr_X_Xk']:.3f}")
        n_cells = len(s4["signal_amplitudes"]) * len(s4["functional_forms"]) * len(s4["signal_sizes"])
        est = (s4["replicates"] * sum(timing["seconds_per_draw"].values())
               + s4["replicates"] * n_cells * len(arms) * timing["seconds_per_fit"]) / 60
        print(f"  timing {timing}  ->  projected full run on this machine: ~{est:.0f} min")
        (rd / "stage4_diagnose.json").write_text(json.dumps(
            {"info": info, "exchangeability": exch, "pilot": pilot, "timing": timing,
             "projected_full_minutes": est, "minutes": (time.time() - t0) / 60}, indent=2, default=float))
        print(f"\nwrote {rd}/stage4_diagnose.json ({(time.time() - t0) / 60:.1f} min)")
        return

    amps, forms, ks = s4["signal_amplitudes"], s4["functional_forms"], s4["signal_sizes"]
    R = args.limit_reps or s4["replicates"]
    print(f"\n=== full: {len(arms)} arms x {len(amps)} amplitudes x {len(forms)} forms x {len(ks)} k x {R} reps ===")
    M, conv, corr, timing = run_grid(cfg, ds, hur, device, R, amps, forms, ks, arms)
    conds, contrasts, summary = aggregate(cfg, M, conv, amps, forms, ks, arms)

    print("\n=== results ===")
    for a, s in summary["arms"].items():
        print(f"  {a:13s} mean power {s['mean_power']:.3f}  mean FDR {s['mean_fdr']:.4f}  max FDR {s['max_fdr']:.3f}  "
              f"FDR-inflated cells {s['n_fdr_inflated']}  converged {s['fraction_converged']:.0%}  "
              f"corr(X,Xk) {np.mean(corr[a]):.3f}")
    for c, s in summary["contrasts"].items():
        print(f"  {c:12s} ({s['a']} - {s['b']}): power {s['mean_power_diff']:+.3f}  "
              f"sig +{s['n_power_sig_positive']}/-{s['n_power_sig_negative']} of {s['n_cells']}  "
              f"FDR {s['mean_fdr_diff']:+.4f} (sig higher in {s['n_fdr_sig_positive']})")
    print(f"  share of the no-atom gap closed by the hurdle: {summary['gap_closed_fraction']}")
    print(f"  E2 hurdle FDR valid: {summary['E2_hurdle_fdr_valid']}")

    out = {"config_hash": str(d["config_hash"]), "master_seed": cfg["master_seed"], "replicates": R,
           "amplitudes": amps, "forms": forms, "signal_sizes": ks, "arms": [list(a) for a in arms],
           "nominal_fdr_targets": s4["nominal_fdr_targets"], "info": info, "exchangeability": exch,
           "mean_corr_X_Xk": {a: float(np.mean(v)) for a, v in corr.items()}, "timing": timing,
           "summary": summary, "conditions": conds, "contrasts": contrasts,
           "minutes": round((time.time() - t0) / 60, 1)}
    (rd / "stage4_hurdle.json").write_text(json.dumps(out, indent=2, default=float))
    np.savez(rd / "stage4_records.npz", **{f"{m}__{a[0]}": M[m][a[0]] for m in METRICS for a in arms},
             **{f"converged__{a[0]}": conv[a[0]] for a in arms})
    make_figure(conds, amps, forms, ks, arms, 0.10, rd)
    print(f"\nwrote {rd}/stage4_hurdle.json, stage4_records.npz, fig17 ({out['minutes']} min)")


if __name__ == "__main__":
    main()
