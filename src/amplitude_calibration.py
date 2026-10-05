"""Stage 3 amplitude calibration (config/preregistration.yaml stage3_amendment_4).

Where do the real SST-2 labels sit on the planted-signal amplitude axis? A ridge-logistic probe is
fitted to the real labels (ridge chosen on validation log-loss) and, with the same ridge and split,
to planted labels at each amplitude. Primary mapping: the amplitude at which planted labels give the
same held-out probe AUC as the real labels. Secondary: the amplitude at which the probe's top-k
coefficient norm matches (the linear generator's coefficient vector has L2 norm = amplitude).

Usage: python src/amplitude_calibration.py --config config/default.yaml --device cuda
Writes results/stage3_calibration.json and results/fig23_stage3_calibration.png.
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
from sklearn.metrics import roc_auc_score

from common import cache_path, load_config, results_dir, rng_for
from knockoff_audit import standardise
from planted_fdr import generate_planted_labels
from planted_fdr_controls import cell_rng
from stage4_repairs import newton_logistic


def logloss(F, y, beta, b) -> float:
    eta = (F @ beta.float() + b).double()
    return float((torch.nn.functional.softplus(eta) - y.double() * eta).mean())


def setting_data(cfg: dict, name: str, X_all: np.ndarray, labels: np.ndarray):
    n_all, p_all = X_all.shape
    if name == "p2048_all":
        rows, cols = np.arange(n_all), np.arange(p_all)
    elif name == "p512_v2":                      # the stage3_v2 subset: first two draws of s3v2_data
        rng = rng_for(cfg, "s3v2_data")
        v2 = cfg["stage3_v2"]
        cols = np.sort(rng.choice(p_all, v2["p"], replace=False))
        rows = np.sort(rng.choice(n_all, v2["n_rows"], replace=False))
    else:
        raise ValueError(name)
    Z, _, _ = standardise(X_all[np.ix_(rows, cols)].astype(np.float64))
    return Z.astype(np.float32), labels[rows].astype(np.float32)


class Probe:
    """Ridge-logistic probe on a fixed train/validation/test split, on the GPU."""

    def __init__(self, Z: np.ndarray, split: dict, sec: dict, device):
        self.sec, self.dev = sec, device
        self.F = {s: torch.from_numpy(Z[idx]).to(device) for s, idx in split.items()}

    def fit(self, y_tr: np.ndarray, ridge: float):
        y = torch.from_numpy(y_tr).to(self.dev)
        beta, b, conv, _, _ = newton_logistic(self.F["train"], y, ridge, self.sec["newton_max_iter"],
                                              self.sec["newton_tol"])
        return beta, b, conv

    def logits(self, part: str, beta, b) -> np.ndarray:
        return (self.F[part] @ beta.float() + b).double().cpu().numpy()


def top_k_norm(beta, k: int) -> float:
    v = beta.abs().double().cpu().numpy()
    return float(np.sqrt(np.sum(np.sort(v)[::-1][:k] ** 2)))


def match_amplitude(amps, values, target: float):
    """Amplitude at which `values` (increasing in amplitude) reaches `target`; log-linear interpolation."""
    x = np.log(np.asarray(amps, float))
    v = np.maximum.accumulate(np.asarray(values, float))
    if target <= v[0]:
        return {"amplitude": float(amps[0]), "bound": "at or below"}
    if target >= v[-1]:
        return {"amplitude": float(amps[-1]), "bound": "at or above"}
    i = int(np.searchsorted(v, target))
    t = (target - v[i - 1]) / max(v[i] - v[i - 1], 1e-12)
    return {"amplitude": float(math.exp(x[i - 1] + t * (x[i] - x[i - 1]))), "bound": None}


def run_setting(cfg, sec, si: int, name: str, X_all, labels, device) -> dict:
    t0 = time.time()
    Z, y_real = setting_data(cfg, name, X_all, labels)
    n, p = Z.shape
    perm = cell_rng(cfg, "s3c_calibration", si).permutation(n)
    a, b_ = int(sec["split"][0] * n), int((sec["split"][0] + sec["split"][1]) * n)
    split = {"train": perm[:a], "val": perm[a:b_], "test": perm[b_:]}
    probe = Probe(Z, split, sec, device)
    print(f"\n=== {name}: n={n} p={p}  positive rate {y_real.mean():.3f} ===", flush=True)

    # real labels: ridge by validation log-loss, then fixed
    yv = torch.from_numpy(y_real[split["val"]]).to(device)
    grid = []
    for r in sec["ridge_grid"]:
        beta, b, conv = probe.fit(y_real[split["train"]], r)
        grid.append({"ridge": r, "val_logloss": logloss(probe.F["val"], yv, beta, b), "converged": bool(conv)})
        print(f"  ridge {r:.0e}: val log-loss {grid[-1]['val_logloss']:.4f}  converged {conv}", flush=True)
    ridge = min(grid, key=lambda g: g["val_logloss"])["ridge"]
    beta, b, conv = probe.fit(y_real[split["train"]], ridge)
    lt = probe.logits("test", beta, b)
    real = {"ridge": ridge, "converged": bool(conv), "test_auc": float(roc_auc_score(y_real[split["test"]], lt)),
            "test_logit_sd": float(lt.std()),
            "top_k_norm": {str(k): top_k_norm(beta, k) for k in sec["signal_sizes"]},
            "full_norm": float(beta.norm())}
    print(f"  real labels: ridge {ridge:.0e}  test AUC {real['test_auc']:.4f}  test logit SD {real['test_logit_sd']:.3f}  "
          f"top-k norms {real['top_k_norm']}", flush=True)

    # planted labels, same probe, ridge and split
    amps, forms, ks, R = sec["signal_amplitudes"], sec["functional_forms"], sec["signal_sizes"], sec["planted_replicates"]
    planted = {}
    for jf, form in enumerate(forms):
        for kk, k in enumerate(ks):
            rows = []
            for ia, amp in enumerate(amps):
                acc = {"probe_auc": [], "bayes_auc": [], "true_logit_sd": [], "top_k_norm": [], "converged": []}
                for rep in range(R):
                    rng = cell_rng(cfg, "s3c_calibration", si, ia, jf, kk, rep, 1)
                    S_idx = np.sort(rng.choice(p, size=k, replace=False))
                    y, prob = generate_planted_labels(Z, S_idx, form, amp, rng)
                    beta_p, b_p, cv = probe.fit(y[split["train"]], ridge)
                    yt, pt = y[split["test"]], prob[split["test"]]
                    acc["probe_auc"].append(roc_auc_score(yt, probe.logits("test", beta_p, b_p)))
                    acc["bayes_auc"].append(roc_auc_score(yt, pt))
                    pt = np.clip(pt, 1e-12, 1 - 1e-12)
                    acc["true_logit_sd"].append(float(np.log(pt / (1 - pt)).std()))
                    acc["top_k_norm"].append(top_k_norm(beta_p, k))
                    acc["converged"].append(bool(cv))
                rows.append({"amplitude": amp, **{m: float(np.mean(v)) for m, v in acc.items()}})
            planted[f"{form}|{k}"] = {
                "by_amplitude": rows,
                "auc_matched": match_amplitude(amps, [r["probe_auc"] for r in rows], real["test_auc"]),
                "norm_matched": match_amplitude(amps, [r["top_k_norm"] for r in rows], real["top_k_norm"][str(k)])}
            am, nm = planted[f"{form}|{k}"]["auc_matched"], planted[f"{form}|{k}"]["norm_matched"]
            print(f"  {form:11s} k={k:2d}: AUC-matched amplitude {am['amplitude']:.2f}{' (' + am['bound'] + ')' if am['bound'] else ''}"
                  f"   norm-matched {nm['amplitude']:.2f}{' (' + nm['bound'] + ')' if nm['bound'] else ''}", flush=True)
    return {"n": n, "p": p, "positive_rate": float(y_real.mean()), "ridge_grid": grid, "real": real,
            "planted": planted, "minutes": (time.time() - t0) / 60}


def make_figure(out: dict, sec: dict, rd: Path) -> None:
    settings = list(out["settings"])
    fig, axes = plt.subplots(1, len(settings), figsize=(6 * len(settings), 4.5), squeeze=False)
    styles = {"linear": "-", "interaction": "--"}
    colors = dict(zip(sec["signal_sizes"], ("#3b6ea5", "#e07b39", "#2e8b57")))
    for ax, s in zip(axes[0], settings):
        res = out["settings"][s]
        for key, v in res["planted"].items():
            form, k = key.split("|")
            ax.plot([r["amplitude"] for r in v["by_amplitude"]], [r["probe_auc"] for r in v["by_amplitude"]],
                    styles[form], marker="o", ms=3, color=colors[int(k)], label=f"planted, {form}, k={k}")
        ax.axhline(res["real"]["test_auc"], color="k", lw=1.5, label=f"real SST-2 labels ({res['real']['test_auc']:.3f})")
        ax.set_xscale("log"); ax.set_xlabel("planted signal amplitude"); ax.set_ylabel("held-out probe AUC")
        ax.set_title(f"{s} (n = {res['n']:,}, p = {res['p']})")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Where real labels sit on the amplitude axis: same probe, same split", y=1.02)
    fig.tight_layout()
    fig.savefig(rd / "fig23_stage3_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3 amplitude calibration")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny grid on the p512 setting; results not written")
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = load_config(args.config)
    rd = results_dir(cfg)
    sec = cfg["stage3_calibration"]
    if args.smoke:
        sec = {**sec, "settings": ["p512_v2"], "signal_amplitudes": [1.0, 8.0], "signal_sizes": [10],
               "planted_replicates": 1, "ridge_grid": sec["ridge_grid"][::3]}
    cp = Path(args.cache) if args.cache else cache_path(cfg)
    if not cp.exists():
        raise SystemExit(f"Cache not found at {cp}. Run src/cache_activations.py first.")
    d = np.load(cp, allow_pickle=True)
    X_all = d[f"X_{cfg['aggregation']['primary']}"].astype(np.float32, copy=False)
    labels = np.asarray(d["labels"])
    print(f"device: {device} | cache {d['config_hash']}")
    t0 = time.time()
    out = {"config_hash": str(d["config_hash"]), "master_seed": cfg["master_seed"], "settings": {}}
    for si, name in enumerate(sec["settings"]):
        out["settings"][name] = run_setting(cfg, sec, si, name, X_all, labels, device)
        torch.cuda.empty_cache() if device.type == "cuda" else None
    out["minutes"] = round((time.time() - t0) / 60, 1)
    if args.smoke:
        print(f"\nsmoke test done ({out['minutes']} min); nothing written")
        return
    (rd / "stage3_calibration.json").write_text(json.dumps(out, indent=2, default=float))
    make_figure(out, sec, rd)
    print(f"\nwrote {rd}/stage3_calibration.json, fig23_stage3_calibration.png ({out['minutes']} min)")


if __name__ == "__main__":
    main()
