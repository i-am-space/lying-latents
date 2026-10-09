"""Per-dataset amplitude calibration for the SAEBench planted-signal benchmark
(config/preregistration.yaml saebench_amendment_4).

The planted amplitudes of saebench_amendment_2 (8 and 32) were calibrated to SST-2's real labels and
applied unchanged to every SAEBench dataset, so the datasets were not tested at the same difficulty.
Here each dataset's own real labels set the amplitude: a ridge-logistic probe is fitted to each of
the dataset's real one-vs-rest tasks (5,000 rows each), and the amplitude at which planted labels
(same probe, ridge and split, same number of rows) reach the same held-out probe AUC is found, per
(form, k). It is the procedure of src/amplitude_calibration.py, per dataset. Parameters only: no
knockoffs, no FDR.

Usage: python src/saebench/calibrate_amplitude.py --config config/default.yaml --device cuda
       [--dataset NAME] [--positive-threshold FRAC]
Writes <results_dir>/saebench/amplitude_calibration[_thr<FRAC>].json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/
sys.path.insert(0, str(Path(__file__).resolve().parent))       # src/saebench/

from amplitude_calibration import Probe, logloss, match_amplitude   # noqa: E402
from common import load_config, stream_rng                           # noqa: E402
from knockoff_audit import standardise                               # noqa: E402
from planted_fdr import generate_planted_labels                      # noqa: E402
from sbutil import binary_tasks, load_cache, positive_threshold, results_dir, slug   # noqa: E402


def split_idx(n: int, frac, rng) -> dict:
    perm = rng.permutation(n)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return {"train": perm[:a], "val": perm[a:b], "test": perm[b:]}


def fit_best_ridge(probe: Probe, y: np.ndarray, split: dict, grid, device):
    yv = torch.from_numpy(y[split["val"]].astype(np.float32)).to(device)
    best = None
    for r in grid:
        beta, b, _ = probe.fit(y[split["train"]].astype(np.float32), r)
        ll = logloss(probe.F["val"], yv, beta, b)
        if best is None or ll < best[0]:
            best = (ll, r)
    return best[1]


def calibrate_dataset(cfg, sec, di: int, name: str, device, frac_thr: float | None) -> dict:
    t0 = time.time()
    C = load_cache(cfg, name)
    X = C["X"]
    zeroed = None
    if frac_thr:
        X, zeroed = positive_threshold(X, frac_thr)
    Z, _, _ = standardise(X.astype(np.float64))
    Z = Z.astype(np.float32)
    n, p = Z.shape
    tasks = binary_tasks(cfg, name, C)
    print(f"\n=== {name}: n={n} p={p}  {len(tasks)} tasks"
          + (f"  (thresholded at {frac_thr} x median positive; {zeroed:.3f} of positives zeroed)" if frac_thr else ""),
          flush=True)

    # real labels: held-out probe AUC per task, ridge by validation log-loss
    real_auc, ridges = [], []
    for ti, t in enumerate(tasks):
        rows, y = t["rows"], t["y"]
        split = split_idx(len(rows), sec["split"], stream_rng(cfg, "sb4_calib", di, 0, ti))
        probe = Probe(Z[rows], split, sec, device)
        r = fit_best_ridge(probe, y, split, sec["ridge_grid"], device)
        beta, b, _ = probe.fit(y[split["train"]], r)
        real_auc.append(float(roc_auc_score(y[split["test"]], probe.logits("test", beta, b))))
        ridges.append(r)
        del probe
    target = float(np.median(real_auc))
    ridge = float(np.exp(np.median(np.log(ridges))))
    ridge = min(sec["ridge_grid"], key=lambda g: abs(np.log(g) - np.log(ridge)))
    print(f"  real tasks: held-out probe AUC median {target:.4f} (range {min(real_auc):.4f}-{max(real_auc):.4f}), "
          f"ridge {ridge:.0e}", flush=True)

    # planted labels on the same number of rows as a real task, same probe, ridge and split
    m = min(sec["task_rows"], n)
    sub = np.sort(stream_rng(cfg, "sb4_calib", di, 1).choice(n, size=m, replace=False))
    split = split_idx(m, sec["split"], stream_rng(cfg, "sb4_calib", di, 2))
    Zs = Z[sub]
    probe = Probe(Zs, split, sec, device)
    amps, R = sec["signal_amplitudes"], sec["planted_replicates"]
    matched, curves = {}, {}
    for jf, form in enumerate(sec["forms"]):
        for kk, k in enumerate(sec["signal_sizes"]):
            aucs = []
            for ia, amp in enumerate(amps):
                a = []
                for rep in range(R):
                    rng = stream_rng(cfg, "sb4_calib", di, 3, jf, kk, ia, rep)
                    S_idx = np.sort(rng.choice(p, size=k, replace=False))
                    y, _ = generate_planted_labels(Zs, S_idx, form, amp, rng)
                    beta, b, _ = probe.fit(y[split["train"]], ridge)
                    a.append(roc_auc_score(y[split["test"]], probe.logits("test", beta, b)))
                aucs.append(float(np.mean(a)))
            mt = match_amplitude(amps, aucs, target)
            matched[f"{form}|{k}"] = {"amplitude": round(mt["amplitude"], 2), "bound": mt["bound"]}
            curves[f"{form}|{k}"] = aucs
            print(f"  {form:11s} k={k:3d}: matched amplitude {mt['amplitude']:6.2f}"
                  f"{' (' + mt['bound'] + ')' if mt['bound'] else ''}   planted AUC by amplitude "
                  + " ".join(f"{x:.3f}" for x in aucs), flush=True)
    del probe
    return {"dataset": name, "n": n, "p": p, "n_tasks": len(tasks), "task_rows": m,
            "real_auc_by_task": real_auc, "real_auc_median": target, "ridge": ridge,
            "positive_threshold": {"frac": frac_thr, "positives_zeroed": zeroed},
            "amplitudes": amps, "planted_auc": curves, "matched": matched,
            "minutes": round((time.time() - t0) / 60, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-dataset amplitude calibration (saebench_amendment_4)")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dataset", default=None, help="one dataset (default: all of config saebench.datasets)")
    ap.add_argument("--positive-threshold", type=float, default=None, metavar="FRAC",
                    help="ReLU SAEs: zero values below FRAC x the latent's median positive value first")
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = load_config(args.config)
    sec = {**cfg["stage4_saebench_matched"]["calibration"], "forms": cfg["stage4_saebench_matched"]["forms"],
           "signal_sizes": cfg["stage4_saebench_matched"]["signal_sizes"],
           "newton_max_iter": cfg["stage4_saebench_matched"]["calibration"]["newton_max_iter"],
           "newton_tol": cfg["stage4_saebench_matched"]["calibration"]["newton_tol"]}
    datasets = cfg["saebench"]["datasets"]
    todo = [args.dataset] if args.dataset else datasets
    suffix = f"_thr{args.positive_threshold:g}" if args.positive_threshold else ""
    out_path = results_dir(cfg) / f"amplitude_calibration{suffix}.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {"datasets": {}}
    for name in todo:
        di = datasets.index(name)
        if name in out["datasets"]:
            print(f"{name}: done already")
            continue
        out["datasets"][name] = calibrate_dataset(cfg, sec, di, name, device, args.positive_threshold)
        out_path.write_text(json.dumps(out, indent=2, default=float))      # resumable per dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
