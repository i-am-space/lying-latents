"""SAEBench Stage 2 — distributional premise and exchangeability of Gaussian knockoffs, per dataset.

Per dataset (config/preregistration.yaml saebench_amendment_1), on all of its cached rows (the
classes' SAEBench train and test texts pooled; knockoffs depend only on the law of X, never on labels):

  premise    zero mass of the retained latents (rule SB2-P: median Pr(X_j = 0) >= 0.80, as in Stage 2)
  knockoffs  second-order Gaussian, equicorrelated S (the audited construction), on standardised
             latents, covariance by the v2 rule (sample unless ill-conditioned, then Ledoit-Wolf)
  tests      knockoff_audit.run_suite, unchanged: the 1{x = 0} rule, swap two-sample classifier
             and MMD over |S| in {0, 1, 10, 50, all}, moments; a label-permutation null band for
             the classifier AUC
  control    Gaussian data with the dataset's covariance at the same n, covariance re-estimated by
             the same rule, same knockoffs and tests: must stay null (rule SB2-V)

Each dataset writes its own JSON (resumable); --aggregate pools them.

Usage: python src/saebench/stage2.py --config config/default.yaml [--dataset NAME]
       python src/saebench/stage2.py --config config/default.yaml --aggregate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/
sys.path.insert(0, str(Path(__file__).resolve().parent))       # src/saebench/

from common import load_config, stream_rng                                      # noqa: E402
from knockoff_audit import (estimate_cov, label_permutation_null, make_knockoffs,  # noqa: E402
                            run_suite, standardise)
from sbutil import load_cache, results_dir, slug                                  # noqa: E402


def choose_covariance(Z: np.ndarray, cfg: dict):
    S = estimate_cov(Z, "sample")
    ev = np.linalg.eigvalsh(S)
    cond = float(ev[-1] / max(ev[0], 1e-300))
    if cond > float(cfg["covariance"]["ill_conditioned_cond_number"]):
        return estimate_cov(Z, "ledoit_wolf"), "ledoit_wolf", cond, float(ev[0])
    return S, "sample", cond, float(ev[0])


def premise(X: np.ndarray) -> dict:
    p0 = (X == 0).mean(axis=0)
    skew, kurt = [], []
    for j in range(X.shape[1]):
        nz = X[X[:, j] > 0, j]
        if nz.size > 3:
            skew.append(stats.skew(nz)); kurt.append(stats.kurtosis(nz))
    return {"median_p_zero": float(np.median(p0)), "frac_p_zero_gt_0.8": float((p0 > 0.8).mean()),
            "frac_p_zero_gt_0.95": float((p0 > 0.95).mean()), "min_p_zero": float(p0.min()),
            "median_nonzero_skew": float(np.median(skew)), "median_nonzero_excess_kurtosis": float(np.median(kurt))}


def auc_at(suite: dict, size: int) -> float:
    v = [s["auc"] for s in suite["swap"] if s["swap_size"] == size]
    return float(np.mean(v)) if v else float("nan")


def run_dataset(cfg: dict, name: str) -> dict:
    s2 = cfg["saebench"]["stage2"]
    di = cfg["saebench"]["datasets"].index(name)
    C = load_cache(cfg, name)
    X_raw = C["X"].astype(np.float64)
    n, p = X_raw.shape
    print(f"\n=== {name}: n = {n}, p = {p}, cache {C['config_hash']} ===", flush=True)
    out = {"dataset": name, "n": n, "p": p, "n_over_p": n / p, "cache_config_hash": C["config_hash"]}
    out["premise"] = premise(X_raw)
    out["SB2_P_premise_holds"] = bool(out["premise"]["median_p_zero"] >= 0.80)
    print(f"  premise: median Pr(X=0) {out['premise']['median_p_zero']:.3f} -> "
          f"{'holds' if out['SB2_P_premise_holds'] else 'FAILS'}", flush=True)

    Z, mu, sd = standardise(X_raw)
    Sigma, est, cond, eigmin = choose_covariance(Z, cfg)
    out["covariance"] = {"estimator": est, "cond_sample": cond, "eig_min_sample": eigmin}
    seed_k = int(stream_rng(cfg, "sb_knockoff", di).integers(2**31))
    t = time.time()
    Zk, S_mat = make_knockoffs(Z, Sigma, s2["s_method"], seed_k)
    Xk_raw = Zk * sd + mu
    out["s_mean"] = float(np.diag(S_mat).mean())
    print(f"  covariance {est} (cond {cond:.3g}); knockoffs {time.time() - t:.0f}s, s = {out['s_mean']:.4f}", flush=True)

    rr = stream_rng(cfg, "sb_row_partition", di)
    real = run_suite(f"{slug(name)}_real", Z, Zk, cfg, rr, s2["swap_sizes"], s2["swap_replicates"],
                     raw=(X_raw, Xk_raw))
    real.pop("_zero_mass_arrays", None)
    null = label_permutation_null(Z, Zk, cfg, rr, s2["n_label_permutations"])
    out["real"] = real
    out["label_permutation_auc"] = {"values": null, "mean": float(np.mean(null)), "sd": float(np.std(null)),
                                    "q975": float(np.quantile(null, 0.975))}
    del Zk, Xk_raw

    print("  Gaussian control (same n, covariance re-estimated)", flush=True)
    rs = stream_rng(cfg, "sb_synthetic_null", di)
    L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(p))
    G, _, _ = standardise(rs.standard_normal((n, p)) @ L.T)
    del L
    Sig_G, est_G, _, _ = choose_covariance(G, cfg)
    Gk, _ = make_knockoffs(G, Sig_G, s2["s_method"], seed_k + 7)
    ctrl = run_suite(f"{slug(name)}_gauss_ctrl", G, Gk, cfg, rs, s2["control_swap_sizes"], s2["swap_replicates"])
    ctrl.pop("_zero_mass_arrays", None)
    out["gaussian_control"] = {**ctrl, "covariance_estimator": est_G}
    del G, Gk

    q975 = out["label_permutation_auc"]["q975"]
    zm = real["zero_mass"]["median_accuracy"]
    out["SB2_violation_detected"] = bool(zm > 0.55 and auc_at(real, 1) > q975)
    out["SB2_V_diagnostics_valid"] = bool(all(auc_at(ctrl, s if s != "all" else p) <= q975
                                              for s in s2["control_swap_sizes"]))
    out["summary"] = {"zero_mass_median_accuracy": zm, "auc_swap1": auc_at(real, 1), "auc_swap10": auc_at(real, 10),
                      "auc_swap_all": auc_at(real, p), "auc_swap0": auc_at(real, 0), "null_q975": q975,
                      "control_auc_swap1": auc_at(ctrl, 1), "control_auc_swap_all": auc_at(ctrl, p),
                      "knockoff_self_correlation": real["knockoff_self_correlation"]}
    print(f"  zero-mass rule median accuracy {zm:.3f} | swap AUC |S|=1 {auc_at(real, 1):.3f} (null q97.5 {q975:.3f}) | "
          f"control |S|=1 {auc_at(ctrl, 1):.3f} | violation {out['SB2_violation_detected']} | "
          f"diagnostics valid {out['SB2_V_diagnostics_valid']}", flush=True)
    return out


def aggregate(cfg: dict) -> None:
    rd = results_dir(cfg, "stage2")
    res = [json.loads(p.read_text()) for p in sorted(rd.glob("*.json"))]
    if not res:
        raise SystemExit(f"no per-dataset results in {rd}")
    summ = {"n_datasets": len(res),
            "per_dataset": {r["dataset"]: {"n": r["n"], "p": r["p"], **r["premise"], **r["summary"],
                                           "premise_holds": r["SB2_P_premise_holds"],
                                           "violation_detected": r["SB2_violation_detected"],
                                           "diagnostics_valid": r["SB2_V_diagnostics_valid"],
                                           "covariance": r["covariance"]["estimator"]} for r in res}}
    valid = [r for r in res if r["SB2_V_diagnostics_valid"]]
    summ["SB2_R_violation_replicates"] = {
        "n_valid": len(valid), "n_violation_among_valid": int(sum(r["SB2_violation_detected"] for r in valid)),
        "replicates": bool(valid and all(r["SB2_violation_detected"] for r in valid if r["SB2_P_premise_holds"]))}
    (results_dir(cfg) / "stage2_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))
    print(f"\nwrote {results_dir(cfg)}/stage2_summary.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="SAEBench Stage 2: premise and exchangeability per dataset")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--debug", action="store_true",
                    help="cut-down run to check the code path and time one dataset; writes to stage2_debug/")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.aggregate:
        aggregate(cfg)
        return
    if args.debug:
        cfg["saebench"]["stage2"].update({"swap_sizes": [0, 1, "all"], "swap_replicates": 1,
                                          "control_swap_sizes": [0, "all"], "n_label_permutations": 2})
        cfg["diagnostics"]["mmd_permutations"] = 100
    rd = results_dir(cfg, "stage2_debug" if args.debug else "stage2")
    for name in ([args.dataset] if args.dataset else cfg["saebench"]["datasets"]):
        path = rd / f"{slug(name)}.json"
        if path.exists():
            print(f"{name}: done, skipping")
            continue
        t = time.time()
        out = run_dataset(cfg, name)
        out["minutes"] = (time.time() - t) / 60
        path.write_text(json.dumps(out, indent=2, default=float))
        print(f"  wrote {path} ({out['minutes']:.1f} min)", flush=True)
    print("\nall requested datasets done; run with --aggregate to pool")


if __name__ == "__main__":
    main()
