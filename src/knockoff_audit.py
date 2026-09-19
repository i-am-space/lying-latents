"""Stage 2, Deliverable 3 — exchangeability audit of Gaussian second-order knockoffs.

Order of operations is deliberate: the diagnostic suite is validated on synthetic
Gaussian data (where second-order knockoffs are exactly valid, so every diagnostic
MUST return null) before it is ever pointed at real latents.

Usage: python src/knockoff_audit.py --config config/default.yaml
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from common import cache_path, load_config, results_dir, rng_for

# ---------------------------------------------------------------------------
# knockoff construction
# ---------------------------------------------------------------------------

def standardise(X: np.ndarray):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (X - mu) / sd, mu, sd


def estimate_cov(Z: np.ndarray, estimator: str):
    if estimator == "sample":
        n = Z.shape[0]
        Zc = Z - Z.mean(axis=0)
        return (Zc.T @ Zc) / (n - 1)
    if estimator == "ledoit_wolf":
        return LedoitWolf(assume_centered=False).fit(Z).covariance_
    raise ValueError(estimator)


def make_knockoffs(Z, Sigma, method, seed):
    """Second-order Gaussian knockoffs via knockpy. Never hand-rolled."""
    from knockpy.knockoffs import GaussianSampler
    np.random.seed(seed)                      # knockpy samples from the global stream
    s = GaussianSampler(Z, mu=Z.mean(axis=0), Sigma=Sigma, method=method)
    Zk = s.sample_knockoffs()
    return Zk, np.asarray(s.fetch_S())


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

def diag_zero_mass(X_raw, Xk_raw):
    """(a) Accuracy of the trivial classifier 1{x = 0} at telling real from knockoff.
    Under exchangeability this is 0.5. Requires no test, only the definition."""
    p0 = (X_raw == 0.0).mean(axis=0)
    q0 = (Xk_raw == 0.0).mean(axis=0)
    acc = 0.5 * (p0 + (1.0 - q0))
    return {"p_zero_real": p0, "p_zero_knockoff": q0, "accuracy": acc}


def _swap(Z, p, S_idx):
    if len(S_idx) == 0:
        return Z
    Zs = Z.copy()
    Zs[:, S_idx] = Z[:, S_idx + p]
    Zs[:, S_idx + p] = Z[:, S_idx]
    return Zs


def classifier_auc(ZA, ZB, rng, max_iter, holdout):
    """Held-out AUC of a gradient-boosted classifier separating the two samples."""
    Z = np.vstack([ZA, ZB]).astype(np.float32)
    y = np.concatenate([np.zeros(len(ZA)), np.ones(len(ZB))])
    perm = rng.permutation(len(y))
    Z, y = Z[perm], y[perm]
    ntr = int(len(y) * (1 - holdout))
    clf = HistGradientBoostingClassifier(max_iter=max_iter,
                                         random_state=int(rng.integers(2**31)))
    clf.fit(Z[:ntr], y[:ntr])
    return float(roc_auc_score(y[ntr:], clf.predict_proba(Z[ntr:])[:, 1]))


def mmd_test(ZA, ZB, rng, m, n_perm):
    """Unbiased MMD^2 with a Gaussian kernel, median-heuristic bandwidth,
    permutation p-value. Kernel is computed once and reused across permutations."""
    a = ZA[rng.choice(len(ZA), min(m, len(ZA)), replace=False)]
    b = ZB[rng.choice(len(ZB), min(m, len(ZB)), replace=False)]
    W = np.vstack([a, b]).astype(np.float64)
    na = len(a)
    sq = (W * W).sum(1)
    D = sq[:, None] + sq[None, :] - 2 * (W @ W.T)
    np.maximum(D, 0, out=D)
    med = np.median(D[np.triu_indices_from(D, k=1)])
    K = np.exp(-D / (med if med > 0 else 1.0))
    np.fill_diagonal(K, 0.0)

    # Vectorised permutation null: each permutation's MMD^2 is a pair of
    # quadratic forms in its group-indicator vector, so all n_perm permutations
    # become one matrix product instead of n_perm gathers.
    N = len(W)
    U = np.zeros((N, n_perm + 1))
    U[:na, 0] = 1.0
    for i in range(n_perm):
        U[rng.permutation(N)[:na], i + 1] = 1.0
    V = 1.0 - U
    KU, KV = K @ U, K @ V
    saa = (U * KU).sum(0) / (na * (na - 1))
    nb = N - na
    sbb = (V * KV).sum(0) / (nb * (nb - 1))
    sab = (U * KV).sum(0) / (na * nb)
    vals = saa + sbb - 2 * sab
    obs, null = vals[0], vals[1:]
    return {"mmd2": float(obs), "p_value": float((1 + (null >= obs).sum()) / (1 + n_perm)),
            "null_mean": float(null.mean()), "null_sd": float(null.std())}


def moment_table(Z, Zk):
    """(c) Moments 1-2 are matched by construction; 3-4 and the tails are not."""
    def stat(A):
        return {"mean": A.mean(0), "var": A.var(0), "skew": stats.skew(A, axis=0),
                "kurtosis": stats.kurtosis(A, axis=0),
                "q90": np.quantile(A, 0.90, axis=0), "q99": np.quantile(A, 0.99, axis=0),
                "q999": np.quantile(A, 0.999, axis=0), "max": A.max(0)}
    r, k = stat(Z), stat(Zk)
    return {m: {"real_median": float(np.median(r[m])),
                "knockoff_median": float(np.median(k[m])),
                "median_abs_gap": float(np.median(np.abs(r[m] - k[m]))),
                "median_signed_gap": float(np.median(r[m] - k[m]))}
            for m in r}, r, k


def swap_sweep(Z, Zk, cfg, rng, sizes, replicates):
    """(b) Swap two-sample tests as a function of |S|. |S| = 0 is the exact null."""
    n, p = Z.shape
    ZZ = np.hstack([Z, Zk])
    d = cfg["diagnostics"]
    out = []
    for size in sizes:
        k = p if size == "all" else int(size)
        reps = 1 if k in (0, p) else replicates
        for rep in range(reps):
            S_idx = (np.array([], dtype=int) if k == 0
                     else np.sort(rng.choice(p, k, replace=False)))
            perm = rng.permutation(n)
            A, B = perm[: n // 2], perm[n // 2:]
            ZA = ZZ[A]
            ZB = _swap(ZZ[B], p, S_idx)
            nc = min(d["n_clf_rows"] // 2, len(ZA), len(ZB))
            auc = classifier_auc(ZA[:nc], ZB[:nc], rng, d["classifier_max_iter"],
                                 d["holdout_fraction"])
            mmd = mmd_test(ZA, ZB, rng, d["mmd_subsample"], d["mmd_permutations"])
            out.append({"swap_size": k, "rep": rep, "auc": auc, **mmd})
            print(f"    |S|={k:<5} rep{rep}  AUC={auc:.4f}  MMD2={mmd['mmd2']:.3e}  "
                  f"p={mmd['p_value']:.4f}", flush=True)
    return out


def label_permutation_null(Z, Zk, cfg, rng, n_perm):
    """Null band for the classifier AUC: shuffle the group labels and refit."""
    n, p = Z.shape
    ZZ = np.hstack([Z, Zk])
    d = cfg["diagnostics"]
    nc = min(d["n_clf_rows"], n)
    sub = ZZ[rng.permutation(n)[:nc]]
    aucs = []
    for _ in range(n_perm):
        y = rng.permutation(np.r_[np.zeros(nc // 2), np.ones(nc - nc // 2)])
        aucs.append(classifier_auc(sub[y == 0], sub[y == 1], rng,
                                   d["classifier_max_iter"], d["holdout_fraction"]))
    return aucs


# ---------------------------------------------------------------------------
# suite runner
# ---------------------------------------------------------------------------

def run_suite(name, Z, Zk, cfg, rng, sizes, replicates, raw=None):
    print(f"  [{name}] suite")
    res = {"name": name, "n": int(Z.shape[0]), "p": int(Z.shape[1])}
    res["knockoff_self_correlation"] = float(np.mean(
        [np.corrcoef(Z[:, j], Zk[:, j])[0, 1] for j in range(0, Z.shape[1], max(1, Z.shape[1] // 200))]))
    mom, _, _ = moment_table(Z, Zk)
    res["moments"] = mom
    res["swap"] = swap_sweep(Z, Zk, cfg, rng, sizes, replicates)
    if raw is not None:
        X_raw, Xk_raw = raw
        zm = diag_zero_mass(X_raw, Xk_raw)
        res["zero_mass"] = {
            "median_accuracy": float(np.median(zm["accuracy"])),
            "mean_accuracy": float(zm["accuracy"].mean()),
            "min_accuracy": float(zm["accuracy"].min()),
            "max_accuracy": float(zm["accuracy"].max()),
            "median_p_zero_real": float(np.median(zm["p_zero_real"])),
            "max_p_zero_knockoff": float(zm["p_zero_knockoff"].max()),
            "frac_latents_accuracy_above_0.55": float((zm["accuracy"] > 0.55).mean()),
        }
        res["_zero_mass_arrays"] = zm
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--stage", default="all", choices=["all", "calibration", "real"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    rd = results_dir(cfg)
    d = np.load(args.cache or cache_path(cfg), allow_pickle=True)
    prim = cfg["aggregation"]["primary"]
    X_raw = d[f"X_{prim}"].astype(np.float64)
    n, p = X_raw.shape
    print(f"real latents: n={n} p={p}  aggregator={prim}  hash={d['config_hash']}")

    dg = cfg["diagnostics"]
    sizes = dg["swap_sizes"]
    reps = dg["swap_replicates"]
    seed_k = int(rng_for(cfg, "knockoff_sampler").integers(2**31))
    out = {"config_hash": str(d["config_hash"]), "master_seed": cfg["master_seed"],
           "aggregator": prim, "n": n, "p": p, "knockoff_seed": seed_k}

    # ---- covariance -------------------------------------------------------
    Z, mu, sd = standardise(X_raw)
    Sig_s = estimate_cov(Z, "sample")
    Sig_lw = estimate_cov(Z, "ledoit_wolf")
    ev = np.linalg.eigvalsh(Sig_s)
    cond = float(ev[-1] / max(ev[0], 1e-300))
    ill = cond > float(cfg["covariance"]["ill_conditioned_cond_number"])
    est = "ledoit_wolf" if ill else "sample"
    Sigma = Sig_lw if ill else Sig_s
    out["covariance"] = {"cond_sample": cond, "eig_min_sample": float(ev[0]),
                         "ill_conditioned": bool(ill), "primary_estimator": est,
                         "cond_ledoit_wolf": float(np.linalg.cond(Sig_lw)),
                         "standardised": True}
    print(f"covariance: cond(sample)={cond:.4g} -> primary estimator = {est}")

    # ---- 7.2 harness validation (MUST run first) --------------------------
    if args.stage in ("all", "calibration"):
        print("\n=== 7.2 harness validation on synthetic Gaussian (must return null) ===")
        rs = rng_for(cfg, "synthetic_null")
        nrows = dg["synthetic_n_rows"]
        L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(p))
        cal = []
        for r in range(dg["synthetic_replicates"]):
            G = rs.standard_normal((nrows, p)) @ L.T
            # oracle: knockoffs built from the KNOWN true covariance -> exactly valid
            Gk, _ = make_knockoffs(G, Sigma, cfg["knockoffs"]["s_method"], seed_k + 100 + r)
            print(f"  replicate {r}")
            cal.append(run_suite(f"synthetic_oracle_rep{r}", G, Gk, cfg, rs, [0, "all"], 1))
        aucs = [s["auc"] for c in cal for s in c["swap"] if s["swap_size"] == p]
        pvals = [s["p_value"] for c in cal for s in c["swap"] if s["swap_size"] == p]
        ks = stats.kstest(pvals, "uniform")
        gaps = [max(abs(c["moments"][m]["median_signed_gap"]) for m in ("skew", "kurtosis"))
                for c in cal]
        ci = (float(np.mean(aucs) - 1.96 * np.std(aucs) / np.sqrt(len(aucs))),
              float(np.mean(aucs) + 1.96 * np.std(aucs) / np.sqrt(len(aucs))))
        passed = (ci[0] <= 0.5 <= ci[1]) and ks.pvalue >= 0.05 and max(gaps) <= 0.10
        out["calibration"] = {
            "replicates": dg["synthetic_replicates"], "n_rows": nrows,
            "auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
            "auc_ci95": ci, "auc_values": aucs,
            "mmd_pvalues": pvals, "mmd_uniformity_ks_p": float(ks.pvalue),
            "max_abs_moment_gap": float(max(gaps)),
            "PASSED": bool(passed),
            "criteria": {"auc_ci_contains_0.5": bool(ci[0] <= 0.5 <= ci[1]),
                         "mmd_uniform_ks_p>=0.05": bool(ks.pvalue >= 0.05),
                         "moment_gap<=0.10": bool(max(gaps) <= 0.10)},
        }
        print(f"\nCALIBRATION: AUC {np.mean(aucs):.4f} CI95 {ci}  "
              f"MMD-p KS={ks.pvalue:.3f}  moment gap {max(gaps):.4f}  -> "
              f"{'PASS' if passed else 'FAIL'}")
        (rd / "stage2_audit.json").write_text(json.dumps(out, indent=2, default=float))
        if not passed:
            raise SystemExit("Harness validation FAILED. Fix the diagnostics before "
                             "running on real latents (preregistration §7.2).")

    if args.stage == "calibration":
        return

    # ---- control: same n, covariance ESTIMATED from the synthetic sample ---
    print("\n=== control: Gaussian data, covariance estimated at the real sample size ===")
    rs2 = rng_for(cfg, "synthetic_null")
    L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(p))
    G = rs2.standard_normal((n, p)) @ L.T
    Gz, _, _ = standardise(G)
    Sig_G = estimate_cov(Gz, est)
    Gk, _ = make_knockoffs(Gz, Sig_G, cfg["knockoffs"]["s_method"], seed_k + 7)
    out["gaussian_control"] = run_suite("gaussian_estimated_cov", Gz, Gk, cfg, rs2,
                                        sizes, reps)
    out["gaussian_control"].pop("_zero_mass_arrays", None)

    # ---- real latents -----------------------------------------------------
    print("\n=== 7.3 diagnostics on real SAE latents ===")
    rr = rng_for(cfg, "row_partition")
    t0 = time.time()
    Zk, S_mat = make_knockoffs(Z, Sigma, cfg["knockoffs"]["s_method"], seed_k)
    print(f"knockoffs generated in {time.time()-t0:.1f}s; "
          f"s = {np.diag(S_mat)[0]:.4f} (equicorrelated)")
    Xk_raw = Zk * sd + mu                    # same affine map as X: swap-equivariant
    out["s_value"] = float(np.diag(S_mat)[0])
    real = run_suite("real_latents", Z, Zk, cfg, rr, sizes, reps, raw=(X_raw, Xk_raw))
    zm_arrays = real.pop("_zero_mass_arrays")
    out["real"] = real

    # null band for the classifier AUC
    out["real"]["label_permutation_auc"] = label_permutation_null(
        Z, Zk, cfg, rr, dg["n_label_permutations"])

    # ---- the reference's unstandardised construction, for comparison ------
    print("\n=== reference-style construction: no standardisation ===")
    Sig_raw = estimate_cov(X_raw, "sample")
    ev_raw = np.linalg.eigvalsh(Sig_raw)
    s_ref = min(2 * max(ev_raw[0], 1e-6), 0.99)     # their rule, verbatim
    out["reference_style"] = {
        "eig_min_raw_cov": float(ev_raw[0]),
        "eig_max_raw_cov": float(ev_raw[-1]),
        "median_raw_variance": float(np.median(np.diag(Sig_raw))),
        "s_value_their_rule": float(s_ref),
        "s_over_median_variance": float(s_ref / np.median(np.diag(Sig_raw))),
    }
    print(f"  their s = {s_ref:.4g} vs median latent variance "
          f"{np.median(np.diag(Sig_raw)):.4g}  -> ratio "
          f"{out['reference_style']['s_over_median_variance']:.3g}")

    (rd / "stage2_audit.json").write_text(json.dumps(out, indent=2, default=float))
    np.savez(rd / "stage2_zero_mass.npz", **zm_arrays)
    make_figures(out, zm_arrays, rd, p)
    print(f"\nwrote {rd}/stage2_audit.json + figures")


def make_figures(out, zm, rd, p):
    cal = out.get("calibration")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    if cal:
        ax.hist(cal["auc_values"], bins=10, color="#5b8c5a", edgecolor="white")
        ax.axvline(0.5, color="k", ls="--", label="exchangeable (0.5)")
        ax.set_title(f"Calibration: synthetic Gaussian\n|S|=all, {cal['replicates']} reps "
                     f"-> {'PASS' if cal['PASSED'] else 'FAIL'}")
        ax.set_xlabel("held-out AUC"); ax.legend(fontsize=8)

    ax = axes[1]
    for key, colour, lab in [("real", "crimson", "real latents"),
                             ("gaussian_control", "#5b8c5a", "Gaussian control")]:
        if key not in out:
            continue
        rows = out[key]["swap"]
        xs = sorted({r["swap_size"] for r in rows})
        mu_ = [np.mean([r["auc"] for r in rows if r["swap_size"] == s]) for s in xs]
        sdv = [np.std([r["auc"] for r in rows if r["swap_size"] == s]) for s in xs]
        ax.errorbar([max(s, 0.5) for s in xs], mu_, yerr=sdv, marker="o",
                    color=colour, label=lab, capsize=3)
    band = out["real"].get("label_permutation_auc")
    if band:
        ax.axhspan(np.quantile(band, 0.025), np.quantile(band, 0.975),
                   color="grey", alpha=0.3, label="label-permutation null (95%)")
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("|S| (swap subset size; 0.5 = no swap)")
    ax.set_ylabel("held-out AUC"); ax.set_title("Swap two-sample test"); ax.legend(fontsize=8)

    ax = axes[2]
    ax.hist(zm["accuracy"], bins=50, color="crimson", edgecolor="white")
    ax.axvline(0.5, color="k", ls="--", label="exchangeable (0.5)")
    ax.axvline(np.median(zm["accuracy"]), color="navy", ls="-",
               label=f"median {np.median(zm['accuracy']):.3f}")
    ax.set_xlabel(r"accuracy of $\mathbb{1}\{x=0\}$"); ax.set_ylabel("latents")
    ax.set_title("Zero-mass discrepancy"); ax.legend(fontsize=8)

    fig.tight_layout(); fig.savefig(rd / "fig4_audit.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
