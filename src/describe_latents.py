"""Stage 2, Deliverable 2 — distributional summary of the cached latents.

Runs the pre-registered premise check: does Pr(X_j = 0) have enough mass at
*exactly* zero for the atom argument to stand?

Usage: python src/describe_latents.py --config config/default.yaml
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from common import cache_path, load_config, results_dir

AGGS = ("mean", "max", "last")


def per_latent_table(X: np.ndarray) -> dict:
    n, p = X.shape
    zero = X == 0.0                                   # exact zeros, not near-zeros
    p0 = zero.mean(axis=0)
    fire = 1.0 - p0
    out = {
        "p_zero": p0,
        "firing_rate": fire,
        "nz_mean": np.zeros(p),
        "nz_median": np.zeros(p),
        "nz_skew": np.full(p, np.nan),
        "nz_kurtosis": np.full(p, np.nan),
        "q90": np.zeros(p), "q99": np.zeros(p), "q999": np.zeros(p),
        "max": X.max(axis=0),
    }
    for j in range(p):
        v = X[~zero[:, j], j]
        if v.size == 0:
            continue
        out["nz_mean"][j] = v.mean()
        out["nz_median"][j] = np.median(v)
        if v.size > 3 and v.std() > 0:
            out["nz_skew"][j] = stats.skew(v)
            out["nz_kurtosis"][j] = stats.kurtosis(v)      # excess
        out["q90"][j], out["q99"][j], out["q999"][j] = np.quantile(v, [0.9, 0.99, 0.999])
    return out


def covariance_summary(X: np.ndarray, label: str) -> dict:
    Xc = X - X.mean(axis=0)
    sd = Xc.std(axis=0)
    Z = Xc / np.where(sd > 0, sd, 1.0)                 # correlation scale
    n = X.shape[0]
    C = (Z.T @ Z) / (n - 1)
    ev = np.linalg.eigvalsh(C)[::-1]
    ev_pos = np.clip(ev, 0, None)
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(Z)
    ev_lw = np.linalg.eigvalsh(lw.covariance_)[::-1]
    return {
        "aggregator": label,
        "n": int(n), "p": int(X.shape[1]),
        "n_over_p": float(n / X.shape[1]),
        "eig_max": float(ev[0]), "eig_min": float(ev[-1]),
        "condition_number": float(ev[0] / ev[-1]) if ev[-1] > 0 else float("inf"),
        "cond_number_clipped": float(ev[0] / max(ev[-1], 1e-12)),
        "n_nonpositive_eigs": int((ev <= 0).sum()),
        "frac_var_top10": float(ev_pos[:10].sum() / ev_pos.sum()),
        "frac_var_top50": float(ev_pos[:50].sum() / ev_pos.sum()),
        "ledoit_wolf_shrinkage": float(lw.shrinkage_),
        "ledoit_wolf_condition_number": float(ev_lw[0] / ev_lw[-1]),
        "eigenvalues_top100": ev[:100].tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rd = results_dir(cfg)
    path = args.cache or cache_path(cfg)
    d = np.load(path, allow_pickle=True)
    print(f"cache: {path}  hash={d['config_hash']}  "
          f"EV={float(d['sae_explained_variance']):.4f}  L0={float(d['sae_mean_l0']):.1f}")

    primary = cfg["aggregation"]["primary"]
    summary = {
        "config_hash": str(d["config_hash"]),
        "master_seed": int(d["master_seed"]),
        "sae_explained_variance": float(d["sae_explained_variance"]),
        "sae_mean_l0": float(d["sae_mean_l0"]),
        "n_sentences": int(d["labels"].shape[0]),
        "p_retained": int(d["retained_idx"].shape[0]),
        "n_tokens_mean": float(d["n_tokens"].mean()),
        "primary_aggregator": primary,
        "aggregators": {},
    }

    tables = {}
    for a in AGGS:
        X = d[f"X_{a}"].astype(np.float64)
        t = per_latent_table(X)
        tables[a] = t
        cov = covariance_summary(X, a)
        med_p0 = float(np.median(t["p_zero"]))
        summary["aggregators"][a] = {
            "median_p_zero": med_p0,
            "mean_p_zero": float(t["p_zero"].mean()),
            "p_zero_quantiles": {q: float(np.quantile(t["p_zero"], float(q)))
                                 for q in ("0.05", "0.25", "0.5", "0.75", "0.95")},
            "frac_latents_p_zero_above_0.95": float((t["p_zero"] > 0.95).mean()),
            "frac_latents_p_zero_above_0.8": float((t["p_zero"] > 0.8).mean()),
            "median_firing_rate": float(np.median(t["firing_rate"])),
            "median_nz_skew": float(np.nanmedian(t["nz_skew"])),
            "median_nz_kurtosis": float(np.nanmedian(t["nz_kurtosis"])),
            "median_q99_over_q90": float(np.median(t["q99"] / np.where(t["q90"] > 0, t["q90"], np.nan))),
            "covariance": cov,
        }
        print(f"[{a:4s}] median Pr(X=0)={med_p0:.4f}  "
              f"frac>0.95={(t['p_zero'] > 0.95).mean():.3f}  "
              f"cond={cov['condition_number']:.3g}  LW shrink={cov['ledoit_wolf_shrinkage']:.4f}")

    # ---- pre-registered premise check ------------------------------------
    m = summary["aggregators"][primary]["median_p_zero"]
    if m >= 0.80:
        verdict, action = "PREMISE_HOLDS", "proceed to the exchangeability audit"
    elif m >= 0.50:
        verdict, action = "PREMISE_WEAKENED", "proceed; report realised value in place of the >0.95 claim"
    else:
        verdict, action = "PREMISE_FAILS", "stop; the atom argument needs rewriting around the aggregator"
    summary["premise_check"] = {
        "statistic": "median_zero_mass_retained_primary",
        "aggregator": primary, "value": m, "verdict": verdict, "action": action,
    }
    print(f"\nPREMISE CHECK [{primary}]: median Pr(X_j=0) = {m:.4f} -> {verdict}\n  {action}")

    cp = summary["aggregators"][primary]["covariance"]
    ill = cp["condition_number"] > float(cfg["covariance"]["ill_conditioned_cond_number"])
    summary["covariance_verdict"] = {
        "ill_conditioned": bool(ill),
        "threshold": float(cfg["covariance"]["ill_conditioned_cond_number"]),
        "primary_estimator": "ledoit_wolf" if ill else "sample",
    }
    print(f"COVARIANCE: cond={cp['condition_number']:.4g} -> "
          f"{'ILL-CONDITIONED, Ledoit-Wolf primary' if ill else 'well-conditioned, sample primary'}")

    (rd / "stage2_describe.json").write_text(json.dumps(summary, indent=2))
    np.savez(rd / "stage2_latent_table.npz",
             **{f"{a}__{k}": v for a, t in tables.items() for k, v in t.items()})

    # ---- figures ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, a in zip(axes, AGGS):
        ax.hist(tables[a]["p_zero"], bins=50, range=(0, 1), color="#3b6ea5", edgecolor="white")
        ax.axvline(np.median(tables[a]["p_zero"]), color="crimson", ls="--",
                   label=f"median {np.median(tables[a]['p_zero']):.3f}")
        ax.set_title(f"{a}-pooled" + ("  (primary)" if a == primary else ""))
        ax.set_xlabel(r"$\Pr(X_j = 0)$"); ax.legend(fontsize=8)
    axes[0].set_ylabel("latents")
    fig.suptitle(f"Zero-mass across {summary['p_retained']} retained latents "
                 f"(SST-2 train, n={summary['n_sentences']})")
    fig.tight_layout(); fig.savefig(rd / "fig1_zero_mass_hist.png", dpi=150); plt.close(fig)

    # marginals for 6 latents spanning the firing-rate range, primary aggregator
    X = d[f"X_{primary}"].astype(np.float64)
    fr = tables[primary]["firing_rate"]
    picks = [int(np.argsort(fr)[int(q * (len(fr) - 1))]) for q in (0.02, 0.2, 0.4, 0.6, 0.8, 0.98)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, j in zip(axes.ravel(), picks):
        v = X[:, j]; nz = v[v > 0]
        ax.hist(nz, bins=60, color="#3b6ea5", label=f"nonzero part (n={nz.size})")
        ax.bar([0], [(v == 0).sum()], width=(nz.max() - 0) / 60 if nz.size else 1,
               color="crimson", label=f"atom at 0: {(v == 0).mean():.3f}")
        ax.set_yscale("log"); ax.set_title(f"latent {int(d['retained_idx'][j])}  "
                                           f"firing rate {fr[j]:.3f}", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f"Marginal law of six retained latents, {primary}-pooled "
                 "(log counts; zero-atom in red)")
    fig.tight_layout(); fig.savefig(rd / "fig2_marginals.png", dpi=150); plt.close(fig)

    # eigenvalue spectrum
    fig, ax = plt.subplots(figsize=(6, 4))
    for a in AGGS:
        ev = summary["aggregators"][a]["covariance"]["eigenvalues_top100"]
        ax.semilogy(np.arange(1, len(ev) + 1), np.clip(ev, 1e-12, None), label=a)
    ax.set_xlabel("component"); ax.set_ylabel("eigenvalue (correlation scale)")
    ax.set_title("Eigenvalue spectrum, top 100"); ax.legend()
    fig.tight_layout(); fig.savefig(rd / "fig3_eigenspectrum.png", dpi=150); plt.close(fig)
    print(f"\nwrote {rd}/stage2_describe.json and 3 figures")


if __name__ == "__main__":
    main()
