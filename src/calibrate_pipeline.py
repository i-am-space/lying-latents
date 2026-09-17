"""Stage 1 — Calibrating the standard search-then-validate pipeline.

Measures the family-wise error rate (FWER) of the conventional workflow under
the global null by permuting SST-2 labels B times, destroying all label-latent
association while preserving the latent covariance exactly.

For each permutation:
  1. Score every retained latent (mean activation difference, AUROC, probe weight)
  2. Take the top-k by each scoring method
  3. Validate each candidate by zeroing its column and checking probe accuracy drop
  4. Record whether any candidate passes → contributes to FWER

The permutation null of the max score across latents yields a Westfall-Young
multiplicity correction usable independently of the knockoff framework.

Usage: python src/calibrate_pipeline.py --config config/default.yaml
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from common import cache_path, load_config, results_dir, rng_for


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def to_gpu(arr: np.ndarray, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr)).to(dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# scoring functions  (all operate on GPU tensors)
# ---------------------------------------------------------------------------

def score_mean_diff(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Unsigned mean activation difference between classes, per latent."""
    mask1 = y.bool()
    return (X[mask1].mean(dim=0) - X[~mask1].mean(dim=0)).abs()


def score_auroc(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Vectorised |AUC - 0.5| per latent via the rank-sum formula (GPU).

    Equivalent to the Mann-Whitney U statistic normalised to [0, 1].
    """
    n, p = X.shape
    n1 = int(y.sum().item())
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return torch.zeros(p, device=X.device)
    mask1 = y.bool()

    # Ranks: argsort twice gives rank array
    order = torch.argsort(X, dim=0, stable=True)               # (n, p)
    ranks = torch.empty_like(X)
    row_idx = torch.arange(1, n + 1, dtype=X.dtype, device=X.device).unsqueeze(1)
    ranks.scatter_(0, order, row_idx.expand_as(order))

    sum_ranks_pos = ranks[mask1].sum(dim=0)                    # (p,)
    auc = (sum_ranks_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return (auc - 0.5).abs()


# ---------------------------------------------------------------------------
# GPU logistic regression probe (L2, gradient descent)
# ---------------------------------------------------------------------------

def fit_probe_gpu(X_tr: torch.Tensor, y_tr: torch.Tensor,
                  C: float = 1.0, lr: float = 0.1, max_iter: int = 300,
                  seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit an L2-logistic probe on GPU with gradient descent.

    Returns (w, b, mean, std) for the StandardScaler equivalent.
    Uses L2 regularisation (C = 1/lambda), matching sklearn's default behaviour
    closely enough for ranking purposes.
    """
    # standardise
    mean = X_tr.mean(dim=0)
    std = X_tr.std(dim=0).clamp(min=1e-8)
    Xs = (X_tr - mean) / std

    n, p = Xs.shape
    w = torch.zeros(p, device=Xs.device, dtype=Xs.dtype)
    b = torch.zeros(1, device=Xs.device, dtype=Xs.dtype)
    lam = 1.0 / (C * n)

    for _ in range(max_iter):
        logits = Xs @ w + b                                    # (n,)
        probs = torch.sigmoid(logits)
        err = probs - y_tr                                     # (n,)
        grad_w = Xs.t() @ err / n + lam * w
        grad_b = err.mean()
        w = w - lr * grad_w
        b = b - lr * grad_b

    return w, b, mean, std


def probe_predict(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
                  mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    Xs = (X - mean) / std
    return (Xs @ w + b >= 0.0).long()


def score_probe_weight(w: torch.Tensor) -> torch.Tensor:
    """Absolute logistic-probe coefficients as a score per latent."""
    return w.abs()


# ---------------------------------------------------------------------------
# ablation (latent-zeroing) validation  — GPU
# ---------------------------------------------------------------------------

def ablation_check_gpu(X_te: torch.Tensor, y_te: torch.Tensor,
                        j: int,
                        w: torch.Tensor, b: torch.Tensor,
                        mean: torch.Tensor, std: torch.Tensor,
                        delta: float) -> bool:
    """Zero column j; return True if held-out accuracy drops by >= delta."""
    preds_full = probe_predict(X_te, w, b, mean, std)
    acc_full = (preds_full == y_te.long()).float().mean().item()

    X_abl = X_te.clone()
    X_abl[:, j] = 0.0
    preds_abl = probe_predict(X_abl, w, b, mean, std)
    acc_abl = (preds_abl == y_te.long()).float().mean().item()
    return float(acc_full - acc_abl) >= delta


# ---------------------------------------------------------------------------
# one permutation
# ---------------------------------------------------------------------------

def run_one_permutation(X: torch.Tensor, y_perm: torch.Tensor,
                        cfg_s1: dict, rng: np.random.Generator,
                        device: torch.device) -> dict:
    """Run the full search-then-validate workflow on (possibly permuted) labels.

    Returns per-method: full score array, max score, top-k indices,
    number certified, and whether any were certified.
    """
    n = len(y_perm)
    ntr = int(n * (1 - cfg_s1["probe_holdout_fraction"]))
    split = rng.permutation(n)
    tr, te = split[:ntr], split[ntr:]

    X_tr = X[tr];  y_tr = y_perm[tr]
    X_te = X[te];  y_te = y_perm[te]

    top_k = cfg_s1["top_k_certify"]
    delta = cfg_s1["ablation_delta"]
    methods = cfg_s1["scoring_methods"]

    # Fit one probe (used for probe_weight scoring AND ablation for all methods)
    w, b, mean, std = fit_probe_gpu(
        X_tr, y_tr, C=cfg_s1["probe_C"],
        max_iter=cfg_s1["probe_max_iter"],
        seed=int(rng.integers(2**31)),
    )

    results = {}
    for method in methods:
        if method == "mean_diff":
            scores = score_mean_diff(X_tr, y_tr)
        elif method == "auroc":
            scores = score_auroc(X_tr, y_tr)
        elif method == "probe_weight":
            scores = score_probe_weight(w)
        else:
            raise ValueError(f"unknown scoring method: {method!r}")

        max_score = float(scores.max().item())
        top_idx = torch.argsort(scores, descending=True)[:top_k].cpu().tolist()

        n_cert = 0
        for j in top_idx:
            if ablation_check_gpu(X_te, y_te, int(j), w, b, mean, std, delta):
                n_cert += 1

        results[method] = {
            "scores": scores.cpu().numpy(),
            "max_score": max_score,
            "top_k_indices": top_idx,
            "n_certified": n_cert,
            "any_certified": n_cert > 0,
        }

    return results


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(out, methods, rd):
    n_m = len(methods)

    # Fig 5: max-score null distribution with real-label max marked
    fig, axes = plt.subplots(1, n_m, figsize=(5 * n_m, 4), squeeze=False)
    for i, m in enumerate(methods):
        ax = axes[0, i]
        null = out["permutation"][m]["max_scores"]
        ax.hist(null, bins=40, color="#5b8c5a", edgecolor="white", alpha=0.8,
                label="permutation null")
        real_max = out["real_baseline"][m]["max_score"]
        ax.axvline(real_max, color="crimson", lw=2,
                   label=f"real labels ({real_max:.4f})")
        q95 = out["permutation"][m]["max_score_null_q95"]
        ax.axvline(q95, color="navy", ls="--", lw=1.5,
                   label=f"95th pctl ({q95:.4f})")
        wy_p = out["real_baseline"][m]["westfall_young_pvalue"]
        ax.set_title(f"{m}\nWY p = {wy_p:.4f}")
        ax.set_xlabel("max score across latents")
        ax.set_ylabel("permutations")
        ax.legend(fontsize=7)
    fig.suptitle(f"Westfall\u2013Young null distribution (B={out['n_permutations']})",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(rd / "fig5_permutation_null.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 6: FWER bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    fwers = [out["permutation"][m]["fwer"] for m in methods]
    colours = ["#3b6ea5", "#e07b39", "#5b8c5a"][:n_m]
    bars = ax.bar(methods, fwers, color=colours, edgecolor="white")
    for bar, v in zip(bars, fwers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("FWER")
    ax.set_title(f"Family-Wise Error Rate under global null\n"
                 f"(top-{out['top_k']}, \u03b4={out['ablation_delta']}, "
                 f"B={out['n_permutations']})")
    ax.set_ylim(0, min(1.15, max(fwers) * 1.3 + 0.05))
    fig.tight_layout()
    fig.savefig(rd / "fig6_fwer_by_method.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 1 — FWER calibration of the standard pipeline")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--limit-perms", type=int, default=None,
                    help="debug: cap number of permutations")
    ap.add_argument("--device", default=None,
                    help="torch device, e.g. 'cuda:1'. Default: cuda if available, else cpu")
    args = ap.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    cfg = load_config(args.config)
    rd = results_dir(cfg)
    cp = args.cache or cache_path(cfg)
    if not cp.exists() if hasattr(cp, "exists") else True:
        raise SystemExit(
            f"Cache not found at {cp}. Run src/cache_activations.py first.")
    d = np.load(cp, allow_pickle=True)
    prim = cfg["aggregation"]["primary"]
    X_np = d[f"X_{prim}"].astype(np.float32)
    y_np = d["labels"].astype(np.float32)
    n, p = X_np.shape
    print(f"loaded: n={n} p={p} aggregator={prim} hash={d['config_hash']}")
    print(f"class balance: {y_np.mean():.3f} (n1={int(y_np.sum())}, n0={n - int(y_np.sum())})")

    # Move full data to GPU once
    X = to_gpu(X_np, device)
    y = to_gpu(y_np, device)

    cfg_s1 = cfg["stage1"]
    B = args.limit_perms or cfg_s1["n_permutations"]
    methods = cfg_s1["scoring_methods"]
    rng = rng_for(cfg, "label_permutation")

    print(f"\n=== Stage 1: FWER calibration, B={B} permutations ===")
    print(f"methods: {methods}  top_k={cfg_s1['top_k_certify']}  "
          f"delta={cfg_s1['ablation_delta']}")

    # ---- Real-label baseline ------------------------------------------------
    print("\n--- real-label baseline ---")
    real = run_one_permutation(X, y, cfg_s1, rng, device)
    real_scores = {}
    for m in methods:
        real_scores[m] = real[m]["scores"]
        print(f"  [{m}] max_score={real[m]['max_score']:.6f}  "
              f"certified={real[m]['n_certified']}/{cfg_s1['top_k_certify']}")

    # ---- Permutation loop ---------------------------------------------------
    print(f"\n--- running {B} permutations ---")
    t0 = time.time()
    perm_results = {m: {"max_scores": [], "any_certified": [], "n_certified": []}
                    for m in methods}

    for b in range(B):
        y_perm_np = rng.permutation(y_np)
        y_perm = to_gpu(y_perm_np, device)
        res = run_one_permutation(X, y_perm, cfg_s1, rng, device)
        for m in methods:
            perm_results[m]["max_scores"].append(res[m]["max_score"])
            perm_results[m]["any_certified"].append(res[m]["any_certified"])
            perm_results[m]["n_certified"].append(res[m]["n_certified"])

        if (b + 1) % 50 == 0 or b == 0:
            elapsed = time.time() - t0
            eta = elapsed / (b + 1) * (B - b - 1)
            fwer_so_far = {m: np.mean(perm_results[m]["any_certified"])
                           for m in methods}
            print(f"  perm {b+1:>4}/{B}  elapsed={elapsed/60:.1f}min  "
                  f"ETA={eta/60:.1f}min  FWER: "
                  + "  ".join(f"{m}={fwer_so_far[m]:.3f}" for m in methods),
                  flush=True)

    elapsed = time.time() - t0
    print(f"\ncompleted {B} permutations in {elapsed/60:.1f} min")

    # ---- Aggregate results --------------------------------------------------
    out = {
        "config_hash": str(d["config_hash"]),
        "master_seed": cfg["master_seed"],
        "aggregator": prim,
        "n": n, "p": p,
        "n_permutations": B,
        "scoring_methods": methods,
        "top_k": cfg_s1["top_k_certify"],
        "ablation_delta": cfg_s1["ablation_delta"],
        "elapsed_minutes": round(elapsed / 60, 1),
        "real_baseline": {},
        "permutation": {},
    }

    for m in methods:
        max_null = np.array(perm_results[m]["max_scores"])
        fwer = float(np.mean(perm_results[m]["any_certified"]))
        real_max = real[m]["max_score"]

        # Westfall-Young p-value for the real-label max score
        wy_pvalue = float((1 + (max_null >= real_max).sum()) / (1 + B))

        # Per-latent WY-adjusted p-values (single-step)
        rs = real_scores[m]
        adjusted_p = (1 + (max_null[:, None] >= rs[None, :]).sum(axis=0)
                      ).astype(np.float64) / (1 + B)

        wy_q95 = float(np.quantile(max_null, 0.95))
        wy_q99 = float(np.quantile(max_null, 0.99))

        out["real_baseline"][m] = {
            "max_score": real_max,
            "n_certified": real[m]["n_certified"],
            "top_k_indices": real[m]["top_k_indices"],
            "westfall_young_pvalue": wy_pvalue,
            "n_surviving_wy_0.05": int((adjusted_p <= 0.05).sum()),
            "n_surviving_wy_0.01": int((adjusted_p <= 0.01).sum()),
        }

        out["permutation"][m] = {
            "fwer": fwer,
            "mean_n_certified": float(np.mean(perm_results[m]["n_certified"])),
            "max_score_null_mean": float(max_null.mean()),
            "max_score_null_sd": float(max_null.std()),
            "max_score_null_q95": wy_q95,
            "max_score_null_q99": wy_q99,
            "max_scores": max_null.tolist(),
        }

        print(f"\n[{m}]")
        print(f"  FWER = {fwer:.4f}  "
              f"(fraction of permutations with >= 1 certification)")
        print(f"  mean certified per perm = "
              f"{np.mean(perm_results[m]['n_certified']):.2f}")
        print(f"  real max score = {real_max:.6f}  |  "
              f"null 95th = {wy_q95:.6f}")
        print(f"  WY p-value (max) = {wy_pvalue:.4f}")
        print(f"  latents surviving WY at alpha=0.05: "
              f"{out['real_baseline'][m]['n_surviving_wy_0.05']}")
        print(f"  latents surviving WY at alpha=0.01: "
              f"{out['real_baseline'][m]['n_surviving_wy_0.01']}")

    # ---- Save ---------------------------------------------------------------
    (rd / "stage1_calibration.json").write_text(
        json.dumps(out, indent=2, default=float))

    # Save per-latent adjusted p-values as npz (too large for JSON)
    np.savez(rd / "stage1_adjusted_pvalues.npz",
             **{m: (1 + (np.array(perm_results[m]["max_scores"])[:, None]
                         >= real_scores[m][None, :]).sum(axis=0)
                    ).astype(np.float64) / (1 + B)
                for m in methods})

    make_figures(out, methods, rd)
    print(f"\nwrote {rd}/stage1_calibration.json, "
          f"{rd}/stage1_adjusted_pvalues.npz, and 2 figures")


if __name__ == "__main__":
    main()
