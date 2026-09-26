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

v2 additions (post-hoc; see config/preregistration.yaml stage1_amendment_1). The v1
computation above is unchanged and consumes the RNG identically, so v1's numbers
must reproduce. Added on top of it, deterministically:
  * the naive per-latent test (Welch |t|) under the null: how many latents pass an
    uncorrected 5% test, versus Bonferroni and Westfall-Young;
  * label-free behavioural effects of ablation (prediction flip rate, mean |dp|;
    top-k jointly and best single latent), on the held-out AND the training split
    (the training-split variant is the conventional, same-data pipeline);
  * null curves FWER(tau) for those effects, with the real-label value marked;
  * a planted-signal power check: the certification step must fire when a real
    signal exists, otherwise its FWER is uninformative.

Usage: python src/calibrate_pipeline.py --config config/default.yaml
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from common import cache_path, load_config, results_dir, rng_for
from planted_fdr import generate_planted_labels

EFFECT_KEYS = ("single_flip_max", "single_dp_max", "acc_drop_single_max",
               "joint_flip", "joint_dp", "acc_drop_joint")


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


def score_auroc(X: torch.Tensor, y: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """Vectorised |AUC - 0.5| per latent via the rank-sum formula (GPU).

    Equivalent to the Mann-Whitney U statistic normalised to [0, 1]. Latents are
    processed `chunk` columns at a time: columns are independent, so the result is
    the same, but the (n, p) int64 argsort output and rank matrix (~1.2 GB at
    n=47k, p=2048, plus sort workspace) never exist at once.
    """
    n, p = X.shape
    n1 = int(y.sum().item())
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return torch.zeros(p, device=X.device)
    mask1 = y.bool()
    row_idx = torch.arange(1, n + 1, dtype=X.dtype, device=X.device).unsqueeze(1)

    auc = torch.empty(p, device=X.device, dtype=X.dtype)
    for s in range(0, p, chunk):
        Xc = X[:, s:s + chunk]
        # Ranks: argsort then scatter gives the rank array (ties by stable order)
        order = torch.argsort(Xc, dim=0, stable=True)          # (n, chunk)
        ranks = torch.empty_like(Xc)
        ranks.scatter_(0, order, row_idx.expand_as(order))
        sum_ranks_pos = ranks[mask1].sum(dim=0)                # (chunk,)
        auc[s:s + chunk] = (sum_ranks_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return (auc - 0.5).abs()


def welch_abs_t(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """|Welch t| per latent: the naive per-latent test a practitioner would run.
    At n ~ 5e4 it is ~N(0, 1) under a permuted label, so no tie/normality fix is needed."""
    m1 = y.bool()
    n1 = int(m1.sum().item())
    n0 = X.shape[0] - n1
    if n1 < 2 or n0 < 2:
        return torch.zeros(X.shape[1], device=X.device)
    X1, X0 = X[m1], X[~m1]
    se = torch.sqrt(X1.var(dim=0, unbiased=True) / n1
                    + X0.var(dim=0, unbiased=True) / n0).clamp(min=1e-12)
    return ((X1.mean(dim=0) - X0.mean(dim=0)) / se).abs()


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


def behaviour_effects(Xs: torch.Tensor, y: torch.Tensor, logits: torch.Tensor,
                      w: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                      top_idx: list[int], delta: float) -> dict:
    """Effects of ablating the top-k latents (raw activation set to 0, as in
    ablation_check_gpu) on the probe, evaluated on the rows of Xs.

    Xs is the already-standardised evaluation matrix and logits its probe logits.
    Ablating latent j moves its standardised value to (0 - mean_j)/std_j, which
    changes the logit by w_j * (that - Xs_j); no matrix clone per candidate.

    Label-free effects (flip rate, mean |dp|) are the 'behavioural change' of the
    probe itself; accuracy drops depend on the labels. Each is given for the best
    SINGLE latent among the top-k and for all top-k ablated JOINTLY.
    """
    idx = torch.as_tensor(top_idx, dtype=torch.long, device=Xs.device)
    yl = y.long()
    pred0 = logits >= 0.0
    p0 = torch.sigmoid(logits)
    acc0 = (pred0.long() == yl).float().mean()

    z0 = -mean[idx] / std[idx]                                          # (k,)
    dj = w[idx].unsqueeze(0) * (z0.unsqueeze(0) - Xs[:, idx])           # (n, k)
    lj = logits.unsqueeze(1) + dj                                       # each latent alone
    lJ = logits + dj.sum(dim=1)                                         # all top-k together

    pred_j = lj >= 0.0
    flip_j = (pred_j != pred0.unsqueeze(1)).float().mean(dim=0)         # (k,)
    dp_j = (torch.sigmoid(lj) - p0.unsqueeze(1)).abs().mean(dim=0)      # (k,)
    drop_j = acc0 - (pred_j.long() == yl.unsqueeze(1)).float().mean(dim=0)

    pred_J = lJ >= 0.0
    flip_J = (pred_J != pred0).float().mean()
    dp_J = (torch.sigmoid(lJ) - p0).abs().mean()
    drop_J = acc0 - (pred_J.long() == yl).float().mean()

    vals = torch.stack([flip_j.max(), dp_j.max(), drop_j.max(), flip_J, dp_J, drop_J,
                        (drop_j >= delta).sum().float()]).tolist()
    out = dict(zip(EFFECT_KEYS, vals[:6]))
    out["n_certified"] = int(round(vals[6]))
    return out


# ---------------------------------------------------------------------------
# one permutation
# ---------------------------------------------------------------------------

def run_one_permutation(X: torch.Tensor, y_perm: torch.Tensor,
                        cfg_s1: dict, rng: np.random.Generator,
                        device: torch.device) -> dict:
    """Run the full search-then-validate workflow on (possibly permuted) labels.

    Returns per-method: full score array, max score, top-k indices,
    number certified, and whether any were certified; plus the v2 behavioural
    effects (held-out and training split) and, under "_welch", the naive test.

    RNG use (split, then probe seed) is deliberately unchanged from v1; every v2
    addition is a deterministic function of the fitted probe and the data.
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
        lr=cfg_s1.get("probe_lr", 0.1),
        max_iter=cfg_s1["probe_max_iter"],
        seed=int(rng.integers(2**31)),
    )

    # v2: standardised matrices and logits, computed once for all methods
    Xs_tr = (X_tr - mean) / std
    Xs_te = (X_te - mean) / std
    lg_tr = Xs_tr @ w + b
    lg_te = Xs_te @ w + b

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

        eff_te = behaviour_effects(Xs_te, y_te, lg_te, w, mean, std, top_idx, delta)
        eff_tr = behaviour_effects(Xs_tr, y_tr, lg_tr, w, mean, std, top_idx, delta)

        results[method] = {
            "scores": scores.cpu().numpy(),
            "max_score": max_score,
            "top_k_indices": top_idx,
            "n_certified": n_cert,
            "any_certified": n_cert > 0,
            "n_certified_vec_heldout": eff_te["n_certified"],
            "n_certified_samedata": eff_tr["n_certified"],
            "effects_heldout": {e: eff_te[e] for e in EFFECT_KEYS},
            "effects_train": {e: eff_tr[e] for e in EFFECT_KEYS},
        }

    # v2: the naive per-latent test on the same training split the scores use
    p = X.shape[1]
    alpha = cfg_s1["naive_alpha"]
    z_naive = statistics.NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_bonf = statistics.NormalDist().inv_cdf(1.0 - alpha / (2.0 * p))
    t_abs = welch_abs_t(X_tr, y_tr)
    results["_welch"] = {
        "t_abs": t_abs.cpu().numpy(),
        "t_max": float(t_abs.max().item()),
        "n_naive": int((t_abs > z_naive).sum().item()),
        "n_bonf": int((t_abs > z_bonf).sum().item()),
    }
    return results


# ---------------------------------------------------------------------------
# planted-signal power check
# ---------------------------------------------------------------------------

def planted_power_check(X: torch.Tensor, X_np: np.ndarray, cfg: dict, cfg_s1: dict,
                        device: torch.device) -> list[dict]:
    """Run the same pipeline on labels generated from a KNOWN set of latents
    (Stage 3's generator). A certification step that never fires here has no power,
    and its FWER under the null says nothing about the pipeline.

    Uses the pre-registered `planted_signal` stream (index 8), not the
    `label_permutation` stream, so the permutations above are unaffected.
    """
    pc = cfg_s1["planted_check"]
    methods = cfg_s1["scoring_methods"]
    rng = rng_for(cfg, "planted_signal")
    sd = X_np.std(axis=0)
    Z = ((X_np - X_np.mean(axis=0)) / np.where(sd > 1e-8, sd, 1.0)).astype(np.float32)
    p = X_np.shape[1]

    rows = []
    for r in range(pc["replicates"]):
        S = np.sort(rng.choice(p, size=pc["k"], replace=False))
        y_np, _ = generate_planted_labels(Z, S, pc["form"], pc["amplitude"], rng)
        res = run_one_permutation(X, to_gpu(y_np, device), cfg_s1, rng, device)
        S_set = set(S.tolist())
        rows.append({m: {
            "recall_at_k": len(S_set & set(res[m]["top_k_indices"])) / len(S_set),
            "n_certified_heldout": int(res[m]["n_certified"]),
            "n_certified_samedata": int(res[m]["n_certified_samedata"]),
            "effects_heldout": res[m]["effects_heldout"],
            "effects_train": res[m]["effects_train"],
        } for m in methods})
        print("  planted rep {:>2}/{}: ".format(r + 1, pc["replicates"])
              + "  ".join(f"{m}: recall={rows[-1][m]['recall_at_k']:.1f} "
                          f"cert(held/same)={rows[-1][m]['n_certified_heldout']}/"
                          f"{rows[-1][m]['n_certified_samedata']}" for m in methods),
              flush=True)
    del Z
    return rows


def summarise_planted(rows: list[dict], v2: dict, methods: list[str], cfg_s1: dict) -> dict:
    pc = cfg_s1["planted_check"]
    out = {"replicates": len(rows), "k": pc["k"], "amplitude": pc["amplitude"],
           "form": pc["form"], "min_detect_fraction": pc["min_detect_fraction"],
           "per_method": {}}
    for m in methods:
        above = {}
        for split, key in (("heldout", "effects_heldout"), ("train", "effects_train")):
            above[split] = {
                e: float(np.mean([r[m][key][e] > v2["behaviour"][m][split][e]["null_q95"]
                                  for r in rows]))
                for e in EFFECT_KEYS}
        out["per_method"][m] = {
            "mean_recall_at_k": float(np.mean([r[m]["recall_at_k"] for r in rows])),
            "detect_fraction_v1_heldout": float(np.mean([r[m]["n_certified_heldout"] >= 1 for r in rows])),
            "detect_fraction_samedata": float(np.mean([r[m]["n_certified_samedata"] >= 1 for r in rows])),
            "fraction_above_null_q95": above,
        }
    out["criterion_has_power"] = bool(any(
        pm["detect_fraction_v1_heldout"] >= pc["min_detect_fraction"]
        for pm in out["per_method"].values()))
    return out


# ---------------------------------------------------------------------------
# v2 aggregation
# ---------------------------------------------------------------------------

def _null_summary(null, real, taus, B):
    null = np.asarray(null, dtype=np.float64)
    return {
        "real": float(real),
        "null_q50": float(np.quantile(null, 0.50)),
        "null_q95": float(np.quantile(null, 0.95)),
        "null_q99": float(np.quantile(null, 0.99)),
        "null_max": float(null.max()),
        "wy_p": float((1 + (null >= real).sum()) / (1 + B)),
        "fwer_at_tau": {f"{t:g}": float((null >= t).mean()) for t in taus},
    }


def aggregate_v2(perm_results: dict, welch_null: dict, real: dict, methods: list[str],
                 cfg_s1: dict, B: int, p: int) -> tuple[dict, dict]:
    taus = cfg_s1["behaviour_thresholds"]
    alpha = cfg_s1["naive_alpha"]
    wy_alpha = cfg_s1["wy_alpha"]
    v2 = {"behaviour": {}, "fwer_samedata": {}, "welch": {}}
    arrays = {}

    for m in methods:
        pr = perm_results[m]
        v2["behaviour"][m] = {}
        for split, real_key, store in (("heldout", "effects_heldout", "eff_heldout"),
                                       ("train", "effects_train", "eff_train")):
            v2["behaviour"][m][split] = {}
            for e in EFFECT_KEYS:
                null = np.array(pr[store][e])
                arrays[f"{m}__{split}__{e}"] = null
                v2["behaviour"][m][split][e] = _null_summary(null, real[m][real_key][e], taus, B)
        ncs = np.array(pr["n_certified_samedata"])
        v2["fwer_samedata"][m] = {
            "fwer": float((ncs > 0).mean()),
            "mean_n_certified": float(ncs.mean()),
            "real_n_certified": int(real[m]["n_certified_samedata"]),
        }
        arrays[f"{m}__n_certified_samedata"] = ncs
        arrays[f"{m}__n_certified_heldout_v1"] = np.array(pr["n_certified"])

    mismatch = sum(int((np.array(perm_results[m]["n_cert_vec_heldout"])
                        != np.array(perm_results[m]["n_certified"])).sum()) for m in methods)
    v2["vectorised_vs_v1_heldout_mismatch"] = {
        "n_permutation_method_pairs": len(methods) * B, "n_mismatch": mismatch}

    tmax = np.array(welch_null["t_max"])
    nn = np.array(welch_null["n_naive"])
    nb = np.array(welch_null["n_bonf"])
    z_naive = statistics.NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_bonf = statistics.NormalDist().inv_cdf(1.0 - alpha / (2.0 * p))
    wy_thr = float(np.quantile(tmax, 1.0 - wy_alpha))
    t_real = real["_welch"]["t_abs"]
    v2["welch"] = {
        "z_naive": z_naive, "z_bonferroni": z_bonf, "wy_threshold": wy_thr,
        "expected_naive_false_passes": float(alpha * p),
        "null_mean_n_naive": float(nn.mean()),
        "null_mean_n_bonferroni": float(nb.mean()),
        "fwer_naive": float((nn >= 1).mean()),
        "fwer_bonferroni": float((nb >= 1).mean()),
        "fwer_wy_insample": float((tmax > wy_thr).mean()),
        "real_n_naive": int((t_real > z_naive).sum()),
        "real_n_bonferroni": int((t_real > z_bonf).sum()),
        "real_n_wy": int((t_real > wy_thr).sum()),
        "real_t_max": float(t_real.max()),
        "null_t_max_q95": wy_thr,
    }
    arrays["welch__t_max"], arrays["welch__n_naive"], arrays["welch__n_bonf"] = tmax, nn, nb
    return v2, arrays


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
    fig.suptitle(f"Westfall–Young null distribution (B={out['n_permutations']})",
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
                 f"(top-{out['top_k']}, δ={out['ablation_delta']}, "
                 f"B={out['n_permutations']})")
    ax.set_ylim(0, min(1.15, max(fwers) * 1.3 + 0.05))
    fig.tight_layout()
    fig.savefig(rd / "fig6_fwer_by_method.png", dpi=150)
    plt.close(fig)


def make_figures_v2(v2, arrays, methods, cfg_s1, rd):
    # Fig 10: FWER(tau) of label-free behavioural effects under the null; the
    # real-label value of each effect is the dashed line.
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4), squeeze=False)
    for i, m in enumerate(methods):
        ax = axes[0, i]
        for e, col in (("joint_flip", "#3b6ea5"), ("single_flip_max", "#e07b39")):
            null = arrays[f"{m}__heldout__{e}"]
            real_v = v2["behaviour"][m]["heldout"][e]["real"]
            grid = np.linspace(0.0, max(float(null.max()), real_v) * 1.1 + 1e-9, 200)
            ax.plot(grid, [(null >= t).mean() for t in grid], color=col,
                    label=f"{e} (null)")
            ax.axvline(real_v, color=col, ls="--", lw=1)
        ax.axhline(cfg_s1["wy_alpha"], color="k", ls=":", lw=1)
        ax.set_title(m)
        ax.set_xlabel("behavioural-change threshold tau")
        ax.set_ylabel("P(effect >= tau | global null)")
        ax.legend(fontsize=7)
    fig.suptitle("Stage 1 v2: FWER as a function of the behavioural threshold "
                 "(dashed = real labels)", y=1.02)
    fig.tight_layout()
    fig.savefig(rd / "fig10_stage1_behaviour_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 11: the proposal's motivating quantity - naive vs corrected per-latent testing
    w = v2["welch"]
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["naive (|t|>1.96)", "Bonferroni", "Westfall-Young"]
    vals = [w["fwer_naive"], w["fwer_bonferroni"], w["fwer_wy_insample"]]
    bars = ax.bar(names, vals, color=["#c0392b", "#e07b39", "#5b8c5a"], edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("FWER under global null")
    ax.set_title(f"Per-latent testing, p={int(round(w['expected_naive_false_passes'] / cfg_s1['naive_alpha']))}\n"
                 f"naive passes per null experiment: {w['null_mean_n_naive']:.0f} "
                 f"(expected {w['expected_naive_false_passes']:.0f})")
    fig.tight_layout()
    fig.savefig(rd / "fig11_stage1_naive_vs_corrected.png", dpi=150)
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
    ap.add_argument("--skip-planted", action="store_true",
                    help="debug: skip the planted-signal power check")
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
    cp = Path(args.cache) if args.cache else cache_path(cfg)
    if not cp.exists():
        raise SystemExit(
            f"Cache not found at {cp}. Run src/cache_activations.py first.")
    d = np.load(cp, allow_pickle=True)
    prim = cfg["aggregation"]["primary"]
    X_np = d[f"X_{prim}"].astype(np.float32, copy=False)
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
              f"certified={real[m]['n_certified']}/{cfg_s1['top_k_certify']} "
              f"(same-data {real[m]['n_certified_samedata']}/{cfg_s1['top_k_certify']})  "
              f"joint flip rate={real[m]['effects_heldout']['joint_flip']:.4f}")
    print(f"  [welch] naive passes={real['_welch']['n_naive']}  "
          f"bonferroni={real['_welch']['n_bonf']}  max|t|={real['_welch']['t_max']:.2f}")

    # ---- Permutation loop ---------------------------------------------------
    print(f"\n--- running {B} permutations ---")
    t0 = time.time()
    perm_results = {m: {"max_scores": [], "any_certified": [], "n_certified": [],
                        "n_certified_samedata": [], "n_cert_vec_heldout": [],
                        "eff_heldout": {e: [] for e in EFFECT_KEYS},
                        "eff_train": {e: [] for e in EFFECT_KEYS}}
                    for m in methods}
    welch_null = {"t_max": [], "n_naive": [], "n_bonf": []}

    for b in range(B):
        y_perm_np = rng.permutation(y_np)
        y_perm = to_gpu(y_perm_np, device)
        res = run_one_permutation(X, y_perm, cfg_s1, rng, device)
        for m in methods:
            perm_results[m]["max_scores"].append(res[m]["max_score"])
            perm_results[m]["any_certified"].append(res[m]["any_certified"])
            perm_results[m]["n_certified"].append(res[m]["n_certified"])
            perm_results[m]["n_certified_samedata"].append(res[m]["n_certified_samedata"])
            perm_results[m]["n_cert_vec_heldout"].append(res[m]["n_certified_vec_heldout"])
            for e in EFFECT_KEYS:
                perm_results[m]["eff_heldout"][e].append(res[m]["effects_heldout"][e])
                perm_results[m]["eff_train"][e].append(res[m]["effects_train"][e])
        for k in welch_null:
            welch_null[k].append(res["_welch"][k])

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

    # ---- v2 aggregation -----------------------------------------------------
    v2, arrays = aggregate_v2(perm_results, welch_null, real, methods, cfg_s1, B, p)
    out["v2"] = v2

    w = v2["welch"]
    print("\n=== v2: naive per-latent testing under the global null ===")
    print(f"  naive (|t|>{w['z_naive']:.2f}): {w['null_mean_n_naive']:.1f} passes per null "
          f"experiment (expected {w['expected_naive_false_passes']:.0f}), "
          f"FWER={w['fwer_naive']:.3f}; real labels: {w['real_n_naive']} pass")
    print(f"  Bonferroni: FWER={w['fwer_bonferroni']:.3f}; real labels: {w['real_n_bonferroni']} pass")
    print(f"  Westfall-Young (|t|>{w['wy_threshold']:.2f}): real labels: {w['real_n_wy']} pass")
    print("\n=== v2: same-data (conventional) certification, FWER ===")
    for m in methods:
        f = v2["fwer_samedata"][m]
        print(f"  [{m}] same-data FWER={f['fwer']:.4f}  real certified={f['real_n_certified']}"
              f"   (held-out v1 FWER={out['permutation'][m]['fwer']:.4f})")
    mm = v2["vectorised_vs_v1_heldout_mismatch"]
    print(f"  vectorised vs v1 held-out certification mismatches: "
          f"{mm['n_mismatch']}/{mm['n_permutation_method_pairs']}")

    if not args.skip_planted:
        print("\n=== v2: planted-signal power check ===")
        rows = planted_power_check(X, X_np, cfg, cfg_s1, device)
        planted = summarise_planted(rows, v2, methods, cfg_s1)
        v2["planted_check"] = planted
        for m in methods:
            pm = planted["per_method"][m]
            print(f"  [{m}] recall@k={pm['mean_recall_at_k']:.2f}  "
                  f"v1 criterion fires in {pm['detect_fraction_v1_heldout']:.0%} of replicates "
                  f"(same-data {pm['detect_fraction_samedata']:.0%})")
        if planted["criterion_has_power"]:
            print("  criterion_has_power = True")
        else:
            print("  criterion_has_power = FALSE: the certification step does not fire on a "
                  "known signal, so the FWER numbers above are UNINFORMATIVE.")

    # ---- Save ---------------------------------------------------------------
    (rd / "stage1_calibration.json").write_text(
        json.dumps(out, indent=2, default=float))

    # Save per-latent adjusted p-values as npz (too large for JSON)
    np.savez(rd / "stage1_adjusted_pvalues.npz",
             **{m: (1 + (np.array(perm_results[m]["max_scores"])[:, None]
                         >= real_scores[m][None, :]).sum(axis=0)
                    ).astype(np.float64) / (1 + B)
                for m in methods})
    np.savez(rd / "stage1_behaviour_null.npz", **arrays)

    make_figures(out, methods, rd)
    make_figures_v2(v2, arrays, methods, cfg_s1, rd)
    print(f"\nwrote {rd}/stage1_calibration.json, "
          f"{rd}/stage1_adjusted_pvalues.npz, {rd}/stage1_behaviour_null.npz, and 4 figures")


if __name__ == "__main__":
    main()
