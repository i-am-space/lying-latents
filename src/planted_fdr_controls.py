"""Stage 3 v2 — realised FDR with controls that isolate the zero atom.

v1 (planted_fdr.py) used equicorrelated S (near-copy knockoffs), had no control for which
Gaussian knockoffs are valid, and stopped its amplitude sweep at 3.0. This re-run
(config/preregistration.yaml stage3_amendment_1, fixed before it ran) applies the same
planted-signal benchmark to three datasets on identical draws:

  real_mvr   real latents, MVR S                                   (primary)
  real_equi  real latents, equicorrelated S (v1's choice)          (isolates near-copy knockoffs)
  gauss_mvr  Gaussian data with the real covariance, MVR S         (Gaussian knockoffs exactly valid:
                                                                   the procedure with NO zero atom)

real_mvr vs gauss_mvr isolates the atom; real_equi vs real_mvr isolates S. Secondary columns:
knockoff offset 0, and the Westfall-Young marginal baseline scored against the conditional truth.

Stages:  --stage diagnose  S statistics, k = 0 harness, small pilot (rule C1), in minutes
         --stage full      the whole grid

--experiment p2048 runs the follow-up on v1's power collapse at p = 2048 instead
(stage3_amendment_2): v1's solver, v1's rows and the zero atom are changed one at a time.

Usage: python src/planted_fdr_controls.py --config config/default.yaml --device cuda --stage diagnose
       python src/planted_fdr_controls.py --config config/default.yaml --device cuda --experiment p2048 --stage full
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

from common import STREAMS, cache_path, load_config, results_dir, rng_for
from knockoff_audit import estimate_cov, standardise
from planted_fdr import compute_knockoff_plus_threshold, evaluate_discoveries, generate_planted_labels

DS_LABEL = {"real_mvr": "real, MVR S (primary)", "real_equi": "real, equicorrelated S",
            "gauss_mvr": "Gaussian control, MVR S"}
DS_COLOR = {"real_mvr": "#c0392b", "real_equi": "#e07b39", "gauss_mvr": "#3b6ea5"}
METRICS = ("ko_fdr", "ko_pow", "ko_nd", "k0_fdr", "k0_pow", "k0_nd", "wy_fdr", "wy_pow", "wy_nd")


# ---------------------------------------------------------------------------
# seeding: per-cell streams, so datasets are compared on identical draws
# ---------------------------------------------------------------------------

def cell_rng(cfg: dict, stream: str, *idx: int) -> np.random.Generator:
    base = np.random.SeedSequence(cfg["master_seed"]).spawn(len(STREAMS))[STREAMS.index(stream)]
    return np.random.default_rng(np.random.SeedSequence(
        entropy=base.entropy, spawn_key=base.spawn_key + tuple(int(i) for i in idx)))


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------

def build_datasets(cfg: dict, X_all: np.ndarray) -> tuple[dict, dict]:
    from knockpy import smatrix
    from knockpy.knockoffs import GaussianSampler

    s3 = cfg["stage3_v2"]
    rng = rng_for(cfg, "s3v2_data")
    n_all, p_all = X_all.shape
    cols = np.sort(rng.choice(p_all, s3["p"], replace=False))
    rows = np.sort(rng.choice(n_all, s3["n_rows"], replace=False))
    X = X_all[np.ix_(rows, cols)].astype(np.float64)
    p0 = (X == 0).mean(axis=0)
    Z, _, _ = standardise(X)
    n, p = Z.shape

    sample_cov = estimate_cov(Z, "sample")
    ev = np.linalg.eigvalsh(sample_cov)
    cond = float(ev[-1] / max(ev[0], 1e-300))
    est = "ledoit_wolf" if cond > float(cfg["covariance"]["ill_conditioned_cond_number"]) else "sample"
    Sigma = estimate_cov(Z, est) if est == "ledoit_wolf" else sample_cov
    info = {"n": n, "p": p, "n_over_p": n / p, "cond_sample": cond, "estimator": est,
            "median_zero_mass": float(np.median(p0)), "q10_zero_mass": float(np.quantile(p0, 0.1)),
            "q90_zero_mass": float(np.quantile(p0, 0.9)), "S": {}}
    print(f"  data: n={n} p={p} n/p={n/p:.1f}  cond(sample)={cond:.3g} -> {est}  "
          f"median Pr(X=0)={info['median_zero_mass']:.3f}", flush=True)

    def timed_S(Sig, method):
        t = time.time()
        S = np.asarray(smatrix.compute_smatrix(Sig, method=method))
        return S, time.time() - t

    S_mvr, t_mvr = timed_S(Sigma, "mvr")
    S_equi, t_equi = timed_S(Sigma, "equicorrelated")

    # Gaussian control: same covariance, same n; the covariance is re-estimated from the
    # Gaussian sample exactly as for the real data.
    Lc = np.linalg.cholesky(Sigma + 1e-10 * np.eye(p))
    G = rng.standard_normal((n, p)) @ Lc.T
    Zg, _, _ = standardise(G)
    Sigma_g = estimate_cov(Zg, est)
    min_ev = float(np.linalg.eigvalsh(2 * Sigma_g - S_mvr).min())
    if min_ev > 1e-8:
        S_g, t_g, g_note = S_mvr, 0.0, "reused primary S"
    else:
        S_g, t_g = timed_S(Sigma_g, "mvr")
        g_note = "recomputed MVR (primary S not valid for the Gaussian covariance)"

    ds = {}
    for name, Zd, Sig, S in (("real_mvr", Z, Sigma, S_mvr), ("real_equi", Z, Sigma, S_equi),
                             ("gauss_mvr", Zg, Sigma_g, S_g)):
        ds[name] = {"Z": Zd, "Sigma": Sig, "S": S,
                    "sampler": GaussianSampler(Zd, mu=Zd.mean(axis=0), Sigma=Sig, S=S)}
        s = np.diag(S)
        np.random.seed(0)
        Zk = ds[name]["sampler"].sample_knockoffs()
        emp = float(np.mean([np.corrcoef(Zd[:, j], Zk[:, j])[0, 1] for j in range(p)]))
        info["S"][name] = {"mean_s": float(s.mean()), "median_s": float(np.median(s)),
                           "min_s": float(s.min()), "max_s": float(s.max()),
                           "mean_1_minus_s": float((1 - s).mean()), "empirical_mean_corr_X_Xk": emp}
        print(f"  S[{name:10s}] mean s={s.mean():.4f} median={np.median(s):.4f} min={s.min():.4f} "
              f"max={s.max():.4f}  mean corr(X_j,Xk_j)={emp:.3f}", flush=True)
    info["S_timing_seconds"] = {"mvr": t_mvr, "equicorrelated": t_equi, "gauss_control": t_g}
    info["gauss_control_S"] = g_note
    print(f"  MVR {t_mvr:.0f}s, equicorrelated {t_equi:.1f}s; Gaussian control: {g_note}", flush=True)
    return ds, info


# ---------------------------------------------------------------------------
# lasso, thresholds, baselines
# ---------------------------------------------------------------------------

def lmax_power(Phi: torch.Tensor, n_iter: int = 50) -> float:
    v = torch.ones(Phi.shape[1], device=Phi.device)
    v /= v.norm()
    lam = 1.0
    for _ in range(n_iter):
        v = Phi.t() @ (Phi @ v) / Phi.shape[0]
        lam = float(v.norm())
        v /= lam
    return lam


def fit_lasso_checked(Phi: torch.Tensor, y: torch.Tensor, lam: float, max_iter: int, tol: float):
    """FISTA L1-logistic with step 1/L (L = 0.25 * (1.1 * lambda_max(Phi'Phi/n) + 1)).

    Every step is coordinate-wise, so the fit is equivariant to swapping a variable with its
    knockoff and W keeps the flip-sign property even if the cap is hit; non-convergence costs
    power, not validity. Returns (w, converged, residual, iterations)."""
    n, d = Phi.shape
    lr = 1.0 / (0.25 * (1.1 * lmax_power(Phi) + 1.0))
    w = torch.zeros(d, device=Phi.device); b = torch.zeros(1, device=Phi.device)
    w_prev, b_prev, t = w.clone(), b.clone(), 1.0
    res, converged, it = float("inf"), False, 0
    for it in range(1, max_iter + 1):
        t_new = 0.5 * (1 + math.sqrt(1 + 4 * t * t))
        mom = (t - 1.0) / t_new
        v = w + mom * (w - w_prev); vb = b + mom * (b - b_prev)
        w_prev, b_prev = w, b
        err = (torch.sigmoid(Phi @ v + vb) - y) / n
        u = v - lr * (Phi.t() @ err)
        w = torch.sign(u) * torch.clamp(u.abs() - lr * lam, min=0.0)
        b = vb - lr * err.sum()
        t = t_new
        if it % 25 == 0:
            res = float((v - w).abs().max() / lr)
            if res < tol:
                converged = True
                break
    return w, converged, res, it


def knockoff_threshold(W: np.ndarray, q: float, offset: int) -> float:
    """Smallest t with (offset + #{W <= -t}) / max(1, #{W >= t}) <= q; offset 1 = Knockoff+."""
    pos = np.sort(W[W > 0]); neg = np.sort(-W[W < 0])
    cand = np.unique(np.abs(W[W != 0]))
    if cand.size == 0:
        return float("inf")
    n_pos = pos.size - np.searchsorted(pos, cand, side="left")
    n_neg = neg.size - np.searchsorted(neg, cand, side="left")
    ok = (offset + n_neg) / np.maximum(1, n_pos) <= q
    return float(cand[np.argmax(ok)]) if ok.any() else float("inf")


def wy_marginal(Z_t: torch.Tensor, y_t: torch.Tensor, qs: list[float], B: int, seed: int):
    """Marginal association scan with a single-step Westfall-Young threshold at alpha = q."""
    n = y_t.numel()
    yc = y_t - y_t.mean()
    r = (Z_t.t() @ yc).abs() / n
    gen = torch.Generator(device=Z_t.device); gen.manual_seed(seed)
    idx = torch.rand(B, n, generator=gen, device=Z_t.device).argsort(dim=1)
    max_null = ((yc[idx] @ Z_t).abs() / n).max(dim=1).values
    thr = {q: float(torch.quantile(max_null, 1 - q)) for q in qs}
    return r.cpu().numpy(), thr


def score_fit(W: np.ndarray, r: np.ndarray, wy_thr: dict, qs, true_set: set) -> dict:
    out = {m: np.zeros(len(qs), dtype=np.float32) for m in METRICS}
    for i, q in enumerate(qs):
        for tag, offset in (("ko", 1), ("k0", 0)):
            tau = knockoff_threshold(W, q, offset)
            disc = set() if math.isinf(tau) else set(np.where(W >= tau)[0].tolist())
            out[f"{tag}_fdr"][i], out[f"{tag}_pow"][i] = evaluate_discoveries(disc, true_set)
            out[f"{tag}_nd"][i] = len(disc)
        disc = set(np.where(r > wy_thr[q])[0].tolist())
        out["wy_fdr"][i], out["wy_pow"][i] = evaluate_discoveries(disc, true_set)
        out["wy_nd"][i] = len(disc)
    return out


def run_fit(ds: dict, y_np: np.ndarray, seed: int, cfg: dict, device, lam: float, true_set: set):
    s3 = cfg["stage3_v2"]; qs = s3["nominal_fdr_targets"]
    Z = ds["Z"]; p = Z.shape[1]
    np.random.seed(seed)
    Zk = ds["sampler"].sample_knockoffs()
    Phi = torch.from_numpy(np.hstack([Z, Zk]).astype(np.float32)).to(device)
    y_t = torch.from_numpy(y_np.astype(np.float32)).to(device)
    w, conv, res, it = fit_lasso_checked(Phi, y_t, lam, s3["lasso"]["max_iter"], s3["lasso"]["tol"])
    w_np = w.cpu().numpy()
    W = np.abs(w_np[:p]) - np.abs(w_np[p:])
    r, thr = wy_marginal(Phi[:, :p], y_t, qs, s3["wy_baseline"]["n_permutations"], seed)
    sc = score_fit(W, r, thr, qs, true_set)
    return sc, conv, res, it


# ---------------------------------------------------------------------------
# harness and grid
# ---------------------------------------------------------------------------

def null_harness(ds: dict, cfg: dict, device) -> dict:
    """k = 0: labels independent of X, so every latent is null and FDR = P(any discovery)."""
    s3 = cfg["stage3_v2"]; qs = s3["nominal_fdr_targets"]; R = s3["null_harness"]["replicates"]
    out = {}
    for name, d in ds.items():
        hits = {m: np.zeros(len(qs)) for m in ("ko", "k0", "wy")}
        for rep in range(R):
            rng = cell_rng(cfg, "s3v2_planted", 99, rep)
            y = (rng.random(d["Z"].shape[0]) < 0.5).astype(np.float32)
            seed = int(cell_rng(cfg, "s3v2_knockoff", 99, rep).integers(2**31))
            sc, *_ = run_fit(d, y, seed, cfg, device, s3["lasso"]["lambda"], set())
            for tag in hits:
                hits[tag] += (sc[f"{tag}_nd"] > 0)
        out[name] = {tag: {str(q): float(hits[tag][i] / R) for i, q in enumerate(qs)} for tag in hits}
        print(f"  k=0 harness [{name:10s}] P(any discovery) Knockoff+ at q={qs}: "
              f"{[round(out[name]['ko'][str(q)], 2) for q in qs]}   WY: "
              f"{[round(out[name]['wy'][str(q)], 2) for q in qs]}", flush=True)
    return out


def run_grid(ds: dict, cfg: dict, device, R: int, lam: float, amps, forms, ks, only_ds=None):
    qs = cfg["stage3_v2"]["nominal_fdr_targets"]
    names = only_ds or list(ds)
    M = {m: {d: np.zeros((len(amps), len(forms), len(ks), R, len(qs)), dtype=np.float32) for d in names}
         for m in METRICS}
    conv = {d: np.zeros((len(amps), len(forms), len(ks), R), dtype=bool) for d in names}
    resid = {d: np.zeros((len(amps), len(forms), len(ks), R), dtype=np.float32) for d in names}
    total = len(names) * len(amps) * len(forms) * len(ks) * R
    done, t0 = 0, time.time()
    for ia, amp in enumerate(amps):
        for jf, form in enumerate(forms):
            for kk, k in enumerate(ks):
                for rep in range(R):
                    seed = int(cell_rng(cfg, "s3v2_knockoff", ia, jf, kk, rep).integers(2**31))
                    for name in names:
                        rng = cell_rng(cfg, "s3v2_planted", ia, jf, kk, rep)
                        S_idx = np.sort(rng.choice(ds[name]["Z"].shape[1], size=k, replace=False))
                        y_np, _ = generate_planted_labels(ds[name]["Z"], S_idx, form, amp, rng)
                        sc, cv, rs, _ = run_fit(ds[name], y_np, seed, cfg, device, lam, set(S_idx.tolist()))
                        for m in METRICS:
                            M[m][name][ia, jf, kk, rep] = sc[m]
                        conv[name][ia, jf, kk, rep], resid[name][ia, jf, kk, rep] = cv, rs
                        done += 1
                if done % 60 < len(names) * R or done == total:
                    el = time.time() - t0
                    print(f"  [{done:>5}/{total}] amp={amp} {form:<11} k={k:<2} elapsed={el/60:.1f}m "
                          f"ETA={el/done*(total-done)/60:.1f}m", flush=True)
    return M, conv, resid


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def mean_se(v: np.ndarray):
    v = np.asarray(v, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0


def aggregate(cfg, M, conv, resid, amps, forms, ks, names):
    qs = cfg["stage3_v2"]["nominal_fdr_targets"]
    conds = []
    for d in names:
        for ia, amp in enumerate(amps):
            for jf, form in enumerate(forms):
                for kk, k in enumerate(ks):
                    for iq, q in enumerate(qs):
                        row = {"dataset": d, "amplitude": amp, "form": form, "k": k, "q": q,
                               "power_capped_k_lt_1_over_q": bool(k < math.ceil(1 / q))}
                        for m in METRICS:
                            mu, se = mean_se(M[m][d][ia, jf, kk, :, iq])
                            row[m] = mu; row[m + "_se"] = se
                        row["ko_inflated"] = bool(row["ko_fdr"] - 1.96 * row["ko_fdr_se"] > q)
                        row["ko_controlled_strict"] = bool(row["ko_fdr"] <= q)
                        conds.append(row)
    idx = {(c["dataset"], c["amplitude"], c["form"], c["k"], c["q"]): c for c in conds}
    contrasts = []
    for (a, b) in (("real_mvr", "gauss_mvr"), ("real_equi", "real_mvr")):
        if a not in names or b not in names:
            continue
        for amp in amps:
            for form in forms:
                for k in ks:
                    for q in qs:
                        x, y = idx[(a, amp, form, k, q)], idx[(b, amp, form, k, q)]
                        row = {"a": a, "b": b, "amplitude": amp, "form": form, "k": k, "q": q}
                        for m in ("ko_fdr", "ko_pow"):
                            row[m + "_diff"] = x[m] - y[m]
                            row[m + "_diff_se"] = math.sqrt(x[m + "_se"] ** 2 + y[m + "_se"] ** 2)
                        contrasts.append(row)
    conv_stats = {d: {"fraction_converged": float(conv[d].mean()),
                      "median_residual": float(np.median(resid[d]))} for d in names}
    return conds, contrasts, conv_stats


def c1_check(conds) -> dict:
    ctrl = [c for c in conds if c["dataset"] == "gauss_mvr"]
    bad = [c for c in ctrl if c["ko_inflated"]]
    return {"n_control_cells": len(ctrl), "n_inflated": len(bad),
            "worst": max(((c["ko_fdr"] - c["q"], c["amplitude"], c["form"], c["k"], c["q"]) for c in ctrl),
                         default=None), "C1_holds": len(bad) == 0}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(conds, info, amps, forms, ks, names, q_ref, rd):
    for fname, met, ylab, ttl in (("fig13_stage3_v2_fdr.png", "ko_fdr", "realised FDR", "FDR"),
                                  ("fig14_stage3_v2_power.png", "ko_pow", "power", "Power")):
        fig, axes = plt.subplots(len(forms), len(ks), figsize=(4.2 * len(ks), 3.6 * len(forms)),
                                 sharex=True, sharey=True, squeeze=False)
        for i, form in enumerate(forms):
            for j, k in enumerate(ks):
                ax = axes[i, j]
                for d in names:
                    rows = sorted([c for c in conds if c["dataset"] == d and c["form"] == form
                                   and c["k"] == k and c["q"] == q_ref], key=lambda c: c["amplitude"])
                    ax.errorbar([c["amplitude"] for c in rows], [c[met] for c in rows],
                                yerr=[1.96 * c[met + "_se"] for c in rows], marker="o", ms=3, capsize=2,
                                color=DS_COLOR[d], label=DS_LABEL[d])
                if met == "ko_fdr":
                    ax.axhline(q_ref, color="k", ls="--", lw=1)
                ax.set_xscale("log"); ax.set_title(f"{form}, |S|={k}", fontsize=9)
                if i == len(forms) - 1: ax.set_xlabel("signal amplitude")
                if j == 0: ax.set_ylabel(ylab)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"Stage 3 v2: {ttl} at q = {q_ref} (95% intervals)", y=1.0)
        fig.tight_layout(); fig.savefig(rd / fname, dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [(d, info["S"][d]["empirical_mean_corr_X_Xk"]) for d in names]
    ax.bar([DS_LABEL[d] for d, _ in vals], [v for _, v in vals], color=[DS_COLOR[d] for d, _ in vals])
    for i, (_, v) in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
    ax.set_ylabel("mean corr(X_j, knockoff_j)"); ax.set_ylim(0, 1.05)
    ax.set_title("Near-copy check: knockoff correlation with its variable")
    plt.setp(ax.get_xticklabels(), fontsize=7)
    fig.tight_layout(); fig.savefig(rd / "fig15_stage3_v2_s_matrix.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# p2048 experiment (stage3_amendment_2): why did v1 see a power collapse that v2 does not?
# Arms change one thing at a time: v1's solver, v1's first-20000 rows, and the zero atom,
# all at v1's p = 2048 with v1's equicorrelated S and Ledoit-Wolf covariance.
# ---------------------------------------------------------------------------

def fit_lasso_v1(Phi: torch.Tensor, y: torch.Tensor, lam: float, lr: float, max_iter: int):
    """v1's solver, copied verbatim from planted_fdr.fit_lasso (fixed step, fixed iteration
    count, no convergence check), except that it also returns the intercept."""
    n, dim = Phi.shape
    w = torch.zeros(dim, device=Phi.device, dtype=Phi.dtype)
    b = torch.zeros(1, device=Phi.device, dtype=Phi.dtype)
    w_prev = w.clone()
    for t in range(1, max_iter + 1):
        beta = (t - 1.0) / (t + 2.0)
        v = w + beta * (w - w_prev)
        w_prev.copy_(w)
        logits = Phi @ v + b
        p = torch.sigmoid(logits)
        err = (p - y) / n
        grad_w = Phi.t() @ err
        grad_b = err.sum()
        u = v - lr * grad_w
        b = b - lr * grad_b
        thresh = lr * lam
        w = torch.sign(u) * torch.clamp(u.abs() - thresh, min=0.0)
    return w, b


def prox_residual(Phi: torch.Tensor, y: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
                  lam: float, lr: float) -> float:
    """Proximal-gradient residual at (w, b) with a safe step: 0 at the lasso optimum."""
    err = (torch.sigmoid(Phi @ w + b) - y) / Phi.shape[0]
    u = w - lr * (Phi.t() @ err)
    w2 = torch.sign(u) * torch.clamp(u.abs() - lr * lam, min=0.0)
    return float((w - w2).abs().max() / lr)


def build_datasets_p2048(cfg: dict, sec: dict, X_all: np.ndarray) -> tuple[dict, dict]:
    from knockpy import smatrix
    from knockpy.knockoffs import GaussianSampler

    rng = rng_for(cfg, "s3p2048_data")
    n_all, p_all = X_all.shape
    if sec["p"] != p_all:
        raise SystemExit(f"stage3_p2048.p = {sec['p']} but the cache has {p_all} latents")
    n = sec["n_rows"]
    rows = {"real_first": np.arange(n),                                   # v1: X_raw[:20000]
            "real_random": np.sort(rng.choice(n_all, n, replace=False))}
    ds = {}
    for name, r in rows.items():
        X = X_all[r].astype(np.float64)
        p0 = (X == 0).mean(axis=0)
        Z, _, _ = standardise(X)
        del X
        Sigma = estimate_cov(Z, sec["covariance"])
        S = np.asarray(smatrix.compute_smatrix(Sigma, method=sec["s_method"]))
        ds[name] = {"Z": Z, "Sigma": Sigma, "S": S, "median_zero_mass": float(np.median(p0))}

    # Gaussian control: same covariance as the random-row real data, same n; the covariance
    # is re-estimated from the Gaussian sample exactly as for the real data.
    base = ds["real_random"]
    Lc = np.linalg.cholesky(base["Sigma"] + 1e-10 * np.eye(p_all))
    Zg, _, _ = standardise(rng.standard_normal((n, p_all)) @ Lc.T)
    Sigma_g = estimate_cov(Zg, sec["covariance"])
    if np.linalg.eigvalsh(2 * Sigma_g - base["S"]).min() > 1e-8:
        S_g, note = base["S"], "reused the real_random S"
    else:
        S_g, note = np.asarray(smatrix.compute_smatrix(Sigma_g, method=sec["s_method"])), "recomputed"
    ds["gauss"] = {"Z": Zg, "Sigma": Sigma_g, "S": S_g, "median_zero_mass": 0.0}

    info = {"n": n, "p": p_all, "n_over_p": n / p_all, "covariance": sec["covariance"],
            "s_method": sec["s_method"], "gauss_control_S": note, "datasets": {}}
    for name, d in ds.items():
        d["sampler"] = GaussianSampler(d["Z"], mu=d["Z"].mean(axis=0), Sigma=d["Sigma"], S=d["S"])
        s = np.diag(d["S"])
        info["datasets"][name] = {"median_zero_mass": d["median_zero_mass"], "mean_s": float(s.mean()),
                                  "min_s": float(s.min()), "max_s": float(s.max())}
        print(f"  [{name:11s}] median Pr(X=0)={d['median_zero_mass']:.3f}  mean s={s.mean():.4f}", flush=True)
    print(f"  n={n} p={p_all} n/p={n/p_all:.1f}  covariance={sec['covariance']}  S={sec['s_method']}  "
          f"Gaussian control: {note}", flush=True)
    return ds, info


def run_grid_p2048(ds: dict, cfg: dict, sec: dict, device, R: int, amps, forms, ks, arms):
    """Draw bank: replicate r uses knockoff draw r of its dataset in every cell, so replicates
    within a cell are independent, and arms on the same dataset share draws and labels (paired).
    A knockpy draw costs ~20 s at p = 2048, so a fresh draw per cell is not affordable."""
    qs = sec["nominal_fdr_targets"]
    lam = sec["lasso"]["lambda"]
    names = [a[0] for a in arms]
    dkeys = list(dict.fromkeys(a[1] for a in arms))
    p = ds[dkeys[0]]["Z"].shape[1]
    shape = (len(amps), len(forms), len(ks), R)
    M = {m: {a: np.zeros(shape + (len(qs),), dtype=np.float32) for a in names} for m in METRICS}
    conv = {a: np.zeros(shape, dtype=bool) for a in names}
    resid = {a: np.zeros(shape, dtype=np.float32) for a in names}
    finite = {a: np.ones(shape, dtype=bool) for a in names}
    draws = {k: {"lambda_max": [], "safe_step": [], "mean_corr_X_Xk": []} for k in dkeys}
    no_r, no_thr = np.zeros(p), {q: float("inf") for q in qs}
    t_draw, t_fit = {k: 0.0 for k in dkeys}, {s: [0.0, 0] for s in ("fixed", "v1")}
    t0 = time.time()

    for rep in range(R):
        for di, dk in enumerate(dkeys):
            d = ds[dk]
            seed = int(cell_rng(cfg, "s3p2048_knockoff", di, rep).integers(2**31))
            t = time.time()
            np.random.seed(seed)
            Zk = d["sampler"].sample_knockoffs()
            t_draw[dk] += time.time() - t
            zc, kc = d["Z"] - d["Z"].mean(axis=0), Zk - Zk.mean(axis=0)
            corr = (zc * kc).sum(0) / np.sqrt((zc ** 2).sum(0) * (kc ** 2).sum(0))
            Phi = torch.from_numpy(np.hstack([d["Z"], Zk]).astype(np.float32)).to(device)
            del Zk, zc, kc
            lmax = lmax_power(Phi)
            safe = 1.0 / (0.25 * (1.1 * lmax + 1.0))
            draws[dk]["lambda_max"].append(lmax)
            draws[dk]["safe_step"].append(safe)
            draws[dk]["mean_corr_X_Xk"].append(float(np.mean(corr)))

            for ia, amp in enumerate(amps):
                for jf, form in enumerate(forms):
                    for kk, k in enumerate(ks):
                        rng = cell_rng(cfg, "s3p2048_planted", ia, jf, kk, rep)
                        S_idx = np.sort(rng.choice(p, size=k, replace=False))
                        true_set = set(S_idx.tolist())
                        y_np, _ = generate_planted_labels(d["Z"], S_idx, form, amp, rng)
                        y_t = torch.from_numpy(y_np.astype(np.float32)).to(device)
                        for name, akey, solver in arms:
                            if akey != dk:
                                continue
                            t = time.time()
                            if solver == "fixed":
                                w, cv, rs, _ = fit_lasso_checked(Phi, y_t, lam, sec["lasso"]["max_iter"],
                                                                 sec["lasso"]["tol"])
                            elif solver == "v1":
                                w, b = fit_lasso_v1(Phi, y_t, lam, sec["v1_solver"]["lr"],
                                                    sec["v1_solver"]["max_iter"])
                                rs = prox_residual(Phi, y_t, w, b, lam, safe)
                                cv = rs < sec["lasso"]["tol"]
                            else:
                                raise ValueError(f"unknown solver {solver!r}")
                            w_np = w.cpu().numpy()
                            if not np.all(np.isfinite(w_np)):
                                w_np, rs, cv = np.zeros_like(w_np), float("inf"), False
                                finite[name][ia, jf, kk, rep] = False
                            t_fit[solver][0] += time.time() - t
                            t_fit[solver][1] += 1
                            sc = score_fit(np.abs(w_np[:p]) - np.abs(w_np[p:]), no_r, no_thr, qs, true_set)
                            for m in METRICS:
                                M[m][name][ia, jf, kk, rep] = sc[m]
                            conv[name][ia, jf, kk, rep], resid[name][ia, jf, kk, rep] = cv, rs
            del Phi
        el = time.time() - t0
        print(f"  rep {rep + 1:>2}/{R}  elapsed {el / 60:.1f}m  ETA {el / (rep + 1) * (R - rep - 1) / 60:.1f}m",
              flush=True)

    n_draws = R
    timing = {"seconds_per_draw": {k: v / n_draws for k, v in t_draw.items()},
              "seconds_per_fit": {s: (v[0] / v[1] if v[1] else None) for s, v in t_fit.items()}}
    return M, conv, resid, finite, draws, timing


def load_v1_reference(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    ref = {}
    for c in json.loads(p.read_text())["conditions"]:
        ref[(float(c["amplitude"]), c["form"], int(c["signal_size"]), float(c["nominal_q"]))] = {
            "power": float(c["power_mean"]), "fdr": float(c["realised_fdr_mean"])}
    return ref


def aggregate_p2048(sec, M, conv, resid, finite, amps, forms, ks, arms, v1ref):
    qs = sec["nominal_fdr_targets"]
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
                            mu, se = mean_se(M[m][a][ia, jf, kk, :, iq])
                            row[m], row[m + "_se"] = mu, se
                        ref = v1ref.get((float(amp), form, int(k), float(q)))
                        if a == sec["v1_arm"] and ref is not None:
                            row["v1_reported_power"], row["v1_reported_fdr"] = ref["power"], ref["fdr"]
                            row["reproduces_v1"] = bool(abs(row["ko_pow"] - ref["power"])
                                                        <= max(2 * row["ko_pow_se"], 0.10))
                        conds.append(row)

    contrasts = []
    for cname, a, b in sec["contrasts"]:
        for ia, amp in enumerate(amps):
            for jf, form in enumerate(forms):
                for kk, k in enumerate(ks):
                    for iq, q in enumerate(qs):
                        row = {"contrast": cname, "a": a, "b": b, "amplitude": amp, "form": form,
                               "k": k, "q": q}
                        for m in ("ko_pow", "ko_fdr"):
                            dif = M[m][a][ia, jf, kk, :, iq] - M[m][b][ia, jf, kk, :, iq]
                            mu, se = mean_se(dif)
                            row[m + "_diff"], row[m + "_diff_se"] = mu, se
                            row[m + "_significant"] = bool(se > 0 and abs(mu) > 1.96 * se)
                        contrasts.append(row)

    summary = {"contrasts": {}, "arms": {}}
    for cname, a, b in sec["contrasts"]:
        cs = [c for c in contrasts if c["contrast"] == cname]
        summary["contrasts"][cname] = {
            "a": a, "b": b, "n_cells": len(cs),
            "mean_power_diff": float(np.mean([c["ko_pow_diff"] for c in cs])),
            "n_power_diff_significantly_positive": sum(c["ko_pow_significant"] and c["ko_pow_diff"] > 0 for c in cs),
            "n_power_diff_significantly_negative": sum(c["ko_pow_significant"] and c["ko_pow_diff"] < 0 for c in cs),
            "mean_fdr_diff": float(np.mean([c["ko_fdr_diff"] for c in cs]))}
    for a in names:
        cs = [c for c in conds if c["arm"] == a]
        summary["arms"][a] = {"mean_power": float(np.mean([c["ko_pow"] for c in cs])),
                              "mean_fdr": float(np.mean([c["ko_fdr"] for c in cs])),
                              "n_fdr_inflated": sum(c["ko_fdr"] - 1.96 * c["ko_fdr_se"] > c["q"] for c in cs),
                              "fraction_converged": float(conv[a].mean()),
                              "median_residual": float(np.median(resid[a])),
                              "fraction_nonfinite": float(1 - finite[a].mean())}
    rep_cells = [c for c in conds if "reproduces_v1" in c]
    summary["v1_reproduction"] = {
        "n_cells_compared": len(rep_cells),
        "n_cells_reproduced": sum(c["reproduces_v1"] for c in rep_cells),
        "D1_reproduced": bool(rep_cells) and sum(c["reproduces_v1"] for c in rep_cells) >= 0.8 * len(rep_cells)}
    return conds, contrasts, summary


def make_figure_p2048(conds, amps, forms, ks, arms, q_ref, rd):
    colors = ["#7f7f7f", "#c0392b", "#e07b39", "#3b6ea5", "#8e44ad", "#2e8b57"]
    fig, axes = plt.subplots(len(forms), len(amps), figsize=(5 * len(amps), 3.8 * len(forms)),
                             sharey=True, squeeze=False)
    for i, form in enumerate(forms):
        for j, amp in enumerate(amps):
            ax = axes[i, j]
            for ci, (name, _, _) in enumerate(arms):
                rows = sorted([c for c in conds if c["arm"] == name and c["form"] == form
                               and c["amplitude"] == amp and c["q"] == q_ref], key=lambda c: c["k"])
                ax.errorbar([c["k"] + (ci - 2) * 0.6 for c in rows], [c["ko_pow"] for c in rows],
                            yerr=[1.96 * c["ko_pow_se"] for c in rows], marker="o", ms=4, capsize=2,
                            color=colors[ci % len(colors)], label=name)
                ref = [(c["k"], c["v1_reported_power"]) for c in rows if "v1_reported_power" in c]
                if ref:
                    ax.scatter([r[0] for r in ref], [r[1] for r in ref], marker="x", s=60, color="k",
                               zorder=5, label="v1 reported" if (i, j) == (0, 0) else None)
            ax.set_title(f"{form}, amplitude {amp}", fontsize=10)
            ax.set_xticks(ks)
            ax.set_ylim(-0.03, 1.05)
            if i == len(forms) - 1:
                ax.set_xlabel("number of planted latents k")
            if j == 0:
                ax.set_ylabel(f"power at q = {q_ref}")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Stage 3, p = 2048: which change removes v1's power collapse? (95% intervals)", y=1.0)
    fig.tight_layout()
    fig.savefig(rd / "fig16_stage3_p2048_power.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main_p2048(args, cfg: dict, d, X_all: np.ndarray, device, rd: Path) -> None:
    sec = cfg["stage3_p2048"]
    arms = [tuple(a) for a in sec["arms"]]
    t0 = time.time()
    print("\n=== p = 2048 datasets ===")
    ds, info = build_datasets_p2048(cfg, sec, X_all)
    del X_all
    v1ref = load_v1_reference(sec["v1_reference"])

    if args.stage == "diagnose":
        amp, form, k = sec["diagnose"]["pilot_cell"]
        R = args.limit_reps or sec["diagnose"]["pilot_replicates"]
        print(f"\n=== pilot: amplitude {amp}, {form}, k = {k}, {R} replicates, all arms ===")
        M, conv, resid, finite, draws, timing = run_grid_p2048(ds, cfg, sec, device, R, [amp], [form], [k], arms)
        iq = sec["nominal_fdr_targets"].index(0.10)
        pilot = {}
        for name, dk, solver in arms:
            pw, fd = M["ko_pow"][name][0, 0, 0, :, iq], M["ko_fdr"][name][0, 0, 0, :, iq]
            pilot[name] = {"dataset": dk, "solver": solver, "power_q0.10": float(pw.mean()),
                           "fdr_q0.10": float(fd.mean()), "fraction_converged": float(conv[name].mean()),
                           "median_residual": float(np.median(resid[name]))}
            print(f"  {name:13s} ({dk}, {solver:5s}) power {pw.mean():.3f}  FDR {fd.mean():.3f}  "
                  f"converged {conv[name].mean():.0%}  median residual {np.median(resid[name]):.2e}")
        ref = v1ref.get((float(amp), form, int(k), 0.10))
        if ref:
            print(f"  v1 reported power for this cell: {ref['power']:.3f}")
        for dk, v in draws.items():
            print(f"  [{dk:11s}] lambda_max {np.mean(v['lambda_max']):.1f}  safe step {np.mean(v['safe_step']):.4f}"
                  f"  (v1 step {sec['v1_solver']['lr']} = {sec['v1_solver']['lr'] / np.mean(v['safe_step']):.1f}x)"
                  f"  mean corr(X_j, Xk_j) {np.mean(v['mean_corr_X_Xk']):.3f}")
        n_cells = len(sec["signal_amplitudes"]) * len(sec["functional_forms"]) * len(sec["signal_sizes"])
        n_fixed = sum(1 for a in arms if a[2] == "fixed")
        n_v1 = len(arms) - n_fixed
        R_full = sec["replicates"]
        est = (R_full * sum(timing["seconds_per_draw"].values())
               + R_full * n_cells * (n_fixed * (timing["seconds_per_fit"]["fixed"] or 0)
                                     + n_v1 * (timing["seconds_per_fit"]["v1"] or 0))) / 60
        print(f"  timing: {timing}  ->  projected full run on this machine: ~{est:.0f} min")
        (rd / "stage3_p2048_diagnose.json").write_text(json.dumps(
            {"info": info, "pilot": pilot, "draws": draws, "timing": timing,
             "projected_full_minutes": est, "minutes": (time.time() - t0) / 60}, indent=2, default=float))
        print(f"\nwrote {rd}/stage3_p2048_diagnose.json ({(time.time() - t0) / 60:.1f} min)")
        return

    amps, forms, ks = sec["signal_amplitudes"], sec["functional_forms"], sec["signal_sizes"]
    R = args.limit_reps or sec["replicates"]
    print(f"\n=== full: {len(arms)} arms x {len(amps)} amplitudes x {len(forms)} forms x {len(ks)} k x {R} reps ===")
    M, conv, resid, finite, draws, timing = run_grid_p2048(ds, cfg, sec, device, R, amps, forms, ks, arms)
    conds, contrasts, summary = aggregate_p2048(sec, M, conv, resid, finite, amps, forms, ks, arms, v1ref)

    rp = summary["v1_reproduction"]
    print(f"\nD1 v1 reproduction: {rp['n_cells_reproduced']}/{rp['n_cells_compared']} cells -> "
          f"{'REPRODUCED' if rp['D1_reproduced'] else 'NOT REPRODUCED'}")
    for a, s in summary["arms"].items():
        print(f"  {a:13s} mean power {s['mean_power']:.3f}  mean FDR {s['mean_fdr']:.3f}  "
              f"converged {s['fraction_converged']:.0%}  non-finite {s['fraction_nonfinite']:.0%}")
    for cname, s in summary["contrasts"].items():
        print(f"  contrast {cname:20s} ({s['a']} - {s['b']}): mean power diff {s['mean_power_diff']:+.3f}, "
              f"significant +{s['n_power_diff_significantly_positive']} / -{s['n_power_diff_significantly_negative']}"
              f" of {s['n_cells']}")

    out = {"config_hash": str(d["config_hash"]), "master_seed": cfg["master_seed"], "replicates": R,
           "amplitudes": amps, "forms": forms, "signal_sizes": ks, "arms": [list(a) for a in arms],
           "nominal_fdr_targets": sec["nominal_fdr_targets"], "info": info, "draws": draws,
           "timing": timing, "summary": summary, "conditions": conds, "contrasts": contrasts,
           "minutes": round((time.time() - t0) / 60, 1)}
    (rd / "stage3_p2048_fdr.json").write_text(json.dumps(out, indent=2, default=float))
    np.savez(rd / "stage3_p2048_records.npz",
             **{f"{m}__{a[0]}": M[m][a[0]] for m in METRICS for a in arms},
             **{f"converged__{a[0]}": conv[a[0]] for a in arms},
             **{f"residual__{a[0]}": resid[a[0]] for a in arms})
    make_figure_p2048(conds, amps, forms, ks, arms, 0.10, rd)
    print(f"\nwrote {rd}/stage3_p2048_fdr.json, stage3_p2048_records.npz, fig16 ({out['minutes']} min)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3 v2 — realised FDR with controls")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--stage", default="diagnose", choices=["diagnose", "full"])
    ap.add_argument("--limit-reps", type=int, default=None)
    ap.add_argument("--experiment", default="v2", choices=["v2", "p2048"],
                    help="v2: the p = 512 benchmark; p2048: the follow-up on v1's power collapse")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = load_config(args.config); rd = results_dir(cfg); s3 = cfg["stage3_v2"]
    cp = Path(args.cache) if args.cache else cache_path(cfg)
    if not cp.exists():
        raise SystemExit(f"Cache not found at {cp}. Run src/cache_activations.py first.")
    d = np.load(cp, allow_pickle=True)
    X_all = d[f"X_{cfg['aggregation']['primary']}"].astype(np.float32, copy=False)
    print(f"device: {device} | cache {d['config_hash']} | stage: {args.stage} | experiment: {args.experiment}")
    if args.experiment == "p2048":
        main_p2048(args, cfg, d, X_all, device, rd)
        return

    t0 = time.time()
    print("\n=== datasets and S matrices ===")
    ds, info = build_datasets(cfg, X_all)
    names = list(ds)
    print("\n=== global-null harness (k = 0) ===")
    harness = null_harness(ds, cfg, device)

    amps, forms, ks = s3["signal_amplitudes"], s3["functional_forms"], s3["signal_sizes"]
    R = args.limit_reps or s3["replicates"]
    lam = s3["lasso"]["lambda"]

    if args.stage == "diagnose":
        print("\n=== pilot (rule C1): amplitude 3.0, linear, k = 20 ===")
        pil = {}
        Rp = args.limit_reps or s3["diagnose"]["pilot_replicates"]
        M, cv, rs = run_grid(ds, cfg, device, Rp, lam, [3.0], ["linear"], [20])
        for n in names:
            pil[n] = {str(q): {"fdr": float(M["ko_fdr"][n][0, 0, 0, :, i].mean()),
                               "power": float(M["ko_pow"][n][0, 0, 0, :, i].mean())}
                      for i, q in enumerate(s3["nominal_fdr_targets"])}
            print(f"  [{n:10s}] " + "  ".join(f"q={q}: FDR {pil[n][str(q)]['fdr']:.3f} "
                  f"pow {pil[n][str(q)]['power']:.2f}" for q in s3["nominal_fdr_targets"])
                  + f"   converged {cv[n].mean():.0%}")
        (rd / "stage3_v2_diagnose.json").write_text(json.dumps(
            {"info": info, "null_harness": harness, "pilot": pil, "minutes": (time.time() - t0) / 60},
            indent=2, default=float))
        print(f"\nwrote {rd}/stage3_v2_diagnose.json ({(time.time()-t0)/60:.1f} min)")
        return

    print(f"\n=== full grid: {len(amps)} amplitudes x {len(forms)} forms x {len(ks)} k x {R} reps x "
          f"{len(names)} datasets ===")
    M, conv, resid = run_grid(ds, cfg, device, R, lam, amps, forms, ks)
    conds, contrasts, conv_stats = aggregate(cfg, M, conv, resid, amps, forms, ks, names)
    c1 = c1_check(conds)
    print(f"\nC1 (Gaussian control FDR <= q): {'HOLDS' if c1['C1_holds'] else 'VIOLATED'} "
          f"({c1['n_inflated']}/{c1['n_control_cells']} cells inflated)")
    for n in names:
        print(f"  convergence [{n}]: {conv_stats[n]['fraction_converged']:.0%} converged, "
              f"median residual {conv_stats[n]['median_residual']:.2e}")

    print("\n=== lambda sensitivity (amplitude {}, linear) ===".format(s3["lasso"]["sensitivity_amplitude"]))
    sens = {}
    for lam2 in s3["lasso"]["lambda_sensitivity"]:
        Ms, _, _ = run_grid(ds, cfg, device, R, lam2, [s3["lasso"]["sensitivity_amplitude"]], ["linear"],
                            ks, only_ds=["real_mvr", "gauss_mvr"])
        sens[str(lam2)] = {n: {str(k): {str(q): {"fdr": float(Ms["ko_fdr"][n][0, 0, kk, :, iq].mean()),
                                                 "power": float(Ms["ko_pow"][n][0, 0, kk, :, iq].mean())}
                                        for iq, q in enumerate(s3["nominal_fdr_targets"])}
                               for kk, k in enumerate(ks)} for n in ("real_mvr", "gauss_mvr")}

    out = {"config_hash": str(d["config_hash"]), "master_seed": cfg["master_seed"], "replicates": R,
           "amplitudes": amps, "forms": forms, "signal_sizes": ks,
           "nominal_fdr_targets": s3["nominal_fdr_targets"], "info": info, "null_harness": harness,
           "convergence": conv_stats, "C1": c1, "conditions": conds, "contrasts": contrasts,
           "lambda_sensitivity": sens, "minutes": round((time.time() - t0) / 60, 1)}
    (rd / "stage3_v2_fdr.json").write_text(json.dumps(out, indent=2, default=float))
    np.savez(rd / "stage3_v2_records.npz",
             **{f"{m}__{n}": M[m][n] for m in METRICS for n in names},
             **{f"converged__{n}": conv[n] for n in names})
    make_figures(conds, info, amps, forms, ks, names, 0.10, rd)
    print(f"\nwrote {rd}/stage3_v2_fdr.json, stage3_v2_records.npz, fig13-15 ({out['minutes']} min)")


if __name__ == "__main__":
    main()
