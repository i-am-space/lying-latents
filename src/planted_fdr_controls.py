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

Usage: python src/planted_fdr_controls.py --config config/default.yaml --device cuda --stage diagnose
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
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3 v2 — realised FDR with controls")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--stage", default="diagnose", choices=["diagnose", "full"])
    ap.add_argument("--limit-reps", type=int, default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = load_config(args.config); rd = results_dir(cfg); s3 = cfg["stage3_v2"]
    cp = Path(args.cache) if args.cache else cache_path(cfg)
    if not cp.exists():
        raise SystemExit(f"Cache not found at {cp}. Run src/cache_activations.py first.")
    d = np.load(cp, allow_pickle=True)
    X_all = d[f"X_{cfg['aggregation']['primary']}"].astype(np.float32, copy=False)
    print(f"device: {device} | cache {d['config_hash']} | stage: {args.stage}")

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
