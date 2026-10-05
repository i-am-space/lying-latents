"""Stage 4 as specified in the proposal — three repairs vs Gaussian knockoffs.

config/preregistration.yaml stage4_amendment_2 (fixed before any real-data run). Arms, all on the
same planted-signal benchmark with identical labels:

  gauss_real     real latents, second-order Gaussian knockoffs              (Stage 3 baseline)
  hurdle         real latents, hurdle knockoffs by sequential conditional independent pairs:
                 Pr(X_j > 0 | X_-j, X~_<j) by ridge logistic regression, log X_j | X_j > 0 by
                 ridge linear regression (log-normal positive part)
  binarised      knockoffs for Z_j = 1[X_j > 0] by the same scheme (logistic part only);
                 the knockoff filter runs on [Z, Z~]
  evalue         sample splitting: lasso selects on half A, logistic Wald p-values on half B,
                 calibrated to e-values, e-BH over all latents
  gauss_ceiling  Gaussian data with the real covariance, Gaussian knockoffs (no zero atom)

Swap two-sample tests (Stage 2) check exchangeability of every knockoff arm; the two Gaussian
arms are the known-violation and valid controls for those tests (rule F0).

Usage: python src/stage4_repairs.py --config config/default.yaml --device cuda --stage diagnose
       python src/stage4_repairs.py --config config/default.yaml --device cuda --stage full
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

# The SCIP loop allocates a slightly larger feature matrix for each of the p latents. With the
# default CUDA caching allocator those differently sized blocks fragment the cache, which then
# grows until the GPU is full (seen on an 8 GB and a 32 GB card). Expandable segments let cached
# blocks grow in place. Memory layout only; no effect on any number.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import cache_path, load_config, results_dir, rng_for
from knockoff_audit import estimate_cov, standardise, swap_sweep
from planted_fdr import evaluate_discoveries, generate_planted_labels
from planted_fdr_controls import METRICS, cell_rng, fit_lasso_checked, mean_se, score_fit

ARM_COLOR = {"gauss_real": "#c0392b", "hurdle": "#2e8b57", "binarised": "#8e44ad",
             "evalue": "#e07b39", "gauss_ceiling": "#3b6ea5", "gauss_mvr": "#f1948a",
             "gauss_ceiling_mvr": "#85c1e9"}
KNOCKOFF_METHODS = ("gaussian", "hurdle_scip", "binary_scip", "gaussian_control",
                    "gaussian_mvr", "gaussian_control_mvr")
CONTROL_METHODS = ("gaussian_control", "gaussian_control_mvr")


# ---------------------------------------------------------------------------
# GPU solvers
# ---------------------------------------------------------------------------

def _penalised_logloss(F: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, b: torch.Tensor,
                       ridge: float) -> float:
    eta = (F @ beta.float() + b.float()).double()
    return float((torch.nn.functional.softplus(eta) - y.double() * eta).mean() + 0.5 * ridge * (beta ** 2).sum())


def newton_logistic(F: torch.Tensor, y: torch.Tensor, ridge: float, max_iter: int, tol: float,
                    beta0: torch.Tensor | None = None, b0: float | None = None):
    """Ridge logistic regression (mean log-loss + ridge/2 * |beta|^2, intercept unpenalised) by
    Newton's method with a backtracking (Armijo) line search. Matrix products in float32, the
    linear solve and the loss in float64. Converged when half the squared Newton decrement,
    g' H^-1 g / 2 (the predicted loss gap to the optimum), is below tol.

    The line search is required, not optional: on the real SCIP loop, full Newton steps diverged
    in most fits (loss rising, steps of ~1e9, fitted firing probability collapsing to 0), because
    once predictions saturate the intercept's curvature falls to ~1e-10.
    Returns (beta, intercept, converged, iterations, augmented Hessian at the last iterate)."""
    n, d = F.shape
    dev = F.device
    beta = beta0.clone() if beta0 is not None else torch.zeros(d, dtype=torch.float64, device=dev)
    if b0 is None:
        ybar = float(y.mean().clamp(1e-6, 1 - 1e-6))
        b0 = math.log(ybar / (1 - ybar))
    b = torch.tensor([b0], dtype=torch.float64, device=dev)
    eye = torch.eye(d, dtype=torch.float64, device=dev)
    f = _penalised_logloss(F, y, beta, b, ridge)
    converged, it, Ha = False, 0, None
    for it in range(1, max_iter + 1):
        p = torch.sigmoid(F @ beta.float() + b.float())
        r = p - y
        w = (p * (1 - p)).clamp(min=1e-10)
        Fw = F * w.unsqueeze(1)
        H = (F.t() @ Fw).double() / n + ridge * eye
        hb = Fw.sum(0).double() / n
        del Fw
        Ha = torch.empty((d + 1, d + 1), dtype=torch.float64, device=dev)
        Ha[:d, :d], Ha[:d, d], Ha[d, :d], Ha[d, d] = H, hb, hb, w.double().mean()
        g = torch.cat([(F.t() @ r).double() / n + ridge * beta, r.double().mean().reshape(1)])
        step = torch.linalg.solve(Ha, g)
        if not torch.isfinite(step).all():
            break
        dec = float(g @ step)                          # squared Newton decrement (> 0: Ha is PD)
        if dec / 2 < tol:
            converged = True
            break
        t, accepted = 1.0, False
        while t >= 1e-8:
            beta_c, b_c = beta - t * step[:d], b - t * step[d:]
            f_c = _penalised_logloss(F, y, beta_c, b_c, ridge)
            if f_c <= f - 1e-4 * t * dec:
                beta, b, f, accepted = beta_c, b_c, f_c, True
                break
            t *= 0.5
        if not accepted:                               # no decrease possible at float32 precision
            converged = dec / 2 < 1e-6
            break
    return beta, float(b), converged, it, Ha


def ridge_regression(F: torch.Tensor, t: torch.Tensor, ridge: float):
    """Ridge linear regression of t on F (intercept unpenalised): returns (gamma, intercept, sigma)."""
    n, d = F.shape
    Fm, tm = F.mean(0), t.mean()
    Fc, tc = F - Fm, t - tm
    A = (Fc.t() @ Fc).double() / n + ridge * torch.eye(d, dtype=torch.float64, device=F.device)
    gamma = torch.linalg.solve(A, (Fc.t() @ tc).double() / n)
    c = float(tm) - float(Fm.double() @ gamma)
    resid = t.double() - (F @ gamma.float()).double() - c
    return gamma, c, float(resid.std()) if n > 2 else 1.0


# ---------------------------------------------------------------------------
# SCIP knockoff sampler (hurdle and binarised)
# ---------------------------------------------------------------------------

class SCIPSampler:
    """Sequential conditional independent pairs with fitted hurdle (or logistic) conditionals.

    For j in index order, the conditional law of latent j given all other real latents and the
    knockoffs already drawn is fitted on the data and X~_j is drawn from it. Covariates are
    log1p(x) (hurdle) or 1[x > 0] (binarised), standardised with the real data's moments, for real
    and knockoff columns alike. The fitted models are exact only if correctly specified; validity is
    checked by the swap tests, not assumed.
    """

    def __init__(self, X: np.ndarray, kind: str, scip: dict, device):
        if kind not in ("hurdle", "binary"):
            raise ValueError(kind)
        self.kind, self.scip, self.device = kind, scip, device
        pos = X > 0
        V = np.log1p(X) if kind == "hurdle" else pos.astype(np.float64)
        self.m = V.mean(0)
        self.s = np.where(V.std(0) > 1e-8, V.std(0), 1.0)
        self.U = torch.from_numpy(((V - self.m) / self.s).astype(np.float32)).to(device)
        self.pos = torch.from_numpy(pos).to(device)
        if kind == "hurdle":
            lx = np.where(pos, np.log(np.where(pos, X, 1.0)), 0.0)
            self.logx = torch.from_numpy(lx.astype(np.float32)).to(device)
            lo = np.where(pos.any(0), np.where(pos, lx, np.inf).min(0), 0.0)
            hi = np.where(pos.any(0), np.where(pos, lx, -np.inf).max(0), 0.0)
            self.clip = (lo - 1.0, hi + 1.0)       # sampled log x stays within the observed range +- 1
        self.m_t = torch.from_numpy(self.m.astype(np.float32)).to(device)
        self.s_t = torch.from_numpy(self.s.astype(np.float32)).to(device)
        self.warm: dict[int, tuple[torch.Tensor, float]] = {}

    def sample(self, seed: int) -> tuple[np.ndarray, dict]:
        """One knockoff draw on the raw scale (hurdle: activations; binary: 0/1)."""
        sc = self.scip
        n, p = self.U.shape
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        Uk = torch.empty((n, p), dtype=torch.float32, device=self.device)
        Xk = torch.zeros((n, p), dtype=torch.float32, device=self.device)
        n_conv, iters, few_pos = 0, 0, 0
        for j in range(p):
            F = torch.cat([self.U[:, :j], self.U[:, j + 1:], Uk[:, :j]], dim=1)
            y = self.pos[:, j].float()
            n_pos = int(y.sum())
            if n_pos in (0, n):                       # constant latent: copy its value
                on = torch.full((n,), n_pos == n, device=self.device)
                few_pos += 1
                n_conv += 1
            else:
                wb = self.warm.get(j)
                beta, b, conv, it, _ = newton_logistic(F, y, sc["logistic_ridge"], sc["newton_max_iter"],
                                                       sc["newton_tol"], *(wb if wb else (None, None)))
                self.warm[j] = (beta, b)
                n_conv += conv
                iters += it
                pi = torch.sigmoid(F @ beta.float() + b)
                on = torch.rand(n, generator=gen, device=self.device) < pi
            if self.kind == "hurdle":
                rows = self.pos[:, j]
                if n_pos >= 5:
                    gamma, c, sigma = ridge_regression(F[rows], self.logx[rows, j], sc["positive_ridge"])
                    mu = F @ gamma.float() + c
                else:
                    mu = torch.full((n,), float(self.logx[rows, j].mean()) if n_pos else 0.0, device=self.device)
                    sigma = 1.0
                lz = mu + sigma * torch.randn(n, generator=gen, device=self.device)
                lz = lz.clamp(float(self.clip[0][j]), float(self.clip[1][j]))
                xk = torch.where(on, torch.exp(lz), torch.zeros_like(lz))
                Uk[:, j] = (torch.log1p(xk) - self.m_t[j]) / self.s_t[j]
            else:
                xk = on.float()
                Uk[:, j] = (xk - self.m_t[j]) / self.s_t[j]
            Xk[:, j] = xk
            del F
        info = {"fraction_logistic_converged": n_conv / p, "mean_newton_iterations": iters / max(1, p - few_pos),
                "n_constant_latents": few_pos}
        return Xk.double().cpu().numpy(), info


# ---------------------------------------------------------------------------
# e-value sample splitting
# ---------------------------------------------------------------------------

def evalue_discoveries(Z: torch.Tensor, y: torch.Tensor, split: np.ndarray, ev: dict, lasso: dict,
                       qs) -> dict:
    """Select on half A with the lasso, test on half B with logistic Wald p-values, calibrate to
    e-values e = kappa * p^(kappa - 1), and run e-BH at each q over all p latents."""
    n, p = Z.shape
    nA = int(n * ev["split_fraction"])
    A = torch.as_tensor(split[:nA], device=Z.device)
    B = torch.as_tensor(split[nA:], device=Z.device)
    w, _, _, _ = fit_lasso_checked(Z[A], y[A], ev["selection_lambda"], lasso["max_iter"], lasso["tol"])
    sel = torch.nonzero(w.abs() > 0).flatten()
    if sel.numel() > ev["max_selected"]:
        sel = sel[torch.argsort(w[sel].abs(), descending=True)[: ev["max_selected"]]]
    if sel.numel() == 0:
        return {q: set() for q in qs}
    beta, _, _, _, Ha = newton_logistic(Z[B][:, sel], y[B], ev["refit_ridge"], 50, 1e-8)
    try:
        cov = torch.linalg.inv(Ha) / B.numel()
    except torch.linalg.LinAlgError:              # singular (collinear selected latents): pseudo-inverse
        cov = torch.linalg.pinv(Ha) / B.numel()
    z = beta / torch.sqrt(torch.diag(cov)[: sel.numel()].clamp(min=1e-300))
    pval = torch.special.erfc(z.abs() / math.sqrt(2.0)).clamp(min=1e-300)   # two-sided
    kap = ev["calibrator_kappa"]
    e = torch.zeros(p, dtype=torch.float64, device=Z.device)
    e[sel] = kap * pval ** (kap - 1.0)
    order = torch.argsort(e, descending=True)
    es = e[order].cpu().numpy()
    ks = np.arange(1, p + 1)
    out = {}
    for q in qs:
        ok = np.nonzero(es >= p / (q * ks))[0]
        k_star = int(ok.max() + 1) if ok.size else 0
        out[q] = set(order[:k_star].cpu().numpy().tolist())
    return out


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def build_block_mvr(Z, Sigma, Zg, Sigma_g, sec: dict, device) -> dict:
    """Block-diagonal MVR S (knockpy's blockdiag approximation with its line search, so 2 Sigma - S
    stays PSD) for the real latents and the Gaussian control, with GPU samplers of the same law as
    knockpy's GaussianSampler (stage3_followups.GaussKnockoffGPU, checked against knockpy there)."""
    from knockpy import smatrix
    from stage3_followups import GaussKnockoffGPU   # imported here: stage3_followups imports this module

    def solve(Sig):
        t = time.time()
        S = np.asarray(smatrix.compute_smatrix(Sig, method="mvr", how_approx="blockdiag",
                                               max_block=sec["mvr_max_block"]))
        return S, time.time() - t

    S, t_real = solve(Sigma)
    if np.linalg.eigvalsh(2 * Sigma_g - S).min() > 1e-8:
        S_g, t_g, note = S, 0.0, "reused the real-latent S"
    else:
        (S_g, t_g), note = solve(Sigma_g), "recomputed for the control covariance"
    Zt = torch.from_numpy(Z.astype(np.float32)).to(device)
    Zgt = torch.from_numpy(Zg.astype(np.float32)).to(device)
    info = {"mean_s": float(np.diag(S).mean()), "seconds": t_real, "control_S": note, "control_seconds": t_g,
            "min_eig_2Sigma_minus_S": float(np.linalg.eigvalsh(2 * Sigma - S).min())}
    print(f"  block MVR: mean s={info['mean_s']:.4f} ({t_real:.0f}s), control: {note}", flush=True)
    return {"gauss_mvr": GaussKnockoffGPU(Zt, Sigma, S), "gauss_ctrl_mvr": GaussKnockoffGPU(Zgt, Sigma_g, S_g),
            "mvr_info": info}


def build(cfg: dict, sec: dict, X_all: np.ndarray, device, methods=None) -> dict:
    """methods: the arm methods that will run; samplers for the others are not built."""
    from knockpy import smatrix
    from knockpy.knockoffs import GaussianSampler

    n_all, p_all = X_all.shape
    # the Stage 3 p2048 rows: the first draw from s3p2048_data
    rows = np.sort(rng_for(cfg, "s3p2048_data").choice(n_all, sec["n_rows"], replace=False))
    rng = rng_for(cfg, "s4r_data")
    cols = np.arange(p_all) if sec["p"] >= p_all else np.sort(rng.choice(p_all, sec["p"], replace=False))
    X = X_all[np.ix_(rows, cols)].astype(np.float64)
    Z, mu, sd = standardise(X)
    n, p = Z.shape
    Sigma = estimate_cov(Z, sec["covariance"])
    S = np.asarray(smatrix.compute_smatrix(Sigma, method=sec["s_method"]))
    Lc = np.linalg.cholesky(Sigma + 1e-10 * np.eye(p))
    Zg, _, _ = standardise(rng.standard_normal((n, p)) @ Lc.T)
    Sigma_g = estimate_cov(Zg, sec["covariance"])
    S_g = S if np.linalg.eigvalsh(2 * Sigma_g - S).min() > 1e-8 else \
        np.asarray(smatrix.compute_smatrix(Sigma_g, method=sec["s_method"]))
    Zb = (X > 0).astype(np.float64)
    mb, sb = Zb.mean(0), np.where(Zb.std(0) > 1e-8, Zb.std(0), 1.0)
    print(f"  n={n} p={p} n/p={n / p:.1f}  median Pr(X=0)={np.median((X == 0).mean(0)):.3f}  "
          f"equicorrelated s={np.diag(S).mean():.4f}", flush=True)
    t = time.time()
    need = set(methods) if methods else set(KNOCKOFF_METHODS)
    samplers = {}
    if "hurdle_scip" in need:
        samplers["hurdle_scip"] = SCIPSampler(X, "hurdle", sec["scip"], device)
    if "binary_scip" in need:
        samplers["binary_scip"] = SCIPSampler(X, "binary", sec["scip"], device)
    mvr = {}
    if need & {"gaussian_mvr", "gaussian_control_mvr"}:
        mvr = build_block_mvr(Z, Sigma, Zg, Sigma_g, sec, device)
    return {"X": X, "Z": Z, "mu": mu, "sd": sd, "Zg": Zg, "Zb": (Zb - mb) / sb, "mb": mb, "sb": sb,
            "gauss": GaussianSampler(Z, mu=Z.mean(0), Sigma=Sigma, S=S),
            "gauss_ctrl": GaussianSampler(Zg, mu=Zg.mean(0), Sigma=Sigma_g, S=S_g),
            "scip": samplers, "cols": cols, **mvr,
            "info": {"n": n, "p": p, "s_equicorrelated": float(np.diag(S).mean()),
                     "median_zero_mass": float(np.median((X == 0).mean(0))),
                     "sampler_setup_seconds": time.time() - t,
                     **({"block_mvr": mvr["mvr_info"]} if mvr else {})}}


def data_and_knockoffs(method: str, D: dict, seed: int):
    """(data matrix used by the filter, its knockoff, extra info) on the standardised scale."""
    if method == "gaussian":
        np.random.seed(seed)
        return D["Z"], D["gauss"].sample_knockoffs(), {}
    if method == "gaussian_control":
        np.random.seed(seed)
        return D["Zg"], D["gauss_ctrl"].sample_knockoffs(), {}
    if method == "gaussian_mvr":
        return D["Z"], D["gauss_mvr"].sample(seed).double().cpu().numpy(), {}
    if method == "gaussian_control_mvr":
        return D["Zg"], D["gauss_ctrl_mvr"].sample(seed).double().cpu().numpy(), {}
    if method == "hurdle_scip":
        Xk, info = D["scip"]["hurdle_scip"].sample(seed)
        info["zero_mass_error_max"] = float(np.abs((D["X"] == 0).mean(0) - (Xk == 0).mean(0)).max())
        return D["Z"], (Xk - D["mu"]) / D["sd"], info
    if method == "binary_scip":
        Zbk, info = D["scip"]["binary_scip"].sample(seed)
        info["zero_mass_error_max"] = float(np.abs((D["X"] > 0).mean(0) - Zbk.mean(0)).max())
        return D["Zb"], (Zbk - D["mb"]) / D["sb"], info
    raise ValueError(method)


def mean_corr(A: np.ndarray, B: np.ndarray) -> float:
    ac, bc = A - A.mean(0), B - B.mean(0)
    den = np.sqrt((ac ** 2).sum(0) * (bc ** 2).sum(0))
    return float(np.mean(np.where(den > 0, (ac * bc).sum(0) / np.where(den > 0, den, 1.0), 0.0)))


# ---------------------------------------------------------------------------
# exchangeability, grid, aggregation
# ---------------------------------------------------------------------------

def exchangeability(cfg: dict, sec: dict, D: dict, arms, arm_ids=None) -> dict:
    ex = sec["exchangeability"]
    rng = rng_for(cfg, "s4r_diagnostics")
    out = {}
    for ai, (name, method) in enumerate(arms):
        if method not in KNOCKOFF_METHODS:
            continue
        seed = int(cell_rng(cfg, "s4r_diagnostics", (arm_ids or {}).get(name, ai)).integers(2**31))
        t = time.time()
        Dm, Dk, info = data_and_knockoffs(method, D, seed)
        print(f"  [{name}] draw {time.time() - t:.0f}s", flush=True)
        rows = swap_sweep(Dm, Dk, cfg, rng, ex["swap_sizes"], ex["swap_replicates"])
        out[name] = {"swap": rows, "mean_corr_X_Xk": mean_corr(Dm, Dk), "draw_seconds": time.time() - t, **info}
    p = D["Z"].shape[1]
    full = {n: next(r["auc"] for r in v["swap"] if r["swap_size"] == p) for n, v in out.items()}
    mmdp = {n: next(r["p_value"] for r in v["swap"] if r["swap_size"] == p) for n, v in out.items()}
    return {"arms": out, "full_swap_auc": full, "full_swap_mmd_p": mmdp,
            "F0_diagnostics_valid": bool(full.get("gauss_real", 0) >= 0.90
                                         and 0.45 <= full.get("gauss_ceiling", 0) <= 0.55),
            "F1_not_detected_as_violated": {n: bool(full[n] <= 0.55 and mmdp[n] >= 0.05)
                                            for n in full if n in ("hurdle", "binarised")}}


def run_grid(cfg: dict, sec: dict, D: dict, device, R: int, amps, forms, ks, arms,
             checkpoint: Path | None = None, arm_ids: dict | None = None):
    """Every replicate's seeds are derived from its own index, so a run resumed from a checkpoint
    gives the same numbers as an uninterrupted one."""
    qs, lam = sec["nominal_fdr_targets"], sec["lasso"]["lambda"]
    names = [a[0] for a in arms]
    p = D["Z"].shape[1]
    shape = (len(amps), len(forms), len(ks), R)
    M = {m: {a: np.zeros(shape + (len(qs),), dtype=np.float32) for a in names} for m in METRICS}
    corr = {a: [] for a in names}
    draw_info = {a: [] for a in names}
    t_draw = {a: 0.0 for a in names}
    t_fit = {a: 0.0 for a in names}
    no_r, no_thr = np.zeros(p), {q: float("inf") for q in qs}
    Zt_real = torch.from_numpy(D["Z"].astype(np.float32)).to(device)
    start = 0
    if checkpoint is not None and checkpoint.exists():
        ck = json.loads(checkpoint.read_text())
        if ck["shape"] == list(shape) and ck["arms"] == [list(a) for a in arms]:
            z = np.load(checkpoint.with_suffix(".npz"))
            for m in METRICS:
                for a in names:
                    M[m][a] = z[f"{m}__{a}"]
            corr, draw_info, start = ck["corr"], ck["draw_info"], ck["reps_done"]
            # the SCIP fits warm-start from the previous replicate's solution; restoring that state
            # makes a resumed run identical to an uninterrupted one
            warm = torch.load(checkpoint.with_suffix(".pt"), map_location=device)
            for k, v in warm.items():
                D["scip"][k].warm = v
            print(f"  resuming from checkpoint: {start}/{R} replicates done", flush=True)
    t0 = time.time()
    for rep in range(start, R):
        for ai, (name, method) in enumerate(arms):
            t = time.time()
            if method in KNOCKOFF_METHODS:
                seed = int(cell_rng(cfg, "s4r_knockoff", (arm_ids or {}).get(name, ai), rep).integers(2**31))
                Dm, Dk, info = data_and_knockoffs(method, D, seed)
                corr[name].append(mean_corr(Dm, Dk))
                draw_info[name].append(info)
                Phi = torch.from_numpy(np.hstack([Dm, Dk]).astype(np.float32)).to(device)
                del Dk
            else:
                split = cell_rng(cfg, "s4r_split", rep).permutation(D["Z"].shape[0])
            t_draw[name] += time.time() - t
            label_Z = D["Zg"] if method in CONTROL_METHODS else D["Z"]
            for ia, amp in enumerate(amps):
                for jf, form in enumerate(forms):
                    for kk, k in enumerate(ks):
                        rng = cell_rng(cfg, "s4r_planted", ia, jf, kk, rep)
                        S_idx = np.sort(rng.choice(p, size=k, replace=False))
                        truth = set(S_idx.tolist())
                        y_np, _ = generate_planted_labels(label_Z, S_idx, form, amp, rng)
                        y_t = torch.from_numpy(y_np.astype(np.float32)).to(device)
                        t = time.time()
                        if method in KNOCKOFF_METHODS:
                            w, _, _, _ = fit_lasso_checked(Phi, y_t, lam, sec["lasso"]["max_iter"], sec["lasso"]["tol"])
                            w_np = w.cpu().numpy()
                            sc = score_fit(np.abs(w_np[:p]) - np.abs(w_np[p:]), no_r, no_thr, qs, truth)
                        else:
                            disc = evalue_discoveries(Zt_real, y_t, split, sec["evalue"], sec["lasso"], qs)
                            sc = {m: np.full(len(qs), np.nan, dtype=np.float32) for m in METRICS}
                            for iq, q in enumerate(qs):
                                sc["ko_fdr"][iq], sc["ko_pow"][iq] = evaluate_discoveries(disc[q], truth)
                                sc["ko_nd"][iq] = len(disc[q])
                        t_fit[name] += time.time() - t
                        for m in METRICS:
                            M[m][name][ia, jf, kk, rep] = sc[m]
            if method in KNOCKOFF_METHODS:
                del Phi
        if checkpoint is not None:
            np.savez(checkpoint.with_suffix(".npz"), **{f"{m}__{a}": M[m][a] for m in METRICS for a in names})
            torch.save({k: smp.warm for k, smp in D["scip"].items()}, checkpoint.with_suffix(".pt"))
            checkpoint.write_text(json.dumps({"shape": list(shape), "arms": [list(a) for a in arms],
                                              "reps_done": rep + 1, "corr": corr, "draw_info": draw_info},
                                             default=float))
        el = time.time() - t0
        done = rep + 1 - start
        print(f"  rep {rep + 1:>2}/{R}  elapsed {el / 60:.1f}m  ETA {el / done * (R - rep - 1) / 60:.1f}m", flush=True)
    n_cells = len(amps) * len(forms) * len(ks)
    timing = {"seconds_per_draw": {a: v / R for a, v in t_draw.items()},
              "seconds_per_cell_fit": {a: v / (R * n_cells) for a, v in t_fit.items()}}
    return M, corr, draw_info, timing


def aggregate(sec: dict, M, amps, forms, ks, arms):
    qs = sec["nominal_fdr_targets"]
    names = [a[0] for a in arms]
    conds = []
    for a in names:
        for ia, amp in enumerate(amps):
            for jf, form in enumerate(forms):
                for kk, k in enumerate(ks):
                    for iq, q in enumerate(qs):
                        row = {"arm": a, "amplitude": amp, "form": form, "k": k, "q": q,
                               "power_capped_k_lt_1_over_q": bool(k < math.ceil(1 / q))}
                        for m in ("ko_fdr", "ko_pow", "ko_nd"):
                            row[m], row[m + "_se"] = mean_se(M[m][a][ia, jf, kk, :, iq])
                        row["fdr_inflated"] = bool(row["ko_fdr"] - 1.96 * row["ko_fdr_se"] > q)
                        conds.append(row)
    contrasts = []
    for cname, a, b in sec["contrasts"]:
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
                              "F2_fdr_valid": bool(not any(c["fdr_inflated"] for c in cs))}
    for cname, a, b in sec["contrasts"]:
        cs = [c for c in contrasts if c["contrast"] == cname]
        summary["contrasts"][cname] = {
            "a": a, "b": b, "n_cells": len(cs),
            "mean_power_diff": float(np.mean([c["ko_pow_diff"] for c in cs])),
            "n_power_sig_positive": int(sum(c["ko_pow_significant"] and c["ko_pow_diff"] > 0 for c in cs)),
            "n_power_sig_negative": int(sum(c["ko_pow_significant"] and c["ko_pow_diff"] < 0 for c in cs)),
            "mean_fdr_diff": float(np.mean([c["ko_fdr_diff"] for c in cs])),
            "n_fdr_sig_positive": int(sum(c["ko_fdr_significant"] and c["ko_fdr_diff"] > 0 for c in cs))}
    gap = summary["contrasts"].get("atom_gap", {}).get("mean_power_diff")
    summary["gap_closed_fraction"] = {c: (float(v["mean_power_diff"] / gap) if gap and abs(gap) > 1e-9 else None)
                                      for c, v in summary["contrasts"].items() if c != "atom_gap"}
    return conds, contrasts, summary


def make_figure(conds, amps, arms, q_ref, rd: Path, fname: str = "fig18_stage4_repairs.png") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (met, lab) in zip(axes, (("ko_pow", "power"), ("ko_fdr", "realised FDR"))):
        for name, _ in arms:
            vals = [np.mean([c[met] for c in conds if c["arm"] == name and c["amplitude"] == a and c["q"] == q_ref])
                    for a in amps]
            ax.plot(amps, vals, marker="o", color=ARM_COLOR.get(name, "gray"), label=name)
        if met == "ko_fdr":
            ax.axhline(q_ref, color="k", ls="--", lw=1)
        ax.set_xscale("log"); ax.set_xlabel("signal amplitude"); ax.set_ylabel(lab)
        ax.set_title(f"{lab} at q = {q_ref} (mean over forms and k)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(rd / fname, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 4 — the proposal's repairs vs Gaussian knockoffs")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--stage", default="diagnose", choices=["diagnose", "full"])
    ap.add_argument("--limit-reps", type=int, default=None)
    ap.add_argument("--skip-exchangeability", action="store_true")
    ap.add_argument("--experiment", default="repairs", choices=["repairs", "mvr"],
                    help="repairs: the five pre-registered arms; mvr: add block-MVR Gaussian arms (stage4_amendment_3)")
    args = ap.parse_args()
    if args.experiment == "mvr":
        main_mvr(args)
        return

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = load_config(args.config)
    rd = results_dir(cfg)
    sec = cfg["stage4_repairs"]
    cp = Path(args.cache) if args.cache else cache_path(cfg)
    if not cp.exists():
        raise SystemExit(f"Cache not found at {cp}. Run src/cache_activations.py first.")
    d = np.load(cp, allow_pickle=True)
    X_all = d[f"X_{cfg['aggregation']['primary']}"].astype(np.float32, copy=False)
    arms = [tuple(a) for a in sec["arms"]]
    print(f"device: {device} | cache {d['config_hash']} | stage: {args.stage}")
    t0 = time.time()

    print("\n=== data and samplers ===")
    D = build(cfg, sec, X_all, device)
    del X_all

    exch = None
    if not args.skip_exchangeability:
        print("\n=== exchangeability swap tests (F0 controls, F1 repairs) ===")
        exch = exchangeability(cfg, sec, D, arms)
        for n in exch["full_swap_auc"]:
            a = exch["arms"][n]
            print(f"  {n:13s} full-swap AUC {exch['full_swap_auc'][n]:.3f}  MMD p {exch['full_swap_mmd_p'][n]:.3f}  "
                  f"corr(X,Xk) {a['mean_corr_X_Xk']:.3f}"
                  + (f"  max zero-mass error {a['zero_mass_error_max']:.3f}" if "zero_mass_error_max" in a else ""))
        print(f"  F0 diagnostics valid: {exch['F0_diagnostics_valid']}  "
              f"F1 not detected as violated: {exch['F1_not_detected_as_violated']}")

    if args.stage == "diagnose":
        amp, form, k = sec["diagnose"]["pilot_cell"]
        R = args.limit_reps or sec["diagnose"]["pilot_replicates"]
        print(f"\n=== pilot: amplitude {amp}, {form}, k = {k}, {R} replicate(s), all arms ===")
        M, corr, dinfo, timing = run_grid(cfg, sec, D, device, R, [amp], [form], [k], arms)
        iq = sec["nominal_fdr_targets"].index(0.10)
        pilot = {a[0]: {"power_q0.10": float(np.nanmean(M["ko_pow"][a[0]][0, 0, 0, :, iq])),
                        "fdr_q0.10": float(np.nanmean(M["ko_fdr"][a[0]][0, 0, 0, :, iq]))} for a in arms}
        for a, v in pilot.items():
            print(f"  {a:13s} power {v['power_q0.10']:.3f}  FDR {v['fdr_q0.10']:.3f}")
        n_cells = len(sec["signal_amplitudes"]) * len(sec["functional_forms"]) * len(sec["signal_sizes"])
        est = sec["replicates"] * (sum(timing["seconds_per_draw"].values())
                                   + n_cells * sum(timing["seconds_per_cell_fit"].values())) / 60
        print(f"  timing {json.dumps(timing, default=float)}\n  -> projected full run on this machine: ~{est:.0f} min")
        (rd / "stage4_repairs_diagnose.json").write_text(json.dumps(
            {"info": D["info"], "exchangeability": exch, "pilot": pilot, "draw_info": dinfo, "timing": timing,
             "projected_full_minutes": est, "minutes": (time.time() - t0) / 60}, indent=2, default=float))
        print(f"\nwrote {rd}/stage4_repairs_diagnose.json ({(time.time() - t0) / 60:.1f} min)")
        return

    amps, forms, ks = sec["signal_amplitudes"], sec["functional_forms"], sec["signal_sizes"]
    R = args.limit_reps or sec["replicates"]
    print(f"\n=== full: {len(arms)} arms x {len(amps)} amplitudes x {len(forms)} forms x {len(ks)} k x {R} reps ===")
    ckpt = rd / "stage4_repairs_checkpoint.json"
    M, corr, dinfo, timing = run_grid(cfg, sec, D, device, R, amps, forms, ks, arms, checkpoint=ckpt)
    conds, contrasts, summary = aggregate(sec, M, amps, forms, ks, arms)
    print("\n=== results ===")
    for a, s in summary["arms"].items():
        print(f"  {a:13s} mean power {s['mean_power']:.3f}  mean FDR {s['mean_fdr']:.4f}  max FDR {s['max_fdr']:.3f}  "
              f"FDR-inflated cells {s['n_fdr_inflated']}" + (f"  corr(X,Xk) {np.mean(corr[a]):.3f}" if corr[a] else ""))
    for c, s in summary["contrasts"].items():
        print(f"  {c:15s} ({s['a']} - {s['b']}): power {s['mean_power_diff']:+.3f}  sig +{s['n_power_sig_positive']}"
              f"/-{s['n_power_sig_negative']} of {s['n_cells']}  FDR {s['mean_fdr_diff']:+.4f} "
              f"(sig higher in {s['n_fdr_sig_positive']})")
    print(f"  share of the no-atom gap closed: {summary['gap_closed_fraction']}")
    out = {"config_hash": str(d["config_hash"]), "master_seed": cfg["master_seed"], "replicates": R,
           "amplitudes": amps, "forms": forms, "signal_sizes": ks, "arms": [list(a) for a in arms],
           "nominal_fdr_targets": sec["nominal_fdr_targets"], "info": D["info"], "exchangeability": exch,
           "mean_corr_X_Xk": {a: (float(np.mean(v)) if v else None) for a, v in corr.items()},
           "draw_info": dinfo, "timing": timing, "summary": summary, "conditions": conds, "contrasts": contrasts,
           "minutes": round((time.time() - t0) / 60, 1)}
    (rd / "stage4_repairs.json").write_text(json.dumps(out, indent=2, default=float))
    np.savez(rd / "stage4_repairs_records.npz", **{f"{m}__{a[0]}": M[m][a[0]] for m in METRICS for a in arms})
    make_figure(conds, amps, arms, 0.10, rd)
    for f in (ckpt, ckpt.with_suffix(".npz"), ckpt.with_suffix(".pt")):
        f.unlink(missing_ok=True)
    print(f"\nwrote {rd}/stage4_repairs.json, stage4_repairs_records.npz, fig18 ({out['minutes']} min)")


# ---------------------------------------------------------------------------
# block-MVR baseline (config/preregistration.yaml stage4_amendment_3)
# ---------------------------------------------------------------------------

def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, min(1.0, (len(p) - i) * p[j]))
        adj[j] = running
    return adj


def main_mvr(args) -> None:
    """Block-MVR Gaussian knockoffs on the Stage 4 benchmark. The new arms reuse Stage 4's label
    streams (identical labels) and get arm ids after the five original arms (new knockoff seeds), so
    they are paired with the saved Stage 4 records replicate by replicate without rerunning those."""
    from math import erfc, sqrt

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = load_config(args.config)
    rd = results_dir(cfg)
    ext = cfg["stage4_mvr"]
    sec = {**cfg["stage4_repairs"], **{k: v for k, v in ext.items() if k not in ("arms", "base")}}
    base_arms = [tuple(a) for a in cfg["stage4_repairs"]["arms"]]
    new_arms = [tuple(a) for a in ext["arms"]]
    arm_ids = {name: i for i, (name, _) in enumerate(base_arms + new_arms)}
    saved_path = Path(ext["saved_records"])
    if not saved_path.exists():
        raise SystemExit(f"{saved_path} not found: the Stage 4 records are needed for the paired comparison")
    saved = np.load(saved_path)
    cp = Path(args.cache) if args.cache else cache_path(cfg)
    if not cp.exists():
        raise SystemExit(f"Cache not found at {cp}. Run src/cache_activations.py first.")
    d = np.load(cp, allow_pickle=True)
    X_all = d[f"X_{cfg['aggregation']['primary']}"].astype(np.float32, copy=False)
    print(f"device: {device} | cache {d['config_hash']} | experiment: mvr | stage: {args.stage}")
    t0 = time.time()
    amps, forms, ks = sec["signal_amplitudes"], sec["functional_forms"], sec["signal_sizes"]
    qs = sec["nominal_fdr_targets"]

    check_arms = [a for a in base_arms if a[0] == "gauss_real"]
    run_arms = new_arms + (check_arms if args.stage == "diagnose" else [])
    print("\n=== data and samplers ===")
    D = build(cfg, sec, X_all, device, methods=[m for _, m in run_arms])
    del X_all

    exch = None
    if not args.skip_exchangeability:
        print("\n=== exchangeability swap tests (block-MVR arms) ===")
        exch = exchangeability(cfg, sec, D, new_arms, arm_ids=arm_ids)

    if args.stage == "diagnose":
        # The full grid, not one pilot cell: label seeds are indexed by a cell's position in the grid,
        # so only the full grid reproduces Stage 4's labels.
        R = args.limit_reps or sec["diagnose"]["pilot_replicates"]
        print(f"\n=== pilot: full grid, {R} replicate(s), block-MVR arms + the Stage 4 gauss_real arm ===")
        M, corr, dinfo, timing = run_grid(cfg, sec, D, device, R, amps, forms, ks, run_arms, arm_ids=arm_ids)
        # M0: the Gaussian baseline rerun must reproduce the saved Stage 4 records (same labels and seeds)
        r0 = {}
        for m in ("ko_fdr", "ko_pow"):
            new_v, old_v = M[m]["gauss_real"], saved[f"{m}__gauss_real"][..., :R, :]
            r0[m] = {"max_abs_diff": float(np.abs(new_v - old_v).max()),
                     "share_identical": float((new_v == old_v).mean())}
        print(f"  M0 gauss_real vs saved Stage 4 records ({R} replicate(s), all cells): {r0}")
        iq = qs.index(0.10)
        for a, _ in run_arms:
            print(f"  {a:18s} power {M['ko_pow'][a][..., iq].mean():.3f}  FDR {M['ko_fdr'][a][..., iq].mean():.3f}"
                  + (f"  corr(X,Xk) {np.mean(corr[a]):.3f}" if corr[a] else ""))
        n_cells = len(amps) * len(forms) * len(ks)
        est = sec["replicates"] * sum(timing["seconds_per_draw"][a] + n_cells * timing["seconds_per_cell_fit"][a]
                                      for a, _ in new_arms) / 60
        print(f"  -> projected full run on this machine: ~{est:.0f} min (plus the S solves)")
        (rd / "stage4_mvr_diagnose.json").write_text(json.dumps(
            {"info": D["info"], "exchangeability": exch, "M0": r0, "timing": timing,
             "projected_full_minutes": est, "minutes": (time.time() - t0) / 60}, indent=2, default=float))
        print(f"\nwrote {rd}/stage4_mvr_diagnose.json ({(time.time() - t0) / 60:.1f} min)")
        return

    R = args.limit_reps or sec["replicates"]
    print(f"\n=== full: {len(new_arms)} block-MVR arms x {len(amps) * len(forms) * len(ks)} cells x {R} reps ===")
    ckpt = rd / "stage4_mvr_checkpoint.json"
    M, corr, dinfo, timing = run_grid(cfg, sec, D, device, R, amps, forms, ks, new_arms, checkpoint=ckpt,
                                      arm_ids=arm_ids)
    # combine with the saved Stage 4 arms (same labels, replicate r paired with replicate r)
    for m in ("ko_fdr", "ko_pow", "ko_nd"):
        for a, _ in base_arms:
            M[m][a] = saved[f"{m}__{a}"][..., :R, :]
    all_arms = base_arms + new_arms
    conds, contrasts, summary = aggregate(sec, M, amps, forms, ks, all_arms)
    # multiplicity: Holm across the 54 cells of each contrast, two-sided normal p-values
    for cname, _, _ in sec["contrasts"]:
        cs = [c for c in contrasts if c["contrast"] == cname]
        for m in ("ko_pow", "ko_fdr"):
            pv = np.array([erfc(abs(c[m + "_diff"]) / (c[m + "_diff_se"] * sqrt(2))) if c[m + "_diff_se"] > 0 else 1.0
                           for c in cs])
            adj = holm(pv)
            for c, a in zip(cs, adj):
                c[m + "_p_holm"] = float(a)
            s = summary["contrasts"][cname]
            s[f"n_{m}_holm_positive"] = int(sum(a < 0.05 and c[m + "_diff"] > 0 for c, a in zip(cs, adj)))
            s[f"n_{m}_holm_negative"] = int(sum(a < 0.05 and c[m + "_diff"] < 0 for c, a in zip(cs, adj)))
    gap = summary["contrasts"].get("atom_gap_mvr", {}).get("mean_power_diff")
    summary["gap_closed_vs_mvr"] = (summary["contrasts"]["hurdle_vs_mvr"]["mean_power_diff"] / gap
                                    if gap and abs(gap) > 1e-9 else None)
    floor = [c for c in conds if not c["power_capped_k_lt_1_over_q"]]
    summary["mean_power_excl_floor"] = {a: float(np.mean([c["ko_pow"] for c in floor if c["arm"] == a]))
                                        for a, _ in all_arms}
    print("\n=== results ===")
    for a, s in summary["arms"].items():
        print(f"  {a:18s} power {s['mean_power']:.3f} (excl. floor {summary['mean_power_excl_floor'][a]:.3f})  "
              f"mean FDR {s['mean_fdr']:.4f}  max {s['max_fdr']:.3f}  inflated {s['n_fdr_inflated']}"
              + (f"  corr(X,Xk) {np.mean(corr[a]):.3f}" if a in corr and corr[a] else ""))
    for c, s in summary["contrasts"].items():
        print(f"  {c:16s} ({s['a']} - {s['b']}): power {s['mean_power_diff']:+.3f}  sig +{s['n_power_sig_positive']}"
              f"/-{s['n_power_sig_negative']}  Holm +{s['n_ko_pow_holm_positive']}/-{s['n_ko_pow_holm_negative']}"
              f"  FDR {s['mean_fdr_diff']:+.4f}")
    print(f"  hurdle's share of the block-MVR atom gap: {summary['gap_closed_vs_mvr']}")
    out = {"config_hash": str(d["config_hash"]), "master_seed": cfg["master_seed"], "replicates": R,
           "amplitudes": amps, "forms": forms, "signal_sizes": ks, "arms": [list(a) for a in all_arms],
           "new_arms": [list(a) for a in new_arms], "arm_ids": arm_ids, "nominal_fdr_targets": qs,
           "info": D["info"], "exchangeability": exch,
           "mean_corr_X_Xk": {a: float(np.mean(v)) for a, v in corr.items() if v},
           "timing": timing, "summary": summary, "conditions": conds, "contrasts": contrasts,
           "minutes": round((time.time() - t0) / 60, 1)}
    (rd / "stage4_mvr.json").write_text(json.dumps(out, indent=2, default=float))
    np.savez(rd / "stage4_mvr_records.npz", **{f"{m}__{a}": M[m][a] for m in METRICS for a, _ in new_arms})
    make_figure(conds, amps, [a for a in all_arms if a[0] in ("gauss_real", "gauss_mvr", "hurdle",
                                                              "gauss_ceiling", "gauss_ceiling_mvr")],
                0.10, rd, fname="fig22_stage4_mvr.png")
    for f in (ckpt, ckpt.with_suffix(".npz"), ckpt.with_suffix(".pt")):
        f.unlink(missing_ok=True)
    print(f"\nwrote {rd}/stage4_mvr.json, stage4_mvr_records.npz, fig22 ({out['minutes']} min)")


if __name__ == "__main__":
    main()
