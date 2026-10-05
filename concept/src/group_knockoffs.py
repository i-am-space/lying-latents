"""Step 5 — group knockoff library: block-S Gaussian sampler, group MVR S, group W, MKF+.

Everything here is device-agnostic (torch on cuda or cpu).

  BlockGaussKnockoff   second-order Gaussian knockoffs with a FULL block-diagonal S.
                       src/stage3_followups.GaussKnockoffGPU keeps only diag(S), which is wrong
                       for group S. With diagonal S the two are identical draw for draw.
  group_smatrix        knockpy group MVR (method passed explicitly: with groups knockpy
                       otherwise defaults to SDP), seeded because merge_groups shuffles.
  fit_lasso_batched    the FISTA L1-logistic of planted_fdr_controls.fit_lasso_checked, run on
                       B label vectors at once against one design (one GEMM per step instead of
                       B GEMVs). Same step size, same coordinate-wise prox, same per-column stop
                       rule, so each column is swap-equivariant exactly as the original is.
  group_W              W_g = sum_{j in g}|b_j| - sum_{j in g}|b~_j| (antisymmetric under group swaps).
  mkf_select           multilayer knockoff filter (Katsevich & Sabatti 2019), MKF(c)+.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

import _paths  # noqa: F401
from planted_fdr_controls import knockoff_threshold, lmax_power


# ---------------------------------------------------------------------------
# sampler and S
# ---------------------------------------------------------------------------

class BlockGaussKnockoff:
    """Xk = mu + (X - mu)(I - Sigma^-1 S) + E V^(1/2),  V = 2S - S Sigma^-1 S, S block-diagonal."""

    def __init__(self, Z_t: torch.Tensor, Sigma: np.ndarray, S: np.ndarray):
        p = Sigma.shape[0]
        SinvS = np.linalg.solve(Sigma, S)
        A = np.eye(p) - SinvS
        V = 2 * S - S @ SinvS
        V = 0.5 * (V + V.T)
        ev, U = np.linalg.eigh(V)
        self.min_eig_V = float(ev.min())
        L = U * np.sqrt(np.clip(ev, 0.0, None))
        dev = Z_t.device
        self.Z = Z_t
        self.mu = Z_t.mean(0, keepdim=True)
        self.A = torch.from_numpy(A.astype(np.float32)).to(dev)
        self.Lt = torch.from_numpy(L.T.astype(np.float32)).to(dev)

    def sample(self, seed: int) -> torch.Tensor:
        gen = torch.Generator(device=self.Z.device)
        gen.manual_seed(int(seed))
        E = torch.randn(self.Z.shape, generator=gen, device=self.Z.device)
        return self.mu + (self.Z - self.mu) @ self.A + E @ self.Lt


def group_smatrix(Sigma: np.ndarray, groups0: np.ndarray | None, seed: int, max_block: int = 512,
                  num_processes: int = 1):
    """MVR S. groups0: 0-indexed group id per variable, or None for per-variable S.
    Returns (S, seconds, min_eig(2 Sigma - S))."""
    from knockpy import smatrix
    t = time.time()
    np.random.seed(seed)                  # merge_groups and the MVR solver use numpy's global state
    kw = dict(method="mvr", how_approx="blockdiag", max_block=max_block, num_processes=num_processes)
    if groups0 is None or len(np.unique(groups0)) == len(groups0):
        S = smatrix.compute_smatrix(Sigma, **kw)
    else:
        S = smatrix.compute_smatrix(Sigma, groups=np.asarray(groups0) + 1, **kw)
    S = np.asarray(S)
    S = 0.5 * (S + S.T)
    return S, time.time() - t, float(np.linalg.eigvalsh(2 * Sigma - S).min())


def check_block_diagonal(S: np.ndarray, groups0: np.ndarray, tol: float = 1e-10) -> float:
    """Largest |S_ij| between different groups (must be ~0 for a valid group S)."""
    off = groups0[:, None] != groups0[None, :]
    return float(np.abs(S[off]).max()) if off.any() else 0.0


# ---------------------------------------------------------------------------
# batched lasso
# ---------------------------------------------------------------------------

def fit_lasso_batched(Phi: torch.Tensor, Y: torch.Tensor, lam: float, max_iter: int, tol: float,
                      lmax: float | None = None):
    """FISTA L1-logistic on B label columns at once. Y: (n, B). Returns (W (d, B), converged (B,),
    residual (B,), iterations (B,)). Each column follows exactly the iteration of
    fit_lasso_checked and is frozen at the step where it meets the stop rule."""
    n, d = Phi.shape
    B = Y.shape[1]
    dev = Phi.device
    lr = 1.0 / (0.25 * (1.1 * (lmax if lmax is not None else lmax_power(Phi)) + 1.0))
    w = torch.zeros(d, B, device=dev); b = torch.zeros(1, B, device=dev)
    w_prev, b_prev, t = w.clone(), b.clone(), 1.0
    active = torch.ones(B, dtype=torch.bool, device=dev)
    res = torch.full((B,), float("inf"), device=dev)
    iters = torch.full((B,), max_iter, dtype=torch.int64, device=dev)
    for it in range(1, max_iter + 1):
        t_new = 0.5 * (1 + math.sqrt(1 + 4 * t * t))
        mom = (t - 1.0) / t_new
        v = w + mom * (w - w_prev); vb = b + mom * (b - b_prev)
        err = (torch.sigmoid(Phi @ v + vb) - Y) / n
        u = v - lr * (Phi.t() @ err)
        w_new = torch.sign(u) * torch.clamp(u.abs() - lr * lam, min=0.0)
        b_new = vb - lr * err.sum(0, keepdim=True)
        a = active[None, :]
        w_prev = torch.where(a, w, w_prev); b_prev = torch.where(a, b, b_prev)
        w = torch.where(a, w_new, w); b = torch.where(a, b_new, b)
        t = t_new
        if it % 25 == 0:
            r = (v - w_new).abs().amax(0) / lr
            res = torch.where(active, r, res)
            done = active & (r < tol)
            iters = torch.where(done, torch.full_like(iters, it), iters)
            active = active & ~done
            if not bool(active.any()):
                break
    return w, ~active, res, iters


# ---------------------------------------------------------------------------
# statistics and filters
# ---------------------------------------------------------------------------

def latent_W(w: np.ndarray, p: int) -> np.ndarray:
    return np.abs(w[:p]) - np.abs(w[p:2 * p])


def group_W(w: np.ndarray, p: int, groups0: np.ndarray, n_groups: int | None = None) -> np.ndarray:
    G = int(n_groups if n_groups is not None else groups0.max() + 1)
    return (np.bincount(groups0, weights=np.abs(w[:p]), minlength=G)
            - np.bincount(groups0, weights=np.abs(w[p:2 * p]), minlength=G))


def select_single(W: np.ndarray, q: float, offset: int = 1) -> np.ndarray:
    tau = knockoff_threshold(W, q, offset)
    return np.zeros(0, dtype=np.int64) if math.isinf(tau) else np.flatnonzero(W >= tau)


def _fdp_hat(Wm: np.ndarray, t: float, n_sel_units: int, offset: int) -> float:
    if math.isinf(t):
        return 0.0                                  # selects nothing: always admissible
    return (offset + np.count_nonzero(Wm <= -t)) / max(1, n_sel_units)


def _selected_latents(Ws, unit_of, t):
    ok = np.ones(unit_of[0].shape[0], dtype=bool)
    for Wm, um, tm in zip(Ws, unit_of, t):
        ok &= Wm[um] >= tm
    return ok


def mkf_select(Ws: list[np.ndarray], unit_of: list[np.ndarray], qs: list[float], c: float = 1.0,
               offset: int = 1, max_rounds: int = 1000):
    """MKF(c)+ (Katsevich & Sabatti 2019, Algorithm 1). Ws[m]: statistic per layer-m unit;
    unit_of[m]: layer-m unit of each latent. Latent j is selected iff W^m_{unit_m(j)} >= t_m for
    every m. Layer-m FDP-hat(t) = (offset + #{W^m <= -t_m}) / max(1, |S_m(t)|), S_m(t) the layer-m
    units containing a selected latent; feasibility is FDP-hat_m <= q_m / c for all m. The
    iteration raises each t_m to the smallest admissible value given the others; it stays below
    every feasible point and stops at a feasible one, i.e. the least element of the feasible set.
    Returns (selected latent indices, thresholds)."""
    M = len(Ws)
    cands = [np.append(np.unique(np.abs(W[W != 0])), np.inf) for W in Ws]
    t = [c_[0] for c_ in cands]
    for _ in range(max_rounds):
        changed = False
        for m in range(M):
            others = np.ones(unit_of[0].shape[0], dtype=bool)
            for k in range(M):
                if k != m:
                    others &= Ws[k][unit_of[k]] >= t[k]
            elig = np.zeros(Ws[m].shape[0], dtype=bool)
            elig[unit_of[m][others]] = True                   # units with a latent passing other layers
            We = np.sort(Ws[m][elig])
            Wneg = np.sort(-Ws[m][Ws[m] < 0])
            cm = cands[m][cands[m] >= t[m]]
            n_sel = We.size - np.searchsorted(We, cm, side="left")   # #{eligible units with W >= t}
            n_neg = Wneg.size - np.searchsorted(Wneg, cm, side="left")
            fdp = np.where(np.isinf(cm), 0.0, (offset + n_neg) / np.maximum(1, n_sel))
            new = float(cm[np.argmax(fdp <= qs[m] / c)])     # inf is always admissible
            if new != t[m]:
                t[m] = new; changed = True
        if not changed:
            break
    return np.flatnonzero(_selected_latents(Ws, unit_of, t)), t


def mkf_bruteforce(Ws, unit_of, qs, c=1.0, offset=1):
    """Exhaustive search over the threshold grid (small problems only), for testing mkf_select.
    Returns the list of feasible threshold tuples."""
    import itertools
    cands = [np.append(np.unique(np.abs(W[W != 0])), np.inf) for W in Ws]
    feas = []
    for t in itertools.product(*cands):
        sel = _selected_latents(Ws, unit_of, t)
        ok = True
        for m in range(len(Ws)):
            n_units = np.unique(unit_of[m][sel]).size
            if _fdp_hat(Ws[m], t[m], n_units, offset) > qs[m] / c:
                ok = False; break
        if ok:
            feas.append(tuple(t))
    return feas


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score_sets(sel_latents: np.ndarray, family: np.ndarray, true_latents: set, true_families: set):
    """Latent- and concept-level FDR/power of a latent selection. A concept (family) counts as
    discovered when at least one of its latents is selected. family[j] < 0 means no family."""
    sl = set(map(int, sel_latents))
    fams = {int(family[j]) for j in sl if family[j] >= 0}
    def fp(disc, truth):
        if not disc:
            return 0.0, 0.0
        return len(disc - truth) / len(disc), len(disc & truth) / max(1, len(truth))
    lf, lp = fp(sl, true_latents)
    cf, cp = fp(fams, true_families)
    return {"lat_fdr": lf, "lat_pow": lp, "lat_nd": len(sl), "con_fdr": cf, "con_pow": cp, "con_nd": len(fams)}


def group_sum_design(Z: torch.Tensor, Zk: torch.Tensor, groups0: np.ndarray, n_groups: int):
    """[s, s~] with s_g = sum_{j in g} Z_j and s~_g = sum_{j in g} Z~_j, each pair scaled by the SAME
    constant sqrt((var s_g + var s~_g) / 2). Swapping group g swaps s_g and s~_g and leaves the scale
    unchanged, so any swap-equivariant fit on this design gives a valid group knockoff statistic.
    Exploratory (concept_amendment_2): a statistic that pools within a group before penalising."""
    G = torch.zeros(Z.shape[1], n_groups, device=Z.device)
    G[torch.arange(Z.shape[1], device=Z.device), torch.from_numpy(groups0).to(Z.device)] = 1.0
    s, sk = Z @ G, Zk @ G
    c = torch.sqrt(0.5 * (s.var(0) + sk.var(0))).clamp(min=1e-12)
    return torch.cat([(s - s.mean(0)) / c, (sk - sk.mean(0)) / c], 1)
