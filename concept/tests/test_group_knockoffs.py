"""Unit tests for concept/src/group_knockoffs.py.  Run: pytest -q concept/tests"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import _paths  # noqa: E402,F401
from group_knockoffs import (BlockGaussKnockoff, check_block_diagonal, fit_lasso_batched, group_smatrix,  # noqa: E402
                             group_W, latent_W, mkf_bruteforce, mkf_select)
from planted_fdr_controls import fit_lasso_checked  # noqa: E402
from stage3_followups import GaussKnockoffGPU  # noqa: E402

torch.set_num_threads(8)


def _ar_sigma(p, rho=0.6):
    i = np.arange(p)
    return rho ** np.abs(i[:, None] - i[None, :])


def _grouped_sigma(rng, sizes, within=0.8, between=0.15):
    p = sum(sizes)
    g = np.repeat(np.arange(len(sizes)), sizes)
    S = np.where(g[:, None] == g[None, :], within, between).astype(float)
    np.fill_diagonal(S, 1.0)
    return S, g


def test_singleton_equivalence():
    """With diagonal S the block sampler must reproduce GaussKnockoffGPU draw for draw."""
    rng = np.random.default_rng(0)
    p, n = 40, 500
    Sig = _ar_sigma(p)
    Z = rng.multivariate_normal(np.zeros(p), Sig, size=n)
    S, _, mineig = group_smatrix(Sig, None, seed=1, max_block=512)
    assert mineig > 0
    Zt = torch.from_numpy(Z.astype(np.float32))
    a = GaussKnockoffGPU(Zt, Sig, S).sample(7)
    b = BlockGaussKnockoff(Zt, Sig, np.diag(np.diag(S))).sample(7)
    assert torch.equal(a, b)


def test_batched_lasso_matches_reference():
    rng = np.random.default_rng(1)
    n, d, B = 2000, 60, 5
    Phi = torch.from_numpy(rng.standard_normal((n, d)).astype(np.float32))
    beta = np.zeros(d); beta[:6] = 1.0
    Y = np.stack([(rng.random(n) < 1 / (1 + np.exp(-(Phi.numpy() @ (beta * s))))).astype(np.float32)
                  for s in np.linspace(0.3, 1.5, B)], 1)
    Wb, conv, _, it_b = fit_lasso_batched(Phi, torch.from_numpy(Y), 0.02, 1000, 1e-4)
    for k in range(B):
        w, cv, _, it = fit_lasso_checked(Phi, torch.from_numpy(Y[:, k]), 0.02, 1000, 1e-4)
        assert bool(conv[k]) == cv and int(it_b[k]) == it
        assert torch.allclose(Wb[:, k], w, atol=1e-5), float((Wb[:, k] - w).abs().max())


def test_group_S_is_block_diagonal_and_valid():
    rng = np.random.default_rng(2)
    Sig, g = _grouped_sigma(rng, [3, 1, 4, 2, 5, 3])
    S, _, mineig = group_smatrix(Sig, g, seed=3, max_block=512)
    assert mineig > 0
    assert check_block_diagonal(S, g) < 1e-8
    assert np.abs(S[g[:, None] == g[None, :]]).max() > 0


def test_group_exchangeability_moments():
    """Swapping whole groups leaves the first two moments of (X, Xk) unchanged; swapping one
    member of a multi-latent group must NOT (negative control that the off-diagonal blocks of S
    are actually used)."""
    rng = np.random.default_rng(3)
    Sig, g = _grouped_sigma(rng, [4, 4, 2, 6], within=0.85)
    p, n = Sig.shape[0], 200_000
    S, _, _ = group_smatrix(Sig, g, seed=4, max_block=512)
    Z = rng.multivariate_normal(np.zeros(p), Sig, size=n)
    Zt = torch.from_numpy(Z.astype(np.float32))
    Zk = BlockGaussKnockoff(Zt, Sig, S).sample(11).numpy()
    J = np.hstack([Z, Zk]).astype(np.float64)
    C0 = np.cov(J.T)

    def swapped(cols):
        Jsw = J.copy()
        Jsw[:, cols], Jsw[:, cols + p] = J[:, cols + p], J[:, cols]
        return np.cov(Jsw.T)

    se = 4.0 / np.sqrt(n)                                   # ~4 MC standard errors for unit-scale entries
    whole = np.flatnonzero(np.isin(g, [0, 3]))
    assert np.abs(swapped(whole) - C0).max() < se, "whole-group swap changed the covariance"
    assert np.abs(J.mean(0)).max() < se
    partial = np.flatnonzero(g == 3)[:1]                    # one member of a 6-latent group
    assert np.abs(swapped(partial) - C0).max() > 10 * se, "partial swap should break exchangeability"


@pytest.mark.parametrize("seed", range(40))
def test_mkf_matches_bruteforce(seed):
    rng = np.random.default_rng(100 + seed)
    p = rng.integers(8, 16)
    groups = np.sort(rng.integers(0, rng.integers(3, 7), p))
    groups = np.unique(groups, return_inverse=True)[1]
    G = groups.max() + 1
    W1 = np.round(rng.normal(0.4, 1, p), 2); W1[rng.random(p) < 0.2] = 0
    W2 = np.round(rng.normal(0.5, 1, G), 2)
    Ws, unit_of = [W1, W2], [np.arange(p), groups]
    for qs, c in (([0.5, 0.5], 1.0), ([0.34, 0.6], 1.0), ([0.9, 0.9], 1.93)):
        sel, t = mkf_select(Ws, unit_of, qs, c=c)
        feas = mkf_bruteforce(Ws, unit_of, qs, c=c)
        assert tuple(t) in feas, "MKF output is not feasible"
        for f in feas:                                       # least element of the feasible set
            assert all(a <= b for a, b in zip(t, f)), (t, f)


def test_group_W_antisymmetric():
    rng = np.random.default_rng(5)
    p = 30
    g = rng.integers(0, 7, p)
    w = rng.normal(size=2 * p)
    W = group_W(w, p, g, 7)
    wsw = w.copy(); idx = np.flatnonzero(np.isin(g, [1, 4]))
    wsw[idx], wsw[idx + p] = w[idx + p], w[idx]
    Wsw = group_W(wsw, p, g, 7)
    flip = np.isin(np.arange(7), [1, 4])
    assert np.allclose(Wsw[flip], -W[flip]) and np.allclose(Wsw[~flip], W[~flip])
    assert np.allclose(latent_W(w, p)[:3], np.abs(w[:3]) - np.abs(w[p:p + 3]))


def test_group_sum_design_swap_equivariant():
    from group_knockoffs import group_sum_design
    rng = np.random.default_rng(9)
    n, p = 300, 12
    g = np.array([0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 4, 4])
    Z = torch.from_numpy(rng.standard_normal((n, p)).astype(np.float32))
    Zk = torch.from_numpy(rng.standard_normal((n, p)).astype(np.float32))
    D = group_sum_design(Z, Zk, g, 5)
    sw = np.flatnonzero(np.isin(g, [1, 3]))
    Z2, Zk2 = Z.clone(), Zk.clone()
    Z2[:, sw], Zk2[:, sw] = Zk[:, sw], Z[:, sw]
    D2 = group_sum_design(Z2, Zk2, g, 5)
    perm = np.arange(10); perm[[1, 3]] += 5; perm[[6, 8]] -= 5        # swapped groups exchange columns
    assert torch.allclose(D2, D[:, perm], atol=1e-5)
