"""Step 1 — the toy proposition, where truth is exact.

K concepts, each carried by a family of m latents; null latents are also organised in families of
m, so p and n are fixed across m.  Two mechanisms:
  dilution    concept fires w.p. pi; when it fires exactly one of its m children fires (uniform),
              with a log-normal magnitude. Per-child zero mass 1 - pi/m (0.95 at m = 1).
  redundancy  concept fires w.p. pi; all m children co-fire as near-duplicates
              (child = a * exp(eta), eta ~ N(0, 0.1^2); corr >= 0.95).
Labels: c_g = standardise(sum_{j in G_g} X_j) for the K planted families, then
planted_fdr.generate_planted_labels(C, arange(K), form, amp, rng). Every child of a planted family
is non-null; everything else is null — exactly.

A Gaussian flavour (X Gaussian with family-block correlation, so Gaussian knockoffs are EXACTLY
valid) is the FDR unit test for group_knockoffs: global null (200 replicates, P(any discovery) <= q)
and planted signal (FDR <= q at both layers).

Phases:  solve  --flavour F --design D --m M     data + Ledoit-Wolf Sigma + per-latent and group MVR S
         fit                                      all replicates, all arms (GPU or CPU)
         report                                   JSON, figure, proposition statement
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

import _paths  # noqa: F401
from cseeds import ROOT, cint, crng, load_concept_config
from gpu import init_cuda
from group_knockoffs import (BlockGaussKnockoff, fit_lasso_batched, group_smatrix, group_W, latent_W,
                             mkf_select, score_sets, select_single)
from knockoff_audit import estimate_cov, standardise
from planted_fdr import generate_planted_labels

MS = [1, 2, 4, 8, 16, 32]
DESIGNS = ["dilution", "redundancy"]
FLAVOURS = ["zinf", "gauss"]
N, P, K, PI = 20_000, 1024, 20, 0.05
AMPS = [1.0, 3.0, 8.0]
QS = [0.05, 0.10, 0.20]
ARMS = ["latent", "group", "mkf_c1", "mkf_c1.93"]
LV = ["lat_fdr", "lat_pow", "lat_nd", "con_fdr", "con_pow", "con_nd"]


def tdir(cfg):
    d = ROOT / cfg["concept"]["cache_dir"] / "toy"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_X(flavour, design, m, rng):
    fam = np.repeat(np.arange(P // m), m)
    G = P // m
    if flavour == "gauss":
        rho = 0.3 if design == "dilution" else 0.95
        F = rng.standard_normal((N, G))
        X = math.sqrt(rho) * F[:, fam] + math.sqrt(1 - rho) * rng.standard_normal((N, P))
        return X, fam
    z = rng.random((N, G)) < PI
    mag = np.exp(rng.normal(0.0, 0.75, (N, G)))
    X = np.zeros((N, P))
    if design == "dilution":
        child = rng.integers(0, m, (N, G))
        r, g = np.nonzero(z)
        X[r, g * m + child[r, g]] = mag[r, g]
    else:
        a = (z * mag)[:, fam]
        X = a * np.exp(rng.normal(0.0, 0.1, (N, P)))
    return X, fam


def solve(cfg, flavour, design, m):
    fi, di, mi = FLAVOURS.index(flavour), DESIGNS.index(design), MS.index(m)
    X, fam = make_X(flavour, design, m, crng(cfg, "toy_data", fi, di, mi))
    keep = X.std(0) > 0
    assert keep.all(), f"{(~keep).sum()} constant columns"
    Z, _, _ = standardise(X)
    Sigma = estimate_cov(Z, "ledoit_wolf")
    seed = cint(cfg, "s_solve", 1, fi, di, mi)
    S_lat, t_lat, me_lat = group_smatrix(Sigma, None, seed, cfg["concept"]["mvr_max_block"], num_processes=4)
    if m == 1:
        S_grp, t_grp, me_grp = S_lat, 0.0, me_lat
    else:
        S_grp, t_grp, me_grp = group_smatrix(Sigma, fam, seed, cfg["concept"]["mvr_max_block"], num_processes=4)
    np.savez(tdir(cfg) / f"{flavour}_{design}_m{m}.npz", Z=Z.astype(np.float32), X=X.astype(np.float32), fam=fam,
             Sigma=Sigma, S_lat=S_lat, S_grp=S_grp,
             info=json.dumps({"t_lat": t_lat, "t_grp": t_grp, "mineig_lat": me_lat, "mineig_grp": me_grp,
                              "mean_s_lat": float(np.diag(S_lat).mean()), "mean_s_grp": float(np.diag(S_grp).mean())}))
    print(f"{flavour} {design} m={m}: S lat {t_lat:.0f}s (min eig {me_lat:.2e}), grp {t_grp:.0f}s "
          f"(min eig {me_grp:.2e}); mean s {np.diag(S_lat).mean():.3f}/{np.diag(S_grp).mean():.3f}", flush=True)


def labels_for(cfg, d, flavour, design, m, rep, null: bool):
    """Labels for every amplitude of one replicate (batched against shared knockoff draws)."""
    fi, di, mi = FLAVOURS.index(flavour), DESIGNS.index(design), MS.index(m)
    rng = crng(cfg, "toy_data", 100 + fi, di, mi, rep)
    G = P // m
    planted = np.sort(rng.choice(G, K, replace=False))
    if null:
        return np.stack([(rng.random(N) < 0.5).astype(np.float32)]), planted, set(), set()
    X = d["X"].astype(np.float64)
    C = np.stack([X[:, d["fam"] == g].sum(1) for g in planted], 1)
    C, _, _ = standardise(C)
    ys = [generate_planted_labels(C, np.arange(K), "linear", a, crng(cfg, "toy_data", 200 + fi, di, mi, rep, ia))[0]
          for ia, a in enumerate(AMPS)]
    true_lat = set(np.flatnonzero(np.isin(d["fam"], planted)).tolist())
    return np.stack(ys), planted, true_lat, set(planted.tolist())


def per_child_effect(d, planted, m):
    """True coefficient on each standardised planted child, per unit amp*w/sqrt(K):
    sd(X_j) / sd(sum of its family). Theory under dilution: proportional to 1/sqrt(m)."""
    X = d["X"].astype(np.float64)
    out = []
    for g in planted:
        cols = np.flatnonzero(d["fam"] == g)
        out.extend((X[:, cols].std(0) / X[:, cols].sum(1).std()).tolist())
    return float(np.mean(out))


def fit_all(cfg, device, R, flavours, null_R):
    lam, mi_, tol = cfg["concept"]["lasso"]["lambda"], cfg["concept"]["lasso"]["max_iter"], cfg["concept"]["lasso"]["tol"]
    res = {}
    for flavour in flavours:
        for design in DESIGNS:
            for m in MS:
                f = tdir(cfg) / f"{flavour}_{design}_m{m}.npz"
                d = dict(np.load(f))
                fam = d["fam"]; G = P // m
                Zt = torch.from_numpy(d["Z"]).to(device)
                smp = {"latent": BlockGaussKnockoff(Zt, d["Sigma"], np.diag(np.diag(d["S_lat"]))),
                       "group": BlockGaussKnockoff(Zt, d["Sigma"], d["S_grp"])}
                key = f"{flavour}/{design}/m{m}"
                runs = [("planted", R)] + ([("null", null_R)] if flavour == "gauss" else [])
                for kind, RR in runs:
                    nA = 1 if kind == "null" else len(AMPS)
                    M = {a: {v: np.full((nA, RR, len(QS)), np.nan, np.float32) for v in LV} for a in ARMS}
                    conv = np.zeros((2, nA, RR), bool); eff = []
                    t0 = time.time()
                    for rep in range(RR):
                        Y, planted, tl, tf = labels_for(cfg, d, flavour, design, m, rep, kind == "null")
                        if kind == "planted" and rep < 5:
                            eff.append(per_child_effect(d, planted, m))
                        Yt = torch.from_numpy(Y.T.copy()).to(device)
                        Wd = {}
                        for ai, arm in enumerate(("latent", "group")):
                            seed = cint(cfg, "toy_knockoff", FLAVOURS.index(flavour), DESIGNS.index(design),
                                        MS.index(m), 0 if kind == "planted" else 1, rep, ai)
                            Phi = torch.cat([Zt, smp[arm].sample(seed)], 1)
                            w, cv, _, _ = fit_lasso_batched(Phi, Yt, lam, mi_, tol)
                            conv[ai, :, rep] = cv.cpu().numpy()
                            Wd[arm] = w.cpu().numpy()
                        for ia in range(nA):
                            W1 = latent_W(Wd["latent"][:, ia], P)
                            W2 = group_W(Wd["group"][:, ia], P, fam, G)
                            for iq, q in enumerate(QS):
                                sels = {"latent": select_single(W1, q),
                                        "group": np.flatnonzero(np.isin(fam, select_single(W2, q))),
                                        "mkf_c1": mkf_select([W1, W2], [np.arange(P), fam], [q, q], c=1.0)[0],
                                        "mkf_c1.93": mkf_select([W1, W2], [np.arange(P), fam], [q, q], c=1.93)[0]}
                                for arm, sel in sels.items():
                                    sc = score_sets(sel, fam, tl, tf)
                                    for v in LV:
                                        M[arm][v][ia, rep, iq] = sc[v]
                    res[f"{key}/{kind}"] = {"M": {a: {v: M[a][v].tolist() for v in LV} for a in ARMS},
                                            "conv_rate": float(conv.mean()),
                                            "per_child_effect": float(np.mean(eff)) if eff else None,
                                            "info": json.loads(str(d["info"]))}
                    print(f"  {key} {kind}: {RR} reps in {time.time() - t0:.0f}s  conv {conv.mean():.3f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["solve", "fit"])
    ap.add_argument("--flavour"); ap.add_argument("--design"); ap.add_argument("--m", type=int)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--R", type=int, default=50)
    ap.add_argument("--null-R", type=int, default=200)
    ap.add_argument("--flavours", nargs="+", default=FLAVOURS)
    args = ap.parse_args()
    cfg = load_concept_config()
    if args.phase == "solve":
        solve(cfg, args.flavour, args.design, args.m)
    else:
        torch.set_num_threads(16)
        res = fit_all(cfg, init_cuda(args.device), args.R, args.flavours, args.null_R)
        out = ROOT / "concept" / "results" / "toy_proposition.json"
        meta = {"N": N, "P": P, "K": K, "pi": PI, "amps": AMPS, "qs": QS, "ms": MS, "arms": ARMS, "R": args.R,
                "null_R": args.null_R, "metrics": LV}
        out.write_text(json.dumps({"meta": meta, "results": res}))
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
