"""Step 4 — candidate sets and groups, label-free, p fixed at p* = 2048 at every width.

  planted pool  seeded random 16k reference parents whose families exist at EVERY width
                (cos >= tau, child firing >= 0.1%), whose union at the widest width fits in p*/2
  filler        whole families of the other reference parents, seeded order (same at every width),
                until p = p*; the last family is truncated
  groups        primary: parent family (decoder-only). sensitivity: average-linkage clustering on
                decoder cosine within the width (distance 1 - cos, cut at 1 - tau). Both capped at 64.
The child floor is 0.1%, not 1%: the 1% filter would delete split children by construction.
This script never reads labels (asserted).

Writes concept/results/candidates_L{layer}.json and, per width, cand_{key}.npz in the cache:
Z (standardised candidates), Sigma (Ledoit-Wolf), S for per-latent / family-group / cluster-group MVR.
Usage: python concept/src/build_candidates.py --layer 12 --tau 0.5 [--no-solve]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sp

import _paths  # noqa: F401
from cseeds import ROOT, cint, crng, load_concept_config
from family_census import ref_parents

_READ_KEYS: set = set()


def read_npz(path, key):
    """The only way this script reads cached arrays; records what it reads so we can assert that
    no label is ever loaded."""
    _READ_KEYS.add(key)
    with np.load(path) as z:
        return z[key]


def cap_groups(g: np.ndarray, cap: int, order: np.ndarray) -> np.ndarray:
    """Split any group larger than cap into consecutive chunks (members ordered by `order`)."""
    out = g.copy(); nxt = g.max() + 1
    for gid in np.unique(g):
        mem = np.flatnonzero(g == gid)
        if mem.size > cap:
            mem = mem[np.argsort(order[mem], kind="stable")]
            for c0 in range(cap, mem.size, cap):
                out[mem[c0:c0 + cap]] = nxt; nxt += 1
    return np.unique(out, return_inverse=True)[1]


def cluster_groups(dec: np.ndarray, tau: float, cap: int, fr: np.ndarray) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    D = pdist(dec.astype(np.float64), metric="cosine")
    lab = fcluster(linkage(D, method="average"), t=1 - tau, criterion="distance") - 1
    return cap_groups(lab, cap, -fr)


def solve_one(args):
    key, which, tau, cdir, seed, max_block = args
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import _paths  # noqa: F401
    from group_knockoffs import group_smatrix
    z = np.load(f"{cdir}/cand_{key}.npz")
    groups = None if which == "lat" else z[f"g_{which}"]
    S, secs, mineig = group_smatrix(z["Sigma"], groups, seed, max_block, num_processes=4)
    np.save(f"{cdir}/S_{which}_{key}.npy", S)
    return key, which, secs, mineig, float(np.diag(S).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--keys", nargs="*", default=None, help="diagnose: subset of widths")
    ap.add_argument("--no-solve", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cfg = load_concept_config()
    cc = cfg["concept"]; ca = cc["candidates"]; floor = cc["census"]["child_floor"]
    cdir = ROOT / cc["cache_dir"]
    keys = args.keys or (cc["sweep_L12"] if args.layer == 12 else cc["bridge_L20"])
    base, widest = keys[0], keys[-1]
    P, cap = ca["p_star"], ca["max_group"]
    fr16 = read_npz(cdir / f"lat_{base}.npz", "firing_rate_all")
    parents = ref_parents(fr16, cfg["latent_filter"])

    fam = {}                                                     # key -> (child idx, parent of child, firing)
    for k in keys:
        fr = read_npz(cdir / f"lat_{k}.npz", "firing_rate_all")
        pm_par, pm_cos = read_npz(cdir / f"parentmap_{k}.npz", "parent"), read_npz(cdir / f"parentmap_{k}.npz", "cos")
        ok = (fr >= floor) & (pm_cos >= args.tau) & np.isin(pm_par, parents)
        idx = np.flatnonzero(ok)
        fam[k] = (idx, pm_par[idx], fr[idx])
    size = {k: np.bincount(fam[k][1], minlength=fr16.size) for k in keys}

    # planted pool
    rng = crng(cfg, "candidates", args.layer, 0)
    elig = parents[np.all([size[k][parents] >= 1 for k in keys], axis=0)]
    order = rng.permutation(elig)
    pool, used = [], 0
    for pa in order:
        s = int(min(size[widest][pa], cap))
        if used + s > ca["pool_max_frac"] * P:
            continue
        pool.append(int(pa)); used += s
        if len(pool) >= ca["k_pool_target"]:
            break
    assert len(pool) >= ca["k_pool_min"], f"planted pool too small: {len(pool)}"
    pool = np.array(sorted(pool))
    filler_order = crng(cfg, "candidates", args.layer, 1).permutation(np.setdiff1d(parents, pool))
    print(f"eligible parents {elig.size}/{parents.size}; pool {pool.size} parents using {used} latents at {widest}", flush=True)

    out = {"layer": args.layer, "tau": args.tau, "p_star": P, "child_floor": floor, "keys": keys,
           "pool_parents": pool.tolist(), "n_eligible_parents": int(elig.size), "per_width": {}}
    for k in keys:
        idx, par, fr = fam[k]
        sel, sel_par = [], []
        def add(pa, room):
            m = np.flatnonzero(par == pa)
            m = m[np.argsort(-fr[m], kind="stable")][: min(cap, room)]
            sel.extend(idx[m].tolist()); sel_par.extend([int(pa)] * m.size)
        for pa in pool:
            add(pa, P - len(sel))
        n_pool = len(sel)
        for pa in filler_order:
            if len(sel) >= P:
                break
            if size[k][pa] >= 1:
                add(pa, P - len(sel))
        sel = np.array(sel); sel_par = np.array(sel_par)
        assert sel.size == P, (k, sel.size)
        o = np.argsort(sel); sel, sel_par = sel[o], sel_par[o]
        g_fam = np.unique(sel_par, return_inverse=True)[1]
        dec = np.load(cdir / f"dec_{k}.npy", mmap_mode="r")[sel]
        fr_sel = read_npz(cdir / f"lat_{k}.npz", "firing_rate_all")[sel]
        g_clu = cluster_groups(np.asarray(dec), args.tau, cap, fr_sel)
        # standardised design and Ledoit-Wolf covariance
        t = time.time()
        from knockoff_audit import estimate_cov, standardise
        Xs = sp.load_npz(cdir / f"lat_{k}.csr.npz").tocsc()[:, sel].toarray().astype(np.float64)
        Z, mu, sd = standardise(Xs)
        Sigma = estimate_cov(Z, "ledoit_wolf")
        np.savez(cdir / f"cand_{k}.npz", Z=Z.astype(np.float32), mu=mu, sd=sd, Sigma=Sigma, latents=sel, parent=sel_par,
                 g_fam=g_fam, g_clu=g_clu, pool=pool, in_pool=np.isin(sel_par, pool))
        fs = np.bincount(g_fam)
        W = {"n_pool_latents": int(n_pool), "n_families": int(g_fam.max() + 1), "n_clusters": int(g_clu.max() + 1),
             "pool_family_size_mean": float(np.mean([np.sum(sel_par == pa) for pa in pool])),
             "family_size_median": float(np.median(fs)), "family_size_max": int(fs.max()),
             "median_zero_mass": float(np.median((Xs == 0).mean(0))), "cov_seconds": time.time() - t,
             "latents": sel.tolist(), "parent": sel_par.tolist(), "g_clu": g_clu.tolist()}
        out["per_width"][k] = W
        print(f"{k:9s} pool latents {n_pool:4d} (mean fam {W['pool_family_size_mean']:.2f})  families {W['n_families']:4d}  "
              f"clusters {W['n_clusters']:4d}  median zero mass {W['median_zero_mass']:.3f}", flush=True)
    assert "labels" not in _READ_KEYS, "build_candidates must never read labels"
    out["keys_read"] = sorted(_READ_KEYS)
    (ROOT / "concept" / "results" / f"candidates_L{args.layer}.json").write_text(json.dumps(out))
    if args.no_solve:
        return
    jobs = []
    for i, k in enumerate(keys):
        for w in ("lat", "fam", "clu"):
            if w != "lat" and k.endswith("_16k"):
                continue                                         # singletons: group S == per-latent S
            if (cdir / f"S_{w}_{k}.npy").exists():
                continue
            jobs.append((k, w, args.tau, str(cdir), cint(cfg, "s_solve", 2, args.layer, i, ("lat", "fam", "clu").index(w)),
                         cc["mvr_max_block"]))
    print(f"{len(jobs)} S solves, {args.workers} at a time", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for k, w, secs, me, ms in ex.map(solve_one, jobs):
            print(f"  S[{w}] {k}: {secs:.0f}s  min eig(2Σ-S) {me:.2e}  mean s {ms:.3f}", flush=True)


if __name__ == "__main__":
    main()
