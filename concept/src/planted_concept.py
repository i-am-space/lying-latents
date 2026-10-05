"""Step 7 — planted-concept benchmark (the core result). Also the engine for the Step 6 gate.

Per width (fixed p* candidates), per replicate: draw K_c planted concepts from the pool (identical
across widths), build every label column of the grid, and fit each arm once per replicate against
its own fresh knockoff draw (all label columns of a replicate share that draw, as in the main
repo's p = 2048 runs; replicates use independent draws). Designs:
  A  exact truth:  c_g = standardise(sum of the concept's children at THIS width)
  B  realistic:    c_g = standardise(the 16k parent's own activation), identical across widths
Labels: planted_fdr.generate_planted_labels(C, arange(K_c), form, amp, rng), rng per cell (not per
width), so widths are paired. Checkpointed after every (width, replicate); rerun to resume.

Usage: python concept/src/planted_concept.py --layer 12 --keys L12_16k L12_32k --tag full --gpu-part 0/2
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import scipy.sparse as sp
import torch

import _paths  # noqa: F401
from bench import ARMS, FIT_ARMS, LV, Width, score_column
from cseeds import ROOT, cint, crng, load_concept_config
from gpu import init_cuda
from planted_fdr import generate_planted_labels

DESIGNS = ["A", "B"]


def parent_columns(cdir, key16, parents):
    X16 = sp.load_npz(cdir / f"lat_{key16}.csr.npz").tocsc()[:, parents].toarray().astype(np.float64)
    return (X16 - X16.mean(0)) / X16.std(0)


def run(cfg, layer, keys, all_keys, grid, device, tag, arms=ARMS, designs=DESIGNS, stream=("planted", "planted_knockoff")):
    cc = cfg["concept"]; cdir = ROOT / cc["cache_dir"]
    lam, mit, tol = cc["lasso"]["lambda"], cc["lasso"]["max_iter"], cc["lasso"]["tol"]
    Kcs, amps, forms, R, qs = grid["K_c"], grid["amplitudes"], grid["forms"], grid["replicates"], grid["qs"]
    fit_arms = [a for a in FIT_ARMS if a in arms or (a in ("latent", "group") and any(x.startswith("mkf") for x in arms))]
    key16 = all_keys[0]
    cols = [(d, iK, ia, jf) for d in designs for iK in range(len(Kcs)) for ia in range(len(amps)) for jf in range(len(forms))]
    Kmax = max(Kcs)
    for key in keys:
        ck = ROOT / cc["cache_dir"] / f"ckpt_{tag}_{key}.npz"
        shape = (len(designs), len(Kcs), len(amps), len(forms), R, len(qs))
        if ck.exists():
            z = dict(np.load(ck)); start = int(z["reps_done"])
            M = {a: {v: z[f"{a}__{v}"] for v in LV} for a in arms}
            REC = {a: z[f"rec__{a}"] for a in arms}; CONV = z["conv"]; FS = z["famsize"]
            print(f"[{key}] resuming at replicate {start}/{R}", flush=True)
        else:
            start = 0
            M = {a: {v: np.full(shape, np.nan, np.float32) for v in LV} for a in arms}
            REC = {a: np.zeros(shape + (Kmax,), bool) for a in arms}
            CONV = np.zeros((len(fit_arms), len(cols), R), bool)
            FS = np.zeros((len(Kcs), R, Kmax), np.int16)
        if start >= R:
            continue
        t0 = time.time()
        wd = Width(cdir, key, device, arms=fit_arms)
        PB = parent_columns(cdir, key16, wd.pool) if "B" in designs else None
        print(f"[{key}] loaded in {time.time() - t0:.0f}s; mean s {wd.mean_s}; min eig V {wd.min_eig_V}", flush=True)
        ki = all_keys.index(key)
        for rep in range(start, R):
            tr = time.time()
            planted = {iK: np.sort(crng(cfg, stream[0], layer, iK, rep).choice(wd.pool, K, replace=False))
                       for iK, K in enumerate(Kcs)}
            for iK, K in enumerate(Kcs):
                FS[iK, rep, :K] = [int((wd.parent == pa).sum()) for pa in planted[iK]]
            Y, CA = [], {}
            for d, iK, ia, jf in cols:
                pl = planted[iK]
                if d == "A":
                    if iK not in CA:
                        CA[iK] = wd.family_sum(pl)
                    C = CA[iK]
                else:
                    C = PB[:, np.searchsorted(wd.pool, pl)]
                rng = crng(cfg, stream[0], layer, 1000 + designs.index(d), iK, ia, jf, rep)
                Y.append(generate_planted_labels(C, np.arange(Kcs[iK]), forms[jf], amps[ia], rng)[0])
            Yt = torch.from_numpy(np.stack(Y, 1).astype(np.float32)).to(device)
            Wd = {}
            for ai, a in enumerate(fit_arms):
                w, cv, _ = wd.fit(a, cint(cfg, stream[1], layer, ki, ai, rep), Yt, lam, mit, tol)
                Wd[a] = w; CONV[ai, :, rep] = cv
            for ci, (d, iK, ia, jf) in enumerate(cols):
                pl = planted[iK]
                true_lat = set(np.flatnonzero(np.isin(wd.parent, pl)).tolist())
                out, rec = score_column(Wd, ci, wd, qs, true_lat, set(pl.tolist()), arms)
                di = designs.index(d)
                for a in arms:
                    for v in LV:
                        M[a][v][di, iK, ia, jf, rep] = out[a][v]
                    REC[a][di, iK, ia, jf, rep, :, :len(pl)] = rec[a]
            arrays = {f"{a}__{v}": M[a][v] for a in arms for v in LV}
            arrays.update({f"rec__{a}": REC[a] for a in arms})
            np.savez(ck, **arrays, conv=CONV, famsize=FS, reps_done=rep + 1)
            print(f"[{key}] rep {rep + 1}/{R}  {time.time() - tr:.0f}s  conv {CONV[:, :, rep].mean():.3f}  "
                  f"ETA {(time.time() - t0) / (rep + 1 - start) * (R - rep - 1) / 60:.0f}m", flush=True)
        del wd; torch.cuda.empty_cache() if device.type == "cuda" else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--keys", nargs="+", required=True)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--replicates", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = load_concept_config()
    cc = cfg["concept"]
    grid = dict(cc["planted"]); grid["qs"] = cc["nominal_fdr_targets"]
    if args.replicates:
        grid["replicates"] = args.replicates
    all_keys = cc["sweep_L12"] if args.layer == 12 else cc["bridge_L20"]
    torch.set_num_threads(8)
    run(cfg, args.layer, args.keys, all_keys, grid, init_cuda(args.device), args.tag)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
