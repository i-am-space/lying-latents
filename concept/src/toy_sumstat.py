"""Pooled group statistics on the Step 1 toy, paired with the group arm on identical labels and knockoff
draws.  --stat sum    group-sum (concept_amendment_2, exploratory)   -> toy_sumstat.json
         --stat glasso unweighted group lasso (concept_amendment_3)   -> toy_glasso.json"""
import argparse
import json
import time

import numpy as np
import torch

import _paths  # noqa: F401
from cseeds import ROOT, cint, load_concept_config
from gpu import init_cuda
from group_knockoffs import (BlockGaussKnockoff, fit_group_lasso_batched, fit_lasso_batched, group_norm_W,
                             group_sum_design, score_sets, select_single)
from toy_proposition import AMPS, DESIGNS, FLAVOURS, LV, MS, P, QS, labels_for, tdir

ap = argparse.ArgumentParser(); ap.add_argument("--stat", choices=["sum", "glasso"], default="sum")
STAT = ap.parse_args().stat
cfg = load_concept_config(); dev = init_cuda("cuda")
lam, mit, tol = cfg["concept"]["lasso"]["lambda"], cfg["concept"]["lasso"]["max_iter"], cfg["concept"]["lasso"]["tol"]
res = {}
for flavour in FLAVOURS:
    for design in DESIGNS:
        for m in MS:
            d = dict(np.load(tdir(cfg) / f"{flavour}_{design}_m{m}.npz"))
            fam = d["fam"]; G = P // m
            Zt = torch.from_numpy(d["Z"]).to(dev)
            smp = BlockGaussKnockoff(Zt, d["Sigma"], d["S_grp"])
            for kind, RR in [("planted", 50)] + ([("null", 200)] if flavour == "gauss" else []):
                nA = 1 if kind == "null" else len(AMPS)
                M = {v: np.full((nA, RR, len(QS)), np.nan, np.float32) for v in LV}
                t0 = time.time()
                for rep in range(RR):
                    Y, planted, tl, tf = labels_for(cfg, d, flavour, design, m, rep, kind == "null")
                    seed = cint(cfg, "toy_knockoff", FLAVOURS.index(flavour), DESIGNS.index(design), MS.index(m),
                                0 if kind == "planted" else 1, rep, 1)               # the group arm's draw
                    Yt = torch.from_numpy(Y.T.copy()).to(dev)
                    if STAT == "sum":
                        w = fit_lasso_batched(group_sum_design(Zt, smp.sample(seed), fam, G), Yt, lam, mit, tol)[0].cpu().numpy()
                    else:
                        w = fit_group_lasso_batched(torch.cat([Zt, smp.sample(seed)], 1), Yt, np.concatenate([fam, fam + G]),
                                                    lam, mit, tol)[0].cpu().numpy()
                    for ia in range(nA):
                        W = (np.abs(w[:G, ia]) - np.abs(w[G:, ia])) if STAT == "sum" else group_norm_W(w[:, ia], P, fam, G)
                        for iq, q in enumerate(QS):
                            sc = score_sets(np.flatnonzero(np.isin(fam, select_single(W, q))), fam, tl, tf)
                            for v in LV:
                                M[v][ia, rep, iq] = sc[v]
                res[f"{flavour}/{design}/m{m}/{kind}"] = {v: M[v].tolist() for v in LV}
                print(f"{flavour}/{design}/m{m}/{kind}: {time.time() - t0:.0f}s  concept power (q=0.1, amps) "
                      f"{np.round(M['con_pow'][:, :, 1].mean(1), 2)}  P(any disc) {np.mean(M['con_nd'][:, :, 1] > 0):.3f}", flush=True)
(ROOT / f"concept/results/toy_{'sumstat' if STAT == 'sum' else 'glasso'}.json").write_text(json.dumps(res))
