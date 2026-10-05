"""Shared engine for Steps 6-8: load a width's candidate set, draw knockoffs per arm, run batched
fits for many label columns against one draw, and score every arm at both levels.

Arms (each layer gets its own knockoff draw):
  latent      per-latent Knockoff+, block-MVR S (diag)
  group       group Knockoff+ on parent families, group MVR S
  cluster     group Knockoff+ on decoder-cosine clusters (sensitivity), group MVR S
  mkf_c1      MKF(1)+: layer 1 = latent W, layer 2 = family-group W
  mkf_c1.93   MKF(1.93)+ (the variant with the theorem's guarantee)
"""
from __future__ import annotations

import numpy as np
import torch

import _paths  # noqa: F401
from group_knockoffs import (BlockGaussKnockoff, fit_lasso_batched, group_W, latent_W, mkf_select,
                             select_single)
from planted_fdr_controls import lmax_power

ARMS = ["latent", "group", "cluster", "mkf_c1", "mkf_c1.93"]
FIT_ARMS = ["latent", "group", "cluster"]
LV = ["lat_fdr", "lat_pow", "lat_nd", "con_fdr", "con_pow", "con_nd"]


class Width:
    def __init__(self, cdir, key, device, arms=FIT_ARMS):
        z = np.load(cdir / f"cand_{key}.npz")
        self.key = key
        self.Z = z["Z"]; self.mu = z["mu"]; self.sd = z["sd"]
        self.parent = z["parent"]; self.g_fam = z["g_fam"]; self.g_clu = z["g_clu"]
        self.pool = z["pool"]; self.latents = z["latents"]
        self.p = self.Z.shape[1]
        self.Zt = torch.from_numpy(self.Z).to(device)
        Sig = z["Sigma"]
        S = {"latent": np.load(cdir / f"S_lat_{key}.npy")}
        singleton = key.endswith("_16k")
        S["group"] = S["latent"] if singleton else np.load(cdir / f"S_fam_{key}.npy")
        S["cluster"] = S["latent"] if singleton else np.load(cdir / f"S_clu_{key}.npy")
        self.smp = {}
        for a in arms:
            Sa = np.diag(np.diag(S[a])) if a == "latent" else S[a]
            self.smp[a] = BlockGaussKnockoff(self.Zt, Sig, Sa)
        self.min_eig_V = {a: s.min_eig_V for a, s in self.smp.items()}
        self.mean_s = {a: float(np.diag(S[a]).mean()) for a in S}
        self.n_fam = int(self.g_fam.max() + 1); self.n_clu = int(self.g_clu.max() + 1)

    def family_sum(self, parents):
        """standardise(sum of raw children) for each parent: design A's concept columns."""
        Xraw = self.Z.astype(np.float64) * self.sd + self.mu
        C = np.stack([Xraw[:, self.parent == pa].sum(1) for pa in parents], 1)
        return (C - C.mean(0)) / C.std(0)

    def fit(self, arm, seed, Yt, lam, max_iter, tol):
        Phi = torch.cat([self.Zt, self.smp[arm].sample(seed)], 1)
        w, conv, _, it = fit_lasso_batched(Phi, Yt, lam, max_iter, tol, lmax=lmax_power(Phi))
        del Phi
        return w.cpu().numpy(), conv.cpu().numpy(), it.cpu().numpy()


def arm_W(Wd: dict, col: int, wd: Width) -> dict:
    p = wd.p; W = {}
    if "latent" in Wd:
        W["latent"] = latent_W(Wd["latent"][:, col], p)
    if "group" in Wd:
        W["group"] = group_W(Wd["group"][:, col], p, wd.g_fam, wd.n_fam)
    if "cluster" in Wd:
        W["cluster"] = group_W(Wd["cluster"][:, col], p, wd.g_clu, wd.n_clu)
    return W


def select_arms(W: dict, wd: Width, q: float, arms=ARMS) -> dict:
    """Selected latent indices per arm at level q (Knockoff+, offset 1)."""
    p = wd.p; sel = {}
    if "latent" in arms:
        sel["latent"] = select_single(W["latent"], q)
    if "group" in arms:
        sel["group"] = np.flatnonzero(np.isin(wd.g_fam, select_single(W["group"], q)))
    if "cluster" in arms:
        sel["cluster"] = np.flatnonzero(np.isin(wd.g_clu, select_single(W["cluster"], q)))
    for c in (1.0, 1.93):
        a = f"mkf_c{c:g}"
        if a in arms:
            sel[a] = mkf_select([W["latent"], W["group"]], [np.arange(p), wd.g_fam], [q, q], c=c)[0]
    return sel


def score_column(Wd: dict, col: int, wd: Width, qs, true_lat: set, true_con: set, arms=ARMS):
    """Metrics for one label column. Returns {arm: {metric: (len(qs),)}} and per-concept recovery
    {arm: (len(qs), n_true_concepts) bool} in sorted(true_con) order. Knockoff+ (offset 1). A concept
    (parent family) counts as discovered when at least one of its latents is selected."""
    p = wd.p
    W = arm_W(Wd, col, wd)
    tc = np.array(sorted(true_con))
    out = {a: {v: np.zeros(len(qs), np.float32) for v in LV} for a in arms}
    rec = {a: np.zeros((len(qs), tc.size), bool) for a in arms}
    tl = np.zeros(p, bool); tl[list(true_lat)] = True
    for iq, q in enumerate(qs):
        for a, s in select_arms(W, wd, q, arms).items():
            m = np.zeros(p, bool); m[s] = True
            fams = np.unique(wd.parent[m])
            nd, cd = int(m.sum()), fams.size
            out[a]["lat_nd"][iq] = nd; out[a]["con_nd"][iq] = cd
            out[a]["lat_fdr"][iq] = (m & ~tl).sum() / nd if nd else 0.0
            out[a]["lat_pow"][iq] = (m & tl).sum() / max(1, tl.sum())
            hit = np.isin(fams, tc)
            out[a]["con_fdr"][iq] = (~hit).sum() / cd if cd else 0.0
            out[a]["con_pow"][iq] = hit.sum() / max(1, tc.size)
            rec[a][iq] = np.isin(tc, fams)
    return out, rec


def real_sweep(cfg, layer, keys, all_keys, arms, draws, qs, device, tag, stream="real_knockoff"):
    """Real SST-2 labels: per width, `draws` knockoff redraws per fitted arm; records the selected
    latents per (arm, q, draw). Checkpointed per width."""
    from cseeds import ROOT, cint
    from planted_fdr_controls import fit_lasso_checked  # noqa: F401  (same solver, B = 1 via the batched path)
    cc = cfg["concept"]; cdir = ROOT / cc["cache_dir"]
    lam, mit, tol = cc["lasso"]["lambda"], cc["lasso"]["max_iter"], cc["lasso"]["tol"]
    fit_arms = [a for a in FIT_ARMS if a in arms or (a in ("latent", "group") and any(x.startswith("mkf") for x in arms))]
    y = np.load(cdir / "resid_meta.npz")["labels"].astype(np.float32)
    for key in keys:
        out = cdir / f"real_{tag}_{key}.npz"
        if out.exists():
            continue
        wd = Width(cdir, key, device, arms=fit_arms)
        Yt = torch.from_numpy(y[:, None]).to(device)
        SEL = {a: np.zeros((len(qs), draws, wd.p), bool) for a in arms}
        conv = np.zeros((len(fit_arms), draws), bool)
        ki = all_keys.index(key)
        import time
        t0 = time.time()
        for r in range(draws):
            Wd = {}
            for ai, a in enumerate(fit_arms):
                w, cv, _ = wd.fit(a, cint(cfg, stream, layer, ki, ai, r), Yt, lam, mit, tol)
                Wd[a] = w; conv[ai, r] = cv[0]
            W = arm_W(Wd, 0, wd)
            for iq, q in enumerate(qs):
                for a, s in select_arms(W, wd, q, arms).items():
                    SEL[a][iq, r, s] = True
        np.savez(out, **{f"sel__{a}": SEL[a] for a in arms}, conv=conv, parent=wd.parent, in_pool=np.isin(wd.parent, wd.pool),
                 latents=wd.latents)
        print(f"[real {tag} {key}] {draws} draws in {(time.time() - t0) / 60:.1f}m; conv {conv.mean():.3f}; "
              f"median per-latent discoveries at q={qs[min(1, len(qs) - 1)]}: "
              f"{np.median(SEL['latent'][min(1, len(qs) - 1)].sum(1)) if 'latent' in SEL else 'n/a'}", flush=True)
        del wd; torch.cuda.empty_cache() if device.type == "cuda" else None
