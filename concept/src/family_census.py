"""Step 3 — family census (label-free). Decides whether there is anything to test.

Reference parents at each layer: 16k latents firing >= 1%, top 2048 by firing rate (the main
config's latent_filter; at layer 20 exactly the main cache's retained set). For each wider SAE,
each latent maps to its argmax-decoder-cosine 16k latent (computed in GPU chunks), keeping the
cosine. Per reference parent and width: children with cos >= tau (and firing >= the 0.1% floor),
their firing, summed child firing vs parent firing, and the family's mechanism: mean pairwise
co-firing Jaccard and activation correlation among children (low co-firing = dilution;
corr >= 0.9 = redundancy).

Writes concept/results/census_L{layer}.json, the parent map per width (cache), fig_c2_census.png.
Usage: python concept/src/family_census.py --layer 12
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch

import _paths  # noqa: F401
from cseeds import ROOT, load_concept_config
from gpu import init_cuda


def ref_parents(fr16: np.ndarray, lf: dict) -> np.ndarray:
    surv = np.flatnonzero(fr16 >= lf["min_firing_rate"])
    if surv.size > lf["max_retained"]:
        surv = surv[np.argsort(-fr16[surv], kind="stable")[: lf["max_retained"]]]
    return np.sort(surv)


def parent_map(dec_w: np.ndarray, dec16: np.ndarray, dev, chunk=16384):
    """argmax_k cos(dec_w[i], dec16[k]) and its value, for every wide latent i. Both row-normalised."""
    D16 = torch.from_numpy(dec16.astype(np.float32)).to(dev)
    arg = np.empty(dec_w.shape[0], np.int32); val = np.empty(dec_w.shape[0], np.float32)
    for s in range(0, dec_w.shape[0], chunk):
        C = torch.from_numpy(dec_w[s:s + chunk].astype(np.float32)).to(dev) @ D16.T
        v, a = C.max(1)
        arg[s:s + chunk] = a.cpu().numpy(); val[s:s + chunk] = v.cpu().numpy()
    return arg, val


def mechanism(Xc: sp.csc_matrix, max_k: int):
    """Mean pairwise co-firing Jaccard and Pearson correlation among a family's children."""
    k = Xc.shape[1]
    if k < 2:
        return np.nan, np.nan
    if k > max_k:                                   # largest-firing children only, to bound cost
        top = np.argsort(-np.diff(Xc.indptr))[:max_k]
        Xc = Xc[:, np.sort(top)]
        k = max_k
    B = (Xc > 0).astype(np.float32)
    inter = (B.T @ B).toarray()
    cnt = np.diag(inter)
    union = cnt[:, None] + cnt[None, :] - inter
    jac = inter / np.maximum(union, 1)
    D = Xc.toarray()
    C = np.corrcoef(D.T)
    iu = np.triu_indices(k, 1)
    return float(np.nanmean(jac[iu])), float(np.nanmean(C[iu]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = load_concept_config()
    cc = cfg["concept"]; cen = cc["census"]
    cdir = ROOT / cc["cache_dir"]
    dev = init_cuda(args.device)
    keys = cc["sweep_L12"] if args.layer == 12 else cc["bridge_L20"]
    keys = [k for k in keys if (cdir / f"lat_{k}.json").exists()]
    base = keys[0]
    assert base.endswith("_16k"), "the 16k SAE is the reference"
    fr16 = np.load(cdir / f"lat_{base}.npz")["firing_rate_all"]
    parents = ref_parents(fr16, cfg["latent_filter"])
    dec16 = np.load(cdir / f"dec_{base}.npy")
    X16 = sp.load_npz(cdir / f"lat_{base}.csr.npz").tocsc()
    out = {"layer": args.layer, "widths": keys, "n_parents": int(parents.size), "taus": cen["taus"],
           "child_floor": cen["child_floor"], "per_width": {}}
    for k in keys:
        info = json.loads((cdir / f"lat_{k}.json").read_text())
        fr = np.load(cdir / f"lat_{k}.npz")["firing_rate_all"]
        if k == base:
            arg, val = np.arange(fr.size, dtype=np.int32), np.ones(fr.size, np.float32)
        else:
            arg, val = parent_map(np.load(cdir / f"dec_{k}.npy", mmap_mode="r"), dec16, dev)
        np.savez(cdir / f"parentmap_{k}.npz", parent=arg, cos=val)
        alive = fr >= cen["child_floor"]
        W = {"mean_l0": info["mean_l0"], "explained_variance": info["explained_variance"],
             "d_sae": info["d_sae"], "n_alive": int(alive.sum()), "tau": {}}
        X = sp.load_npz(cdir / f"lat_{k}.csr.npz").tocsc() if k != base else X16
        for tau in cen["taus"]:
            ch = alive & (val >= tau)
            sizes = np.bincount(arg[ch], minlength=fr16.size)[parents]
            sumfr = np.bincount(arg[ch], weights=fr[ch], minlength=fr16.size)[parents]
            ratio = sumfr / fr16[parents]
            # mechanism, on families with >= 2 children
            jac, cor = [], []
            fam_idx = {}
            for j in np.flatnonzero(ch):
                fam_idx.setdefault(int(arg[j]), []).append(j)
            for pi in parents:
                cols = fam_idx.get(int(pi), [])
                if len(cols) >= 2:
                    a, b = mechanism(X[:, sorted(cols)], cen["max_children_pairs"])
                    jac.append(a); cor.append(b)
            jac, cor = np.array(jac), np.array(cor)
            W["tau"][str(tau)] = {
                "size_median": float(np.median(sizes)), "size_q25": float(np.quantile(sizes, 0.25)),
                "size_q75": float(np.quantile(sizes, 0.75)), "size_mean": float(sizes.mean()),
                "frac_size0": float((sizes == 0).mean()), "frac_size_ge2": float((sizes >= 2).mean()),
                "sumfire_over_parent_median": float(np.median(ratio)),
                "sumfire_over_parent_q25": float(np.quantile(ratio, 0.25)),
                "sumfire_over_parent_q75": float(np.quantile(ratio, 0.75)),
                "n_multi": int(jac.size),
                "jaccard_median": float(np.nanmedian(jac)) if jac.size else None,
                "corr_median": float(np.nanmedian(cor)) if cor.size else None,
                "frac_redundancy": float(np.nanmean(cor >= cen["redundancy_corr"])) if cor.size else None,
                "frac_dilution": float(np.nanmean((jac < 0.2) & (cor < cen["redundancy_corr"]))) if jac.size else None,
            }
        out["per_width"][k] = W
        t = W["tau"]
        print(f"{k:9s} L0 {W['mean_l0']:6.1f} EV {W['explained_variance']:.3f} alive {W['n_alive']:7d} | median family "
              + "  ".join(f"t{tau}:{t[str(tau)]['size_median']:.0f}" for tau in cen["taus"])
              + f" | sumfire/parent (t0.5) {t['0.5']['sumfire_over_parent_median']:.2f}"
              + f" | redund {t['0.5']['frac_redundancy']} dilut {t['0.5']['frac_dilution']}", flush=True)
    # Gate C0
    widest = keys[-1]
    c0 = all(out["per_width"][widest]["tau"][str(t)]["size_median"] <= 1 for t in cen["taus"])
    out["gate_C0"] = {"widest": widest, "median_family_size_is_1_for_every_tau": c0,
                      "verdict": "STOP: splitting too rare" if c0 else "PROCEED"}
    print("GATE C0:", out["gate_C0"], flush=True)
    rd = ROOT / "concept" / "results"
    (rd / f"census_L{args.layer}.json").write_text(json.dumps(out, indent=1))
    make_fig(out, rd, cen["taus"])


def make_fig(out, rd, taus):
    keys = out["widths"]
    wid = [out["per_width"][k]["d_sae"] for k in keys]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    cols = plt.cm.viridis(np.linspace(0.1, 0.9, len(taus)))
    for c, tau in zip(cols, taus):
        T = [out["per_width"][k]["tau"][str(tau)] for k in keys]
        med = [t["size_median"] for t in T]; lo = [t["size_q25"] for t in T]; hi = [t["size_q75"] for t in T]
        ax[0].plot(wid, med, "o-", color=c, label=f"τ={tau}")
        ax[0].fill_between(wid, lo, hi, color=c, alpha=0.12)
        ax[1].plot(wid, [t["sumfire_over_parent_median"] for t in T], "o-", color=c, label=f"τ={tau}")
    ax[0].set(xscale="log", yscale="symlog", xlabel="SAE width", ylabel="children per parent (median, IQR)",
              title=f"Family size vs width, layer {out['layer']}")
    ax[1].axhline(1, color="k", ls="--", lw=1)
    ax[1].set(xscale="log", xlabel="SAE width", ylabel="Σ child firing / parent firing (median)",
              title="Do children cover the parent?")
    tau = 0.5
    T = [out["per_width"][k]["tau"][str(tau)] for k in keys]
    ax[2].plot(wid, [t["frac_dilution"] if t["frac_dilution"] is not None else np.nan for t in T], "o-", label="dilution")
    ax[2].plot(wid, [t["frac_redundancy"] if t["frac_redundancy"] is not None else np.nan for t in T], "s-", label="redundancy")
    ax[2].set(xscale="log", ylim=(0, 1), xlabel="SAE width", ylabel="fraction of multi-child families",
              title=f"Mechanism (τ={tau})")
    for a in ax:
        a.legend(fontsize=7); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(rd / f"fig_c2_census_L{out['layer']}.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
