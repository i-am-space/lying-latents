"""Step 8 — real SST-2 width sweep and cross-width stability.

Per width (Step 4 candidates; layer 12 sweep plus the layer-20 bridge): per-latent, group, cluster
and MKF+ at q in {0.05, 0.10, 0.20}, 30 knockoff redraws each. Records discoveries and the
selection frequency of each latent and concept across redraws. Stability: discoveries are mapped to
16k parents; stable parent set = parents selected in >= 50% of redraws; Jaccard between every width
pair. Clean comparison: the fixed planted-pool families (identical parents at every width);
secondary: all families (filler families differ across widths).

Usage: python concept/src/width_sweep_real.py run --layer 12 --keys ...  |  ... analyse
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import _paths  # noqa: F401
from bench import ARMS, real_sweep
from cseeds import ROOT, load_concept_config
from gpu import init_cuda

ARM_COLOR = {"latent": "#c0392b", "group": "#2e86c1", "cluster": "#7d3c98", "mkf_c1": "#27ae60", "mkf_c1.93": "#a9cce3", "group_sum": "#f39c12"}


def jac(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "analyse"])
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--draws", type=int, default=None)
    ap.add_argument("--tag", default="real")
    args = ap.parse_args()
    cfg = load_concept_config(); cc = cfg["concept"]; qs = cc["nominal_fdr_targets"]
    cdir = ROOT / cc["cache_dir"]; rd = ROOT / "concept" / "results"
    if args.phase == "run":
        all_keys = cc["sweep_L12"] if args.layer == 12 else cc["bridge_L20"]
        dev = init_cuda(args.device); torch.set_num_threads(8)
        real_sweep(cfg, args.layer, args.keys, all_keys, ARMS, args.draws or cc["real"]["draws"], qs, dev, args.tag)
        print("DONE", flush=True)
        return
    res = {}
    for layer, keys in ((12, cc["sweep_L12"]), (20, cc["bridge_L20"])):
        keys = [k for k in keys if (cdir / f"real_real_{k}.npz").exists()]
        if not keys:
            continue
        R = {}
        for k in keys:
            z = np.load(cdir / f"real_real_{k}.npz")
            par, pool = z["parent"], np.unique(z["parent"][z["in_pool"]])
            R[k] = {"par": par, "pool": pool, "sel": {a: z[f"sel__{a}"] for a in ARMS}, "conv": float(z["conv"].mean())}
        L = {"keys": keys, "widths": [cc["saes"][k]["width"] for k in keys], "per_width": {}, "jaccard": {}}
        for k in keys:
            W = {"conv_rate": R[k]["conv"]}
            for a in ARMS:
                S = R[k]["sel"][a]                                           # (q, draws, p)
                par = R[k]["par"]
                conc = np.array([[np.unique(par[S[iq, r]]).size for r in range(S.shape[1])] for iq in range(len(qs))])
                pfreq = {}
                for iq in range(len(qs)):
                    f = {}
                    for r in range(S.shape[1]):
                        for pa in np.unique(par[S[iq, r]]):
                            f[int(pa)] = f.get(int(pa), 0) + 1
                    pfreq[iq] = {pa: c / S.shape[1] for pa, c in f.items()}
                W[a] = {"median_latent_discoveries": np.median(S.sum(2), 1).tolist(),
                        "median_concept_discoveries": np.median(conc, 1).tolist(),
                        "frac_latents_selected_ge_half": (S.mean(1) >= 0.5).sum(1).tolist(),
                        "stable_parents": {str(q): sorted(pa for pa, f in pfreq[iq].items() if f >= cc["real"]["stable_freq"])
                                           for iq, q in enumerate(qs)}}
            L["per_width"][k] = W
        for a in ARMS:
            for scope in ("pool", "all", "shared"):         # shared: post hoc, see concept_findings.md
                Mj = np.full((len(keys), len(keys)), np.nan)
                for i, ki in enumerate(keys):
                    for j, kj in enumerate(keys):
                        si = set(L["per_width"][ki][a]["stable_parents"]["0.1"])
                        sj = set(L["per_width"][kj][a]["stable_parents"]["0.1"])
                        if scope == "pool":
                            pool = set(R[ki]["pool"].tolist())
                            si, sj = si & pool, sj & pool
                        elif scope == "shared":                  # parents in both widths' candidate sets
                            sh = set(R[ki]["par"].tolist()) & set(R[kj]["par"].tolist())
                            si, sj = si & sh, sj & sh
                        Mj[i, j] = jac(si, sj)
                L["jaccard"][f"{a}/{scope}"] = Mj.tolist()
        res[f"L{layer}"] = L
        print(f"layer {layer}:", flush=True)
        for k in keys:
            print(f"  {k:9s} " + "  ".join(f"{a}: {L['per_width'][k][a]['median_latent_discoveries'][1]:.0f} lat / "
                                           f"{L['per_width'][k][a]['median_concept_discoveries'][1]:.0f} con" for a in ARMS), flush=True)
    (rd / "real_sweep.json").write_text(json.dumps(res, indent=1, default=float))
    L = res.get("L12")
    if not L:
        return
    keys, wid = L["keys"], L["widths"]
    fig = plt.figure(figsize=(17, 9))
    ax = fig.add_subplot(2, 3, 1)
    for a in ARMS:
        ax.plot(wid, [L["per_width"][k][a]["median_concept_discoveries"][1] for k in keys], "o-", color=ARM_COLOR[a], label=a)
    ax.set(xscale="log", xlabel="SAE width (layer 12)", ylabel="concepts discovered (median of 30 draws)", title="Real SST-2, q = 0.1: concepts")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = fig.add_subplot(2, 3, 2)
    for a in ARMS:
        ax.plot(wid, [L["per_width"][k][a]["median_latent_discoveries"][1] for k in keys], "o-", color=ARM_COLOR[a], label=a)
    ax.set(xscale="log", yscale="symlog", xlabel="SAE width (layer 12)", ylabel="latents selected (median)", title="Real SST-2, q = 0.1: latents")
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(2, 3, 3)
    for a in ARMS:
        Mj = np.array(L["jaccard"][f"{a}/shared"])
        ax.plot(wid, Mj[0], "o-", color=ARM_COLOR[a], label=a)
    ax.set(xscale="log", ylim=(0, 1.02), xlabel="SAE width", ylabel="Jaccard of stable parent set vs 16k",
           title="Stability vs 16k (parents present at both widths; post hoc)")
    ax.grid(alpha=0.3)
    for i, a in enumerate(("latent", "group", "group_sum")):
        ax = fig.add_subplot(2, 3, 4 + i)
        Mj = np.array(L["jaccard"][f"{a}/shared"])
        im = ax.imshow(Mj, vmin=0, vmax=1, cmap="viridis")
        lab = [k.split("_")[1] for k in keys]
        ax.set_xticks(range(len(keys)), lab, rotation=45); ax.set_yticks(range(len(keys)), lab)
        for (r, c), v in np.ndenumerate(Mj):
            ax.text(c, r, "–" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=7, color="w" if (np.isnan(v) or v < 0.6) else "k")
        ax.set_title(f"{a}: stable-parent Jaccard (shared, q=0.1)", fontsize=10)
    fig.colorbar(im, ax=fig.axes[-3:], shrink=0.7)
    fig.savefig(rd / "fig_c6_real_sweep.png", dpi=130, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
