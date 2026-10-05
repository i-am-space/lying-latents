"""Step 7 analysis: pre-registered contrasts and figures from the planted-concept checkpoints.

  (i)   slope of concept power vs log2(width), per arm, paired over replicates, per cell
  (ii)  group - per-latent concept power, paired, per width and cell
  (iii) FDR breaches (latent- and concept-level), one-sided vs q, claimed only after
        Benjamini-Yekutieli across cells (main repo rule K3)
Stratification: per planted concept, recovery vs its family size at that width, and by mechanism
(multi-child family: redundancy if mean child correlation >= 0.9, else dilution).

Usage: python concept/src/report_concept.py --tag full
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _paths  # noqa: F401
from bench import ARMS
from cseeds import ROOT, load_concept_config
from reanalysis_multiplicity import by_adjust, one_sided_exceed, two_sided_diff

ARM_COLOR = {"latent": "#c0392b", "group": "#2e86c1", "cluster": "#7d3c98", "mkf_c1": "#27ae60", "mkf_c1.93": "#82e0aa", "group_sum": "#f39c12"}
ARM_LABEL = {"latent": "per-latent Knockoff+", "group": "group Knockoff+ (families)", "cluster": "group Knockoff+ (clusters)",
             "mkf_c1": "MKF+ (c=1)", "mkf_c1.93": "MKF+ (c=1.93)",
             "group_sum": "group-sum statistic (amendment 2)"}


def load(cdir, tag, keys):
    Z = {k: dict(np.load(cdir / f"ckpt_{tag}_{k}.npz")) for k in keys if (cdir / f"ckpt_{tag}_{k}.npz").exists()}
    return {k: z for k, z in Z.items() if int(z["reps_done"]) > 0}


def mechanism_labels(cdir, key, pool):
    z = np.load(cdir / f"cand_{key}.npz")
    Zm, par = z["Z"], z["parent"]
    lab = {}
    for pa in pool:
        cols = np.flatnonzero(par == pa)
        if cols.size < 2:
            lab[int(pa)] = "single"
        else:
            C = np.corrcoef(Zm[:, cols].T)
            lab[int(pa)] = "redundancy" if C[np.triu_indices(cols.size, 1)].mean() >= 0.9 else "dilution"
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    cfg = load_concept_config(); cc = cfg["concept"]; gr = cc["planted"]
    qs = cc["nominal_fdr_targets"]; cdir = ROOT / cc["cache_dir"]; rd = ROOT / "concept" / "results"
    designs = ["A", "B"]
    out = {"grid": gr, "qs": qs}
    for layer, keys in ((12, cc["sweep_L12"]), (20, cc["bridge_L20"])):
        Z = load(cdir, args.tag, keys)
        keys = [k for k in keys if k in Z]
        if len(keys) < 2:
            continue
        R = min(int(Z[k]["reps_done"]) for k in keys)
        x = np.log2([cc["saes"][k]["width"] for k in keys])
        get = lambda a, v: np.stack([Z[k][f"{a}__{v}"][..., :R, :] for k in keys])   # (W, D, K, A, F, R, Q)
        L = {"keys": keys, "replicates": R, "slopes": {}, "group_minus_latent": {}, "fdr": {}, "means": {}}
        cells = [(di, iK, ia, jf, iq) for di in range(2) for iK in range(len(gr["K_c"])) for ia in range(len(gr["amplitudes"]))
                 for jf in range(len(gr["forms"])) for iq in range(len(qs))]
        name = lambda c: f"{designs[c[0]]}/K{gr['K_c'][c[1]]}/amp{gr['amplitudes'][c[2]]:g}/{gr['forms'][c[3]]}/q{qs[c[4]]}"
        for a in ARMS:
            P = get(a, "con_pow")
            L["means"][a] = {v: get(a, v).mean(axis=5).tolist() for v in ("con_pow", "lat_pow", "con_fdr", "lat_fdr", "con_nd", "lat_nd")}
            # (i) slope per replicate
            rows = []
            for c in cells:
                y = P[:, c[0], c[1], c[2], c[3], :, c[4]]                          # (W, R)
                s = np.polyfit(x, y, 1)[0]                                       # (R,)
                m, se, p = two_sided_diff(s, np.zeros_like(s), paired=True)
                rows.append({"cell": name(c), "slope": m, "se": se, "p": p})
            padj = by_adjust(np.array([r["p"] for r in rows]))
            for r, pa in zip(rows, padj):
                r["p_by"] = float(pa)
            L["slopes"][a] = {"cells": rows,
                              "n_neg_by": int(sum(r["p_by"] < 0.05 and r["slope"] < 0 for r in rows)),
                              "n_pos_by": int(sum(r["p_by"] < 0.05 and r["slope"] > 0 for r in rows)),
                              "median_slope": float(np.median([r["slope"] for r in rows]))}
            # (iii) FDR breaches
            for lvl in ("lat_fdr", "con_fdr"):
                F = get(a, lvl)
                rows = []
                for wi, k in enumerate(keys):
                    for c in cells:
                        v = F[wi, c[0], c[1], c[2], c[3], :, c[4]]
                        m, se, p = one_sided_exceed(v, qs[c[4]])
                        rows.append({"width": k, "cell": name(c), "mean": m, "se": se, "p": p})
                padj = by_adjust(np.array([r["p"] for r in rows]))
                L["fdr"][f"{a}/{lvl}"] = {"n_cells": len(rows), "n_breach_by": int((padj < 0.05).sum()),
                                          "n_mean_above_q": int(sum(r["mean"] > float(r["cell"].split("/q")[1]) for r in rows)),
                                          "max_mean": float(max(r["mean"] for r in rows)),
                                          "breaches": [dict(r, p_by=float(pp)) for r, pp in zip(rows, padj) if pp < 0.05]}
        # (ii) group - latent, paired, per width
        Pl, Pg = get("latent", "con_pow"), get("group", "con_pow")
        rows = []
        for wi, k in enumerate(keys):
            for c in cells:
                a_, b_ = Pg[wi, c[0], c[1], c[2], c[3], :, c[4]], Pl[wi, c[0], c[1], c[2], c[3], :, c[4]]
                m, se, p = two_sided_diff(a_, b_, paired=True)
                rows.append({"width": k, "cell": name(c), "diff": m, "se": se, "p": p})
        padj = by_adjust(np.array([r["p"] for r in rows]))
        for r, pa in zip(rows, padj):
            r["p_by"] = float(pa)
        L["group_minus_latent"] = {"by_width": {k: {"mean_diff": float(np.mean([r["diff"] for r in rows if r["width"] == k])),
                                                    "n_pos_by": int(sum(r["p_by"] < 0.05 and r["diff"] > 0 for r in rows if r["width"] == k)),
                                                    "n_neg_by": int(sum(r["p_by"] < 0.05 and r["diff"] < 0 for r in rows if r["width"] == k)),
                                                    "n_cells": int(sum(r["width"] == k for r in rows))} for k in keys}}
        # stratification by family size and mechanism (q = 0.1)
        iq = qs.index(0.1)
        strat = {}
        for k in keys:
            pool = np.unique(np.load(cdir / f"cand_{k}.npz")["pool"])
            mech = mechanism_labels(cdir, k, pool)
            FS = Z[k]["famsize"][:, :R]                                               # (K, R, Kmax)
            for a in ARMS:
                REC = Z[k][f"rec__{a}"][..., :R, iq, :]                              # (D, K, A, F, R, Kmax)
                for iK, K in enumerate(gr["K_c"]):
                    fs = FS[iK, :, :K]                                                 # (R, K)
                    rec = REC[:, iK, :, :, :, :K]                                       # (D, A, F, R, K)
                    for size_bin, lo, hi in (("1", 1, 1), ("2-3", 2, 3), ("4-7", 4, 7), ("8+", 8, 10**6)):
                        msk = (fs >= lo) & (fs <= hi)
                        if msk.any():
                            v = rec[..., msk]                                          # (D, A, F, n)
                            strat.setdefault(f"{k}/{a}/size{size_bin}", []).append(v.mean(axis=(2, 3)).tolist())
            out.setdefault("mechanism_counts", {})[k] = {m: int(sum(1 for v in mech.values() if v == m)) for m in ("single", "dilution", "redundancy")}
        L["recovery_by_family_size"] = {s: np.mean(np.array(v), axis=0).tolist() for s, v in strat.items()}
        out[f"L{layer}"] = L
        print(f"=== layer {layer} ({R} replicates) ===")
        for a in ARMS:
            print(f"  {a:10s} slope cells BY-neg {L['slopes'][a]['n_neg_by']:3d} / BY-pos {L['slopes'][a]['n_pos_by']:3d} "
                  f"(median slope {L['slopes'][a]['median_slope']:+.3f}/doubling)  FDR breaches lat {L['fdr'][a + '/lat_fdr']['n_breach_by']} "
                  f"con {L['fdr'][a + '/con_fdr']['n_breach_by']} (max mean con FDR {L['fdr'][a + '/con_fdr']['max_mean']:.3f})")
        for k, v in L["group_minus_latent"]["by_width"].items():
            print(f"  group - latent concept power at {k}: {v['mean_diff']:+.3f}  (BY +{v['n_pos_by']}/-{v['n_neg_by']} of {v['n_cells']})")
    (rd / "planted_concept.json").write_text(json.dumps(out, indent=1, default=float))
    figures(out, cc, rd)


def figures(out, cc, rd):
    gr = cc["planted"]; qs = cc["nominal_fdr_targets"]; iq = qs.index(0.1)
    L = out.get("L12")
    if not L:
        return
    keys = L["keys"]; wid = [cc["saes"][k]["width"] for k in keys]
    amps_show = [1.0, 3.0, 8.0]
    # fig_c3: concept power vs width
    fig, axes = plt.subplots(2, len(amps_show), figsize=(5.2 * len(amps_show), 8.5), sharey=True)
    for di, d in enumerate(["A", "B"]):
        for j, amp in enumerate(amps_show):
            ia = gr["amplitudes"].index(amp); ax = axes[di, j]
            for a in ARMS:
                P = np.array(L["means"][a]["con_pow"])[:, di, :, ia, :, iq]               # (W, K, F)
                ax.plot(wid, P.mean(axis=(1, 2)), "o-", color=ARM_COLOR[a], label=ARM_LABEL[a], lw=2 if a in ("latent", "group") else 1.3)
            ax.set(xscale="log", ylim=(-0.02, 1.02), title=f"design {d}, amplitude {amp:g}")
            ax.grid(alpha=0.3)
            if di == 1: ax.set_xlabel("SAE width (layer 12)")
            if j == 0: ax.set_ylabel("concept power (q = 0.1)\nmean over K_c × form")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Concept discovery vs SAE width: per-latent vs group vs multilayer knockoffs", fontsize=13)
    fig.tight_layout(); fig.savefig(rd / "fig_c3_power_vs_width.png", dpi=140); plt.close(fig)
    # fig_c4: FDR vs width (max over amplitudes of the mean, and mean over cells)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    for di, d in enumerate(["A", "B"]):
        for li, lvl in enumerate(("con_fdr", "lat_fdr")):
            ax = axes[li, di]
            for a in ARMS:
                F = np.array(L["means"][a][lvl])[:, di, :, :, :, iq]                    # (W, K, A, F)
                ax.plot(wid, F.reshape(len(keys), -1).mean(1), "o-", color=ARM_COLOR[a], label=ARM_LABEL[a])
                ax.plot(wid, F.reshape(len(keys), -1).max(1), ":", color=ARM_COLOR[a], lw=1)
            ax.axhline(0.1, color="k", ls="--", lw=1)
            ax.set(xscale="log", ylim=(0, 0.35), title=f"design {d}: {'concept' if lvl == 'con_fdr' else 'latent'}-level FDR (q = 0.1)"
                   + (" — approximate truth, not an error rate" if d == "B" else ""))
            if a == ARMS[-1] and d == "A" and lvl == "con_fdr":
                ax.text(0.02, 0.97, "cluster arm: unit mismatch (clusters span several parents)", transform=ax.transAxes,
                        fontsize=8, va="top", color=ARM_COLOR["cluster"])
            ax.grid(alpha=0.3)
            if li == 1: ax.set_xlabel("SAE width (layer 12)")
    axes[0, 0].legend(fontsize=8); axes[0, 0].set_ylabel("solid: mean over cells; dotted: worst cell")
    fig.tight_layout(); fig.savefig(rd / "fig_c4_fdr_vs_width.png", dpi=140); plt.close(fig)
    # fig_c5: recovery by family size
    S = L["recovery_by_family_size"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    bins = ["1", "2-3", "4-7", "8+"]
    for di, d in enumerate(["A", "B"]):
        ax = axes[di]
        for a in ARMS:
            vals = []
            for b in bins:
                v = [np.mean(np.array(S[f"{k}/{a}/size{b}"])[di, gr["amplitudes"].index(1.0)]) for k in keys if f"{k}/{a}/size{b}" in S]
                vals.append(np.mean(v) if v else np.nan)
            ax.plot(range(len(bins)), vals, "o-", color=ARM_COLOR[a], label=ARM_LABEL[a])
        ax.set(xticks=range(len(bins)), xticklabels=bins, xlabel="children of the planted concept at that width",
               title=f"design {d}, amplitude 1, q = 0.1 (pooled over widths)", ylim=(-0.02, 1.02))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("concept recovery"); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(rd / "fig_c5_mechanism.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
