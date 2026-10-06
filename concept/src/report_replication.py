"""Replication report (concept_amendment_3): pre-registered rules R1-R5, T1G and SEC, and figures.

Reads ckpt_rep_{key}.npz (planted, designs A-D) and real_rep_{key}.npz (real labels) for layers 19
(primary), 5 (secondary) and 12 (consistency). Works on partial runs: a layer is analysed when all its
widths have >= 2 replicates, using the replicates every width has.
Writes concept/results/replication.json and fig_r1..fig_r3.

Usage: python concept/src/report_replication.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import _paths  # noqa: F401
from cseeds import ROOT, layer_keys, load_concept_config
from reanalysis_multiplicity import by_adjust, holm, one_sided_exceed

cc = load_concept_config()["concept"]
rep = cc["replication"]; gr = cc["planted"]; qs = cc["nominal_fdr_targets"]
cdir = ROOT / cc["cache_dir"]; rd = ROOT / "concept" / "results"
DES = rep["designs"]; IQ = qs.index(rep["q_primary"])
WEAK = [gr["amplitudes"].index(a) for a in rep["weak_amplitudes"]]
LAYERS = [(19, "primary"), (5, "secondary"), (12, "consistency")]
COL = {"latent": "#c0392b", "group": "#2e86c1", "group_sum": "#f39c12", "group_lasso": "#8e44ad", "mkf_c1": "#27ae60"}
LAB = {"latent": "per-latent", "group": "group (lasso W)", "group_sum": "group-sum (primary)",
       "group_lasso": "group lasso (challenger)", "mkf_c1": "MKF+ (c=1)"}
ARMS = ["latent", "group", "group_sum", "group_lasso", "mkf_c1"]


def load_layer(layer):
    keys = layer_keys(cc, layer)
    Z = {}
    for k in keys:
        f = cdir / f"ckpt_rep_{k}.npz"
        if not f.exists():
            return None
        Z[k] = dict(np.load(f))
    R = min(int(z["reps_done"]) for z in Z.values())
    if R < 2:
        return None
    return keys, Z, R


def endpoint(Z, keys, R, arm, di, metric="con_pow"):
    """(W, R): per replicate, metric at q_primary averaged over K_c, weak amplitudes and forms."""
    return np.stack([Z[k][f"{arm}__{metric}"][di][:, WEAK][..., :R, IQ].mean(axis=(0, 1, 2)) for k in keys])


def diff_tests(a, b, x):
    """R1/R2 statistics for (a - b): one-sided t at the widest width, and of per-replicate slopes."""
    d = a - b
    sl = np.array([np.polyfit(x, d[:, r], 1)[0] for r in range(d.shape[1])])
    p1 = float(stats.ttest_1samp(d[-1], 0.0, alternative="greater").pvalue)
    p2 = float(stats.ttest_1samp(sl, 0.0, alternative="greater").pvalue)
    se = lambda v: float(v.std(ddof=1) / np.sqrt(v.size))
    return {"diff_widest": float(d[-1].mean()), "se_widest": se(d[-1]), "p_R1": p1,
            "slope": float(sl.mean()), "se_slope": se(sl), "p_R2": p2, "diff_by_width": d.mean(1).tolist()}


def t1g():
    f = rd / "toy_glasso.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text()); rows = []
    for k, v in r.items():
        if not k.startswith("gauss"):
            continue
        for lvl in ("lat", "con"):
            if k.endswith("null"):
                nd = np.array(v[f"{lvl}_nd"])[0]
                for j, q in enumerate(qs):
                    kk = int((nd[:, j] > 0).sum())
                    rows.append(stats.binomtest(kk, nd.shape[0], q, alternative="greater").pvalue)
            else:
                F = np.array(v[f"{lvl}_fdr"])
                for ia in range(F.shape[0]):
                    for j, q in enumerate(qs):
                        rows.append(one_sided_exceed(F[ia, :, j], q)[2])
    padj = by_adjust(np.array(rows))
    return {"cells": len(rows), "by_breaches": int((padj < 0.05).sum()), "PASS": bool((padj >= 0.05).all())}


def real_stability(layer, keys):
    out = {}
    try:
        Zs = {k: np.load(cdir / f"real_rep_{k}.npz") for k in (keys[0], keys[-1])}
    except FileNotFoundError:
        return None
    par0, par1 = Zs[keys[0]]["parent"], Zs[keys[-1]]["parent"]
    shared = set(par0.tolist()) & set(par1.tolist())
    for a in ("latent", "group", "group_sum", "group_lasso"):
        st = []
        for k, par in ((keys[0], par0), (keys[-1], par1)):
            S = Zs[k][f"sel__{a}"][IQ]                                     # (draws, p)
            freq = {}
            for r in range(S.shape[0]):
                for pa in np.unique(par[S[r]]):
                    freq[int(pa)] = freq.get(int(pa), 0) + 1
            st.append({pa for pa, c in freq.items() if c / S.shape[0] >= cc["real"]["stable_freq"]} & shared)
        u = st[0] | st[1]
        out[a] = {"jaccard_16k_vs_widest": len(st[0] & st[1]) / len(u) if u else float("nan"),
                  "n_stable_16k": len(st[0]), "n_stable_widest": len(st[1]),
                  "median_concepts_widest": float(np.median([np.unique(par1[s]).size for s in Zs[keys[-1]][f"sel__{a}"][IQ]]))}
    return out


def main():
    res = {"amendment": "concept_amendment_3", "T1G": t1g(), "layers": {}}
    plot = {}
    for layer, role in LAYERS:
        got = load_layer(layer)
        if got is None:
            continue
        keys, Z, R = got
        x = np.log2([cc["saes"][k]["width"] for k in keys])
        E = {(a, d): endpoint(Z, keys, R, a, di) for a in ARMS for di, d in enumerate(DES)}
        X = {(a, d): endpoint(Z, keys, R, a, di, "lat_pow") for a in ARMS for di, d in enumerate(DES)}
        L = {"role": role, "keys": keys, "replicates": R}
        prim = diff_tests(E[("group_sum", "A")], E[("latent", "A")], x)
        ph = holm(np.array([prim["p_R1"], prim["p_R2"]]))
        prim.update({"p_R1_holm": float(ph[0]), "p_R2_holm": float(ph[1]), "PASS": bool((ph < 0.05).all())})
        L["primary_R1_R2"] = prim
        # R3: design A group_sum concept FDR at q = 0.1, every cell of this layer, BY
        F = np.stack([Z[k]["group_sum__con_fdr"][0][..., :R, IQ] for k in keys])            # (W, K, A, F, R)
        cells = [one_sided_exceed(F[w, i, a, f], qs[IQ]) for w in range(len(keys)) for i in range(F.shape[1])
                 for a in range(F.shape[2]) for f in range(F.shape[3])]
        padj = by_adjust(np.array([c[2] for c in cells]))
        L["R3_fdr"] = {"cells": len(cells), "by_breaches": int((padj < 0.05).sum()),
                       "mean_fdr": float(F.mean()), "max_cell_mean": float(max(c[0] for c in cells)),
                       "PASS": bool((padj >= 0.05).all())}
        # SEC: secondary contrasts, BY within this layer
        sec = {}
        for d in DES:
            sec[f"{d}: group_sum - latent"] = diff_tests(E[("group_sum", d)], E[("latent", d)], x)
            sec[f"{d}: group_lasso - latent"] = diff_tests(E[("group_lasso", d)], E[("latent", d)], x)
            sec[f"{d}: group_lasso - group_sum"] = diff_tests(E[("group_lasso", d)], E[("group_sum", d)], x)
            sec[f"{d}: group - latent"] = diff_tests(E[("group", d)], E[("latent", d)], x)
        sec.pop("A: group_sum - latent")                                                # that is the primary
        ps = np.array([v[k] for v in sec.values() for k in ("p_R1", "p_R2")])
        padj = by_adjust(ps).reshape(-1, 2)
        for (name, v), pa in zip(sec.items(), padj):
            v["p_R1_by"], v["p_R2_by"] = float(pa[0]), float(pa[1])
        L["secondary"] = sec
        L["extent_widest"] = {d: {a: float(X[(a, d)][-1].mean()) for a in ARMS} for d in DES}
        L["concept_power_widest_all_amps"] = {
            d: {a: float(Z[keys[-1]][f"{a}__con_pow"][di][..., :R, IQ].mean()) for a in ARMS} for di, d in enumerate(DES)}
        L["real_stability"] = real_stability(layer, keys)
        res["layers"][str(layer)] = L
        plot[layer] = (keys, E, R)
        print(f"== layer {layer} ({role}, {R} replicates)")
        print(f"   R1 diff at {keys[-1]}: {prim['diff_widest']:+.3f} ± {prim['se_widest']:.3f}  p_holm {prim['p_R1_holm']:.2g} | "
              f"R2 slope {prim['slope']:+.4f}/doubling  p_holm {prim['p_R2_holm']:.2g} | R1&R2 {'PASS' if prim['PASS'] else 'FAIL'}")
        print(f"   R3 FDR: {L['R3_fdr']['by_breaches']} BY breaches / {L['R3_fdr']['cells']} cells, mean {L['R3_fdr']['mean_fdr']:.3f}")
        for name, v in sec.items():
            print(f"   {name:28s} {v['diff_widest']:+.3f} (BY p {v['p_R1_by']:.2g})  slope {v['slope']:+.4f} (BY p {v['p_R2_by']:.2g})")

    # outcomes
    lay = res["layers"]
    if "19" in lay:
        ok19 = lay["19"]["primary_R1_R2"]["PASS"] and lay["19"]["R3_fdr"]["PASS"]
        r1 = lay["19"]["primary_R1_R2"]["p_R1_holm"] < 0.05
        out = "success" if ok19 else ("failure" if not r1 else "partial")
        if ok19 and "5" in lay and lay["5"]["primary_R1_R2"]["PASS"]:
            out = "strong_success"
        res["outcome"] = out
        print("OUTCOME:", out, "| T1G:", res["T1G"])
    (rd / "replication.json").write_text(json.dumps(res, indent=1, default=float))
    figures(plot, res)


def figures(plot, res):
    if not plot:
        return
    layers = [l for l, _ in LAYERS if l in plot]
    fig, axes = plt.subplots(len(DES), len(layers), figsize=(5.3 * len(layers), 3.6 * len(DES)), squeeze=False, sharey="row")
    dname = {"A": "A: sum of children", "B": "B: parent activation", "C": "C: unequal + weights", "D": "D: mixed-sign weights"}
    for j, layer in enumerate(layers):
        keys, E, R = plot[layer]
        wid = [cc["saes"][k]["width"] for k in keys]
        for i, d in enumerate(DES):
            ax = axes[i, j]
            for a in ARMS:
                v = E[(a, d)]
                ax.errorbar(wid, v.mean(1), yerr=v.std(1, ddof=1) / np.sqrt(R), marker="o", capsize=3, color=COL[a],
                            label=LAB[a], lw=2.2 if a in ("latent", "group_sum") else 1.3, ls=":" if a == "mkf_c1" else "-")
            ax.set(xscale="log", ylim=(0, 1.02), title=f"layer {layer} — design {dname[d]}")
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel("concept power, weak signals\n(amps 0.5-2, q = 0.1)")
            if i == len(DES) - 1:
                ax.set_xlabel("SAE width")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Weak-concept power vs width (outcome: {res.get('outcome', 'interim')})", fontsize=12)
    fig.tight_layout(); fig.savefig(rd / "fig_r1_replication.png", dpi=130); plt.close(fig)

    # forest plot of effects at the widest width
    fig, ax = plt.subplots(figsize=(9, 0.5 + 0.42 * len(layers) * len(DES) * 2))
    y, ticks = 0, []
    for layer in layers:
        L = res["layers"][str(layer)]
        for d in DES:
            for a, c in (("group_sum", COL["group_sum"]), ("group_lasso", COL["group_lasso"])):
                v = L["primary_R1_R2"] if (a == "group_sum" and d == "A") else L["secondary"][f"{d}: {a} - latent"]
                ax.errorbar(v["diff_widest"], y, xerr=1.96 * v["se_widest"], fmt="o", color=c, capsize=3)
                ticks.append((y, f"L{layer} {d} {'sum' if a == 'group_sum' else 'glasso'}"
                                 + ("  ← primary" if (a == "group_sum" and d == "A" and layer == 19) else "")))
                y -= 1
        y -= 0.6
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=8)
    ax.set_xlabel("pooled minus per-latent concept power at 1M (weak signals, q = 0.1), mean ± 95% CI")
    ax.set_title("Effect of pooling at the widest width"); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(rd / "fig_r2_effects.png", dpi=130); plt.close(fig)

    # extent and real-label stability
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    w = 0.8 / len(ARMS)
    for i, a in enumerate(ARMS):
        ax[0].bar(np.arange(len(layers)) + i * w, [res["layers"][str(l)]["extent_widest"]["A"][a] for l in layers], w,
                  color=COL[a], label=LAB[a])
        st = [res["layers"][str(l)]["real_stability"] for l in layers]
        if a != "mkf_c1":
            ax[1].bar(np.arange(len(layers)) + i * w, [s[a]["jaccard_16k_vs_widest"] if s else np.nan for s in st], w,
                      color=COL[a], label=LAB[a])
    for k, (ttl, yl) in enumerate((("Concept extent at 1M (design A, weak signals)", "share of a concept's latents certified"),
                                   ("Real SST-2: stable concepts at 16k vs 1M", "Jaccard (parents in both candidate sets)"))):
        ax[k].set_xticks(np.arange(len(layers)) + 0.4 - w / 2, [f"layer {l}" for l in layers])
        ax[k].set(ylim=(0, 1.02), title=ttl, ylabel=yl); ax[k].grid(axis="y", alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(rd / "fig_r3_extent_stability.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
