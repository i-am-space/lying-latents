"""Step 1 report: fig_c1_toy.png, rule T1 (library FDR unit test on the exactly-valid Gaussian toy),
and the numbers behind results/proposition.md."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import _paths  # noqa: F401
from cseeds import ROOT
from reanalysis_multiplicity import by_adjust, one_sided_exceed

rd = ROOT / "concept" / "results"
d = json.loads((rd / "toy_proposition.json").read_text())
meta, res = d["meta"], d["results"]
MS, AMPS, QS, ARMS = meta["ms"], meta["amps"], meta["qs"], meta["arms"]
COL = {"latent": "#c0392b", "group": "#2e86c1", "mkf_c1": "#27ae60", "mkf_c1.93": "#82e0aa"}
LAB = {"latent": "per-latent", "group": "group (family)", "mkf_c1": "MKF+ c=1", "mkf_c1.93": "MKF+ c=1.93"}
iq = QS.index(0.1)
A = lambda key, arm, v: np.array(res[key]["M"][arm][v], float)          # (amps, reps, qs)

# ---- rule T1 -------------------------------------------------------------
t1 = {"null": [], "planted": []}
for design in ("dilution", "redundancy"):
    for m in MS:
        nk = f"gauss/{design}/m{m}/null"
        for arm in ("latent", "group", "mkf_c1.93", "mkf_c1"):
            for lvl in ("lat", "con"):
                nd = A(nk, arm, f"{lvl}_nd")[0]                             # (reps, qs)
                for j, q in enumerate(QS):
                    k = int((nd[:, j] > 0).sum()); n = nd.shape[0]
                    p = stats.binomtest(k, n, q, alternative="greater").pvalue
                    t1["null"].append({"design": design, "m": m, "arm": arm, "level": lvl, "q": q, "p_any": k / n, "p": p})
        pk = f"gauss/{design}/m{m}/planted"
        for arm in ARMS:
            for lvl in ("lat", "con"):
                F = A(pk, arm, f"{lvl}_fdr")
                for ia, amp in enumerate(AMPS):
                    for j, q in enumerate(QS):
                        mu, se, p = one_sided_exceed(F[ia, :, j], q)
                        t1["planted"].append({"design": design, "m": m, "arm": arm, "level": lvl, "amp": amp, "q": q, "fdr": mu, "p": p})
for kind in ("null", "planted"):
    padj = by_adjust(np.array([r["p"] for r in t1[kind]]))
    for r, pa in zip(t1[kind], padj):
        r["p_by"] = float(pa)
guaranteed = ("latent", "group", "mkf_c1.93")
breach = lambda kind: [r for r in t1[kind] if r["p_by"] < 0.05 and r["arm"] in guaranteed]
T1 = {"PASS": not breach("null") and not breach("planted"),
      "null_cells": len(t1["null"]), "null_breaches_guaranteed_arms": breach("null"),
      "planted_cells": len(t1["planted"]), "planted_breaches_guaranteed_arms": breach("planted"),
      "mkf_c1_breaches": [r for k in t1 for r in t1[k] if r["p_by"] < 0.05 and r["arm"] == "mkf_c1"],
      "max_null_p_any": {a: max(r["p_any"] for r in t1["null"] if r["arm"] == a and r["q"] == 0.1) for a in ("latent", "group", "mkf_c1.93", "mkf_c1")}}
print("T1:", "PASS" if T1["PASS"] else "FAIL", {k: (len(v) if isinstance(v, list) else v) for k, v in T1.items()})

# ---- proposition numbers -----------------------------------------------------
prop = {}
for design in ("dilution", "redundancy"):
    for m in MS:
        k = f"zinf/{design}/m{m}/planted"
        prop[f"{design}/m{m}"] = {"per_child_effect": res[k]["per_child_effect"], "conv_rate": res[k]["conv_rate"],
                                  "mean_s_latent": res[k]["info"]["mean_s_lat"], "mean_s_group": res[k]["info"]["mean_s_grp"],
                                  **{f"{arm}/{v}": A(k, arm, v)[:, :, iq].mean(1).tolist() for arm in ARMS
                                     for v in ("con_pow", "lat_pow", "con_fdr", "lat_fdr")}}
(rd / "toy_report.json").write_text(json.dumps({"T1": T1, "proposition": prop, "t1_cells": t1}, indent=1, default=float))

# ---- figure ------------------------------------------------------------------
ia = AMPS.index(3.0)
fig, ax = plt.subplots(2, 4, figsize=(19, 8.4))
for r, design in enumerate(("dilution", "redundancy")):
    eff = [prop[f"{design}/m{m}"]["per_child_effect"] for m in MS]
    ax[r, 0].plot(MS, eff, "o-", color="k", label="measured")
    ax[r, 0].plot(MS, eff[0] / np.sqrt(MS), "--", color="grey", label="eff(1)/√m")
    ax[r, 0].plot(MS, eff[0] / np.array(MS, float), ":", color="grey", label="eff(1)/m")
    ax[r, 0].set(xscale="log", yscale="log", title=f"{design}: per-child standardised effect", xlabel="children per concept m")
    for c, (v, title) in enumerate((("con_pow", "concept power"), ("lat_pow", "latent power"), ("con_fdr", "concept FDR"))):
        a = ax[r, c + 1]
        for arm in ARMS:
            a.plot(MS, [prop[f"{design}/m{m}"][f"{arm}/{v}"][ia] for m in MS], "o-", color=COL[arm], label=LAB[arm])
        if v == "con_fdr":
            for arm in ARMS:
                a.plot(MS, [prop[f"{design}/m{m}"][f"{arm}/lat_fdr"][ia] for m in MS], ":", color=COL[arm])
            a.axhline(0.1, color="k", ls="--", lw=1); a.set_ylim(0, 0.3); title += " (solid) / latent FDR (dotted)"
        else:
            a.set_ylim(-0.02, 1.02)
        a.set(xscale="log", title=f"{design}: {title}", xlabel="children per concept m")
        a.grid(alpha=0.3)
    ax[r, 0].legend(fontsize=8); ax[r, 0].grid(alpha=0.3)
ax[0, 1].legend(fontsize=8)
fig.suptitle(f"Toy proposition (zero-inflated, n={meta['N']}, p={meta['P']}, K={meta['K']} concepts, amplitude 3, q = 0.1, "
             f"{meta['R']} replicates). Library test T1: {'PASS' if T1['PASS'] else 'FAIL'}", fontsize=12)
fig.tight_layout(); fig.savefig(rd / "fig_c1_toy.png", dpi=140); plt.close(fig)
for design in ("dilution", "redundancy"):
    print(design)
    for m in MS:
        P = prop[f"{design}/m{m}"]
        print(f"  m={m:2d} eff {P['per_child_effect']:.3f}  s lat/grp {P['mean_s_latent']:.3f}/{P['mean_s_group']:.3f}  "
              + "  ".join(f"{arm}: con {P[arm + '/con_pow'][ia]:.2f} lat {P[arm + '/lat_pow'][ia]:.2f}" for arm in ARMS))
