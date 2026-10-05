"""Headline summary figure for concept_findings.md: fig_c0_headline.png."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _paths  # noqa: F401
from cseeds import ROOT, load_concept_config

cc = load_concept_config()["concept"]; rd = ROOT / "concept/results"
P = json.loads((rd / "planted_concept.json").read_text())["L12"]
C = json.loads((rd / "census_L12.json").read_text())
R = json.loads((rd / "real_sweep.json").read_text())["L12"]
tau = json.loads((rd / "tau.json").read_text())["tau"]
keys = P["keys"]; wid = np.array([cc["saes"][k]["width"] for k in keys])
COL = {"latent": "#c0392b", "group": "#2e86c1", "group_sum": "#f39c12", "mkf_c1": "#27ae60"}
LAB = {"latent": "per-latent Knockoff+", "group": "group Knockoff+ (lasso W)", "group_sum": "group-sum statistic", "mkf_c1": "MKF+ (c=1)"}
iq = 1
fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))
T = [C["per_width"][k]["tau"][str(tau)] for k in keys]
ax[0].plot(wid, [t["size_median"] for t in T], "o-", color="k", label="median")
ax[0].fill_between(wid, [t["size_q25"] for t in T], [t["size_q75"] for t in T], color="grey", alpha=0.25, label="IQR")
ax2 = ax[0].twinx()
ax2.plot(wid, [t["frac_dilution"] or np.nan for t in T], "s--", color="#8e44ad", label="dilution share")
ax2.set_ylim(0, 1.05); ax2.set_ylabel("share of multi-child families that dilute", color="#8e44ad")
ax[0].set(xscale="log", xlabel="SAE width (layer 12)", ylabel=f"children per 16k parent (τ = {tau})",
          title="1. Concepts split as the SAE widens")
ax[0].legend(loc="upper left", fontsize=8)
for i, (v, ttl) in enumerate((("lat_pow", "2. Share of a concept's latents certified"), ("con_pow", "3. Concept found via ≥ 1 latent"))):
    a = ax[i + 1]
    for arm in ("latent", "group", "group_sum", "mkf_c1"):
        M = np.array(P["means"][arm][v])[:, 0]                        # design A, (W, K, A, F, Q)
        a.plot(wid, M[..., iq].reshape(len(keys), -1).mean(1), "o-" if arm != "mkf_c1" else "s:", color=COL[arm],
               label=LAB[arm], lw=2.2 if arm != "mkf_c1" else 1.4)
    a.set(xscale="log", ylim=(0, 1.02), xlabel="SAE width (layer 12)", ylabel="power (design A, exact truth; q = 0.1)", title=ttl)
    a.grid(alpha=0.3)
ax[1].legend(fontsize=8, loc="lower left")
for arm in ("latent", "group", "group_sum"):
    ax[3].plot(wid, np.array(R["jaccard"][f"{arm}/shared"])[0], "o-", color=COL[arm], label=LAB[arm])
ax[3].set(xscale="log", ylim=(0, 1.02), xlabel="SAE width (layer 12)", ylabel="Jaccard of stable parent set vs 16k",
          title="4. Real SST-2: which concepts are found? (q = 0.1)")
ax[3].grid(alpha=0.3); ax[3].legend(fontsize=8)
fig.suptitle("Per-latent discovery keeps the concept but loses its extent; group methods keep the extent, "
             "but no method keeps the same concepts across widths", fontsize=13)
fig.tight_layout(); fig.savefig(rd / "fig_c0_headline.png", dpi=140); plt.close(fig)
print("wrote fig_c0_headline.png")
