"""Figure: how interventional e-BH works, and where a relative null differs.

(a) schematic: what each intervention does to one latent on one input.
(b, c, d) real data, Gemma-3 Bias in Bios, resampling effects: wealth of four latents under each null on one
random ordering of 4,000 inputs, and the latents each procedure selects (alpha = 0.05, K = 50).

    python scripts/generate_pipeline.py   # writes results/figures/fig_pipeline.pdf and .png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": "DejaVu Sans", "font.size": 6.2,
                     "pdf.fonttype": 42, "axes.linewidth": 0.5, "axes.edgecolor": "#94A3B8",
                     "xtick.major.width": 0.4, "ytick.major.width": 0.4, "xtick.major.size": 2,
                     "ytick.major.size": 2, "xtick.color": "#6B7280", "ytick.color": "#6B7280"})
INK, MUTED, FRAME = "#1F2937", "#6B7280", "#CBD5E1"
OURS, THEM, KEY = "#047857", "#B45309", "#7C3AED"
ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/models/gemma3_1b/stage5/bias_in_bios"
K, ALPHA, N = 50, 0.05, 4000
PRIOR = 0.25


def wealth(X, delta, cap):
    """Log10 wealth path of the bet against E[x] <= delta (aGRAPA, as betting_log_evalues in stage5_interventional)."""
    n, k = X.shape
    logw, sx, sxx = np.zeros(k), np.zeros(k), np.zeros(k)
    path = np.zeros((n, k))
    lam_max = cap / (1 + delta)
    for t in range(n):
        mu = (delta + sx) / (t + 1)
        var = (PRIOR + sxx - (sx * sx / t if t else 0.0)) / (t + 1)
        d = mu - delta
        lam = np.clip(d / (np.maximum(var, 1e-12) + d * d), 0, lam_max)
        logw += np.log1p(lam * (X[t] - delta))
        sx += X[t]
        sxx += X[t] ** 2
        path[t] = logw
    return path / np.log(10)


def ebh(log10e):
    o = np.argsort(-log10e)
    ok = log10e[o] >= np.log10(K / (ALPHA * np.arange(1, K + 1)))
    k = np.nonzero(ok)[0].max() + 1 if ok.any() else 0
    return set(o[:k].tolist())


z = np.load(D / "delta.npz")
delta = json.loads((D / "pilot.json").read_text())["delta"]
P = np.vstack([z["B__resample"], z["O__resample"]])[:, :K].astype(float)
tau = P.mean(0) / delta
rng = np.random.default_rng(3)
X = P[rng.permutation(len(P))[:N]]
r = rng.integers(0, K - 1, size=X.shape)
r = r + (r >= np.arange(K))
D_j = (X - np.take_along_axis(X, r, axis=1)) / 2
ours, khera = wealth(X, delta, 0.9), wealth(D_j, 0.0, 0.5)
sel_o, sel_k = ebh(ours[-1]), ebh(khera[-1])
print("selected ours", sorted(np.round(tau[list(sel_o)], 2)), "khera", sorted(np.round(tau[list(sel_k)], 2)))
order = np.argsort(-tau)
comp = tau.mean()

fig = plt.figure(figsize=(7.0, 2.35), dpi=300)


def title(ax, s):
    ax.set_title(s, fontsize=6.6, color=INK, fontweight="bold", loc="left", pad=4)


# (a) what an intervention does
ax = fig.add_axes([0.0, 0.13, 0.275, 0.74])
ax.set_xlim(0, 30)
ax.set_ylim(-0.4, 12.2)
ax.axis("off")
title(ax, "(a) editing one latent on one input")
tokens = ["the", "film", "is", "a", "quiet", "delight"]
rows = [("original", [0, 0, 0, 0, 2.7, 3.9], "#1E3A8A", 0.92), ("resample", [0, 0, 0, 0.7, 1.2, 2.2], OURS, 0.86),
        ("zero", [0, 0, 0, 0, 0, 0], THEM, 0.74)]
for i, (name, a, col, m) in enumerate(rows):
    y0 = 8.3 - i * 3.9
    ax.text(0, y0 + 1.0, name, color=col, fontsize=6.0, fontweight="bold", va="center")
    for t, v in enumerate(a):
        x = 6.6 + t * 2.45
        ax.bar(x, v * 0.72, width=1.7, bottom=y0, color=col, lw=0)
    ax.plot([5.5, 20.4], [y0, y0], color=FRAME, lw=0.5)
    ax.add_patch(FancyArrowPatch((20.7, y0 + 1), (22.3, y0 + 1), arrowstyle="-|>", mutation_scale=5, color=MUTED, lw=0.6))
    ax.barh(y0 + 1, 6.6, height=1.5, left=22.8, color="white", edgecolor=FRAME, lw=0.4)
    ax.barh(y0 + 1, 6.6 * m, height=1.5, left=22.8, color=col, alpha=0.8, lw=0)
    ax.text(22.8 + 6.6 * m - 0.2, y0 + 1, f"{m:.2f}", ha="right", va="center", color="white", fontsize=5.2)
for t, w in enumerate(tokens):
    ax.text(6.6 + t * 2.45, 0.2, w, rotation=38, ha="right", va="top", fontsize=4.9, color=MUTED, family="DejaVu Sans Mono")
ax.text(26.1, 12.0, "$m(x)$", ha="center", va="center", fontsize=5.8, color=MUTED)
ax.text(13, 12.0, "activation of latent $j$", ha="center", va="center", fontsize=5.4, color=MUTED)

# (b, c) wealth
picks = [(order[0], "#1E3A8A"), (order[1], "#2563EB"), (order[3], KEY), (order[8], "#94A3B8")]
xs = np.arange(1, N + 1)
for frame, path, name, col, ttl in [([0.33, 0.2, 0.165, 0.67], ours, "ours", OURS, "(b) bet: effect above $\\delta$"),
                                    ([0.515, 0.2, 0.165, 0.67], khera, "Khera", THEM, "(c) bet: effect above a random candidate")]:
    ax = fig.add_axes(frame)
    for j, c in picks:
        ax.plot(xs[::8], np.maximum(path[::8, j], -3), color=c, lw=0.9)
    ax.axhline(np.log10(K / ALPHA), color="#B91C1C", lw=0.7, ls=(0, (3, 1.5)))
    ax.axhline(0, color=FRAME, lw=0.5)
    ax.set_xlim(0, N)
    ax.set_ylim(-3, max(ours[-1].max(), khera[-1].max()) * 1.08)
    ax.set_xticks([0, 2000, 4000], ["0", "2k", "4k"])
    ax.set_xlabel("inputs", fontsize=5.6, color=MUTED, labelpad=1)
    ax.tick_params(labelsize=5.2, pad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    title(ax, "(b) ours: beat $\\delta$" if name == "ours" else "(c) Khera: beat a random one")
    if name != "ours":
        ax.set_yticklabels([])
    if name == "ours":
        ax.set_ylabel("log$_{10}$ wealth", fontsize=5.6, color=MUTED, labelpad=1)
        ax.text(60, np.log10(K / ALPHA) + 0.3, "$K/\\alpha$", color="#B91C1C", fontsize=5.2, va="bottom")
    if name != "ours":
        for j, c in picks:
            ax.annotate(f"$\\tau$ = {tau[j]:.2f}$\\delta$", (N, path[-1, j]), xytext=(1.5, 0), textcoords="offset points",
                        fontsize=4.8, color=c, va="center", annotation_clip=False)

# (d) who is selected
ax = fig.add_axes([0.79, 0.2, 0.205, 0.67])
n_show = 12
ids = order[:n_show]
ax.bar(range(n_show), tau[ids], width=0.72, color="#CBD5E1", lw=0)
for i, j in enumerate(ids):
    top = tau[j] + 0.09
    if j in sel_o:
        ax.scatter([i - 0.17], [top], s=5, color=OURS, lw=0, zorder=5)
    if j in sel_k:
        ax.scatter([i + 0.17], [top], s=5, color=THEM, lw=0, zorder=5)
ax.axhline(1, color=KEY, lw=0.8)
ax.axhline(comp, color=THEM, lw=0.8, ls=(0, (3, 1.5)))
ax.text(n_show - 0.5, 1.03, "$\\delta$", color=KEY, fontsize=6, ha="right", va="bottom")
ax.text(n_show - 0.5, comp + 0.03, "random candidate", color=THEM, fontsize=5.0, ha="right", va="bottom")
ax.scatter([], [], s=5, color=OURS, label="selected, ours")
ax.scatter([], [], s=5, color=THEM, label="selected, Khera")
ax.legend(frameon=False, fontsize=5.0, loc="upper right", handletextpad=0.1, borderpad=0, labelspacing=0.2)
ax.set_xticks([])
ax.set_xlabel("top latents, by measured effect", fontsize=5.6, color=MUTED, labelpad=2)
ax.set_ylabel("effect $\\tau_j / \\delta$", fontsize=5.6, color=MUTED, labelpad=1)
ax.tick_params(labelsize=5.2, pad=1)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
title(ax, "(d) who is selected")

out = ROOT / "results" / "figures"
fig.savefig(out / "fig_pipeline.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig(out / "fig_pipeline.png", bbox_inches="tight", pad_inches=0.03, dpi=220)
print("wrote", out / "fig_pipeline.pdf")
