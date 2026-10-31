"""Pipeline figure: interventional FDR control, and where it differs from Khera (2026).

Two lanes. A step drawn as two boxes differs between the procedures; a step drawn as one box across
both lanes is shared. The effect dots and bars are schematic.

    python scripts/generate_pipeline.py   # writes results/figures/fig_pipeline.pdf and .png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": "DejaVu Sans", "font.size": 7,
                     "pdf.fonttype": 42})

INK, MUTED, LINE, FRAME = "#1F2937", "#6B7280", "#94A3B8", "#CBD5E1"
OURS, OURS_BG = "#047857", "#ECFDF5"
THEM, THEM_BG = "#475569", "#F1F5F9"
SHARED, SHARED_BG = "#4338CA", "#EEF2FF"
KEY = "#7C3AED"

rng = np.random.default_rng(5)
fig = plt.figure(figsize=(7.0, 2.95), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 42)
ax.axis("off")

W, H, GAP = 14.0, 15.2, 1.45
XS = [6.9 + i * (W + GAP) for i in range(6)]
Y_O, Y_K = 20.4, 2.4                      # bottom of each lane


def box(x, y, w, h, fc, ec, lw=0.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.7", facecolor=fc, edgecolor=ec, lw=lw))


def text(x, y, s, color=INK, size=5.7, ha="center", va="center", **kw):
    ax.text(x, y, s, color=color, fontsize=size, ha=ha, va=va, linespacing=kw.pop("ls", 1.22), **kw)


def arrow(x0, y0, x1, y1, color=LINE):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=5.5, color=color, lw=0.7,
                                 shrinkA=0, shrinkB=0))


titles = ["① candidates", "② intervene", "③ measure", "④ null tested", "⑤ bet and select", "⑥ checked against"]
for x, t in zip(XS, titles):
    text(x + W / 2, 37.8, t, size=6.1, fontweight="bold")

text(3.4, Y_O + H / 2, "Ours", color=OURS, size=7, fontweight="bold")
text(3.4, Y_K + H / 2, "Khera\n(2026)", color=THEM, size=6.2, fontweight="bold")

# highlight the main difference
ax.add_patch(FancyBboxPatch((XS[3] - 0.6, Y_K - 0.7), W + 1.2, 2 * H + 3.6 + 1.3, boxstyle="round,pad=0,rounding_size=0.9",
                            facecolor="none", edgecolor=KEY, lw=1.0, ls=(0, (3, 1.6))))
text(XS[3] + W / 2, 40.6, "main difference", color=KEY, size=5.9, fontweight="bold")

for col in (0, 1, 2, 3, 5):
    box(XS[col], Y_O, W, H, OURS_BG, OURS)
    box(XS[col], Y_K, W, H, THEM_BG, THEM)
for col in range(5):
    for y in (Y_O, Y_K):
        arrow(XS[col] + W + 0.1, y + H / 2, XS[col + 1] - 0.1, y + H / 2)

ym_o, ym_k = Y_O + H / 2, Y_K + H / 2
# 1
text(XS[0] + W / 2, ym_o, "attribution on a\nseparate split\n\ntop 50 of 16k")
text(XS[0] + W / 2, ym_k, "active on $\\geq$20%\nof prompts\n\n89 of 24k")
# 2
for y, new, note, col in [(ym_o, "1.1", "resampled\nfrom a fitted model", OURS), (ym_k, "0", "zero\nablation", THEM)]:
    text(XS[1] + W / 2, y + 3.3, f"$a_j$:  2.8 $\\rightarrow$ {new}", color=col, size=6.8, fontweight="bold")
    text(XS[1] + W / 2, y - 2.6, note)
# 3
text(XS[2] + W / 2, ym_o, "drop in correct-\nclass probability\n\n$\\Delta_j(x)$")
text(XS[2] + W / 2, ym_k, "drop in logit gap,\ncorrect vs. wrong\n\n$\\Delta_j(x)$")

# 4: same latents, two bars
tau = np.sort(np.concatenate([rng.uniform(0.02, 0.2, 13), [0.36, 0.72, 0.86]]))[::-1]
for y, bar, name, col, formula in [(ym_o, 0.60, "$\\delta$: fixed", OURS, "$H_j$: $\\tau_j \\leq \\delta$   absolute"),
                                   (ym_k, 0.13, "mean of tested set", THEM, "$H_j$: $\\tau_j \\leq$ random candidate\nrelative to the set")]:
    sx, sw, base, sh = XS[3] + 1.1, W - 2.2, y - 1.5, 7.2
    bw = sw / len(tau)
    for i, v in enumerate(tau):
        ax.add_patch(Rectangle((sx + i * bw, base), bw * 0.72, v * sh, lw=0, facecolor=col if v > bar else "#CBD5E1"))
    ax.plot([sx - 0.2, sx + sw], [base + bar * sh] * 2, color=KEY, lw=0.8)
    text(sx + sw, base + bar * sh + 0.25 + (1.3 if bar < 0.3 else 0), name, color=KEY, size=4.9, ha="right", va="bottom")
    text(XS[3] + W / 2, y - 4.5, formula, size=4.9 if "\n" in formula else 5.4, color=col)

# 5 shared
ys, hs = Y_K, Y_O + H - Y_K
box(XS[4], ys, W, hs, SHARED_BG, SHARED)
text(XS[4] + W / 2, ys + hs - 2.6, "the same in both", color=SHARED, size=5.8, fontweight="bold")
t = np.arange(90)
good = 0.05 * t + np.cumsum(rng.normal(0, 0.1, 90))
bad = -0.012 * t + np.cumsum(rng.normal(0, 0.09, 90))
px, pw, py, ph = XS[4] + 1.4, W - 2.8, ys + hs / 2 + 1.4, 9.5
lo, hi = min(bad.min(), -0.6), good.max()
f = lambda v: py + (v - lo) / (hi - lo) * ph
ax.plot(px + t / 89 * pw, f(good), color=SHARED, lw=0.9)
ax.plot(px + t / 89 * pw, f(bad), color=LINE, lw=0.9)
ax.plot([px, px + pw], [f(0), f(0)], color=LINE, lw=0.4, ls=(0, (2, 1.5)))
text(XS[4] + W / 2, ys + 9.0, "betting e-value\nper latent (aGRAPA)\n\ne-BH: FDR $\\leq \\alpha$\nunder any dependence", size=5.5)

# 6
text(XS[5] + W / 2, ym_o, "measured effects\non 2 real models,\n4 tasks; stress test")
text(XS[5] + W / 2, ym_k, "synthetic model;\nno ground truth\non GPT-2")

out = Path(__file__).resolve().parents[1] / "results" / "figures"
fig.savefig(out / "fig_pipeline.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig(out / "fig_pipeline.png", bbox_inches="tight", pad_inches=0.03, dpi=220)
print("wrote", out / "fig_pipeline.pdf")
