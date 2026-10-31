"""Architecture of interventional FDR control.

Top: one input through the model, with one latent edited between the SAE encoder and decoder.
Bottom: how the data splits feed the procedure. Schematic: no values are results.

    python scripts/generate_pipeline.py   # writes results/figures/fig_pipeline.pdf and .png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": "DejaVu Sans", "font.size": 6,
                     "pdf.fonttype": 42})
INK, MUTED, LINE, FRAME = "#1F2937", "#6B7280", "#94A3B8", "#CBD5E1"
ACCENT, GREEN, GREEN_BG = "#7C3AED", "#047857", "#ECFDF5"
SAE_FILL, SAE_EDGE = "#EEF2FF", "#818CF8"
SLATE = LinearSegmentedColormap.from_list("slate", ["#E2E8F0", "#64748B"])
BLUE = LinearSegmentedColormap.from_list("blue", ["#BFDBFE", "#1E3A8A"])

rng = np.random.default_rng(4)
fig = plt.figure(figsize=(7.0, 3.4), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(1.6, 50.3)
ax.axis("off")


def arrow(x0, y0, x1, y1, color=INK, lw=0.7, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=5.5, color=color, lw=lw, ls=ls,
                                 shrinkA=0, shrinkB=0))


def text(x, y, s, color=MUTED, size=5.5, ha="center", va="center", **kw):
    ax.text(x, y, s, color=color, fontsize=size, ha=ha, va=va, linespacing=1.2, **kw)


def column(x, y0, h, w, v, cmap, vmax=2.2, zero_white=False):
    n = len(v)
    ch = h / n
    for i, a in enumerate(v):
        fc = "white" if (zero_white and a == 0) else cmap(min(a / vmax, 1))
        ax.add_patch(Rectangle((x, y0 + i * ch), w, ch, facecolor=fc, edgecolor=FRAME, lw=0.25))


def slabs(x, y0, h, n, label):
    for i in range(n):
        xi = x + i * 1.35
        ax.add_patch(Polygon([[xi, y0], [xi + 0.95, y0 + 1.5], [xi + 0.95, y0 + h + 1.5], [xi, y0 + h]], closed=True,
                             facecolor="#F1F5F9", edgecolor=LINE, lw=0.5))
    text(x + (n * 1.35) / 2 + 0.2, y0 - 1.6, label)


# ---------------------------------------------------------------------------
# top: the model
# ---------------------------------------------------------------------------
ym, y0, H = 37.0, 31.0, 12.0
for i, t in enumerate(["Review:", "quiet", "delight", "Sentiment:"]):
    y = 41.6 - i * 3.0
    ax.add_patch(FancyBboxPatch((0.8, y), 9.6, 2.3, boxstyle="round,pad=0,rounding_size=0.5", facecolor="white",
                                edgecolor=LINE, lw=0.45))
    text(5.6, y + 1.15, t, color=INK, size=5.4, family="DejaVu Sans Mono")
text(5.6, y0 - 1.6, "one input $x$")
arrow(10.7, ym, 12.2, ym)
slabs(12.5, y0, H, 4, "layers $\\leq L$")
arrow(18.3, ym, 19.8, ym)

v = np.abs(rng.normal(1, 0.6, 16))
column(20.1, y0, H, 2.0, v, SLATE)
text(21.1, y0 - 1.6, "$h$", color=INK, size=6.2)
arrow(22.4, ym, 23.7, ym)

ax.add_patch(Polygon([[24, ym - 2.6], [24, ym + 2.6], [29, ym + 6.6], [29, ym - 6.6]], closed=True, facecolor=SAE_FILL,
                     edgecolor=SAE_EDGE, lw=0.6))
text(26.5, ym, "SAE\nencoder", color="#4338CA", size=4.9, fontweight="bold")
arrow(29.2, ym, 30.5, ym)

n_l, J = 26, 16
lat = np.zeros(n_l)
lat[[3, 9, 21]] = [1.4, 2.0, 1.1]
lat[J] = 2.3
col_x, col_w = 30.9, 2.3
column(col_x, y0 - 1, H + 2, col_w, lat, BLUE, vmax=2.6, zero_white=True)
ch = (H + 2) / n_l
ax.add_patch(Rectangle((col_x - 0.15, y0 - 1 + J * ch - 0.1), col_w + 0.3, ch + 0.2, facecolor="none", edgecolor=ACCENT, lw=0.9, zorder=5))
text(col_x + col_w / 2, y0 - 2.6, "16k latents", size=5.2)
yj = y0 - 1 + J * ch + ch / 2

# the edit, as an inset
ax.add_patch(FancyBboxPatch((24.6, 44.4), 19.4, 4.2, boxstyle="round,pad=0,rounding_size=0.6", facecolor="#F5F3FF", edgecolor=ACCENT, lw=0.6))
ax.plot([col_x + col_w / 2, col_x + col_w / 2], [yj + ch / 2 + 0.1, 44.4], color=ACCENT, lw=0.5, ls=(0, (2, 1.5)))
text(34.3, 47.3, "edit latent $j$ only", color=ACCENT, size=5.2, fontweight="bold")
text(34.3, 45.6, "$a_j \\rightarrow \\tilde a_j$, resampled from a fitted model", color=ACCENT, size=4.8)

arrow(33.5, ym, 34.8, ym)
ax.add_patch(Polygon([[35.1, ym - 6.6], [35.1, ym + 6.6], [40.1, ym + 2.6], [40.1, ym - 2.6]], closed=True, facecolor=SAE_FILL,
                     edgecolor=SAE_EDGE, lw=0.6))
text(37.6, ym, "SAE\ndecoder", color="#4338CA", size=4.9, fontweight="bold")
arrow(40.3, ym, 41.6, ym)

v2 = v.copy()
v2[[5, 6, 7]] += 0.7
column(41.9, y0, H, 2.0, v2, SLATE)
ax.add_patch(Rectangle((41.75, y0 + 5 * H / 16), 2.3, 3 * H / 16, facecolor="none", edgecolor=ACCENT, lw=0.7))
text(42.9, y0 - 1.6, "edited $h$", color=ACCENT, size=5.4)

# error term: skip connection
ax.plot([21.1, 21.1], [y0 - 3.2, y0 - 4.6], color=LINE, lw=0.6)
ax.plot([21.1, 42.9], [y0 - 4.6, y0 - 4.6], color=LINE, lw=0.6)
arrow(42.9, y0 - 4.6, 42.9, y0 - 2.9, color=LINE, lw=0.6)
text(32.0, y0 - 5.6, "SAE error and all other latents kept", size=4.9)

arrow(44.3, ym, 46.0, ym)
slabs(46.3, y0, H, 4, "layers $> L$")
arrow(52.1, ym, 53.8, ym)

# output
for y, name, w_, col in [(ym + 2.5, "correct", 7.4, GREEN), (ym - 2.5, "wrong", 1.6, LINE)]:
    ax.add_patch(Rectangle((54.1, y - 1.2), 8.6, 2.4, facecolor="white", edgecolor=FRAME, lw=0.4))
    ax.add_patch(Rectangle((54.1, y - 1.2), w_, 2.4, facecolor=col, lw=0, alpha=0.8))
    text(58.4, y + 2.0, name, size=4.9)
text(58.4, y0 - 1.6, "class probabilities")

# unedited run and difference
arrow(62.9, ym, 67.0, ym)
text(64.8, ym + 1.1, "$m(x;j)$", color=ACCENT, size=5.2)
ax.add_patch(Circle((68.0, ym), 1.0, facecolor="white", edgecolor=INK, lw=0.6))
text(68.0, ym, "−", color=INK, size=7)
ax.plot([21.1, 21.1], [y0 + H + 0.2, 49.5], color=LINE, lw=0.6, ls=(0, (2, 1.5)))
ax.plot([21.1, 68.0], [49.5, 49.5], color=LINE, lw=0.6, ls=(0, (2, 1.5)))
arrow(68.0, 49.5, 68.0, ym + 1.1, color=LINE, lw=0.6, ls=(0, (2, 1.5)))
text(69.0, 47.8, "same input, no edit: $m(x)$", size=4.9, ha="left")
arrow(69.1, ym, 72.6, ym)
ax.add_patch(FancyBboxPatch((72.9, ym - 3.6), 25.6, 7.2, boxstyle="round,pad=0,rounding_size=0.7", facecolor=GREEN_BG, edgecolor=GREEN, lw=0.6))
text(85.7, ym + 1.4, "effect of latent $j$ on input $x$", color=GREEN, size=5.2)
text(85.7, ym - 1.3, "$\\Delta_j(x) = m(x) - m(x;j)$", color=INK, size=6.2)

# ---------------------------------------------------------------------------
# bottom: data splits feed the procedure
# ---------------------------------------------------------------------------
y_h, h_h = 19.4, 2.5
segs = [(0.8, 15.0, "pilot"), (17.0, 22.0, "screening split A"), (40.2, 44.8, "test split B"), (86.2, 12.8, "oracle split O")]
for x, w, name in segs:
    ax.add_patch(Rectangle((x, y_h), w, h_h, facecolor="#E2E8F0", edgecolor=LINE, lw=0.4))
    text(x + w / 2, y_h + h_h / 2, name, color=INK, size=5.3, fontweight="bold")
y_t, y_b = 18.4, 6.2

# pilot: delta
ax.plot([2.5, 14.2], [12.5, 12.5], color=LINE, lw=0.6)
ax.plot([2.5, 2.5], [11.9, 13.1], color=LINE, lw=0.6)
ax.plot([6.0, 6.0], [11.2, 13.8], color=ACCENT, lw=1.0)
text(2.5, 10.7, "0", size=4.9)
text(6.0, 15.0, "$\\delta$", color=ACCENT, size=6.6)
text(8.3, 3.3, "fix the smallest\neffect that counts")

# A: ranking
n = 34
h_ = np.exp(-np.arange(n) / 7.0) * 11.0
for i, v_ in enumerate(h_):
    ax.add_patch(Rectangle((18.2 + i * 0.57, y_b), 0.42, v_, lw=0, facecolor=BLUE(0.7) if i < 11 else FRAME))
ax.plot([24.6, 24.6], [y_b, 17.9], color=ACCENT, lw=0.6, ls=(0, (2, 1.5)))
text(25.1, 16.6, "top $K$", color=ACCENT, size=5.0, ha="left")
text(28.0, 3.3, "attribution ranks all 16k,\nkeep the top 50")

# B: effect matrix -> bet -> e-BH
mx, my, mw, mh, cols, rows_ = 41.4, y_b, 9.5, 11.0, 12, 9
M = np.abs(rng.normal(0.4, 0.3, (rows_, cols)))
for r in range(rows_):
    for c in range(cols):
        ax.add_patch(Rectangle((mx + c * mw / cols, my + r * mh / rows_), mw / cols, mh / rows_, lw=0, facecolor=SLATE(min(M[r, c], 1))))
ax.add_patch(Rectangle((mx, my), mw, mh, facecolor="none", edgecolor=FRAME, lw=0.4))
ax.add_patch(Rectangle((mx + 4 * mw / cols, my), mw / cols, mh, facecolor="none", edgecolor=ACCENT, lw=0.8))
text(46.1, 3.3, "$\\Delta_j(x)$ for every\ninput and candidate")

arrow(51.4, 11.7, 53.8, 11.7)
px, pw = 54.2, 12.5
ax.plot([px, px], [y_b, y_b + 11], color=LINE, lw=0.5)
ax.plot([px, px + pw], [y_b + 2.4, y_b + 2.4], color=LINE, lw=0.4, ls=(0, (2, 1.5)))
t = np.linspace(0, 1, 40)
ax.plot(px + t * pw, y_b + 2.4 + 8.2 * t ** 1.6, color=GREEN, lw=1.0)
ax.plot(px + t * pw, y_b + 2.4 - 1.6 * t, color=LINE, lw=1.0)
text(px + 0.6, y_b + 11.0, "wealth", size=4.8, ha="left", va="top")
text(px + pw / 2 + 0.3, 3.3, "bet each effect beats $\\delta$:\nfinal wealth = e-value")

arrow(67.1, 11.7, 69.5, 11.7)
vals = 10.5 * np.exp(-np.arange(9) / 2.6)
for i, v_ in enumerate(vals):
    ax.add_patch(Rectangle((69.9 + i * 1.25, y_b), 0.95, v_, lw=0, facecolor=GREEN if i < 3 else FRAME))
line_x = 69.9 + np.arange(9) * 1.25 + 0.5
ax.plot(line_x, y_b + 6.6 / (1 + np.arange(9) * 0.32), color="#B91C1C", lw=0.8)
text(75.5, 3.3, "e-BH keeps a list\nwith FDR $\\leq \\alpha$")

# O
arrow(82.0, 11.7, 86.2, 11.7)
for i, name in enumerate(["$j_1$", "$j_2$", "$j_3$"]):
    y = 14.2 - i * 3.5
    ax.add_patch(FancyBboxPatch((87.8, y), 3.2, 2.4, boxstyle="round,pad=0,rounding_size=0.4", facecolor=GREEN_BG, edgecolor=GREEN, lw=0.5))
    text(89.4, y + 1.2, name, color=GREEN, size=5.2)
    arrow(91.3, y + 1.2, 92.6, y + 1.2, color=LINE, lw=0.5)
    text(93.0, y + 1.2, "$\\tau_j > \\delta$ ?", color=INK, size=5.3, ha="left")
text(92.6, 3.3, "check realised FDR\non held-out inputs")

# connector from the effect into the matrix
ax.plot([85.7, 85.7], [ym - 3.8, 26.2], color=INK, lw=0.7)
ax.plot([85.7, 46.1], [26.2, 26.2], color=INK, lw=0.7)
arrow(46.1, 26.2, 46.1, y_h + h_h + 0.1)
text(66.0, 27.3, "repeat for every input in B and every candidate latent", color=INK, size=5.2)

out = Path(__file__).resolve().parents[1] / "results" / "figures"
fig.savefig(out / "fig_pipeline.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig(out / "fig_pipeline.png", bbox_inches="tight", pad_inches=0.03, dpi=220)
print("wrote", out / "fig_pipeline.pdf")
