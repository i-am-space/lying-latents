"""Pipeline figure: one intervention on one input, then the interventional FDR procedure.

Companion to generate_overview.py, in the same style. Top row: how one latent is edited inside the
model and what is measured. Bottom row: the five steps from all latents to a selected list.
Schematic: all values are synthetic.

    python scripts/generate_pipeline.py   # writes results/figures/fig_pipeline.pdf and .png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": "DejaVu Sans", "font.size": 7,
                     "pdf.fonttype": 42})

INK, MUTED, LINE, FRAME = "#1F2937", "#6B7280", "#94A3B8", "#CBD5E1"
ACCENT, RED, GREEN, GREEN_BG = "#7C3AED", "#B91C1C", "#047857", "#ECFDF5"
SAE_FILL, SAE_EDGE = "#EEF2FF", "#818CF8"
SLATE = LinearSegmentedColormap.from_list("slate", ["#E2E8F0", "#64748B"])
BLUE = LinearSegmentedColormap.from_list("blue", ["#BFDBFE", "#1E3A8A"])
DIVERGE = LinearSegmentedColormap.from_list("div", ["#FECACA", "#FFFFFF", "#1E3A8A"])

rng = np.random.default_rng(11)
fig = plt.figure(figsize=(7.0, 3.75), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(2.6, 56.4)
ax.axis("off")


def arrow(x0, y0, x1, y1, color=INK, lw=0.8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6.5, color=color,
                                 lw=lw, shrinkA=0, shrinkB=0))


def label(x, y, text, color=MUTED, size=6.0, **kw):
    ax.text(x, y, text, ha=kw.pop("ha", "center"), va=kw.pop("va", "top"), color=color, fontsize=size,
            linespacing=kw.pop("linespacing", 1.2), **kw)


def rows(M, x, y_top, w, h, gap, cmap, vmax, zero_white=True):
    for r in range(M.shape[0]):
        y = y_top - r * (h + gap) - h
        cw = w / M.shape[1]
        for c in range(M.shape[1]):
            if not (zero_white and M[r, c] == 0):
                ax.add_patch(Rectangle((x + c * cw, y), cw, h, lw=0, facecolor=cmap(min(M[r, c] / vmax, 1))))
        ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=FRAME, lw=0.35))


def layers(x0, x1, ym, text, sub):
    ax.add_patch(FancyBboxPatch((x0, ym - 6.4), x1 - x0, 12.8, boxstyle="round,pad=0,rounding_size=0.7",
                                facecolor="#F8FAFC", edgecolor=LINE, lw=0.6))
    ax.text((x0 + x1) / 2, ym + 0.8, text, ha="center", va="center", fontsize=6.3, color=INK, fontweight="bold",
            linespacing=1.15)
    ax.text((x0 + x1) / 2, ym - 3.6, sub, ha="center", va="center", fontsize=5.3, color=MUTED, linespacing=1.15)


# ===========================================================================
# Top row: one input
# ===========================================================================
tokens = ["Review:", "quiet", "delight", "Sentiment:"]
T, row_h, gap, y_top = 4, 2.4, 1.0, 50.6
y_bot = y_top - T * row_h - (T - 1) * gap
ym = (y_top + y_bot) / 2
ax.text(0.8, 55.2, "one intervention on one input", ha="left", va="center", fontsize=7, color=INK, fontweight="bold")

for i, t in enumerate(tokens):
    y = y_top - i * (row_h + gap) - row_h
    ax.add_patch(FancyBboxPatch((0.8, y), 10.2, row_h, boxstyle="round,pad=0,rounding_size=0.6", facecolor="white",
                                edgecolor=LINE, lw=0.5))
    ax.text(5.9, y + row_h / 2, t, ha="center", va="center", fontsize=6.0, color=INK, family="DejaVu Sans Mono")
label(5.9, y_bot - 1.4, "query tokens\n(only these are edited)")
layers(13.4, 21.4, ym, "layers\n$\\leq L$", "cached")
arrow(11.3, ym, 13.1, ym)
arrow(21.7, ym, 23.5, ym)

x_h, w_h = 23.9, 10.5
res = np.abs(rng.normal(0, 1, (T, 16)))
rows(res, x_h, y_top, w_h, row_h, gap, SLATE, 2.2, zero_white=False)
label(x_h + w_h / 2, y_bot - 1.4, "residual stream $h$")
arrow(x_h + w_h + 0.4, ym, 36.5, ym)

x0, x1 = 36.9, 41.6
ax.add_patch(Polygon([[x0, ym - 3.5], [x0, ym + 3.5], [x1, ym + 6.4], [x1, ym - 6.4]], closed=True,
                     facecolor=SAE_FILL, edgecolor=SAE_EDGE, lw=0.7))
ax.text((x0 + x1) / 2, ym, "SAE", ha="center", va="center", fontsize=6.4, fontweight="bold", color="#4338CA")
arrow(x1 + 0.3, ym, 43.3, ym)

x_l, w_l, P, J = 43.7, 15.0, 30, 21
codes = np.zeros((T, P))
for i in range(T):
    idx = rng.choice(P, size=3, replace=False)
    codes[i, idx] = rng.gamma(2.2, 1.0, size=3)
codes[:, J] = [0, 2.8, 3.6, 0]
rows(codes, x_l, y_top, w_l, row_h, gap, BLUE, 4.0)
cw = w_l / P
ax.add_patch(Rectangle((x_l + J * cw - 0.15, y_bot - 0.15), cw + 0.3, y_top - y_bot + 0.3, facecolor="none",
                       edgecolor=ACCENT, lw=0.8, zorder=5))
ax.text(x_l + J * cw + cw / 2, y_top + 0.4, "latent $j$", ha="center", va="bottom", fontsize=5.8, color=ACCENT)
label(x_l + w_l / 2, y_bot - 1.4, "resample $a_j \\rightarrow \\tilde a_j$\nkeep every other latent")

x_e, w_e = 62.3, 10.5
arrow(x_l + w_l + 0.4, ym, x_e - 0.4, ym, color=ACCENT)
edited = res.copy()
edited[1:3] = np.clip(edited[1:3] + rng.normal(0, 0.35, (2, 16)), 0, None)
rows(edited, x_e, y_top, w_e, row_h, gap, SLATE, 2.2, zero_white=False)
for i in (1, 2):
    y = y_top - i * (row_h + gap) - row_h
    ax.add_patch(Rectangle((x_e - 0.15, y - 0.15), w_e + 0.3, row_h + 0.3, facecolor="none", edgecolor=ACCENT, lw=0.8))
label(x_e + w_e / 2, y_bot - 1.4, "$h + (\\tilde a_j - a_j)\\,d_j$", color=ACCENT)
arrow(x_e + w_e + 0.4, ym, 74.8, ym)
layers(75.2, 82.0, ym, "layers\n$> L$", "rerun")
arrow(82.3, ym, 84.0, ym)

bx, bw = 89.8, 8.6
for y, v, name, col in [(ym + 2.4, 0.92, "$m(x)$", "#64748B"), (ym - 2.4, 0.85, "$m(x;j)$", ACCENT)]:
    ax.add_patch(Rectangle((bx, y - 1.1), bw, 2.2, facecolor="white", edgecolor=FRAME, lw=0.4))
    ax.add_patch(Rectangle((bx, y - 1.1), bw * v, 2.2, facecolor=col, alpha=0.75, lw=0))
    ax.text(bx - 0.5, y, name, ha="right", va="center", fontsize=5.6, color=col)
    ax.text(bx + bw * v - 0.4, y, f"{v:.2f}", ha="right", va="center", fontsize=5.3, color="white")
label(92.2, y_bot - 1.4, "effect\n$\\Delta_j(x) = 0.07$", color=INK)

# connector
y_t = 32.0
ax.plot([92.2, 92.2], [y_bot - 5.4, y_t], color=INK, lw=0.8)
ax.plot([92.2, 0.3], [y_t, y_t], color=INK, lw=0.8)
ax.plot([0.3, 0.3], [y_t, 17.3], color=INK, lw=0.8)
arrow(0.3, 17.3, 1.1, 17.3)
ax.text(1.6, y_t - 0.6, "from all latents to a list with FDR $\\leq \\alpha$", ha="left", va="top", fontsize=7,
        color=INK, fontweight="bold")

# ===========================================================================
# Bottom row: five steps
# ===========================================================================
PW, PH, PY = 16.6, 14.8, 8.2
xs = [1.2 + i * 20.05 for i in range(5)]
titles = ["①  screen", "②  intervene", "③  bet", "④  e-BH", "⑤  verify"]
subs = ["split A: rank all 16k\nby attribution, keep $K$ = 50", "split B: $\\Delta_j(x)$ for each\ninput and candidate",
        "wealth grows only if\n$\\tau_j > \\delta$: the e-value", "keep the largest $e$ above\n$K/(\\alpha k)$: FDR $\\leq \\alpha$",
        "split O: measure $\\tau_j$,\n200 re-splits, stress test"]
for x, t, s in zip(xs, titles, subs):
    ax.text(x + PW / 2, 27.9, t, ha="center", va="center", fontsize=7, color=INK, fontweight="bold")
    label(x + PW / 2, PY - 1.1, s, size=5.7)
    ax.add_patch(Rectangle((x, PY), PW, PH + 3.4, facecolor="white", edgecolor=FRAME, lw=0.5))
for x in xs[:-1]:
    arrow(x + PW + 0.4, PY + PH / 2 + 1.7, x + 20.05 - 0.4, PY + PH / 2 + 1.7)
PH += 3.4

# 1 ranking
x = xs[0]
n = 34
h = np.exp(-np.arange(n) / 8.0)
bwid = (PW - 2) / n
for i, v in enumerate(h):
    ax.add_patch(Rectangle((x + 1 + i * bwid, PY + 1), bwid * 0.8, v * (PH - 3), lw=0,
                           facecolor=BLUE(0.75) if i < 12 else "#CBD5E1"))
ax.plot([x + 1 + 12 * bwid] * 2, [PY + 1, PY + PH - 1.6], color=ACCENT, lw=0.7, ls=(0, (2, 1.5)))
ax.text(x + 1 + 12.6 * bwid, PY + PH - 2.0, "top 50", ha="left", va="center", fontsize=5.5, color=ACCENT)

# 2 effect matrix
x = xs[1]
M = rng.normal(0.3, 0.25, (9, 14))
M[:, 3] += 0.5
mx, mw, my, mh = x + 1.6, PW - 3.2, PY + 1.2, PH - 2.4
cw2, ch = mw / 14, mh / 9
for r in range(9):
    for c in range(14):
        ax.add_patch(Rectangle((mx + c * cw2, my + mh - (r + 1) * ch), cw2, ch, lw=0, facecolor=DIVERGE(np.clip(M[r, c], 0, 1))))
ax.add_patch(Rectangle((mx, my), mw, mh, facecolor="none", edgecolor=FRAME, lw=0.4))
ax.add_patch(Rectangle((mx + 3 * cw2, my), cw2, mh, facecolor="none", edgecolor=ACCENT, lw=0.8))

# 3 wealth
x = xs[2]
t = np.arange(120)
good = 0.05 * t + np.cumsum(rng.normal(0, 0.12, 120))
bad = -0.012 * t + np.cumsum(rng.normal(0, 0.1, 120))
sx = (PW - 2.4) / 119
top = max(good.max(), 1)
low = min(bad.min(), -1.0)
sy = (PH - 2.6) / (top - low)
ty = lambda v: PY + 1.3 + (v - low) * sy
ax.plot([x + 1.2, x + PW - 1.2], [ty(0), ty(0)], color=LINE, lw=0.5, ls=(0, (2, 1.5)))
ax.plot(x + 1.2 + t * sx, ty(good), color=GREEN, lw=0.9)
ax.plot(x + 1.2 + t * sx, ty(bad), color=MUTED, lw=0.9)
ax.text(x + 8.2, ty(good[-1]) - 0.6, "$\\tau_j > \\delta$", ha="right", va="top", fontsize=5.6, color=GREEN)
ax.text(x + PW - 1.3, ty(bad[-1]) + 1.0, "$\\tau_j \\leq \\delta$", ha="right", va="bottom", fontsize=5.6, color=MUTED)

# 4 e-BH
x = xs[3]
e = np.array([0.95, 0.8, 0.66, 0.38, 0.27, 0.2, 0.15, 0.1, 0.07, 0.05])
thr = 0.58 / np.arange(1, 11) ** 0.42
kk = 3
bwid = (PW - 2.6) / 10
for i, v in enumerate(e):
    ax.add_patch(Rectangle((x + 1.3 + i * bwid, PY + 1.2), bwid * 0.78, v * (PH - 3), lw=0,
                           facecolor=GREEN if i < kk else "#CBD5E1"))
ax.plot(x + 1.3 + (np.arange(10) + 0.4) * bwid, PY + 1.2 + thr * (PH - 3), color=RED, lw=0.8)
ax.text(x + PW - 1.2, PY + 1.2 + thr[-1] * (PH - 3) + 0.7, "$K/(\\alpha k)$", ha="right", va="bottom", fontsize=5.4, color=RED)

# 5 oracle
x = xs[4]
tau = np.array([1.9, 1.5, 1.25, 0.75, 0.55, 0.42, 0.33, 0.25, 0.18, 0.1])
dy = 0.45
bwid = (PW - 2.6) / 10
yy = lambda v: PY + 1.2 + v / 2.1 * (PH - 3)
ax.plot([x + 1, x + PW - 1], [yy(1)] * 2, color=ACCENT, lw=0.8, ls=(0, (2, 1.5)))
ax.text(x + PW - 1.1, yy(1) + 0.3, "$\\delta$", ha="right", va="bottom", fontsize=6.0, color=ACCENT)
for i, v in enumerate(tau):
    ax.scatter([x + 1.3 + (i + 0.4) * bwid], [yy(v)], s=7, color=GREEN if i < kk else "#94A3B8", zorder=5, lw=0)

out = Path(__file__).resolve().parents[1] / "results" / "figures"
fig.savefig(out / "fig_pipeline.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig(out / "fig_pipeline.png", bbox_inches="tight", pad_inches=0.03, dpi=220)
print("wrote", out / "fig_pipeline.pdf")
