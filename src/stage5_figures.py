"""Stage 5 figures and tables (stage5_amendment_1 / _2), from the saved stage outputs; CPU only, no model.

  fig20_stage5_power.png      power vs n (resampling, primary delta, alpha 0.1): prereg vs v2, per model and task
  fig21_stage5_zero_null.png  realised FDR vs n against the minimum-effect truth: ours vs the zero-null baseline
  fig22_stage5_effects.png    the candidates' oracle effects tau_j, sorted, against delta
  table_stage5_baselines.tex  power against the oracle: knockoffs, observational e-values, ours
  table_stage5_validity.tex   breaches, max realised FDR, max stress-test P(any rejection)

Usage: python src/stage5_figures.py   (writes to results/figures/)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures"
MODELS = {"Gemma-2-2B": ROOT / "results" / "stage5", "Gemma-3-1B": ROOT / "results" / "models" / "gemma3_1b" / "stage5"}
TASKS = {"sst2": "SST-2", "ag_news": "AG News", "bias_in_bios": "Bias in Bios", "europarl": "EuroParl"}
A = "0.1"
SPEC = {"prereg": ("K = 250, cap 0.5", "#8c8c8c", "--"), "v2": ("K = 50, cap 0.9", "#1f5fa8", "-")}


def load(base: Path, task: str, name: str):
    p = base / task / name
    if not p.exists():
        return None
    return np.load(p) if p.suffix == ".npz" else json.loads(p.read_text())


def curve(gr: dict, key: str, view: str = "vs_oracle"):
    ns = sorted(gr, key=int)
    return [int(n) for n in ns], [gr[n][A][view][key] for n in ns]


def fig_power():
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True, sharey=True)
    for r, (m, base) in enumerate(MODELS.items()):
        for c, (t, tl) in enumerate(TASKS.items()):
            ax = axes[r, c]
            ev = load(base, t, "evaluate2.json")
            if ev is None:
                ax.set_axis_off()
                continue
            rs = ev["interventions"]["resample"]["resplit"]
            for sp, (lab, col, ls) in SPEC.items():
                n, pw = curve(rs[sp]["primary"], "power")
                ax.plot(n, pw, ls, color=col, marker="o", ms=4, label=lab)
            ntrue = ev["interventions"]["resample"]["counts"]["tau_above_delta"]
            ax.set_title(f"{m}: {tl}  ({ntrue} with $\\tau>\\delta$)", fontsize=11)
            ax.set_xscale("log")
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.3)
            if c == 0:
                ax.set_ylabel("power")
            if r == 1:
                ax.set_xlabel("inputs n")
    axes[0, 0].legend(fontsize=10, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig20_stage5_power.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_zero_null():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    cols = dict(zip(TASKS, ["#1f5fa8", "#d1495b", "#2a9d8f", "#e9a03b"]))
    for ax, (m, base) in zip(axes, MODELS.items()):
        for t, tl in TASKS.items():
            ev = load(base, t, "evaluate2.json")
            if ev is None:
                continue
            rs = ev["interventions"]["resample"]["resplit"]["v2"]
            n, f0 = curve(rs["zero_null"], "fdr")
            ax.plot(n, f0, "-", color=cols[t], marker="o", ms=4, label=f"zero null: {tl}")
            n, f = curve(rs["primary"], "fdr")
            ax.plot(n, f, ":", color=cols[t], marker="s", ms=3, alpha=0.8)
        ax.axhline(0.1, color="k", lw=0.8, ls="--")
        ax.text(105, 0.12, "$\\alpha$ = 0.1", fontsize=8)
        ax.set_xscale("log")
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("inputs n")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("FDR against $\\tau_j > \\delta$")
    axes[0].plot([], [], ":", color="k", marker="s", ms=3, label="ours ($H_j: \\tau_j \\leq \\delta$), all tasks")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig21_stage5_zero_null.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_effects():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    cols = dict(zip(TASKS, ["#1f5fa8", "#d1495b", "#2a9d8f", "#e9a03b"]))
    for ax, (m, base) in zip(axes, MODELS.items()):
        for t, tl in TASKS.items():
            z, pilot = load(base, t, "delta.npz"), load(base, t, "pilot.json")
            if z is None:
                continue
            K = int(z["n_candidates"])
            P = np.vstack([z["B__resample"][:, :K], z["O__resample"][:, :K]])
            tau = np.sort(P.mean(0))[::-1] / pilot["delta"]
            ax.plot(np.arange(1, K + 1), tau, color=cols[t], lw=1.5, label=tl)
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("candidate rank (by oracle effect)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("oracle $\\tau_j$ / $\\delta$")
    axes[0].legend(fontsize=8)
    fig.suptitle("Few latents matter alone, most matter a little: effects of the 250 screened candidates "
                 "(resampling; dashed line = $\\delta$)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig22_stage5_effects.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def obs_power(base: Path, task: str, name: str, q: str = A):
    """Mean over draws of the share of above-delta latents (in U) that baselines2's method selects."""
    b2, z, pilot = load(base, task, "baselines2.json"), load(base, task, "delta.npz"), load(base, task, "pilot.json")
    if b2 is None or "baselines" not in b2:
        return None
    U = z["U"]
    tau = np.vstack([z["B__resample"], z["O__resample"]]).mean(0)
    true = {int(j) for j, t in zip(U, tau) if t > pilot["delta"]}
    return float(np.mean([len(set(d[q]) & true) / max(len(true), 1) for d in b2["baselines"][name]["draws"]]))


def tables():
    rows = []
    for m, base in MODELS.items():
        for t, tl in TASKS.items():
            ev = load(base, t, "evaluate2.json")
            if ev is None:
                continue
            r = ev["interventions"]["resample"]
            ko, eo = obs_power(base, t, "knockoff"), obs_power(base, t, "evalue_obs")
            ours = r["resplit"]["v2"]["primary"]["2000"][A]["vs_oracle"]["power"]
            f = lambda x: "--" if x is None else f"{x:.2f}"
            rows.append(f"{m} & {tl} & {r['counts']['tau_above_delta']} & {f(ko)} & {f(eo)} & {ours:.2f} \\\\")
    (OUT / "table_stage5_baselines.tex").write_text(
        "\\begin{tabular}{llrrrr}\n\\toprule\nModel & Task & $\\#\\{\\tau_j>\\delta\\}$ & Knockoffs & Obs.\\ e-values "
        "& Ours \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    rows = []
    for m, base in MODELS.items():
        for t, tl in TASKS.items():
            ev = load(base, t, "evaluate2.json")
            if ev is None:
                continue
            br, fmax, smax = 0, 0.0, 0.0
            for kind, r in ev["interventions"].items():
                for sp, gr in r["resplit"].items():
                    for g, gv in gr.items():
                        for nv in gv.values():
                            for w in nv.values():
                                br += bool(w["vs_oracle"].get("breach"))
                                if g != "zero_null":
                                    fmax = max(fmax, w["vs_oracle"]["fdr"])
                    for nv in r["stress"][sp].values():
                        smax = max(smax, max(v["p_any_rejection"] for v in nv.values()))
            rows.append(f"{m} & {tl} & {br} & {fmax:.3f} & {smax:.3f} \\\\")
    (OUT / "table_stage5_validity.tex").write_text(
        "\\begin{tabular}{llrrr}\n\\toprule\nModel & Task & Breaches & Max FDR & Max stress $P(\\text{any rejection})$ "
        "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


plt.rcParams.update({"font.size": 12, "axes.titlesize": 12, "legend.fontsize": 10})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_power()
    fig_zero_null()
    fig_effects()
    tables()
    print(f"wrote fig20-22 and table_stage5_*.tex to {OUT}")


if __name__ == "__main__":
    main()
