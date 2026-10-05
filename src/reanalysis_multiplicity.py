"""Multiplicity reanalysis of the saved Stage 3 and Stage 4 results (no new runs).

The findings count "cells significant at 1.96 SE" across 54-126 conditions without correction.
This script recomputes those counts with Holm and Benjamini-Hochberg corrections within each family
of cells, and replaces cell counts with trend tests where the claim is about a trend:

  1. FDR exceedance: per cell, one-sided t-test of H0: E[FDP] <= q over the 30 replicates
     (df = 29). The findings' rule "mean - 1.96 SE > q" is the uncorrected version at about 2.5%.
  2. Contrasts between arms: per cell, two-sided t-test of the mean difference; paired over
     replicates where the arms share labels, unpaired otherwise (as in the original analyses).
  3. Trends in amplitude: for each replicate, the FDR averaged over the cells at each amplitude,
     regressed on log amplitude; the 30 per-replicate slopes are tested against 0 (one t-test per
     arm, so no multiplicity). Replicates are independent, so the slopes are too.

Usage: python src/reanalysis_multiplicity.py --config config/default.yaml
Writes results/multiplicity_reanalysis.json and prints a summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from common import load_config, results_dir

ALPHA = 0.05


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, running = np.empty_like(p), 0.0
    for i, j in enumerate(order):
        running = max(running, min(1.0, (len(p) - i) * p[j]))
        adj[j] = running
    return adj


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1.0)
    return out


def by_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Yekutieli: BH scaled by the harmonic sum, valid under arbitrary dependence (the cells
    share one activation matrix, and Stage 4's cells share knockoff draws)."""
    n = len(p)
    return np.minimum(bh(p) * np.sum(1.0 / np.arange(1, n + 1)), 1.0)


def one_sided_exceed(v: np.ndarray, q: float) -> tuple[float, float, float]:
    """mean, SE, one-sided p for H0: mean <= q."""
    v = np.asarray(v, float)
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    if se == 0:
        return m, se, (0.0 if m > q else 1.0)
    return m, se, float(stats.t.sf((m - q) / se, df=len(v) - 1))


def two_sided_diff(a: np.ndarray, b: np.ndarray, paired: bool) -> tuple[float, float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if paired:
        d = a - b
        m, se, df = d.mean(), d.std(ddof=1) / np.sqrt(len(d)), len(d) - 1
    else:
        m = a.mean() - b.mean()
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        se = np.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if va + vb > 0 else 1
    if se == 0:
        return m, se, (0.0 if m != 0 else 1.0)
    return m, se, float(2 * stats.t.sf(abs(m / se), df=df))


def load(rd: Path, rec: str, meta: str):
    z = np.load(rd / f"{rec}.npz")
    d = json.loads((rd / f"{meta}.json").read_text())
    return z, d


def cells(d: dict):
    for ia, a in enumerate(d["amplitudes"]):
        for jf, f in enumerate(d["forms"]):
            for kk, k in enumerate(d["signal_sizes"]):
                for iq, q in enumerate(d["nominal_fdr_targets"]):
                    yield (ia, jf, kk, iq), {"amplitude": a, "form": f, "k": k, "q": q}


def fdr_exceedance(z, d, arm: str, label: str) -> dict:
    rows = []
    for (ia, jf, kk, iq), c in cells(d):
        m, se, p = one_sided_exceed(z[f"ko_fdr__{arm}"][ia, jf, kk, :, iq], c["q"])
        rows.append({**c, "mean": m, "se": se, "p": p})
    p = np.array([r["p"] for r in rows])
    ph, pb, py = holm(p), bh(p), by_adjust(p)
    unc = [r for r in rows if r["mean"] - 1.96 * r["se"] > r["q"]]
    return {"experiment": label, "arm": arm, "n_cells": len(rows),
            "n_uncorrected_1.96SE": len(unc),
            "n_holm": int((ph < ALPHA).sum()), "n_bh": int((pb < ALPHA).sum()), "n_by": int((py < ALPHA).sum()),
            "min_p": float(p.min()), "min_p_holm": float(ph.min()), "min_p_bh": float(pb.min()),
            "uncorrected_cells": [{k: r[k] for k in ("amplitude", "form", "k", "q", "mean", "se", "p")} for r in unc]}


def contrast(z, d, a: str, b: str, metric: str, paired: bool, label: str, by=None, where=None) -> dict:
    rows = []
    for (ia, jf, kk, iq), c in cells(d):
        if where and not where(c):
            continue
        m, se, p = two_sided_diff(z[f"{metric}__{a}"][ia, jf, kk, :, iq], z[f"{metric}__{b}"][ia, jf, kk, :, iq], paired)
        rows.append({**c, "diff": m, "se": se, "p": p})
    p = np.array([r["p"] for r in rows])
    ph, pb, py = holm(p), bh(p), by_adjust(p)
    for r, x, y, w in zip(rows, ph, pb, py):
        r["p_holm"], r["p_bh"], r["p_by"] = x, y, w

    def count(rs):
        return {"n_cells": len(rs),
                "uncorrected_pos": sum(r["diff"] > 1.96 * r["se"] > 0 for r in rs),
                "uncorrected_neg": sum(-r["diff"] > 1.96 * r["se"] > 0 for r in rs),
                "holm_pos": sum(r["p_holm"] < ALPHA and r["diff"] > 0 for r in rs),
                "holm_neg": sum(r["p_holm"] < ALPHA and r["diff"] < 0 for r in rs),
                "bh_pos": sum(r["p_bh"] < ALPHA and r["diff"] > 0 for r in rs),
                "bh_neg": sum(r["p_bh"] < ALPHA and r["diff"] < 0 for r in rs),
                "by_pos": sum(r["p_by"] < ALPHA and r["diff"] > 0 for r in rs),
                "by_neg": sum(r["p_by"] < ALPHA and r["diff"] < 0 for r in rs),
                "mean_diff": float(np.mean([r["diff"] for r in rs]))}

    out = {"experiment": label, "contrast": f"{a} - {b}", "metric": metric, "paired": paired, **count(rows)}
    if by:
        out["by_" + by] = {v: count([r for r in rows if r[by] == v]) for v in sorted({r[by] for r in rows}, key=str)}
    return out


def pooled(z, d, a: str, b: str, metric: str, paired: bool, label: str, where=None) -> dict:
    """One test per comparison: each replicate's mean over the selected cells, then a t-test of the
    difference across replicates (paired when the arms share labels)."""
    sel = [idx for idx, c in cells(d) if not where or where(c)]
    va = np.array([np.mean([z[f"{metric}__{a}"][ia, jf, kk, r, iq] for ia, jf, kk, iq in sel])
                   for r in range(z[f"{metric}__{a}"].shape[3])])
    vb = np.array([np.mean([z[f"{metric}__{b}"][ia, jf, kk, r, iq] for ia, jf, kk, iq in sel])
                   for r in range(z[f"{metric}__{b}"].shape[3])])
    m, se, p = two_sided_diff(va, vb, paired)
    return {"experiment": label, "contrast": f"{a} - {b}", "metric": metric, "paired": paired,
            "n_cells": len(sel), "mean_diff": m, "se": se, "p_two_sided": p}


def trend(z, d, arm: str, label: str, minus: str | None = None, where=None) -> dict:
    """Per-replicate slope of mean FDR (over the selected cells at each amplitude) on log amplitude."""
    amps = np.log(np.array(d["amplitudes"], float))
    A, F, K, R, Q = z[f"ko_fdr__{arm}"].shape
    sel = [(jf, kk, iq) for (ia, jf, kk, iq), c in cells(d) if ia == 0 and (not where or where(c))]
    x = z[f"ko_fdr__{arm}"].astype(float)
    if minus:
        x = x - z[f"ko_fdr__{minus}"].astype(float)
    per_rep = np.stack([np.mean([x[ia, jf, kk, :, iq] for jf, kk, iq in sel], axis=0) for ia in range(A)])  # (A, R)
    slopes = np.array([np.polyfit(amps, per_rep[:, r], 1)[0] for r in range(R)])
    t = stats.ttest_1samp(slopes, 0.0)
    return {"experiment": label, "arm": arm + (f" - {minus}" if minus else ""), "n_cells_per_amplitude": len(sel),
            "mean_fdr_by_amplitude": dict(zip(map(str, d["amplitudes"]), per_rep.mean(axis=1).round(4).tolist())),
            "slope_per_log_amplitude": float(slopes.mean()), "slope_se": float(slopes.std(ddof=1) / np.sqrt(R)),
            "t": float(t.statistic), "p_two_sided": float(t.pvalue)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Multiplicity reanalysis of saved Stage 3/4 results")
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    rd = results_dir(load_config(args.config))
    out = {"alpha": ALPHA, "fdr_exceedance": [], "contrasts": [], "trends": []}

    zv, dv = load(rd, "stage3_v2_records", "stage3_v2_fdr")
    zs, ds = load(rd, "stage3_stress_records", "stage3_stress")
    zd, dd = load(rd, "stage3_dims_records", "stage3_dims")
    z4, d4 = load(rd, "stage4_repairs_records", "stage4_repairs")

    # 1. FDR exceedance, one family per arm
    for a in ("real_mvr", "real_equi", "gauss_mvr"):
        out["fdr_exceedance"].append(fdr_exceedance(zv, dv, a, "stage3_v2 (amplitudes 0.5-8)"))
    for a in ("real_mvr", "real_equi", "gauss_mvr", "real_hurdle"):
        out["fdr_exceedance"].append(fdr_exceedance(zs, ds, a, "stage3_stress (amplitudes 1-20)"))
    for a in [k.split("__", 1)[1] for k in zd.files if k.startswith("ko_fdr__")]:
        out["fdr_exceedance"].append(fdr_exceedance(zd, dd, a, "stage3_dims"))
    for a in ("gauss_real", "hurdle", "binarised", "evalue", "gauss_ceiling"):
        out["fdr_exceedance"].append(fdr_exceedance(z4, d4, a, "stage4"))

    # 2. contrasts
    out["contrasts"] += [
        contrast(zv, dv, "real_mvr", "gauss_mvr", "ko_pow", False, "stage3_v2: atom power cost", by="form"),
        contrast(zv, dv, "real_mvr", "gauss_mvr", "ko_fdr", False, "stage3_v2: atom FDR excess"),
        contrast(zs, ds, "real_mvr", "gauss_mvr", "ko_fdr", False, "stage3_stress: atom FDR excess", by="amplitude"),
        contrast(zs, ds, "real_hurdle", "real_equi", "ko_fdr", True, "stage3_stress: hurdle FDR, amplitude >= 5",
                 where=lambda c: c["amplitude"] >= 5),
        contrast(z4, d4, "hurdle", "gauss_real", "ko_pow", True, "stage4: hurdle_gain power"),
        contrast(z4, d4, "hurdle", "gauss_real", "ko_fdr", True, "stage4: hurdle_gain FDR"),
        contrast(z4, d4, "binarised", "gauss_real", "ko_pow", True, "stage4: binarised_gain power"),
        contrast(z4, d4, "binarised", "gauss_real", "ko_fdr", True, "stage4: binarised_gain FDR"),
        contrast(z4, d4, "evalue", "gauss_real", "ko_pow", True, "stage4: evalue_gain power"),
        contrast(z4, d4, "gauss_ceiling", "gauss_real", "ko_pow", True, "stage4: atom_gap power"),
    ]
    for lab, a, b in (("rows, p2048 real equicorr", "p2048_hi_real_equicorrelated", "p2048_lo_real_equicorrelated"),
                      ("rows, p2048 real mvr", "p2048_hi_real_mvr", "p2048_lo_real_mvr"),
                      ("S, p2048 lo real", "p2048_lo_real_mvr", "p2048_lo_real_equicorrelated"),
                      ("S, p2048 hi real", "p2048_hi_real_mvr", "p2048_hi_real_equicorrelated")):
        out["contrasts"].append(contrast(zd, dd, a, b, "ko_pow", lab.startswith("S"), f"stage3_dims: {lab}"))

    # 2b. pooled tests (one per comparison)
    nofloor = lambda c: not (c["k"] == 10 and c["q"] == 0.05)
    out["pooled"] = [
        pooled(zv, dv, "real_mvr", "gauss_mvr", "ko_pow", False, "stage3_v2: atom power cost, linear",
               where=lambda c: c["form"] == "linear"),
        pooled(zv, dv, "real_mvr", "gauss_mvr", "ko_pow", False, "stage3_v2: atom power cost, interaction",
               where=lambda c: c["form"] == "interaction"),
        pooled(zv, dv, "real_mvr", "gauss_mvr", "ko_pow", False, "stage3_v2: atom power cost, amplitude <= 2",
               where=lambda c: c["amplitude"] <= 2),
        pooled(z4, d4, "hurdle", "gauss_real", "ko_pow", True, "stage4: hurdle_gain power, all cells"),
        pooled(z4, d4, "hurdle", "gauss_real", "ko_pow", True, "stage4: hurdle_gain power, excl. floor", where=nofloor),
        pooled(z4, d4, "hurdle", "gauss_real", "ko_fdr", True, "stage4: hurdle_gain FDR"),
        pooled(z4, d4, "gauss_ceiling", "gauss_real", "ko_pow", True, "stage4: atom_gap power, excl. floor", where=nofloor),
        pooled(z4, d4, "binarised", "gauss_real", "ko_fdr", True, "stage4: binarised_gain FDR"),
        pooled(z4, d4, "evalue", "gauss_real", "ko_pow", True, "stage4: evalue_gain power, excl. floor", where=nofloor),
        pooled(z4, d4, "evalue", "hurdle", "ko_pow", True, "stage4: evalue - hurdle power, excl. floor", where=nofloor),
        pooled(zs, ds, "real_hurdle", "real_equi", "ko_fdr", True, "stage3_stress: hurdle FDR, amplitude >= 5",
               where=lambda c: c["amplitude"] >= 5),
        pooled(zd, dd, "p2048_hi_real_mvr", "p2048_lo_real_mvr", "ko_pow", False, "stage3_dims: rows, p2048 real mvr"),
        pooled(zd, dd, "p2048_lo_real_mvr", "p2048_lo_real_equicorrelated", "ko_pow", True, "stage3_dims: S, p2048 lo real"),
    ]

    # 2c. the interaction-vs-linear claim as one direct test (not two separately corrected counts):
    # per replicate, the atom's mean power cost on interaction cells minus that on linear cells
    cost = zv["ko_pow__real_mvr"].astype(float) - zv["ko_pow__gauss_mvr"].astype(float)
    dlin = cost[:, 0].mean(axis=(0, 1, 3))
    dint = cost[:, 1].mean(axis=(0, 1, 3))
    m, se, pv = two_sided_diff(dint, dlin, True)
    out["interaction_vs_linear_atom_cost"] = {"interaction_minus_linear": m, "se": se, "p_two_sided": pv,
                                              "linear_cost": float(dlin.mean()), "interaction_cost": float(dint.mean())}

    # 3. trends in amplitude
    for a in ("real_mvr", "real_equi", "gauss_mvr", "real_hurdle"):
        out["trends"].append(trend(zs, ds, a, "stage3_stress"))
    out["trends"].append(trend(zs, ds, "real_mvr", "stage3_stress", minus="gauss_mvr"))
    out["trends"].append(trend(zs, ds, "real_hurdle", "stage3_stress", minus="real_equi"))
    for a in ("gauss_real", "hurdle", "binarised", "gauss_ceiling"):
        out["trends"].append(trend(z4, d4, a, "stage4"))

    (rd / "multiplicity_reanalysis.json").write_text(json.dumps(out, indent=2, default=float))

    print("=== FDR exceedance (cells with FDR significantly above q) ===")
    print(f"{'experiment':34s} {'arm':32s} cells  unc  Holm  BH  BY   min p_Holm")
    for r in out["fdr_exceedance"]:
        if r["n_uncorrected_1.96SE"] or r["experiment"] != "stage3_dims":
            print(f"{r['experiment']:34s} {r['arm']:32s} {r['n_cells']:5d} {r['n_uncorrected_1.96SE']:4d} "
                  f"{r['n_holm']:5d} {r['n_bh']:3d} {r['n_by']:3d}   {r['min_p_holm']:.3f}")
    print("\n=== contrasts (significant cells: positive/negative) ===")
    for r in out["contrasts"]:
        print(f"{r['experiment']:42s} {r['metric']:6s} mean {r['mean_diff']:+.4f}  cells {r['n_cells']:3d}  "
              f"unc +{r['uncorrected_pos']}/-{r['uncorrected_neg']}  Holm +{r['holm_pos']}/-{r['holm_neg']}  "
              f"BH +{r['bh_pos']}/-{r['bh_neg']}  BY +{r['by_pos']}/-{r['by_neg']}")
        for v, c in r.get("by_form", {}).items():
            print(f"    {v:12s} unc +{c['uncorrected_pos']}/-{c['uncorrected_neg']}  Holm +{c['holm_pos']}/-{c['holm_neg']}"
                  f"  BH +{c['bh_pos']}/-{c['bh_neg']}  mean {c['mean_diff']:+.4f}")
    iv = out["interaction_vs_linear_atom_cost"]
    print(f"\n=== atom power cost, interaction - linear (one direct test): {iv['interaction_minus_linear']:+.4f} "
          f"± {iv['se']:.4f}  p {iv['p_two_sided']:.2g}")
    print("\n=== pooled tests (one per comparison) ===")
    for r in out["pooled"]:
        print(f"{r['experiment']:46s} {r['metric']:6s} {r['mean_diff']:+.4f} ± {r['se']:.4f}  p {r['p_two_sided']:.2g}")
    print("\n=== FDR trend in amplitude (per-replicate slopes on log amplitude) ===")
    for r in out["trends"]:
        print(f"{r['experiment']:14s} {r['arm']:24s} slope {r['slope_per_log_amplitude']:+.4f} ± {r['slope_se']:.4f}  "
              f"t {r['t']:+.1f}  p {r['p_two_sided']:.2g}")
    print(f"\nwrote {rd}/multiplicity_reanalysis.json")


if __name__ == "__main__":
    main()
