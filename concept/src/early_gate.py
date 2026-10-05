"""Step 6 — early gate, per-latent only (block-MVR knockoffs, the best existing baseline).

  planted (design B): per-latent concept recovery vs width at layer 12, amplitudes {1, 3, 8},
                      K_c {20, 30}, both forms, 10 replicates
  real SST-2:         per-latent discoveries at q = 0.1 vs width (median over 10 knockoff draws) and
                      Jaccard of discovered parent sets vs 16k
Gate G1 (pre-registered): proceed only if per-latent concept recovery falls with log2(width): per
replicate, recovery at q = 0.1 averaged over (form, K_c), slope vs log2(width) by least squares;
one-sided t-test of mean slope < 0 per amplitude; Holm across the three amplitudes; G1 passes if any
amplitude's Holm-adjusted p < 0.05.

Usage: python concept/src/early_gate.py run --keys ...   (per GPU)   |   python concept/src/early_gate.py analyse
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

import _paths  # noqa: F401
from bench import real_sweep
from cseeds import ROOT, load_concept_config
from gpu import init_cuda
from planted_concept import run
from reanalysis_multiplicity import holm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "analyse"])
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = load_concept_config(); cc = cfg["concept"]; g = cc["gate"]
    all_keys = cc["sweep_L12"]
    cdir = ROOT / cc["cache_dir"]
    if args.phase == "run":
        dev = init_cuda(args.device); torch.set_num_threads(8)
        grid = {"K_c": g["K_c"], "amplitudes": g["amplitudes"], "forms": g["forms"], "replicates": g["replicates"], "qs": [g["q"]]}
        run(cfg, args.layer, args.keys, all_keys, grid, dev, "gate", arms=["latent"], designs=["B"],
            stream=("gate_planted", "gate_knockoff"))
        real_sweep(cfg, args.layer, args.keys, all_keys, ["latent"], g["real_draws"], [g["q"]], dev, "gate")
        print("DONE", flush=True)
        return
    widths = np.array([cc["saes"][k]["width"] for k in all_keys]); x = np.log2(widths)
    rec = np.stack([np.load(cdir / f"ckpt_gate_{k}.npz")["latent__con_pow"] for k in all_keys])   # (W, 1, K, A, F, R, 1)
    r = rec[:, 0, :, :, :, :, 0].mean(axis=(1, 3))                                                  # (W, A, R)
    slopes = np.array([[np.polyfit(x, r[:, ia, rep], 1)[0] for rep in range(r.shape[2])] for ia in range(r.shape[1])])
    p1 = np.array([stats.ttest_1samp(s, 0.0, alternative="less").pvalue for s in slopes])
    ph = holm(p1)
    res = {"G1": {"amplitudes": g["amplitudes"], "mean_slope_per_log2_width": slopes.mean(1).tolist(),
                  "se": (slopes.std(1, ddof=1) / np.sqrt(slopes.shape[1])).tolist(), "p_one_sided": p1.tolist(),
                  "p_holm": ph.tolist(), "PASS": bool((ph < 0.05).any())},
           "recovery_mean": {k: r[i].mean(1).tolist() for i, k in enumerate(all_keys)}, "real": {}}
    base = None
    for k in all_keys:
        z = np.load(cdir / f"real_gate_{k}.npz")
        S = z["sel__latent"][0]                                       # (draws, p)
        par = z["parent"]
        stable = set(par[S.mean(0) >= 0.5].tolist())
        per_draw = [set(par[s].tolist()) for s in S]
        if base is None:
            base = (stable, per_draw)
        jac = lambda a, b: len(a & b) / max(1, len(a | b))
        res["real"][k] = {"median_discoveries": float(np.median(S.sum(1))), "median_parents": float(np.median([len(s) for s in per_draw])),
                          "stable_parents": len(stable), "jaccard_stable_vs_16k": jac(stable, base[0]),
                          "median_jaccard_draw_vs_16k": float(np.median([jac(a, b) for a, b in zip(per_draw, base[1])]))}
    print(json.dumps(res["G1"], indent=1)); print(json.dumps(res["real"], indent=1))
    rd = ROOT / "concept" / "results"
    (rd / "early_gate.json").write_text(json.dumps(res, indent=1))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for ia, a in enumerate(g["amplitudes"]):
        m = r[:, ia].mean(1); se = r[:, ia].std(1, ddof=1) / np.sqrt(r.shape[2])
        ax[0].errorbar(widths, m, yerr=se, marker="o", capsize=3,
                       label=f"amp {a:g} (slope {slopes[ia].mean():+.3f}/doubling, Holm p {ph[ia]:.3g})")
    ax[0].set(xscale="log", ylim=(0, 1.02), xlabel="SAE width (layer 12)", ylabel="per-latent concept recovery (q=0.1)",
              title=f"Gate G1: {'PASS' if res['G1']['PASS'] else 'FAIL'} — design B, mean over form × K_c")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(widths, [res["real"][k]["median_discoveries"] for k in all_keys], "o-", label="per-latent discoveries")
    ax[1].plot(widths, [res["real"][k]["median_parents"] for k in all_keys], "s-", label="distinct parents")
    ax2 = ax[1].twinx(); ax2.plot(widths, [res["real"][k]["jaccard_stable_vs_16k"] for k in all_keys], "^--", color="C3", label="Jaccard vs 16k")
    ax2.set_ylim(0, 1.02); ax2.set_ylabel("Jaccard of stable parent sets vs 16k", color="C3")
    ax[1].set(xscale="log", xlabel="SAE width (layer 12)", ylabel="median over 10 draws", title="Real SST-2 labels, q = 0.1")
    ax[1].legend(loc="upper left", fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(rd / "fig_c_gate.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
