"""Khera (2026)-style baseline for Stage 5 (CPU only, from each task's delta.npz).

Khera tests whether zero-ablating latent j hurts the task more than zero-ablating a randomly chosen latent: per
input, D_j(x) = Delta_j(x) - Delta_r(x) with r drawn uniformly from the other candidates (Khera, Appendix A), a
betting e-value (aGRAPA, one-sided) against E[D_j] <= 0, and e-BH. Khera's metric is a logit difference; ours is
the correct-class probability. We run it on the zero-ablation effects of the screened candidates
on the same re-split replicates as evaluate2 (same permutation streams), next to our procedure on the same inputs
(zero ablation, minimum effect delta), and score both against:
  - the minimum-effect truth tau_j > delta (our target), and
  - Khera's own target: tau_j > the mean tau of the other candidates.
Writes results/stage5_khera_baseline.json.

Usage: python scripts/stage5_khera_baseline.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import load_config, results_dir, stream_rng                        # noqa: E402
from stage5_interventional import _selections, _summ_rep                       # noqa: E402

CONFIGS = [ROOT / "config" / "default.yaml", ROOT / "config" / "models" / "gemma3_1b.yaml"]
KINDS = {"zero": 1, "resample": 0}    # intervention -> evaluate2's index ([primary] + sensitivity): same permutations
EXTRA_N = {"resample": [3000, 4000]}  # the resampling pool has 5,000 inputs: follow both tests to larger n
CHUNK = 25


def khera_stat(X: np.ndarray, Kk: int, Kr: int, rng: np.random.Generator) -> np.ndarray:
    """(c, n, Kc) effects -> (c, n, Kk) paired differences Delta_j - Delta_r, r uniform over the first Kr latents
    except j, drawn per replicate, input and hypothesis; halved into [-1, 1] for the bounded-mean bet (null: mean <= 0).
    Khera draws r from the tested set, so Kr = Kk; Kr > Kk is the comparator-pool sensitivity."""
    c, n, _ = X.shape
    r = rng.integers(0, Kr - 1, size=(c, n, Kk))
    r = r + (r >= np.arange(Kk))                                                  # skip j itself
    ref = np.take_along_axis(X[:, :, :Kr], r, axis=2)
    return (X[:, :, :Kk] - ref) / 2.0


def run_task(cfg: dict, task: str, kind: str) -> dict | None:
    s5, v2 = cfg["stage5"], cfg["stage5"]["v2"]
    d = results_dir(cfg) / "stage5" / task
    if not (d / "delta.npz").exists() or not (d / "pilot.json").exists():
        return None
    z = np.load(d / "delta.npz")
    if f"B__{kind}" not in z.files:
        return None
    delta0 = json.loads((d / "pilot.json").read_text())["delta"]
    ti = list(s5["tasks"]).index(task)
    ev, alphas, R = s5["evalue"], s5["alphas"], s5["replicates"]
    Kc = int(z["n_candidates"])
    P = np.vstack([z[f"B__{kind}"], z[f"O__{kind}"]])[:, :Kc].astype(np.float64)     # candidates only
    N = P.shape[0]
    K50 = min(int(v2["candidates"]), Kc)
    # (tested K, bet cap, comparator pool). Khera caps the bet at c = 0.5 / B with |X| <= B, so every factor stays in
    # [0.5, 1.5]: cap 0.5 on our halved difference. Khera's comparator comes from the tested set: pool = K.
    specs = {"prereg": (Kc, float(ev["bet_cap"]), Kc), "v2": (K50, float(v2["bet_cap"]), K50),
             "v2_cap0.5": (K50, 0.5, K50), "v2_cap0.5_pool250": (K50, 0.5, Kc)}
    tau = P.mean(0)
    out = {"delta": delta0, "N": N, "candidates": Kc,
           "counts_full_pool": {sp: {"tau_above_delta": int((tau[:Kk] > delta0).sum()),
                                     "tau_above_random_candidate": int((tau[:Kk] > (tau[:Kr].sum() - tau[:Kk]) / (Kr - 1)).sum()),
                                     "mean_tau_comparator_pool": float(tau[:Kr].mean()), "K": Kk, "comparator_pool": Kr}
                                for sp, (Kk, _, Kr) in specs.items()},
           "resplit": {sp: {} for sp in specs}}
    for n in list(s5["test_sizes"]) + EXTRA_N.get(kind, []):
        if n > N - v2["min_oracle"]:
            continue
        acc = {(sp, m): {a: [] for a in alphas} for sp in specs for m in ("ours", "khera")}
        tau_r = []
        for r0 in range(0, R, CHUNK):
            reps = range(r0, min(R, r0 + CHUNK))
            perms = [stream_rng(cfg, "s5_order", ti, 1000 + KINDS[kind], n, r).permutation(N) for r in reps]
            tau_r += [P[pm[n:]].mean(0) for pm in perms]
            Xte = P[np.stack([pm[:n] for pm in perms])]                                # (c, n, Kc)
            rng = stream_rng(cfg, "s5_order", ti, 4000 + KINDS[kind], n, r0)
            for sp, (Kk, cap, Kr) in specs.items():
                for a, s in _selections(Xte[:, :, :Kk], delta0, cap, ev, alphas).items():
                    acc[(sp, "ours")][a].append(s)
                for a, s in _selections(khera_stat(Xte, Kk, Kr, rng), 0.0, cap, ev, alphas).items():
                    acc[(sp, "khera")][a].append(s)
        tau_r = np.stack(tau_r)                                                       # (R, Kc) oracle per replicate
        for sp, (Kk, _, Kr) in specs.items():
            oth_r = (tau_r[:, :Kr].sum(1, keepdims=True) - tau_r[:, :Kk]) / (Kr - 1)
            t_delta, t_own = tau_r[:, :Kk] > delta0, tau_r[:, :Kk] > oth_r
            rec = {}
            for a in alphas:
                ours, kh = np.concatenate(acc[(sp, "ours")][a]), np.concatenate(acc[(sp, "khera")][a])
                rec[str(a)] = {"ours_vs_delta": _summ_rep(ours, t_delta, alpha=a),
                               "khera_vs_delta": _summ_rep(kh, t_delta),
                               "khera_vs_own_target": _summ_rep(kh, t_own, alpha=a)}
            out["resplit"][sp][str(n)] = rec
    return out


def main() -> None:
    t0 = time.time()
    res = {"_note": __doc__.strip().split("\n\n")[0], "interventions": {}}
    for kind, cp in ((k, c) for k in KINDS for c in CONFIGS):
        cfg = load_config(str(cp))
        model = cfg["model"]["name"]
        for task in cfg["stage5"]["tasks"]:
            r = run_task(cfg, task, kind)
            if r is None:
                continue
            res["interventions"].setdefault(kind, {}).setdefault(model, {})[task] = r
            v = r["resplit"]["v2"]
            nmax = max(v, key=int)
            a = v[nmax]["0.1"]
            print(f"{kind:8s} {model:22s} {task:13s} n={nmax:>4}  K=50 alpha=0.1 | "
                  f"ours FDR {a['ours_vs_delta']['fdr']:.3f} power {a['ours_vs_delta']['power']:.2f} "
                  f"sel {a['ours_vs_delta']['mean_selected']:.1f} | "
                  f"Khera FDR(vs delta) {a['khera_vs_delta']['fdr']:.3f} power {a['khera_vs_delta']['power']:.2f} "
                  f"sel {a['khera_vs_delta']['mean_selected']:.1f} | Khera FDR(own) {a['khera_vs_own_target']['fdr']:.3f}  "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    path = ROOT / "results" / "stage5_khera_baseline.json"
    path.write_text(json.dumps(res, indent=2))
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
