"""SAEBench Stage 1 — does search-then-validate certify noise on the 35 sparse-probing tasks?

Per task (config/preregistration.yaml saebench_amendment_1): the SST-2 Stage 1 workflow, unchanged,
on the task's 5,000 rows (its SAEBench train and test texts pooled):

  search    naive per-latent Welch |t| scan (uncorrected, Bonferroni, Westfall-Young), and the
            top-10 by mean difference / AUROC / probe weight on a fresh 70% split
  validate  calibrate_validation.run_one: joint-ablation paired log-loss test, same-data and
            held-out regimes (identical code, probe and settings to the SST-2 run)

under B label permutations (the global null) plus the real labels. A planted-signal power gate per
dataset (rule SB1-B) says whether that dataset's validate-step FWERs can be interpreted. Each task
writes its own JSON, so the run resumes where it stopped; --aggregate pools them.

Usage: python src/saebench/stage1.py --config config/default.yaml --device cuda [--dataset NAME] [--limit-perms 20]
       python src/saebench/stage1.py --config config/default.yaml --aggregate
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/
sys.path.insert(0, str(Path(__file__).resolve().parent))       # src/saebench/

from calibrate_pipeline import to_gpu, welch_abs_t                      # noqa: E402
from calibrate_validation import REGIMES, VARIANTS, certified, run_one, wilson   # noqa: E402
from common import load_config, stream_rng                               # noqa: E402
from planted_fdr import generate_planted_labels                         # noqa: E402
from sbutil import binary_tasks, cache_path, load_cache, results_dir, slug   # noqa: E402


def welch_counts(X: torch.Tensor, y: torch.Tensor, z_naive: float, z_bonf: float) -> dict:
    t = welch_abs_t(X, y)
    return {"n_naive": int((t > z_naive).sum()), "n_bonf": int((t > z_bonf).sum()), "t_max": float(t.max()),
            "t_abs": t}


def run_task(cfg: dict, task: dict, X_np: np.ndarray, device, B: int, ti: int) -> dict:
    s1, sv = cfg["stage1"], cfg["stage1_validate"]
    methods, alpha, k = s1["scoring_methods"], sv["alpha"], sv["top_k"]
    X = to_gpu(X_np, device)
    y_np = task["y"]
    n, p = X_np.shape
    z_naive = statistics.NormalDist().inv_cdf(1.0 - s1["naive_alpha"] / 2.0)
    z_bonf = statistics.NormalDist().inv_cdf(1.0 - s1["naive_alpha"] / (2.0 * p))
    di = cfg["saebench"]["datasets"].index(task["dataset"])
    rng = stream_rng(cfg, "sb_validate_permutation", di, ti)

    real = run_one(X, to_gpu(y_np, device), cfg, rng)
    w_real = welch_counts(X, to_gpu(y_np, device), z_naive, z_bonf)
    null = {m: {r: {"joint_p": [], "single_p_min": [], "joint_mean_d": []} for r in REGIMES} for m in methods}
    wnull = {"n_naive": [], "n_bonf": [], "t_max": []}
    for _ in range(B):
        yp = to_gpu(rng.permutation(y_np), device)
        res = run_one(X, yp, cfg, rng)
        for m in methods:
            for r in REGIMES:
                for key in null[m][r]:
                    null[m][r][key].append(res[m][r][key])
        wc = welch_counts(X, yp, z_naive, z_bonf)
        for key in wnull:
            wnull[key].append(wc[key])

    out = {"task": task["name"], "dataset": task["dataset"], "class": task["class"], "n": n, "p": p,
           "n_positive": int(y_np.sum()), "n_permutations": B, "alpha": alpha, "top_k": k,
           "validate": {}, "real": {}}
    for m in methods:
        out["validate"][m], out["real"][m] = {}, {}
        for r in REGIMES:
            pj, ps = np.array(null[m][r]["joint_p"]), np.array(null[m][r]["single_p_min"])
            out["validate"][m][r] = {}
            for v, hits in (("joint", pj <= alpha), ("any_uncorrected", ps <= alpha),
                            ("any_bonferroni", ps <= alpha / k)):
                out["validate"][m][r][v] = {"fwer": float(hits.mean()), "n_certified": int(hits.sum()),
                                            "ci95": list(wilson(int(hits.sum()), B))}
            out["real"][m][r] = {**real[m][r], **{v: bool(c) for v, c in certified(real[m][r], alpha, k).items()}}
    tmax, nn, nb = np.array(wnull["t_max"]), np.array(wnull["n_naive"]), np.array(wnull["n_bonf"])
    wy_thr = float(np.quantile(tmax, 1.0 - s1["wy_alpha"]))
    t_real = w_real["t_abs"].cpu().numpy()
    out["search"] = {"z_naive": z_naive, "z_bonferroni": z_bonf, "wy_threshold": wy_thr,
                     "expected_naive_false_passes": float(s1["naive_alpha"] * p),
                     "null_mean_n_naive": float(nn.mean()), "null_mean_n_bonferroni": float(nb.mean()),
                     "fwer_naive": float((nn >= 1).mean()), "fwer_bonferroni": float((nb >= 1).mean()),
                     "fwer_wy_insample": float((tmax > wy_thr).mean()),
                     "real_n_naive": int((t_real > z_naive).sum()),
                     "real_n_bonferroni": int((t_real > z_bonf).sum()),
                     "real_n_wy": int((t_real > wy_thr).sum())}
    return out


def power_gate(cfg: dict, dataset: str, task: dict, X_np: np.ndarray, device) -> dict:
    """Rule SB1-B: on the dataset's first task's rows, labels from k known latents; the joint
    validate test must certify in >= min_detect_fraction of replicates for some method, per regime."""
    pg = cfg["saebench"]["stage1"]["planted_gate"]
    sv = cfg["stage1_validate"]
    methods = cfg["stage1"]["scoring_methods"]
    di = cfg["saebench"]["datasets"].index(dataset)
    rng = stream_rng(cfg, "sb_validate_planted", di)
    sd = X_np.std(axis=0)
    Z = ((X_np - X_np.mean(axis=0)) / np.where(sd > 1e-8, sd, 1.0)).astype(np.float32)
    X = to_gpu(X_np, device)
    p = X_np.shape[1]
    rows = []
    for _ in range(pg["replicates"]):
        S = np.sort(rng.choice(p, size=pg["k"], replace=False))
        y_np, _ = generate_planted_labels(Z, S, pg["form"], pg["amplitude"], rng)
        res = run_one(X, to_gpu(y_np, device), cfg, rng)
        S_set = set(S.tolist())
        rows.append({m: {"recall": len(S_set & set(res[m]["top"])) / len(S_set),
                         **{r: certified(res[m][r], sv["alpha"], sv["top_k"]) for r in REGIMES}} for m in methods})
    per = {m: {"mean_recall_at_k": float(np.mean([r[m]["recall"] for r in rows])),
               **{reg: {v: float(np.mean([r[m][reg][v] for r in rows])) for v in VARIANTS} for reg in REGIMES}}
           for m in methods}
    has_power = {reg: bool(any(per[m][reg]["joint"] >= pg["min_detect_fraction"] for m in methods)) for reg in REGIMES}
    return {"dataset": dataset, "task_rows_from": task["name"], **pg, "by_method": per,
            "criterion_has_power": has_power, "criterion_has_power_both": all(has_power.values())}


def aggregate(cfg: dict) -> None:
    rd = results_dir(cfg, "stage1")
    methods = cfg["stage1"]["scoring_methods"]
    tasks = [json.loads(p.read_text()) for p in sorted(rd.glob("*.json")) if not p.name.startswith("gate__")]
    gates = {json.loads(p.read_text())["dataset"]: json.loads(p.read_text()) for p in sorted(rd.glob("gate__*.json"))}
    if not tasks:
        raise SystemExit(f"no per-task results in {rd}")
    alpha = tasks[0]["alpha"]
    summ = {"n_tasks": len(tasks), "n_permutations": sorted({t["n_permutations"] for t in tasks}),
            "datasets": sorted({t["dataset"] for t in tasks}),
            "power_gate": {d: g["criterion_has_power"] for d, g in gates.items()}, "validate": {}, "search": {}}
    for m in methods:
        summ["validate"][m] = {}
        for r in REGIMES:
            ok = [t for t in tasks if gates.get(t["dataset"], {}).get("criterion_has_power", {}).get(r, False)]
            f = np.array([t["validate"][m][r]["joint"]["fwer"] for t in ok]) if ok else np.array([])
            lo = np.array([t["validate"][m][r]["joint"]["ci95"][0] for t in ok]) if ok else np.array([])
            summ["validate"][m][r] = {
                "n_tasks_gate_passed": len(ok),
                "median_fwer": float(np.median(f)) if f.size else None,
                "min_fwer": float(f.min()) if f.size else None, "max_fwer": float(f.max()) if f.size else None,
                "n_tasks_fwer_ge_0.5": int((f >= 0.5).sum()),
                "n_tasks_ci_lower_above_alpha": int((lo > alpha).sum()),
                "real_labels_certified": int(sum(t["real"][m][r]["joint"] for t in ok))}
    s = [t["search"] for t in tasks]
    summ["search"] = {"median_null_mean_n_naive": float(np.median([x["null_mean_n_naive"] for x in s])),
                      "median_expected_naive": float(np.median([x["expected_naive_false_passes"] for x in s])),
                      "n_tasks_fwer_naive_ge_0.95": int(sum(x["fwer_naive"] >= 0.95 for x in s)),
                      "max_fwer_bonferroni": float(max(x["fwer_bonferroni"] for x in s)),
                      "median_real_n_wy": float(np.median([x["real_n_wy"] for x in s]))}
    # rule SB1-R: replication of the SST-2 same-data result
    same = summ["validate"]
    summ["SB1_R_same_data_replicates"] = {
        m: (None if not same[m]["same_data"]["n_tasks_gate_passed"] else
            bool(same[m]["same_data"]["n_tasks_fwer_ge_0.5"] >= (2 / 3) * same[m]["same_data"]["n_tasks_gate_passed"]))
        for m in methods}
    summ["SB1_H_held_out_above_nominal"] = {
        m: {"n_tasks": same[m]["held_out"]["n_tasks_ci_lower_above_alpha"],
            "of": same[m]["held_out"]["n_tasks_gate_passed"]} for m in methods}
    summ["per_task"] = {t["task"]: {m: {r: t["validate"][m][r]["joint"]["fwer"] for r in REGIMES} for m in methods}
                        for t in tasks}
    (results_dir(cfg) / "stage1_summary.json").write_text(json.dumps(summ, indent=2))
    make_figure(tasks, methods, alpha, results_dir(cfg))
    print(json.dumps({k: v for k, v in summ.items() if k != "per_task"}, indent=2))
    print(f"\nwrote {results_dir(cfg)}/stage1_summary.json and fig_sb1_validate.png")


def make_figure(tasks, methods, alpha, rd: Path) -> None:
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.5), squeeze=False, sharey=True)
    names = [t["task"].split("__")[-1] + " (" + t["dataset"].split("/")[-1][:14] + ")" for t in tasks]
    y = np.arange(len(tasks))
    for ax, m in zip(axes[0], methods):
        for r, col, off in (("same_data", "#c0392b", -0.18), ("held_out", "#3b6ea5", 0.18)):
            v = [t["validate"][m][r]["joint"]["fwer"] for t in tasks]
            ax.barh(y + off, v, 0.34, color=col, label=r.replace("_", "-"))
        ax.axvline(alpha, color="k", ls="--", lw=1)
        ax.set_xlim(0, 1.02); ax.set_title(f"{m}: FWER under permuted labels"); ax.set_xlabel("FWER")
    axes[0][0].set_yticks(y); axes[0][0].set_yticklabels(names, fontsize=6); axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(rd / "fig_sb1_validate.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="SAEBench Stage 1: search-then-validate under the global null")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--limit-perms", type=int, default=None, help="debug: fewer permutations (written to a debug dir)")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.aggregate:
        aggregate(cfg)
        return
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    B = args.limit_perms or cfg["saebench"]["stage1"]["n_permutations"]
    rd = results_dir(cfg, "stage1" if not args.limit_perms else "stage1_debug")
    names = [args.dataset] if args.dataset else cfg["saebench"]["datasets"]
    print(f"device {device} | B = {B} | writing to {rd}")
    for name in names:
        if not cache_path(cfg, name).exists():
            print(f"\n=== {name}: no cache, SKIPPED ===", flush=True)
            continue
        C = load_cache(cfg, name)
        tasks = binary_tasks(cfg, name, C)
        print(f"\n=== {name}: n = {len(C['X'])}, p = {C['X'].shape[1]}, {len(tasks)} task(s), cache {C['config_hash']} ===",
              flush=True)
        gp = rd / f"gate__{slug(name)}.json"
        if not gp.exists():
            t = time.time()
            g = power_gate(cfg, name, tasks[0], C["X"][tasks[0]["rows"]], device)
            gp.write_text(json.dumps(g, indent=2))
            print(f"  power gate: {g['criterion_has_power']}  ({time.time() - t:.0f}s)", flush=True)
        for ti, task in enumerate(tasks):
            tp = rd / f"{task['name']}.json"
            if tp.exists():
                print(f"  {task['name']}: done, skipping")
                continue
            t = time.time()
            res = run_task(cfg, task, C["X"][task["rows"]], device, B, ti)
            res["cache_config_hash"] = C["config_hash"]
            res["minutes"] = (time.time() - t) / 60
            tp.write_text(json.dumps(res, indent=2))
            v = res["validate"]
            print(f"  {task['name']}: " + "  ".join(
                f"{m} same {v[m]['same_data']['joint']['fwer']:.2f} held {v[m]['held_out']['joint']['fwer']:.2f}"
                for m in cfg["stage1"]["scoring_methods"])
                + f"  | naive passes {res['search']['null_mean_n_naive']:.0f} (exp {res['search']['expected_naive_false_passes']:.0f})"
                + f"  ({res['minutes']:.1f} min)", flush=True)
        del C
    print("\nall requested datasets done; run with --aggregate to pool")


if __name__ == "__main__":
    main()
