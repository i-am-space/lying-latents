"""Collect the results of the final experimental design into one file.

Reads only the result files of the final design (CV-lambda knockoff runs, the final Stage 5
specification) and writes results/FINAL_RESULTS.json. Every section names its source files.
Superseded runs are deliberately not read: stage3_planted_fdr / stage3_v2 / p2048 / largek / dims
(fixed lambda, replaced by stage3_cvlambda_*), stage4_hurdle / stage4_mvr (replaced by stage4_cv),
and the first Stage 5 run (`tasks` in stage5_summary.json, replaced by `tasks_v2`).

Usage: python scripts/collect_final_results.py
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = R / "FINAL_RESULTS.json"

MODELS = {
    "gemma2_2b": {"name": "google/gemma-2-2b", "sae": "Gemma Scope, layer 20, 16k, JumpReLU",
                  "saebench": R / "saebench", "stage5": R / "stage5"},
    "gemma3_1b": {"name": "google/gemma-3-1b-pt", "sae": "Gemma Scope 2, layer 22, 16k, JumpReLU",
                  "saebench": R / "models/gemma3_1b/saebench", "stage5": R / "models/gemma3_1b/stage5"},
    "pythia70m": {"name": "EleutherAI/pythia-70m-deduped", "sae": "ReLU SAE",
                  "saebench": R / "models/pythia70m/saebench", "stage5": None},
}
SHORT = {
    "Helsinki-NLP/europarl": "europarl",
    "LabHC/bias_in_bios_class_set1": "bias_in_bios_1",
    "LabHC/bias_in_bios_class_set2": "bias_in_bios_2",
    "LabHC/bias_in_bios_class_set3": "bias_in_bios_3",
    "canrager/amazon_reviews_mcauley_1and5": "amazon_category",
    "canrager/amazon_reviews_mcauley_1and5_sentiment": "amazon_sentiment",
    "codeparrot/github-code": "github_code",
    "fancyzhx/ag_news": "ag_news",
}
STAGE5_TASKS = ["sst2", "ag_news", "bias_in_bios", "europarl"]
NS = ["100", "250", "500", "1000", "2000"]


def rnd(x, nd=4):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: rnd(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [rnd(v, nd) for v in x]
    return x


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def rel(p):
    return os.path.relpath(p, ROOT)


def contrast(c):
    return None if c is None else {"diff": c.get("mean_diff"), "se": c.get("se"), "p": c.get("p_two_sided")}


# ---------------------------------------------------------------------------------------------
# 1. Selection reuse (Stage 1)
# ---------------------------------------------------------------------------------------------
def stage1():
    cal = load(R / "stage1_calibration.json")
    val = load(R / "stage1_validation.json")
    welch = cal["v2"]["welch"]
    sst2 = {
        "_source": [rel(R / "stage1_calibration.json"), rel(R / "stage1_validation.json")],
        "n_texts": cal["n"], "n_latents": cal["p"], "n_permutations": cal["n_permutations"],
        "naive_scan_null": {
            "fwer_naive": welch["fwer_naive"],
            "mean_false_passes_naive": welch["null_mean_n_naive"],
            "expected_false_passes": welch["expected_naive_false_passes"],
            "fwer_bonferroni": welch["fwer_bonferroni"],
        },
        "top10_validation_under_null": {
            m: {"same_data_rate": val["null"][m]["same_data"]["joint"]["fwer"],
                "held_out_rate": val["null"][m]["held_out"]["joint"]["fwer"]}
            for m in val["null"]
        },
        "nominal_alpha": val["alpha"],
    }
    saeb = {}
    for key, m in MODELS.items():
        f = m["saebench"] / "stage1_summary.json"
        s = load(f)
        if s is None:
            continue
        saeb[key] = {
            "_source": rel(f),
            "n_tasks": s["n_tasks"],
            "naive_scan_null": s["search"],
            "top10_validation_under_null": {
                sc: {"same_data_median": v["same_data"]["median_fwer"],
                     "same_data_range": [v["same_data"]["min_fwer"], v["same_data"]["max_fwer"]],
                     "held_out_median": v["held_out"]["median_fwer"],
                     "held_out_range": [v["held_out"]["min_fwer"], v["held_out"]["max_fwer"]],
                     "held_out_tasks_ci_above_alpha": v["held_out"]["n_tasks_ci_lower_above_alpha"]}
                for sc, v in s["validate"].items()
            },
        }
    return {"sst2_gemma2": sst2, "saebench": saeb}


# ---------------------------------------------------------------------------------------------
# 2. Exchangeability
# ---------------------------------------------------------------------------------------------
def swap_mean(rows, size):
    v = [r["auc"] for r in rows if r["swap_size"] == size]
    return sum(v) / len(v) if v else None


def exch_summary(s):
    return {"auc_swap1": s["swap_auc"].get("1"), "auc_swap10": s["swap_auc"].get("10"),
            "auc_swap_all": s["swap_auc"].get("all"), "auc_swap0_null": s["swap_auc"].get("0"),
            "null_q975": s.get("null_q975"), "zero_rule_accuracy": s.get("zero_mass_median_accuracy"),
            "self_correlation": s.get("knockoff_self_correlation")}


def exchangeability():
    a = load(R / "stage2_audit.json")
    rep = load(R / "stage4_repairs.json")["exchangeability"]
    sizes = sorted({r["swap_size"] for r in a["real"]["swap"]}, key=lambda x: (isinstance(x, str), x))
    sst2 = {
        "_source": [rel(R / "stage2_audit.json"), rel(R / "stage4_repairs.json")],
        "median_zero_mass": a["real"]["zero_mass"]["median_p_zero_real"],
        "harness_validation": {"auc_mean": a["calibration"]["auc_mean"],
                               "mmd_uniformity_ks_p": a["calibration"]["mmd_uniformity_ks_p"],
                               "passed": a["calibration"]["PASSED"]},
        "gaussian_equicorrelated": {
            "zero_rule_accuracy": a["real"]["zero_mass"]["median_accuracy"],
            "auc_by_swap_size": {str(k): swap_mean(a["real"]["swap"], k) for k in sizes},
        },
        "gaussian_control": {"auc_by_swap_size": {str(k): swap_mean(a["gaussian_control"]["swap"], k)
                                                  for k in sizes}},
        "full_swap_auc": rep["full_swap_auc"],
        "full_swap_mmd_p": rep["full_swap_mmd_p"],
    }
    out = {"sst2_gemma2": sst2, "saebench": {}}
    for key, m in MODELS.items():
        f = m["saebench"] / "stage2_summary.json"
        s = load(f)
        if s is None:
            continue
        per = {}
        for ds, v in s["per_dataset"].items():
            per[SHORT[ds]] = {
                "median_zero_mass": v["median_p_zero"],
                "gaussian_equicorrelated": {
                    "zero_rule_accuracy": v["zero_mass_median_accuracy"], "auc_swap1": v["auc_swap1"],
                    "auc_swap10": v["auc_swap10"], "auc_swap_all": v["auc_swap_all"],
                    "null_q975": v["null_q975"]},
                "gaussian_control": {"auc_swap1": v["control_auc_swap1"],
                                     "auc_swap_all": v["control_auc_swap_all"]},
            }
        # deep samplers and block MVR, from the per-dataset knockoff summaries
        st = load(m["saebench"] / "stage34_summary.json")
        if st:
            for ds, v in st["datasets"].items():
                ex = (v.get("DK_v2") or v.get("DK") or {}).get("DK1_exchange", {})
                for arm in ("gauss_mvr", "deep", "deep2"):
                    if arm in ex:
                        per.setdefault(SHORT[ds], {})[arm] = exch_summary(ex[arm])
        # hurdle (and thresholded-Pythia) exchange tests, one file per run
        for sub, tag in (("stage34", ""), ("stage34_matched_thr0.19", "_thr0.19")):
            for fx in sorted((m["saebench"] / sub).glob("*_exchange.json")) if (m["saebench"] / sub).exists() else []:
                x = load(fx)
                ds = SHORT[x["dataset"]]
                # Gaussian reruns duplicate the Stage 2 audit / DK1 values where those exist
                if x.get("group") != "hurdle" and not tag and "gauss_mvr" in per.get(ds, {}):
                    continue
                for arm, body in x["arms"].items():
                    if "summary" in body:
                        e = exch_summary(body["summary"])
                        e["_source"] = rel(fx)
                        per.setdefault(ds, {})[arm + tag] = e
        out["saebench"][key] = {"_source": rel(f), "datasets": per}
    return out


# ---------------------------------------------------------------------------------------------
# 3-5. Knockoff FDR on the planted benchmark (CV lambda), samplers, observational e-values
# ---------------------------------------------------------------------------------------------
def sst2_knockoffs():
    stress = load(R / "stage3_cvlambda_stress.json")
    largek = load(R / "stage3_cvlambda_largek.json")
    fixed_stress = load(R / "stage3_stress.json")
    cv = load(R / "stage4_cv.json")
    hs = load(R / "stage4_stress.json")
    rep = load(R / "stage4_repairs.json")["summary"]["arms"]
    return {
        "_source": [rel(R / p) for p in ("stage3_cvlambda_stress.json", "stage3_cvlambda_largek.json",
                                         "stage3_stress.json", "stage4_cv.json", "stage4_stress.json",
                                         "stage4_repairs.json")],
        "strong_signals_k10_30": {
            "cells": stress["analysis"]["pooled_fdr_real_minus_control_cv"]["n_cells"],
            "breaches_BY": stress["analysis"]["breaches_BY"],
            "fdr_real_minus_control": contrast(stress["analysis"]["pooled_fdr_real_minus_control_cv"]),
            "mean_power_q0.10": stress["analysis"]["mean_pow_q0.10"]},
        "weak_matched_signals_k100_300": {
            "cells": largek["analysis"]["pooled_fdr_real_minus_control_cv"]["n_cells"],
            "breaches_BY": largek["analysis"]["breaches_BY"],
            "fdr_real_minus_control": contrast(largek["analysis"]["pooled_fdr_real_minus_control_cv"]),
            "mean_power_q0.10": largek["analysis"]["mean_pow_q0.10"]},
        "null_coin_lasso": {
            "real_amplitude_ge5": fixed_stress["rules"]["R2"]["real_pooled_frac_pos_amp_ge5"][0],
            "gaussian_control": fixed_stress["rules"]["R1"]["control_pooled_frac_pos"][0],
            "_note": "mechanism diagnostic; computed on the strong-signal design"},
        "samplers_cv": {
            "_design": "weak matched signals, CV lambda",
            "by_arm": cv["by_lambda"]["cv"],
            "hurdle_minus_block_mvr": rnd({k: contrast(v) for k, v in cv["hurdle_minus_mvr"]["cv"].items()}),
            "evalue_minus_block_mvr": rnd({k: contrast(v) for k, v in cv["evalue_minus_mvr"]["cv"].items()}),
        },
        "hurdle_stress_cv": {"breaches_BY": hs["S2_hurdle_breaches_cv"],
                             "hurdle_minus_block_mvr": {d: {k: contrast(c) for k, c in v.items()}
                                                        for d, v in hs["S3_hurdle_minus_mvr_cv"].items()}},
        "binarised": {
            "_design": "strong signals, fixed lambda 0.02 (arm not run under CV)",
            "mean_fdr": rep["binarised"]["mean_fdr"],
            "mean_fdr_gaussian_equicorrelated": rep["gauss_real"]["mean_fdr"]},
    }


def per_dataset_knockoffs(v):
    d = {
        "median_zero_mass": v.get("median_zero_mass"),
        "breaches_BY": v.get("breaches_cv"),
        "block_mvr_minus_control_fdr": contrast(v.get("mvr_minus_control_fdr_cv")),
        "mean_fdr_q0.10": v.get("mean_fdr_q0.10_cv"),
        "mean_power": v.get("mean_power_cv"),
    }
    if "coin" in v:
        d["null_coin"] = {arm: {f: c[f]["rate"] for f in ("all", "linear", "interaction") if f in c}
                          for arm, c in v["coin"].items()}
        d["null_coin_block_mvr_minus_control"] = {f: contrast(v["coin_mvr_minus_control"][f])
                                                  for f in ("all", "linear", "interaction")}
    if "DK_v2" in v:
        dk2 = v["DK_v2"]
        d["deep_2nd_order_exact"] = {
            "minus_block_mvr": {k: contrast(c) for k, c in dk2["DK3_deep2_minus_gauss_mvr"].items()},
            "minus_residual_deep": {k: contrast(c) for k, c in dk2["DK3_deep2_minus_deep"].items()},
            "repairs": dk2.get("DK2_repairs")}
    if "EV2_evalue_minus_gauss_mvr" in v:
        d["observational_evalue"] = {
            "power_minus_block_mvr": contrast(v["EV2_evalue_minus_gauss_mvr"]["cv"]["ko_pow"]),
            "fdr_minus_block_mvr": contrast(v["EV2_evalue_minus_gauss_mvr"]["cv"]["ko_fdr"]),
        }
    if "V_evalue_plus" in v:
        vp = v["V_evalue_plus"]
        d["evalue_variants"] = {
            "no_cap_power_minus_E0": contrast(vp["V2_power_gain_over_E0"]["evalue_nocap"]),
            "avg5_power_minus_E0": contrast(vp["V2_power_gain_over_E0"]["evalue_k5"]),
            "avg5_gap_to_hurdle_recovered": vp["V3"]["evalue_k5"]["gap_recovered"]}
    return d


def saebench_knockoffs():
    out = {}
    runs = [("original_cells", "stage34_summary.json"),
            ("matched_amplitude_sweep", "stage34_matched_summary.json"),
            ("matched_sweep_thresholded_0.19", "stage34_matched_thr0.19_summary.json")]
    for key, m in MODELS.items():
        for run, fn in runs:
            s = load(m["saebench"] / fn)
            if s is None:
                continue
            agg = s["aggregate"]
            out.setdefault(key, {})[run] = {
                "_source": rel(m["saebench"] / fn),
                "aggregate": {k: agg[k] for k in agg if k.startswith(("SB3", "SB4", "DK", "EV1", "M2", "M3"))},
                "datasets": {SHORT[ds]: per_dataset_knockoffs(v) for ds, v in s["datasets"].items()},
            }
    return out


# ---------------------------------------------------------------------------------------------
# 6. Interventional FDR control (Stage 5)
# ---------------------------------------------------------------------------------------------
def grid_max(node, leaf):
    """Max of `leaf` over every nested entry (spec, delta grid, n, alpha) below `node`."""
    best = 0.0
    if isinstance(node, dict):
        for k, v in node.items():
            if k == leaf and isinstance(v, (int, float)):
                best = max(best, v)
            else:
                best = max(best, grid_max(v, leaf))
    return best


def full_grid_validity(ev):
    """Max FDR against the oracle and max stress-test rejection rate, from evaluate2.json."""
    out = {}
    for iv, body in ev["interventions"].items():
        tests = {sp: {k: v for k, v in b.items() if k != "zero_null"} for sp, b in body["resplit"].items()}
        out[iv] = {"max_fdr_vs_oracle": grid_max(tests, "fdr"),
                   "max_stress_p_any_rejection": grid_max(body["stress"], "p_any_rejection")}
    return out


def stage5():
    out = {"_spec": {"K": 50, "bet_cap": 0.9, "delta": "0.05 * SD(m) on the pilot split",
                     "comparison_setting": {"K": 250, "bet_cap": 0.5}, "q": 0.1,
                     "intervention_primary": "resample", "sensitivity": ["zero", "mean"]}}
    b3 = load(R / "stage5_baselines3_summary.json") or {"tasks": {}}
    for key, m in MODELS.items():
        if m["stage5"] is None:
            continue
        f = m["stage5"] / "stage5_summary.json"
        s = load(f)
        tasks = {}
        for t in STAGE5_TASKS:
            v = s["tasks_v2"][t]
            pilot = load(m["stage5"] / t / "pilot.json")
            r = v["resample"]
            n_true = r["counts"]["tau_above_delta"]

            def power_curve(iv, spec):
                return {n: iv[spec]["primary"][n]["power"] for n in NS if n in iv[spec]["primary"]}

            def max_fdr(iv):
                return max(iv[sp]["primary"][n]["fdr"] for sp in ("v2", "prereg") for n in iv[sp]["primary"])

            bl = {k: b for k, b in r.get("baselines2", {}).items() if isinstance(b, dict)}
            tasks[t] = {
                "balanced_accuracy": pilot["balanced_accuracy"],
                "delta": v["delta"],
                "candidates": r["counts"]["candidates"],
                "n_tau_positive": r["counts"]["tau_positive"],
                "n_tau_above_2se": r["counts"]["tau_above_2se"],
                "n_tau_above_delta": n_true,
                "n_tau_above_delta_in_top50": r["counts"]["tau_above_delta_in_top_v2"],
                "max_fdr_all_n_both_settings": max_fdr(r),
                "full_grid_validity": full_grid_validity(load(m["stage5"] / t / "evaluate2.json")),
                "breaches": r["v2"]["breaches"] + r["prereg"]["breaches"],
                "stress_p_any_rejection": r["v2"]["stress_p_any_rejection"],
                "power_by_n": power_curve(r, "v2"),
                "power_by_n_comparison_K250_cap0.5": power_curve(r, "prereg"),
                "zero_null_fdr_by_n": {n: r["v2"]["zero_null"][n]["fdr_vs_min_effect"] for n in NS},
                "zero_null_fdr_vs_own_target_by_n": {n: r["v2"]["zero_null"][n]["fdr_vs_tau_above_2se"]
                                                     for n in NS},
                "power_n1000_by_intervention": {iv: v[iv]["v2"]["primary"].get("1000", {}).get("power")
                                                for iv in ("resample", "zero", "mean")},
                "max_fdr_by_intervention": {iv: max_fdr(v[iv]) for iv in ("resample", "zero", "mean")},
                "delta_grid": pilot.get("delta_grid"),
                "observational_baselines_q0.1": {
                    name: {"mean_selected": b["mean_selected"],
                           "share_draws_selecting": b["share_draws_selecting"],
                           "power_vs_oracle": (b["mean_true_selected"] / n_true) if n_true else None,
                           "fdr_vs_min_effect": b["fdr_vs_min_effect"],
                           "fdr_vs_tau_above_2se": b["fdr_vs_tau_above_2se"],
                           "fdr_vs_tau_positive": b["fdr_vs_tau_positive"]}
                    for name, b in ((k.replace("_q0.1", ""), bl[k]) for k in bl if k.endswith("_q0.1"))
                },
            }
            b3t = b3["tasks"].get(f"{m['name']}/{t}")
            if b3t and b3t.get("summary"):
                tasks[t]["knockoff_samplers_vs_oracle_q0.1"] = {
                    arm: q.get("0.1") for arm, q in b3t["summary"].items()}
            else:
                tasks[t]["knockoff_samplers_vs_oracle_q0.1"] = "PENDING"
        out[key] = {"_source": [rel(f), rel(m["stage5"]) + "/<task>/pilot.json",
                                rel(m["stage5"]) + "/<task>/evaluate2.json",
                                rel(R / "stage5_baselines3_summary.json")],
                    "tasks": tasks}
    return out


def _fps(d):
    return None if d is None else {"fdr": d["fdr"], "power": d["power"], "mean_selected": d["mean_selected"]}


def khera():
    """Khera (2026) run on our zero-ablation and resampling effects (scripts/stage5_khera_baseline.py), next to ours
    on the same re-splits; plus the attribution top-k baseline (Marks et al.) against the same oracle."""
    f = R / "stage5_khera_baseline.json"
    kb = load(f)
    if kb is None:
        return "PENDING: run scripts/stage5_khera_baseline.py"
    iv = kb["interventions"]
    out = {"_source": [rel(f), "results/<model>/stage5/<task>/evaluate.json (attribution top-k)"],
           "_spec": {"khera": "D_j = drop(random other candidate) - drop(j), comparator redrawn per input from the "
                              "tested set; aGRAPA, cap 0.5/B; e-BH at alpha 0.05. Reimplementation on our candidates "
                              "and metric (probability; Khera: logit difference, features active on >= 20% of prompts)",
                     "setups": {"test50_pool50": "K = 50, comparator from those 50",
                                "test250_pool250": "K = 250, comparator from those 250",
                                "sensitivity_test50_pool250": "K = 50, comparator from 250 (not Khera's design)"},
                     "ours": "resampling, K = 50, cap 0.9",
                     "caution": "n = 4000 leaves 1000 inputs for the oracle; re-splits share one finite pool",
                     "scored_against": "tau_j > delta (the minimum-effect truth); khera_fdr_own_target against "
                                       "tau_j > mean tau of the other latents in the comparator pool"}}
    for key, m in MODELS.items():
        if m["stage5"] is None:
            continue
        tasks = {}
        for t in STAGE5_TASKS:
            z, r = iv["zero"][m["name"]][t], iv["resample"][m["name"]][t]
            def cell(res, spec, n, a, k):
                return res["resplit"][spec].get(str(n), {}).get(str(a), {}).get(k)

            def setup(res, spec, ns):
                return {n: {"khera": _fps(cell(res, spec, n, 0.05, "khera_vs_delta")),
                            "khera_fdr_own_target": (cell(res, spec, n, 0.05, "khera_vs_own_target") or {}).get("fdr")}
                        for n in ns}
            same = ("1000", "2000", "3000", "4000")
            specs = {"test50_pool50": "v2_cap0.5", "test250_pool250": "prereg",
                     "sensitivity_test50_pool250": "v2_cap0.5_pool250"}
            ours = {n: {"alpha0.1": _fps(cell(r, "v2", n, 0.1, "ours_vs_delta")),
                        "alpha0.05": _fps(cell(r, "v2", n, 0.05, "ours_vs_delta"))} for n in same}
            ev = load(m["stage5"] / t / "evaluate.json")
            topk = {k: {"fdr": v["fdr"], "power": v["power"]}
                    for k, v in ev["interventions"]["resample"]["baselines"].items() if k.startswith("atp_top")}
            tasks[t] = {
                "delta": r["delta"],
                "comparator_pools_resample": {k: r["counts_full_pool"][sp] for k, sp in specs.items()},
                "khera_zero_ablation": {k: setup(z, sp, ("1000",)) for k, sp in specs.items()},
                "ours_zero_ablation_same_K_cap_n1000_alpha0.05": _fps(cell(z, "v2_cap0.5", 1000, 0.05, "ours_vs_delta")),
                "khera_resample_by_n": {k: setup(r, sp, same) for k, sp in specs.items()},
                "ours_resample_by_n": ours,
                "attribution_topk_vs_oracle": topk,
            }
        out[key] = tasks
    return out


def pending():
    want = {
        "hurdle exchange tests": [
            ("gemma2_2b", ["ag_news", "amazon_sentiment", "bias_in_bios_3"]),
            ("gemma3_1b", ["europarl", "bias_in_bios_1", "amazon_category"]),
            ("pythia70m", ["europarl", "amazon_category"]),
        ],
    }
    ex = exchangeability()["saebench"]
    miss = []
    for model, dss in want["hurdle exchange tests"]:
        for ds in dss:
            if "hurdle" not in ex.get(model, {}).get("datasets", {}).get(ds, {}):
                miss.append(f"hurdle exchange: {model} {ds}")
    if "hurdle_thr0.19" not in ex.get("pythia70m", {}).get("datasets", {}).get("bias_in_bios_1", {}):
        miss.append("hurdle exchange (thresholded 0.19): pythia70m bias_in_bios_1")
    b3 = load(R / "stage5_baselines3_summary.json") or {}
    for t in ("sst2", "ag_news"):
        if not (load(R / "stage5" / t / "evaluate3.json") or {}).get("draws"):
            miss.append(f"oracle scoring of knockoff-sampler picks: gemma2_2b {t}")
    if not b3.get("B3_1") or "undefined" in str(b3["B3_1"].get("note", "")):
        miss.append("hurdle - block MVR contrast on oracle effects (needs the two runs above)")
    g3 = MODELS["gemma3_1b"]["stage5"]
    for t in STAGE5_TASKS:
        b = load(g3 / t / "baselines3.json") or {}
        if "baselines" in b and not (load(g3 / t / "evaluate3.json") or {}).get("draws"):
            miss.append(f"oracle scoring of knockoff-sampler picks: gemma3_1b {t} "
                        f"(baselines3 done, {len(b.get('extra', []))} new latents need delta3)")
    return miss


def main():
    res = {
        "_about": ("Final results of the Lying Latents project, extracted from the final-design result "
                   "files by scripts/collect_final_results.py. All knockoff FDR results use the "
                   "cross-validated lasso penalty; Stage 5 uses K = 50, bet cap 0.9. Superseded runs "
                   "are not included."),
        "models": {k: {"name": m["name"], "sae": m["sae"]} for k, m in MODELS.items()},
        "1_selection_reuse": stage1(),
        "2_exchangeability": exchangeability(),
        "3_knockoff_fdr_sst2": sst2_knockoffs(),
        "4_knockoff_fdr_saebench": saebench_knockoffs(),
        "5_interventional_fdr": stage5(),
        "6_khera_comparison": khera(),
        "pending": pending(),
    }
    OUT.write_text(json.dumps(rnd(res), indent=1))
    print(f"wrote {rel(OUT)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
