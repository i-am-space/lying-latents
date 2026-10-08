# Stage 1 on SAEBench findings — search-then-validate across 35 sparse-probing tasks

Pre-registered in `config/preregistration.yaml` (`saebench_amendment_1`, committed before any
SAEBench text, cache or result existed) · master seed `20260904` · B = 500 label permutations per
task · Gemma-2-2B, Gemma Scope layer 20 / 16k, unchanged from SST-2. Scripts: `src/saebench/`.
Results: `results/saebench/stage1/` (one JSON per task, `gate__*.json` per dataset),
`results/saebench/stage1_summary.json`, `results/saebench/fig_sb1_validate.png`.

## Summary

1. **The SST-2 same-data result replicates (rule SB1-R, all three scoring methods).** Validating
   the selected top-10 on the rows used to select them certifies pure noise in a median **92% /
   100% / 100%** of permuted-label experiments (mean_diff / auroc / probe_weight). This is at least
   0.5 in 28 / 35 / 29 of the 35 tasks; the rule needed 24.
2. **Held-out validation is above nominal almost everywhere (SB1-H, descriptive).** The median
   held-out FWER is **0.146 / 0.156 / 0.150**, about 3× α = 0.05. The Wilson 95% interval lies
   entirely above 0.05 in **33 / 33 / 30 of 35** tasks. This matches SST-2 (0.274 / 0.096 / 0.106),
   where rule B3 traced it to the joint-ablation test's null hypothesis ("ablation changes
   nothing"), which is false even under permuted labels.
3. **The naive scan behaves as theory predicts (SB1-S).** Under the null it passes **98.1–105.9
   latents per experiment** (median 101.8; α·p = 102.4), with FWER ≥ 0.95 on all 35 tasks.
   Bonferroni FWER is **0.020–0.060** (5 tasks above 0.05, all within Monte Carlo error at B = 500).
   On the real labels, Westfall–Young passes **527–1,745 of 2,048 latents** (median 1,283).
4. **The power gate passes on all 8 datasets in both regimes (SB1-B)**, through auroc or
   probe_weight. mean_diff alone would fail the held-out gate on every dataset (see §2).
5. **New relative to SST-2:** on GitHub code and Bias in Bios set 2, held-out FWER for auroc
   rises to **0.42–0.69**, while same-data FWER *falls* to 0.4–0.6 for mean_diff and probe_weight.
   Held-out FWER also tracks the zero atom. Across tasks, Spearman ρ between held-out FWER and
   median Pr(X = 0) is −0.30 / −0.63 / −0.85; Amazon, which has the longest texts and a 0.30 zero
   atom, is worst for probe_weight (0.32–0.41). Not pre-registered; see §4.

**What this establishes:** across eight text domains, the conventional same-data validate step
certifies noise in most experiments; sample splitting reduces but does not remove the excess; the
naive scan gives about α·p false passes; and Bonferroni / WY control the search step.
**What it does not establish:** a calibrated validate step (none here has a valid null, as on
SST-2; a null-intervention reference is still needed); generality across models, layers or SAE
widths (one model, layer and SAE); 35 independent replications. Tasks within a dataset share
texts, and the three Bias in Bios sets share a source.

## 1. Setup

| Item | Value |
|---|---|
| Datasets | 8 SAEBench 0.6.0 sparse-probing datasets: Bias in Bios class sets 1–3, Amazon category, Amazon sentiment, GitHub code, AG News, EuroParl |
| Texts | SAEBench's own loader and filter; 2,000 train + 500 test per class; hashes in `results/saebench/tasks_manifest.json` |
| Encoding | `cache_activations.encode_texts` unchanged: ≤ 128 tokens, BOS / EOS / padding excluded, mean pooling |
| Latents | per dataset: firing rate ≥ 1% on its rows, top 2,048 by firing rate |
| Tasks | 35 one-vs-rest tasks (SAEBench construction, our `sb_tasks` seed), 2,500 positives + 2,500 negatives each |
| Validate step | `calibrate_validation.run_one` unchanged: fresh 70/30 split, L2 probe, top-10, joint-ablation paired log-loss test at α = 0.05 |
| Search step | Welch \|t\| on all 5,000 rows: uncorrected, Bonferroni, WY from the permutation max-\|t\| null |
| Power gate | per dataset on its first task's rows: 20 replicates, k = 10 planted latents, amplitude 3, linear |
| Runtime | 20 min for all 35 tasks and 8 gates on one shared GPU |

Per-dataset cache statistics (`results/saebench/cache_summary.json`):

| dataset | n | mean tokens | median Pr(X=0) | SAE expl. var. | mean L0 |
|---|---|---|---|---|---|
| EuroParl | 12,500 | 35.9 | 0.730 | 0.780 | 71.4 |
| AG News | 10,000 | 51.6 | 0.669 | 0.764 | 75.9 |
| Bias in Bios set 1 | 12,500 | 74.0 | 0.590 | 0.797 | 70.5 |
| Bias in Bios set 2 | 12,500 | 73.9 | 0.590 | 0.772 | 74.2 |
| Bias in Bios set 3 | 12,500 | 77.8 | 0.581 | 0.788 | 71.7 |
| GitHub code | 12,500 | 126.9 | 0.490 | 0.709 | 81.4 |
| Amazon sentiment | 5,000 | 126.7 | 0.319 | 0.791 | 79.0 |
| Amazon category | 12,500 | 126.7 | 0.304 | 0.788 | 78.9 |
| *SST-2 (reference)* | 67,349 | — | 0.891 | — | — |

## 2. Power gate (SB1-B)

Fraction of 20 planted replicates in which the joint test certifies (same-data / held-out), and
mean recall@10 of the planted latents:

| dataset | mean_diff | auroc | probe_weight |
|---|---|---|---|
| EuroParl | 0.80 / 0.10 (r 0.02) | 1.00 / 0.95 (r 0.33) | 1.00 / 1.00 (r 0.99) |
| Bias in Bios 1 | 0.85 / 0.30 (r 0.02) | 1.00 / 1.00 (r 0.43) | 1.00 / 1.00 (r 0.98) |
| Bias in Bios 2 | 0.85 / 0.20 (r 0.02) | 1.00 / 0.95 (r 0.36) | 1.00 / 1.00 (r 0.98) |
| Bias in Bios 3 | 0.85 / 0.20 (r 0.01) | 1.00 / 1.00 (r 0.34) | 1.00 / 1.00 (r 0.99) |
| Amazon category | 1.00 / 0.65 (r 0.06) | 1.00 / 1.00 (r 0.55) | 1.00 / 1.00 (r 0.99) |
| Amazon sentiment | 1.00 / 0.50 (r 0.04) | 1.00 / 1.00 (r 0.57) | 1.00 / 1.00 (r 0.99) |
| GitHub code | 0.35 / 0.40 (r 0.01) | 0.75 / 0.70 (r 0.16) | 1.00 / 1.00 (r 0.90) |
| AG News | 0.95 / 0.55 (r 0.06) | 1.00 / 1.00 (r 0.52) | 1.00 / 1.00 (r 0.99) |

The gate passes everywhere because probe_weight passes everywhere. As on SST-2, mean_diff almost
never ranks the planted latents in its top-10 (recall 0.01–0.06), so its same-data "power" comes
from ablating whichever latents it picks. The rule is per dataset, not per method, so all three
methods' FWERs are interpreted. A method's type-I error rate does not depend on its power, but
mean_diff's held-out FWERs should be read with its weak gate in mind.

## 3. Validate-step FWER under permuted labels (SB1-R, SB1-H)

Median per dataset (same-data / held-out); per-task values in `stage1_summary.json` → `per_task`
and in the figure:

| dataset (tasks) | mean_diff | auroc | probe_weight |
|---|---|---|---|
| EuroParl (5) | 0.92 / 0.12 | 1.00 / 0.07 | 1.00 / 0.05 |
| Bias in Bios 1 (5) | 0.95 / 0.09 | 1.00 / 0.11 | 1.00 / 0.15 |
| Bias in Bios 2 (5) | 0.55 / 0.34 | 0.90 / 0.57 | 0.59 / 0.16 |
| Bias in Bios 3 (5) | 0.96 / 0.08 | 1.00 / 0.10 | 1.00 / 0.15 |
| Amazon category (5) | 1.00 / 0.16 | 1.00 / 0.18 | 1.00 / 0.41 |
| Amazon sentiment (1) | 1.00 / 0.14 | 1.00 / 0.15 | 1.00 / 0.32 |
| GitHub code (5) | 0.49 / 0.38 | 0.84 / 0.64 | 0.42 / 0.17 |
| AG News (4) | 0.99 / 0.14 | 1.00 / 0.08 | 1.00 / 0.10 |
| **All 35, median** | **0.92 / 0.146** | **1.00 / 0.156** | **1.00 / 0.150** |
| *SST-2 (reference)* | *0.984 / 0.274* | *1.000 / 0.096* | *1.000 / 0.106* |

- **SB1-R:** same-data FWER ≥ 0.5 in 28 / 35 / 29 tasks against a threshold of 24 → **replicates
  for all three methods**. Below 0.5: mean_diff on all five GitHub code tasks, Bias in Bios 2
  classes 13 / 19 and Bias in Bios 1 class 6; probe_weight on all five GitHub code tasks and Bias in
  Bios 2 class 19; auroc on none. Even these are at 0.39–0.49, about 8–10× nominal.
- **SB1-H:** held-out Wilson interval entirely above 0.05 in **33 / 33 / 30 of 35** tasks. The
  exceptions are Bias in Bios 1 class 1 and Bias in Bios 3 class 26 (mean_diff, 0.064–0.066),
  EuroParl en / es (auroc) and all five EuroParl tasks (probe_weight, 0.038–0.060). EuroParl is the
  only dataset where held-out validation is near nominal.
- **Real labels:** certified on same-data in 34 / 35 / 35 tasks and held-out in 30 / 30 / 34. The
  held-out failures are AG News classes 0–2, Bias in Bios 3 classes 20–21 and Bias in Bios 2 class
  11. Given the null FWERs above, real-label certification carries little information on its own.

## 4. Unregistered observations (exploratory; no rule tests them)

1. **Held-out and same-data FWER move in opposite directions across tasks.** Spearman ρ across the
   35 tasks: −0.52 / −0.83 / −0.29. The datasets with the worst held-out FWER (GitHub code, Bias in
   Bios 2) have the lowest same-data FWER, so the gap between regimes narrows. One reading is the
   SST-2 B3 mechanism: under the null, ablation shifts the probe's loss by a systematic, probe-
   dependent amount, so the held-out test picks it up regardless of selection. Where that shift is
   larger, the regimes look more alike. This is a hypothesis; a null-intervention control (random
   latents) on these tasks would test it.
2. **Held-out FWER tracks the zero atom / text length**, most strongly for probe_weight (ρ with
   median Pr(X = 0): −0.85; with mean tokens: +0.74). The 35 tasks are clustered in 8 datasets, so
   these correlations are effectively over 8 points and are descriptive only.
3. GitHub code is the weakest dataset for auroc and mean_diff, both in the gate and under the null.
   It also has the lowest SAE explained variance (0.709) and the most latents firing on ≥ 1% of rows
   (16,162).

## 5. Deviations and notes

- **No deviations from `saebench_amendment_1`** in tasks, latents, thresholds, seeds or permutation
  counts (SB-T).
- **Code fixes during the run, before any result:** the cache step called the decoder without the
  LM head (`model.model`), to avoid an OOM from 256k-vocab logits. Hidden states are identical, and
  only the cache step's memory changed. The cache batch size was set to 32 (not part of the
  config hash).
- **Interpretation constraints from the amendment apply:** one model / layer / SAE; ablation is in a
  probe on cached latents, not the model; negatives are drawn with our seed, not SAEBench's global
  torch RNG; tasks are not independent.
- The figure's row labels truncate the Bias in Bios set name; the set can be read from the class
  ids (0–9: set 1, 11–19: set 2, 20–26: set 3).
- **Stage 2 (same amendment) is reported separately.** Every dataset is below the SB2-P premise of
  median Pr(X = 0) ≥ 0.80, so SB2-R's replication clause is vacuous. A reporting bug in
  `src/saebench/stage2.py --aggregate` would have printed `replicates: True` for an empty set; it
  was fixed before any Stage 2 result was aggregated and now reports `None` with `n_eligible: 0`.
