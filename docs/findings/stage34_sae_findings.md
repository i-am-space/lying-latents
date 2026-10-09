# Stages 3–4 on SAEBench findings — realised FDR of Gaussian and hurdle knockoffs on 8 datasets

Pre-registered in `config/preregistration.yaml` (`saebench_amendment_2`, committed before any run) ·
master seed `20260904` · Gemma-2-2B, Gemma Scope layer 20 / 16k, unchanged from SST-2. Script:
`src/stage4_repairs.py --experiment saebench`. Results: `results/saebench/stage34/` (per dataset: one
JSON, plus records and info for the `gauss` and `hurdle` arm groups), `results/saebench/stage34_summary.json`.

## Summary

1. **The SST-2 result does not replicate (SB3-R: false).** Under the cross-validated penalty,
   block-MVR Gaussian knockoffs have **no** Benjamini–Yekutieli breach of the FDR target on **any**
   of the 8 datasets (the rule needed ≥ 4). The equicorrelated construction breaches on one dataset
   (EuroParl, 2 of 24 cell-q combinations). On SST-2 the same procedure gave 8 and 3 breaches.
2. **The bias is consistent but small.** Block-MVR's realised FDR exceeds that of its own Gaussian
   control on **8 / 8** datasets, by **+0.007 to +0.026** (pooled over cells and q), significantly
   (p < 0.05, two-sided paired test) on 5. Pooled over datasets, block MVR sits **at** the targets
   (0.046 / 0.097 / 0.199 at q = 0.05 / 0.10 / 0.20) while the control sits below them
   (0.037 / 0.083 / 0.184). SST-2's excess was similar (+0.027) but started from a control already at
   the target (0.099 at q = 0.10), which is why it crossed. The 8 / 8 sign pattern (sign test
   p ≈ 0.008) is descriptive: no rule tests it.
3. **Hurdle knockoffs restore control on 1 / 8 datasets (SB4-R)** and breach once themselves
   (Amazon category, the smallest zero atom: amplitude 32, linear, k = 100, FDR 0.165 at q = 0.10).
   Their FDR is about the control's level, below block MVR's on 6 / 8 datasets but significantly only
   on Bias in Bios 1 (p = 0.005; EuroParl p = 0.050), and significantly *above* it on Amazon category
   (+0.010, p = 0.032). Their power matches block MVR's (−0.012 to +0.035).
4. **The zero atom does not predict the excess within SAEBench (SB-Z, exploratory).** Spearman
   ρ = −0.52 (p = 0.18) between median Pr(X = 0) and the block-MVR minus control FDR, over 8 points
   spanning 0.30–0.73, in the opposite direction to the zero-atom account. The largest excess is on
   GitHub code (Pr(X = 0) 0.49).
5. **The control is valid everywhere (SB3-V: 8 / 8)**, so every dataset is interpreted.

**What this establishes:** on 8 SAEBench domains, the non-exchangeability that Stage 2 detects on
every dataset translates into a small, consistent upward bias in realised FDR relative to a valid
Gaussian control, but not into a breach of the nominal target after multiplicity correction. The
SST-2 breach is not a general property of Gaussian knockoffs on these SAE latents. SST-2 is the only
setting run with a heavy zero atom (0.89 against ≤ 0.73) and the largest n (20,000 against
≤ 12,500); which of the two (or something else about SST-2) carries the breach is not identified
here. **What it does not establish:** that the hurdle construction is better in general (it is not,
here), or anything about other models (`saebench_amendment_3`, running).

## 1. Setup

| Item | Value |
|---|---|
| Data | each dataset's Stage 1–2 cache (all texts, its 2,048 retained latents), standardised; Ledoit–Wolf covariance |
| Labels | planted, not SAEBench's: `realistic` designs, amplitudes 8 and 32 × linear / interaction × k = 100, 300 (8 cells) |
| Arms | `gauss_real` (equicorrelated S), `gauss_mvr` (seeded block MVR), `hurdle` (SCIP), `gauss_ceiling_mvr` (Gaussian control: N(0, Σ̂) data with block-MVR knockoffs) |
| Penalty | every fit at the fixed λ = 0.02 and at the 5-fold CV λ (grid relative to λ_max of [X, X̃]); the rules use CV |
| Targets | q ∈ {0.05, 0.10, 0.20}: 24 cell-q combinations per arm and dataset |
| Replicates | 10 per dataset; seeds `sb34_knockoff` (arm, replicate), `sb34_planted` and `sb34_folds` (cell, replicate) |
| Breach | mean realised FDR above q, one-sided test, Benjamini–Yekutieli over the 24 combinations |
| Runtime | 16 jobs (8 datasets × 2 arm groups) on 2 L40S in parallel, sharing the GPUs with two other users' jobs: `gauss` 231–627 min, `hurdle` 785–966 min per dataset |

## 2. Rule outcomes

| Rule | Outcome |
|---|---|
| SB3-V: control has no BY breach at CV λ | **8 / 8 valid** |
| SB3-R: block MVR breaches on ≥ half of valid datasets | **0 / 8 → does not replicate**; equicorrelated: 1 / 8 (EuroParl) |
| SB4-R: hurdle has no breach and FDR significantly below block MVR | **1 / 8** (Bias in Bios 1) |
| SB-Z (exploratory): Spearman, median Pr(X = 0) vs MVR − control FDR | ρ = −0.52, p = 0.18 |

## 3. Per dataset (CV λ)

Mean realised FDR at q = 0.10 and mean power, over the 8 cells and 10 replicates. Breaches are BY
breaches over 24 cell-q combinations.

| dataset | Pr(X=0) | n | FDR: equi / MVR / hurdle / control | power: equi / MVR / hurdle / control | breaches (equi, MVR, hurdle, control) |
|---|---|---|---|---|---|
| EuroParl | 0.73 | 12,500 | 0.086 / 0.104 / 0.094 / 0.094 | 0.910 / 0.969 / 0.957 / 0.965 | 2, 0, 0, 0 |
| AG News | 0.67 | 10,000 | 0.083 / 0.095 / 0.087 / 0.087 | 0.890 / 0.947 / 0.940 / 0.939 | 0, 0, 0, 0 |
| Bias in Bios 1 | 0.59 | 12,500 | 0.077 / 0.109 / 0.094 / 0.093 | 0.860 / 0.943 / 0.944 / 0.933 | 0, 0, 0, 0 |
| Bias in Bios 2 | 0.59 | 12,500 | 0.041 / 0.095 / 0.091 / 0.083 | 0.719 / 0.920 / 0.931 / 0.913 | 0, 0, 0, 0 |
| Bias in Bios 3 | 0.58 | 12,500 | 0.077 / 0.102 / 0.099 / 0.088 | 0.873 / 0.948 / 0.949 / 0.940 | 0, 0, 0, 0 |
| GitHub code | 0.49 | 12,500 | 0.020 / 0.097 / 0.103 / 0.070 | 0.437 / 0.858 / 0.893 / 0.841 | 0, 0, 0, 0 |
| Amazon sentiment | 0.32 | 5,000 | 0.049 / 0.075 / 0.069 / 0.064 | 0.631 / 0.736 / 0.734 / 0.687 | 0, 0, 0, 0 |
| Amazon category | 0.30 | 12,500 | 0.055 / 0.095 / 0.113 / 0.083 | 0.810 / 0.955 / 0.969 / 0.955 | 0, 0, 1, 0 |
| **SST-2** (Stage 4 CV, for reference) | **0.89** | 20,000 | 0.100 / **0.125** / 0.096 / 0.099 | 0.965 / 0.990 / 0.974 / 0.991 | 3, **8**, 0, 0 |

Paired differences (pooled over cells and q; two-sided p):

| dataset | MVR − control FDR | hurdle − MVR FDR | hurdle − MVR power |
|---|---|---|---|
| EuroParl | +0.009 (0.106) | −0.009 (0.050) | −0.012 (< 0.001) |
| AG News | +0.007 (0.117) | −0.008 (0.183) | −0.007 |
| Bias in Bios 1 | +0.015 (0.013) | −0.017 (0.005) | +0.001 |
| Bias in Bios 2 | +0.011 (0.003) | −0.010 (0.115) | +0.011 |
| Bias in Bios 3 | +0.011 (0.051) | −0.005 (0.284) | +0.001 |
| GitHub code | +0.026 (< 0.001) | +0.001 (0.895) | +0.035 |
| Amazon sentiment | +0.011 (0.015) | −0.005 (0.215) | −0.002 |
| Amazon category | +0.012 (0.004) | +0.010 (0.032) | +0.013 |

The breaches:
- EuroParl, equicorrelated: amplitude 32, interaction, k = 100, at q = 0.05 (FDR 0.086, p_BY 0.003)
  and q = 0.10 (0.144, p_BY 0.033). The equicorrelated construction is otherwise conservative
  everywhere (pooled FDR 0.025 / 0.061 / 0.147), at a large power cost on GitHub code (0.44 vs 0.86).
- Amazon category, hurdle: amplitude 32, linear, k = 100, q = 0.10 (FDR 0.165).

## 4. Pooled over datasets (exploratory)

Mean realised FDR at q = 0.05 / 0.10 / 0.20, averaged over the 8 datasets (CV λ):

| arm | FDR | cell-q combinations with mean FDR > q, uncorrected (of 192) |
|---|---|---|
| equicorrelated | 0.025 / 0.061 / 0.147 | 28 |
| block MVR | 0.046 / 0.097 / 0.199 | 65 |
| hurdle | 0.047 / 0.094 / 0.184 | 49 |
| control | 0.037 / 0.083 / 0.184 | 35 |

Under the CV penalty a valid knockoff filter can still sit below q (Knockoff+ controls FDR at most
q, not exactly q), as the control does. Block MVR removes that slack and lands on the target; it
does not pass it. The uncorrected counts are per cell with 10 replicates each and are not tests.

At the fixed λ = 0.02 (not used by any rule), no arm breaches on any dataset, as on SST-2; hurdle
FDR is far below block MVR's (e.g. EuroParl 0.030 vs 0.093 at q = 0.10) at lower power (0.41 vs 0.50).

## 5. Reading against SST-2

- **What replicates:** the direction. Block MVR ≥ control in FDR on every dataset, as on SST-2, and
  hurdle ≤ block MVR on most.
- **What does not:** the breach, and the hurdle's advantage as a rule.
- **Why SST-2 differs is open.** SST-2 has the heaviest zero atom (0.89), the largest n (20,000),
  short sentences (Stage 2: the atom shrinks with text length under mean pooling), and a control
  already at the target under CV. Within SAEBench the excess does not grow with the atom (SB-Z), so
  the atom alone does not account for it over 0.30–0.73; a threshold above 0.73, or n, remain
  possible. Separating them needs a design that varies one at a time (not run).
- **The hurdle's one breach is on the smallest atom**, consistent with its known failure mode in
  the SST-2 stress tests (`stage4_amendment_5`: labels driven by activation size): with few exact
  zeros, more of the law sits in the positive part, where its log-normal size model is the weak link.

## 6. Deviations and notes

- Planned design, seeds, replicates and datasets were run unchanged; no rule was changed after results.
- The 16 jobs ran in parallel (two per dataset, eight per GPU) instead of in two queues; seeds depend
  only on (dataset, arm, cell, replicate), so the parallel run is identical in design. Hurdle
  logistic fits converged in ≥ 99.9% of latents on every dataset.
- Mean knockoff self-correlation (Amazon category): equicorrelated 0.965, block MVR 0.840, hurdle 0.709.
- Limits: one model and SAE (two more in `saebench_amendment_3`); planted labels on real latents; 10
  replicates per dataset, so per-cell FDR is noisy (se ≈ 0.005–0.010).
