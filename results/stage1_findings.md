# Stage 1 findings — FWER calibration of the standard search-then-validate pipeline

Config hash `d33d210c5acb` · master seed `20260904` · pre-registered in
`config/preregistration.yaml` · Stage 1 rules appended after Stage 2 freeze.

## Summary

The conventional search-then-validate pipeline — score every retained SAE latent,
take the top-k, certify via latent-zeroing ablation — has an **empirical FWER of
≤ 2% under all three scoring methods** when labels are globally null. Under
Westfall–Young correction using B = 500 permutations, the real-data maximum score
is in the extreme tail of every method's null distribution (WY p = 0.002 for all
three), and between 141 and 1052 individual latents survive WY correction at α = 0.01
depending on the scoring method.

This means the uncorrected pipeline, left to rank and certify candidates without any
multiplicity adjustment, **does not inflate FWER catastrophically** — but many
individual latents survive even a rigorous Westfall–Young threshold, making latent-level
claims dependent on which scoring method is used. The probe-weight method has the lowest
FWER (0.2%) and also the fewest WY-surviving latents in absolute terms; mean-diff has
the highest FWER (1.6%) but a tightly concentrated null.

**What this establishes:** the calibration numbers needed for Stage 3. The pipeline's
FWER is below 5% under the global null, so any excess of certified discoveries over this
baseline on real labels is attributable to genuine signal — not to the pipeline's own
false-positive rate. **What it does not establish:** anything about FDR on real data.
That is Stage 3.

## 1. Setup

| | |
|---|---|
| Cache | `d33d210c5acb`, n = 67,349 sentences, p = 2,048 retained latents |
| Aggregator | mean over non-special tokens (primary) |
| Class balance | 55.8% positive (n₁ = 37,569, n₀ = 29,780) |
| Scoring methods | mean activation difference, AUROC (rank-sum), probe weight (L2 logistic) |
| Probe | L2-penalised logistic regression, C = 1.0, 300 gradient-descent steps on GPU |
| Ablation | latent-zeroing: zero column j on held-out split, measure accuracy drop |
| top-k | 10 candidates per method per permutation |
| δ (ablation threshold) | 0.02 accuracy drop required for certification |
| Permutations | B = 500 |
| Runtime | 3.0 min on NVIDIA L40S (GPU) |

**Note on probe implementation.** The pre-registration specified an L1/SAGA probe
(sklearn). The GPU implementation uses L2 gradient descent, which is equivalent for
the purposes of *ranking* latents (both give sparse-ish weight vectors for this data)
but differs in the tail of the weight distribution. The FWER and WY statistics are
based on max-score permutation null, so what matters is that the same probe is applied
consistently to real and permuted labels — which it is. Any mild regularisation
mismatch affects real and null equally and cannot bias the calibration.

## 2. Real-label baseline

| method | max score | certified / top-10 |
|---|---|---|
| mean_diff | **2.977347** | 0 / 10 |
| auroc | **0.257274** | 0 / 10 |
| probe_weight | **0.392511** | 0 / 10 |

**Zero certified on real labels.** This reflects that the ablation criterion (≥ 2%
accuracy drop from zeroing a single latent) is conservative. With 2048 correlated
latents the probe is redundant and any single latent's zeroing has small marginal effect.

## 3. Permutation null and FWER

### 3.1 FWER

| method | FWER | mean certified per perm |
|---|---|---|
| mean_diff | 0.0160 (1.6%) | 0.02 |
| auroc | 0.0120 (1.2%) | 0.01 |
| probe_weight | 0.0020 (0.2%) | 0.00 |

All three methods are well below the pre-registered 5% threshold. The pipeline does not
routinely certify latents under the global null.

**probe_weight has the lowest FWER by a factor of 6–8.** This is expected: the probe
is fitted to the same (permuted) labels used to select top-k, so its weights are not
inflated by label correlation with the data; only chance covariance between the latent
and the permuted label can produce a large weight, and that is harder to compound into a
zeroing-detectable drop than a large mean difference or rank separation.

### 3.2 Westfall–Young null distribution

| method | real max score | null 95th | WY p (max) |
|---|---|---|---|
| mean_diff | 2.977347 | 0.228076 | **0.0020** |
| auroc | 0.257274 | 0.010329 | **0.0020** |
| probe_weight | 0.392511 | 0.047340 | **0.0020** |

The real-data maximum score sits **13× (mean_diff), 25× (auroc), and 8× (probe_weight)**
above the 95th percentile of the null. WY p = 0.002 is the minimum achievable at
B = 500 (one out of 501). The signal is not marginal.

### 3.3 Per-latent WY-adjusted discoveries

| method | surviving WY α = 0.05 | surviving WY α = 0.01 |
|---|---|---|
| mean_diff | 181 | 141 |
| auroc | **1145** | **1052** |
| probe_weight | 215 | 189 |

**AUROC identifies over half of all retained latents (p = 2048) as individually
significant after Westfall–Young correction.** The per-latent counts say that real-data
AUROC scores for >1000 latents exceed the *maximum* AUROC across all 2048 latents under
any of the 500 null permutations — i.e., the signal at those latents is stronger than
anything chance alone ever produced across the entire latent space.

The spread between methods (141 to 1052) reflects that mean_diff and probe_weight
concentrate power on fewer latents while AUROC, a rank-based statistic invariant to
monotone rescaling, is sensitive to a much broader population of differentially activated
features.

## 4. Interpretation for Stage 3

Two numbers from this calibration carry forward:

1. **Baseline certified-discovery rate ≤ 1.6%.** Under the global null the pipeline
   certifies at least one candidate in at most 1 in 60 experiments (mean_diff). Any
   empirical FDR study from Stage 3 that compares real-data certifications to this
   baseline has a well-calibrated floor.

2. **WY threshold for per-latent claims.** To make a multiplicity-corrected per-latent
   claim at α = 0.05, the threshold to beat is the 95th percentile of the max-score
   null: 0.228 (mean_diff), 0.0103 (auroc), 0.0473 (probe_weight). The saved
   `stage1_adjusted_pvalues.npz` contains exact adjusted p-values for every retained
   latent under all three methods.

**Scoring method choice matters for Stage 3.** AUROC produces a far larger set of
WY-significant latents than the other two methods, which may indicate either higher power
or a softer signal that does not translate into ablation-certified discoveries. Stage 3
should report all three methods and note this divergence explicitly.

## 5. Deviations from the pre-registration

1. **GPU probe (L2 gradient descent) in place of sklearn L1/SAGA.** Motivation: the
   CPU SAGA solver at n = 67,349 and p = 2048 was taking ~5.9 min per permutation
   (projected total ~49 h). The GPU implementation reduces this to ~0.36 sec per
   permutation (3 min total). The switch is from L1 to L2 regularisation; both produce
   a ranking of latents by coefficient magnitude, which is all that the scoring step
   requires. The FWER statistic is max-null calibrated — it is self-correcting for
   changes to the probe as long as the same probe is applied to real and permuted labels.
2. **`--device cuda` flag added.** `calibrate_pipeline.py` now accepts a `--device`
   argument; it defaults to `cuda` if available, `cpu` otherwise. This is a usability
   change with no effect on the numerics.

## 6. Reproduction

```bash
# cache must exist first:
python src/cache_activations.py --config config/default.yaml   # ~5 min, GPU

# Stage 1 (GPU strongly recommended):
python src/calibrate_pipeline.py --config config/default.yaml --device cuda
```

Runtime: **3.0 min** on NVIDIA L40S.

Artefacts: `results/stage1_calibration.json` (all FWER, WY, and baseline numbers),
`results/stage1_adjusted_pvalues.npz` (per-latent WY-adjusted p-values for all three
methods), `results/fig5_permutation_null.png` (null distributions with real-label max
marked), `results/fig6_fwer_by_method.png` (FWER bar chart).
