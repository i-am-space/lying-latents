# Stage 2 on SAEBench findings — premise and exchangeability of Gaussian knockoffs on 8 datasets

Pre-registered in `config/preregistration.yaml` (`saebench_amendment_1`, committed before any
SAEBench text, cache or result existed) · master seed `20260904` · Gemma-2-2B, Gemma Scope layer 20 /
16k, unchanged from SST-2. Script: `src/saebench/stage2.py`. Results: `results/saebench/stage2/` (one
JSON per dataset), `results/saebench/stage2_summary.json`.

## Summary

1. **The zero-atom premise (SB2-P) fails on every dataset.** The median Pr(X_j = 0) over retained
   latents is **0.30–0.73**, against SST-2's 0.89 and the 0.80 threshold. It shrinks with text
   length: EuroParl (36 tokens) 0.73, AG News (52) 0.67, Bias in Bios (74–78) 0.58–0.59, GitHub
   code and Amazon (127, at the 128-token cap) 0.30–0.49. Mean pooling over more tokens gives a
   latent more chances to fire somewhere in the text, so fewer text-level values are exactly zero.
2. **Gaussian knockoffs are still detectably non-exchangeable on every dataset.** The violation
   rule fires on **8 / 8** datasets, and on **6 / 6** of those whose diagnostics are valid. The
   `1{x = 0}` rule separates real from knockoff columns at median accuracy **0.65–0.87**. Swapping a
   *single* latent out of 2,048 is detected at classifier AUC **0.88–0.94** (0.70 on Amazon
   sentiment, the only dataset with n = 5,000), against a label-permutation 97.5th percentile of
   0.51–0.52. Swapping 10 latents is detected at AUC ≥ 0.99997 everywhere.
3. **The diagnostics are valid on 6 / 8 datasets (SB2-V).** The Gaussian control (same n,
   covariance re-estimated) stays at AUC 0.485–0.509 at |S| = 1 and 0.485–0.505 at |S| = all on every
   dataset. On EuroParl and Amazon sentiment its |S| = 0 AUC (0.516, 0.523) is slightly above a null
   band built from only 10 permutations (q97.5 0.514, 0.519), so per the rule those two datasets'
   violation results are not interpreted. See §4.
4. **SB2-R's replication clause is vacuous.** It asks for a violation on every dataset that passes
   *both* SB2-P and SB2-V, and none passes SB2-P (`n_eligible: 0`, `replicates: null`). Reported as
   is: the pre-registered replication test has no data to apply to.
5. **The zero-mass accuracy equals (1 + Pr(X = 0)) / 2 to three decimals on every dataset**
   (e.g. AG News: (1 + 0.669) / 2 = 0.834, observed 0.834; Amazon: 0.652 vs 0.652). This is what the
   rule must give when the knockoff has no exact zeros: it classifies every zero correctly and is at
   chance on the rest. So the knockoffs never produce an exact zero, on any dataset, as on SST-2.

**What this establishes:** across eight domains, the Gaussian knockoff construction is not
exchangeable with SAE latents, and the failure does not need a large atom. With 30% zeros the
single-latent swap is still detected at AUC 0.91. The theoretical argument (any atom of positive
mass makes the KL term in the robustness bound infinite) covers every dataset here; none has a
latent-level atom of zero. **What it does not establish:** that realised FDR exceeds the target on
these datasets (that is Stage 3, not yet run on SAEBench). It also does not establish anything
about other models, layers or widths, or about the original SST-2 claims beyond their direction.

## 1. Setup

| Item | Value |
|---|---|
| Rows | per dataset, all its cached texts (train + test pooled; knockoffs depend only on the law of X) |
| Latents | per dataset, p = 2,048 (firing rate ≥ 1%, top 2,048 by firing rate) |
| Knockoffs | second-order Gaussian, equicorrelated S, standardised latents; covariance sample unless cond > 10⁴, then Ledoit–Wolf |
| Tests | `knockoff_audit.run_suite` unchanged: `1{x = 0}` rule, swap classifier AUC and MMD at \|S\| ∈ {0, 1, 10, 50, all} (3 replicates for 1 / 10 / 50), moments |
| Null band | 10 label permutations of the swap classifier |
| Control | N(0, Σ̂) at the same n, covariance re-estimated by the same rule, same knockoffs; \|S\| ∈ {0, 1, 10, all} |
| Runtime | 18–29 min per dataset with 8 threads per process, 5 run in parallel on the CPU; Bias in Bios sets ran sequentially in the main queue (111, 59 and 31 min, unrestricted threads, sharing the CPU with Stage 1 and then the parallel jobs) |

## 2. Premise (SB2-P)

| dataset | n | mean tokens | median Pr(X=0) | frac latents Pr(X=0) > 0.8 | nonzero skew | nonzero excess kurtosis | covariance |
|---|---|---|---|---|---|---|---|
| EuroParl | 12,500 | 35.9 | 0.730 | 0.182 | 3.28 | 17.4 | Ledoit–Wolf |
| AG News | 10,000 | 51.6 | 0.669 | 0.000 | 2.55 | 9.8 | sample |
| Bias in Bios 1 | 12,500 | 74.0 | 0.590 | 0.000 | 2.73 | 11.9 | Ledoit–Wolf |
| Bias in Bios 2 | 12,500 | 73.9 | 0.590 | 0.000 | 2.82 | 12.7 | Ledoit–Wolf |
| Bias in Bios 3 | 12,500 | 77.8 | 0.581 | 0.000 | 2.83 | 12.9 | sample |
| GitHub code | 12,500 | 126.9 | 0.490 | 0.000 | 3.10 | 15.5 | Ledoit–Wolf |
| Amazon sentiment | 5,000 | 126.7 | 0.319 | 0.000 | 2.40 | 8.7 | Ledoit–Wolf |
| Amazon category | 12,500 | 126.7 | 0.304 | 0.000 | 2.38 | 8.8 | Ledoit–Wolf |
| *SST-2 (reference)* | *67,349* | — | *0.891* | *0.826* | *2.77* | *11.9* | *Ledoit–Wolf* |

Every dataset fails SB2-P. Two (GitHub code, Amazon) are also below the 0.50 *stop* threshold of
the original Stage 2 rule; the amendment did not carry that stop rule over, and its design runs the
tests on every dataset, so they were run. The nonzero part is as heavy-tailed as on SST-2
(skew 2.4–3.3, kurtosis 8.7–17.4). Whatever the atom, the latents are not Gaussian.

## 3. Exchangeability (SB2-R) and control (SB2-V)

| dataset | zero-mass acc. | (1 + Pr(X=0))/2 | AUC \|S\|=0 | AUC \|S\|=1 | AUC \|S\|=10 | null q97.5 | control \|S\|=1 | control \|S\|=all | violation | diagnostics valid |
|---|---|---|---|---|---|---|---|---|---|---|
| EuroParl | 0.865 | 0.865 | 0.497 | 0.927 | 1.0000 | 0.514 | 0.500 | 0.494 | yes | **no** |
| AG News | 0.834 | 0.834 | 0.508 | 0.927 | 1.0000 | 0.525 | 0.485 | 0.497 | yes | yes |
| Bias in Bios 1 | 0.795 | 0.795 | 0.495 | 0.923 | 1.0000 | 0.518 | 0.508 | 0.492 | yes | yes |
| Bias in Bios 2 | 0.795 | 0.795 | 0.513 | 0.908 | 1.0000 | 0.512 | 0.509 | 0.502 | yes | yes |
| Bias in Bios 3 | 0.791 | 0.791 | 0.512 | 0.937 | 1.0000 | 0.510 | 0.499 | 0.491 | yes | yes |
| GitHub code | 0.745 | 0.745 | 0.509 | 0.878 | 1.0000 | 0.513 | 0.497 | 0.505 | yes | yes |
| Amazon sentiment | 0.659 | 0.659 | 0.491 | 0.699 | 1.0000 | 0.519 | 0.502 | 0.485 | yes | **no** |
| Amazon category | 0.652 | 0.652 | 0.504 | 0.910 | 1.0000 | 0.514 | 0.502 | 0.495 | yes | yes |
| *SST-2 (reference)* | *0.945* | *0.945* | *0.495* | *0.997* | *1.0000* | *0.515* | *0.508* | *0.491* | *yes* | *yes* |

AUC at |S| = 1 is the mean of 3 replicates; |S| = 10 is shown to 4 decimals (all ≥ 0.99997);
|S| = all is 1.0000 on every dataset. Knockoff–original correlation is 0.92–0.99 (s small, as on
SST-2), so each knockoff is close to its original plus small noise. That noise smears the atom,
which is what the zero-mass rule and the classifier detect.

- **The swap AUC at |S| = 1 does not track the atom size much.** Amazon category (30% zeros) gives
  0.910, close to EuroParl's 0.927 (73% zeros). The single-latent signal is not just the zero
  indicator. The one low value, Amazon sentiment at 0.699, has the same latents' distribution as
  Amazon category (same texts, median Pr(X = 0) 0.32 vs 0.30) but n = 5,000 instead of 12,500, so
  the classifier has less data. This is most likely a sample-size effect, not a smaller violation.
- **MMD**, as on SST-2, has no power at this dimension. In the five run logs inspected, its p-values
  at |S| = all are 0.17–0.83 on real data. Per-|S| values are in the per-dataset JSONs.

## 4. The null band is too narrow (exploratory; no rule tests it)

SB2-V failed on two datasets only at |S| = 0 (no swap), where the control is null by construction:
EuroParl 0.516 vs q97.5 0.514, Amazon sentiment 0.523 vs 0.519. On the *real* data, |S| = 0 (also
null by construction) also exceeds its band on Bias in Bios 2 (0.513 vs 0.512) and 3 (0.512 vs
0.510). The 97.5th percentile of 10 permutations is close to their maximum, and every |S| = 0 AUC is
a single replicate. So the band is narrower than the classifier's own run-to-run spread (control
AUCs at one |S| within one dataset spread by up to ±0.035). The pre-registration set 10 permutations (SST-2 used 20).
Per SB-T, nothing is changed here; the two datasets stay "not interpreted".

The practical consequence is small. On both datasets the |S| = 1 AUC (0.927, 0.699) is far
outside any plausible band, so a wider band (e.g. more permutations, or the control's own spread)
would not change the violation verdict on any dataset. A future amendment that wants SB2-V to
discriminate should pre-register more permutations and replicate |S| = 0.

## 5. Deviations and notes

- **No deviations from `saebench_amendment_1`** in datasets, latents, thresholds, seeds or
  permutation counts (SB-T).
- **Parallel execution:** Bias in Bios sets 1–3 ran in the main queue; the other five datasets were
  launched as separate processes (one per dataset, `OMP_NUM_THREADS=8`) once Stage 1 had finished,
  to use idle cores. Each process uses its dataset's own seed streams (`sb_knockoff`,
  `sb_row_partition`, `sb_synthetic_null`, indexed by dataset), so results do not depend on how the
  datasets were scheduled. Thread counts can change floating-point summation order in BLAS and the
  classifier, so the numbers are reproducible to that level, not bitwise.
- **Aggregate fix before aggregation:** `--aggregate` would have printed `replicates: True` for SB2-R
  over an empty set of eligible datasets; it was changed to report `None` with `n_eligible` before
  the final aggregate. The queue's own aggregate ran once with the old code and was overwritten.
- **The SAEBench debug run** (`results/saebench/stage2_debug/`, AG News, cut-down settings) is a code
  path check only and is not a result.
- **Interpretation constraints from the amendment apply:** one model / layer / SAE; the diagnostics
  can falsify exchangeability on the (unknown) null latents but not confirm it.
