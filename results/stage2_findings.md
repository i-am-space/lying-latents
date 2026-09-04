# Stage 2 findings — exchangeability audit of Gaussian knockoffs for SAE latents

Config hash `d33d210c5acb` · master seed `20260904` · pre-registered in
`config/preregistration.yaml` (commit `ecbf77d`, which precedes every results commit).

## Summary

Exchangeability is violated for Gemma Scope SAE latents under second-order Gaussian
knockoffs. The violation is unarguable in the sense that it needs no statistical test:
89% of the mass of a typical retained latent sits at *exactly* zero, and a Gaussian
knockoff places zero probability there, so the indicator `1{x = 0}` separates a real
column from its knockoff at an accuracy no exchangeable pair could reach. The swap
two-sample test confirms this with power to spare, against a harness that returns
exactly null on Gaussian data where the same knockoffs are provably valid.

Separately, and reported separately because they are defects of one implementation
rather than of the Gaussian surrogate itself, the reference pipeline
(Enkhbayar, 2025) reads ~93% of its feature vectors at a padding position, and its
knockoff construction produces decoys that are 97.7–99.2% correlated with the
variables they are supposed to be decoys for.

**What this establishes:** the finite-sample FDR guarantee of Model-X knockoffs does
not hold for this construction on this data. **What it does not establish:** that the
realised FDR exceeds the nominal target. That is Stage 3 and was not run.

## 1. Setup

| | |
|---|---|
| Model | `google/gemma-2-2b`, float32, `blocks.20.hook_resid_post` (HF `hidden_states[21]`) |
| SAE | `gemma-scope-2b-pt-res-canonical` / `layer_20/width_16k/canonical`, JumpReLU, 16384 latents |
| Data | GLUE SST-2 **train**, all 67,349 sentences, one row per sentence |
| Aggregation | mean over non-special tokens (primary); max and last-real-token also cached |
| Retained | firing rate ≥ 1%, then top 2048 by firing rate → **p = 2048**, n/p = 32.9 |

The cache is validated by two numbers that would not appear if the activations were
wrong: **SAE explained variance 0.780** and **mean L0 69.4**, both matching the
canonical Gemma Scope figures for this SAE. Re-running the pipeline reproduces the
matrix bitwise.

**BOS must be excluded from aggregation.** At layer 20 the `<bos>` position is an
attention sink with residual norm ≈ 2900 against ≈ 350 for content tokens, and the SAE
does not model it: L0 ≈ 7000 there against ≈ 69 on real tokens, with relative
reconstruction error 3.9 against 0.3. Including BOS in a mean drives measured
explained variance to −13 and would inject thousands of spurious "firing" latents into
every sentence vector.

## 2. Distributional premise (Deliverable 2)

Pre-registered rule: proceed if median `Pr(X_j = 0)` over retained latents under the
primary aggregator is ≥ 0.80; stop if < 0.50.

| aggregator | median `Pr(X_j=0)` | frac > 0.95 | frac > 0.8 | nonzero skew | nonzero excess kurtosis |
|---|---|---|---|---|---|
| **mean (primary)** | **0.8909** | 0.000 | 0.826 | 2.77 | 11.89 |
| max | 0.8909 | 0.000 | 0.826 | 1.97 | 5.03 |
| last real token | 0.9998 | 0.940 | 0.983 | 1.49 | 2.11 |

**Verdict: PREMISE_HOLDS** (0.8909 ≥ 0.80). The atom is large and the audit proceeds.

Three things here matter for the paper.

**(a) Mean and max pooling have identical zero-mass — necessarily so.** Verified
bitwise on the real matrix: the two aggregators share their zero pattern exactly. For
non-negative activations `mean(x) = 0` iff every token is zero iff `max(x) = 0`, so no
choice between mean and max can move the atom. It changes only the nonzero part (skew
2.77 vs 1.97). The build instructions anticipated that max-pooling would "drive the
zero-mass down sharply"; that is not possible, and the two aggregators need not be
argued between on this axis.

**(b) The proposal's ">95%" figure is a per-token statement and does not survive
aggregation to the analysis unit.** Per token, `Pr(X_j = 0) = 1 − 69.4/16384 = 0.9958`.
Per sentence, over all 16,384 latents the median is 0.9916; restricted to the 7,696
latents firing on ≥1% of sentences it is 0.9669; and over the retained top-2048 it is
**0.8909**. The retention step is what pulls it down, because it selects the most
frequently firing latents by construction. The paper should quote 0.89 for the analysed
set, not >0.95. The argument is unaffected — any atom of positive mass makes the KL
divergence to a Gaussian infinite — but the number as written describes a different
population from the one analysed.

**(c) Selecting latents the reference's way changes nothing.** Their statistic is
top-k by mean |activation| ("energy") rather than firing rate. That set overlaps ours
by 1692/2048 (82.6%) and has median `Pr(X_j = 0) = 0.8918` against our 0.8909. The
atom is not an artefact of our filter.

### Covariance conditioning

On the correlation scale, the sample covariance of the retained latents has condition
number **1.30 × 10⁴** with smallest eigenvalue 0.0042 and no non-positive eigenvalues.
That exceeds the pre-registered 10⁴ threshold, so **Ledoit–Wolf is the primary
estimator** (shrinkage 0.0072, condition number 4773). The top 10 components carry
11.9% of variance and the top 50 carry 25.8%.

Under the last-real-token aggregator the covariance is genuinely **singular** — 220
non-positive eigenvalues at n = 67,349 — which is the regime the reference operates in,
at n = 872 and p = 512.

## 3. Harness validation (§7.2) — done before any real diagnostic

Every diagnostic below is designed to detect a violation of exchangeability. A buggy
one detects violations everywhere, including where none exist, which would look like a
finding and would not be one. So the entire suite was first run on synthetic data drawn
from a multivariate Gaussian with the *same* covariance as the real latents, where
second-order knockoffs are exactly valid and exchangeability genuinely holds.

**All three pre-registered criteria pass** (10 replicates, n = 20,000, |S| = all):

| criterion | required | observed | |
|---|---|---|---|
| classifier AUC 95% CI contains 0.50 | yes | 0.4973 ± 0.0078, CI (0.4925, 0.5021) | ✅ |
| MMD p-values uniform (KS test) | p ≥ 0.05 | p = 0.958 | ✅ |
| max abs standardised moment gap | ≤ 0.10 | 0.0009 | ✅ |

**A second control, not pre-registered, rules out the obvious alternative explanation.**
The calibration above builds knockoffs from the *known* covariance. Real data has only
an *estimated* one, so a sceptic could attribute any real-data signal to covariance
estimation error at p = 2048. To close that off, Gaussian data was generated at the real
sample size (n = 67,349), its covariance estimated exactly as the real pipeline does,
and the full swap sweep run on it. Every AUC lands in **0.4909–0.5173** across
|S| ∈ {0, 1, 10, 50, 2048}. Covariance estimation error at this n/p does not produce a
detectable violation. Whatever the real data shows is not that.

## 4. Diagnostics on real latents (§7.3)

### (a) Zero-mass discrepancy

The trivial classifier `1{x = 0}`, per latent, separating a real column from its knockoff:

| statistic | value |
|---|---|
| median accuracy | **0.9454** |
| mean accuracy | 0.9242 |
| range | 0.5000 – 0.9677 |
| latents with accuracy > 0.55 | **99.17%** (2031 / 2048) |
| max `Pr(X̃_j = 0)` over knockoff columns | **0.0000** |

Under exchangeability this must be 0.50 for every latent. It is 0.945 at the median.
No knockoff column contains a single exact zero anywhere in 67,349 rows, against 89% of
rows in a typical real column.

This is deliberately the simplest possible falsification, and it should not be oversold
as a sophisticated finding. Its force is that it requires no test at all, only the
definition of exchangeability: swapping `X_j` with `X̃_j` must leave the joint law
unchanged, and here it changes a coordinate that is exactly zero 89% of the time into
one that is never exactly zero. The single latent at accuracy 0.500 is the one latent
in the retained set that fires on every sentence, so it has no atom to destroy.

### (b) Swap two-sample tests

Held-out AUC of a gradient-boosted classifier separating swapped from unswapped
samples, with the row partition redrawn per replicate:

| \|S\| | real latents | Gaussian control | |
|---|---|---|---|
| **0** (negative control) | **0.4948** | 0.5012 | null, as it must be |
| 1 | **0.9960, 0.9975, 0.9962** | 0.5025, 0.5142, 0.5069 | |
| 10 | **1.0000, 1.0000, 1.0000** | 0.5063, 0.5029, 0.4987 | |
| 50 | **1.0000, 1.0000, 1.0000** | 0.5079, 0.5173, 0.5020 | |
| 2048 (all) | **1.0000** | 0.4909 | |

Null band from 20 label permutations: mean 0.4995, sd 0.0077, 97.5th percentile 0.5153.

**Swapping a single latent out of 2048 is detected at AUC 0.9966** — 65 standard
deviations above the label-permutation null. Swapping ten is detected perfectly. The
|S| = 0 control sits inside the null band, so the classifier is not picking up the row
partition itself, and the Gaussian control at the same n, p and covariance estimator
stays null throughout.

**The MMD test detects nothing** (p-values scattered across 0.04–0.98 with no trend in
|S|), and this is reported rather than tuned away. It is not a bug: the implementation
recovers mean shifts at p = 0.005 in both 10 and 200 dimensions, and it is correctly
calibrated (uniform p-values on the synthetic null, KS p = 0.958). It is a power
limitation. A Gaussian kernel with median-heuristic bandwidth loses power as dimension
grows, and here the two samples differ in |S| of 4096 coordinates. Verified directly:
against a matched-moment Gaussian, MMD detects zero-inflation at p = 0.005 in 10
dimensions but only p = 0.06 in 200. The pre-registered choice was kept; the classifier
test is the one carrying the evidence.

### (c) Higher moments and tails

| moment | real (median) | knockoff (median) | median abs gap |
|---|---|---|---|
| mean | −0.0000 | 0.0000 | **0.0005** |
| variance | 1.0000 | 1.0005 | **0.0012** |
| skewness | 6.978 | 6.464 | 0.514 |
| excess kurtosis | 71.46 | 64.30 | 7.30 |
| q90 | 0.2165 | 0.3047 | 0.0877 |
| q99 | 4.554 | 4.466 | 0.0850 |
| q999 | 10.656 | 10.401 | 0.251 |
| max | 23.77 | 22.99 | 0.730 |

**Moments 1 and 2 match** to 0.0005 and 0.0012 against a pre-registered 5% tolerance, so
the construction is correctly configured and the rest is interpretable.

The higher moments differ, but only modestly (skew 6.98 vs 6.46, kurtosis 71.5 vs 64.3),
and this deserves a careful reading rather than a triumphant one. Because `s` is tiny
(§5), the knockoff is close to `X` plus small independent noise, so it *inherits* most
of the real marginal shape, heavy tails included. **The violation is not driven by the
tails. It is driven by the atom** — the noise smears 89% of the mass out of a single
point and across a window of width ≈ 0.21, which the classifier detects immediately
while the moment summary barely registers it. That is precisely the paper's argument,
and it is worth stating that the moment comparison is the *weakest* of the three
diagnostics here, not the strongest.

## 5. Two defects of the reference implementation, reported separately

These concern the specific pipeline of Enkhbayar (2025) rather than the Gaussian
surrogate itself. They would survive a fix to the atom problem, and the atom problem
would survive a fix to them. They must not be merged into the same claim.

**(i) ~93% of their feature vectors are read at a padding position.** Their aggregation
is `cache[hook][:, -1, :]` after `model.to_tokens(texts)`, and TransformerLens
right-pads by default. Only rows attaining the batch maximum length get a real final
token, so with `activation_batch_size = 16` at most one row in sixteen does. Verified on
their exact model, and simulated under their exact defaults: **476/512 = 93.0%** of rows.
The per-example vector is therefore largely a function of how long the *other fifteen*
sentences in the batch happened to be. Details in `NOTES_reference.md` §2.

**(ii) Their knockoffs are near-copies of the variables they are decoys for.** They
apply an equicorrelated `S = s·I` with `s = min(2λ_min, 0.99)` to an **unstandardised**
covariance. On our data that rule gives `s = 0.00261` against a median latent variance
of 0.490 — a ratio of **0.0053**, so `corr(X_j, X̃_j) ≈ 0.995`. This is a **power**
defect, not a validity defect: `W_j = |β_j| − |β̃_j|` compares each variable against a
99.5% copy of itself, which drives the statistic toward zero for signal and null alike.
It is confounded with raw activation energy, exactly as anticipated.

Note that this is not fixed by standardising. On the correlation scale the equicorrelated
rule still gives `s = 0.0228` (Ledoit–Wolf) or `0.0084` (sample), i.e.
`corr(X_j, X̃_j)` of 0.977 and 0.992. The cause is the ill-conditioned correlation matrix
(λ_min = 0.0042), and the fix would be a different `S` (MVR or SDP), not a different
scale. We report it because a Stage 3 power measurement against this baseline would be
measuring the `S` choice as much as the Gaussian assumption.

## 6. What this establishes, and what it does not

Written here so it survives into the paper.

- **Exchangeability is violated.** The zero-mass diagnostic falsifies it by definition,
  and the swap classifier confirms it at AUC 0.9966 for a single swapped coordinate
  against a validated null harness. The finite-sample FDR guarantee of Model-X knockoffs
  **does not hold** for this construction on this data.
- **These are necessary-condition tests.** Exchangeability is required only on *null*
  latents, and the null set is unknown on real data. The diagnostics are sufficient to
  falsify, never to confirm. We have **not** "confirmed the violation on nulls", and no
  claim in the paper may say so.
- **A void guarantee is not a demonstrated failure.** Nothing here shows that realised
  FDR exceeds the nominal target. The procedure could still control FDR by accident. That
  question is Stage 3, it was not run, and the distinction is the one the whole paper
  rests on.
- **The atom is 0.891, not >0.95,** for the analysed unit. The KL argument is unaffected
  (any atom of positive mass gives infinite divergence), but the figure quoted in the
  proposal is a per-token statistic and should be restated.

## 7. Deviations from the pre-registration

All changes are recorded in git; `config/preregistration.yaml` at commit `ecbf77d`
predates every results commit.

1. **`n_clf_rows = 20000`** — the two-sample classifier is fitted on a 20,000-row
   subsample rather than all 67,349. Cost only.
2. **`swap_replicates = 3`** — independent draws of `S` for |S| ∈ {1, 10, 50}, to report
   variability rather than a single draw. This is strictly more information.
3. **A Gaussian control at the real sample size with estimated covariance** was added
   (§3). Not pre-registered; it can only make the real-data result harder to believe,
   not easier, and it is the control a reviewer would ask for.
4. **`last`-token aggregator cached** as a third option (pre-registered in the config's
   `aggregation.cached`, but not required by the build instructions), so the reference's
   *intended* aggregator can be reported separately from their padding bug.
5. **`ill_conditioned_cond_number` written as `10000.0`** rather than `1.0e4`. PyYAML
   parses an unsigned exponent as a string. Value unchanged.

6. **`s_method_secondary: mvr` was pre-registered and is reported separately below**
   rather than as part of the main audit. It bears only on power, not on validity: the
   zero-mass diagnostic gives accuracy `(1 + p₀)/2` for *any* second-order Gaussian
   construction, because every such knockoff is continuous and so never attains an exact
   zero. No choice of `S` can repair that.

Items 1–2 are applied identically to the |S| = 0 negative control and to the Gaussian
control, both of which return null, so they cannot manufacture a positive result.

### Correction to the commit record

Commit `5e0559d` ("Add Stage 2 pipeline") carries the line *"No results committed at
this commit"*. **That statement is false.** A `git add -A` swept the distributional
summary outputs — `results/fig1_zero_mass_hist.png`, `fig2_marginals.png`,
`fig3_eigenspectrum.png`, `stage2_describe.json`, `stage2_latent_table.npz` — into that
commit alongside the code. The history is not rewritten to hide this; the error is
recorded here instead.

What the error does **not** affect is the claim the pre-registration exists to support.
The pre-registration commit `ecbf77d` (2026-09-04 10:27:20 +0000) contains
`config/preregistration.yaml` and **no results of any kind** — verified by
`git ls-tree -r --name-only ecbf77d`, which lists nothing under `results/` or `data/`.
Every result in this repository post-dates it. The mislabelled commit is the *code*
commit, one commit later, and the results it accidentally contains are the
distributional summary, which was itself produced after the pre-registration was
frozen.

## 8. Reproduction

```bash
python src/cache_activations.py --config config/default.yaml   # ~5 min, 1 GPU
python src/describe_latents.py  --config config/default.yaml   # ~2 min, CPU
python src/knockoff_audit.py    --config config/default.yaml   # ~85 min, CPU
```

Cache hash `d33d210c5acb`, master seed `20260904`. The cache is reproduced bitwise
across runs (verified: `X_mean`, `X_max`, `X_last`, `retained_idx`, `firing_rate_all`
all identical over two independent runs).

Artefacts: `stage2_describe.json`, `stage2_audit.json`, `stage2_latent_table.npz`,
`stage2_zero_mass.npz`, and figures `fig1_zero_mass_hist.png` (zero-mass across
latents), `fig2_marginals.png` (six representative latents with the atom shown
explicitly), `fig3_eigenspectrum.png`, `fig4_audit.png` (calibration, AUC vs |S|,
zero-mass discrepancy).
