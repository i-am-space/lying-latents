# Stage 3 findings — realised FDR of Gaussian knockoffs on real SAE latents (v2, with controls)

Config hash `d33d210c5acb` · master seed `20260904` · design fixed in
`config/preregistration.yaml` `stage3_amendment_1` (post-hoc, committed in `13c81c5` before
the full run) · script `src/planted_fdr_controls.py` · 76 min on an RTX 5050 laptop GPU.
The teammates' v1 (`src/planted_fdr.py`, `results/stage3_planted_fdr.json`) is kept as
history; §7 lists what was wrong with it. §9 reports a follow-up at v1's p = 2,048
(`stage3_amendment_2`) that explains v1's power collapse. §10–§12 report three further
follow-ups (`stage3_amendment_3`, script `src/stage3_followups.py`): the amplitude sweep
extended to 20, a direct test of why FDR rises, and a factorial that separates the causes of
the power collapse. §13–§16 add a multiplicity reanalysis of all cell counts, a calibration
of the amplitude axis against the real labels, a many-weak-signal benchmark, and the same
benchmarks under the cross-validated penalty (`stage3_amendment_4`–`6`).

## Summary

**Headline (updated 6 October, §16).** Under the lasso penalty practitioners use, chosen by 5-fold
cross-validation, Gaussian knockoffs on real SAE latents **exceed the FDR target**, while the same
procedure on a Gaussian control with the same covariance does not. Every earlier Stage 3 run used a
fixed penalty (λ = 0.02, about 8× larger than the cross-validated one), which made Knockoff+ so
conservative that the control used almost none of its error budget (FDR ≈ 0.004 at q = 0.10); that
margin absorbed the effect of the invalid knockoffs and made Gaussian knockoffs look safe.

| Setting | Arm | Mean FDR at q = 0.10, fixed λ | Mean FDR at q = 0.10, CV λ | Cells above q after Benjamini–Yekutieli, CV λ |
|---|---|---|---|---|
| 10–30 strong signals, p = 512 (amplitudes 1, 5, 20) | real latents | 0.012 → 0.086 | 0.094 → 0.143 | **7 of 54** (all amplitude 20, interaction) |
| | Gaussian control | 0.004–0.008 | 0.077–0.083 | 0 |
| 100–300 weak signals, p = 2,048, designs matched to real labels (§14) | real latents | 0.048–0.098 | 0.098–0.357 | **12 of 24** (all interaction) |
| | Gaussian control | 0.025–0.069 | 0.088–0.118 | 0 |

The benchmark plants a known set S of latents in synthetic labels Y = f(X_S) + noise, keeps the real
activations, and runs Gaussian knockoff+ at q ∈ {0.05, 0.10, 0.20}, always next to a **Gaussian
control** (N(0, Σ) with the real covariance) on which Gaussian knockoffs are valid, so real-vs-control
isolates the non-Gaussian structure of the latents (the zero atom and co-firing).

1. **At the fixed penalty, no cell exceeds q after correction.** The two amplitude-20 cells
   reported in §10 (0.083 ± 0.016, 0.076 ± 0.011 at q = 0.05) do not survive Holm,
   Benjamini–Hochberg or Benjamini–Yekutieli correction across the 126 cells (§13).
2. **The margin erodes with signal strength, robustly.** Real-latent FDR exceeds the control's
   (Holm-significant in 47 of 126 cells; BY 63) and rises with amplitude: per-replicate slope of
   FDR on log amplitude +0.024 ± 0.001 (t = 18.7) for real latents, −0.001 for the control (§13).
3. **Why (§11):** null latents correlated with the signal latents beat their Gaussian knockoffs
   73% of the time at amplitude ≥ 5, against 47% on the control at the same correlations; nulls
   with correlation below 0.05 never became false discoveries. The pre-registered rule (R2) is
   formally unmet because one of its three checks could not be computed (see §11).
4. **Real labels sit where the benchmark had not looked (§14).** A probe on the real SST-2 labels
   reaches held-out AUC 0.972, as predictable as planted signals at amplitude 5 to over 50, but its
   top-k coefficients are as small as planted amplitudes 1 to 3: real sentiment is many weak
   signals that add up. With 100–300 weak planted latents (§15), at the fixed penalty FDR stays below
   q (no breach after correction) but the margin is already small (0.06–0.09 at q = 0.10 on the
   realistic designs, 0.03–0.07 on the control); under CV λ it is exceeded (table above).
5. **The p = 2,048 power collapse is mostly the S matrix (§12).** Block-diagonal MVR knockoffs
   instead of equicorrelated ones raise real-latent power by 0.12 (pooled p = 8 × 10⁻¹³); 3.4× more
   rows by 0.05 (no single cell Holm-significant). Together: 0.45 → 0.69 at q = 0.10. The zero atom
   costs 0.11–0.14 power with near-copy (equicorrelated) knockoffs but 0.05–0.06 with MVR.
6. **At p = 512 the zero atom's power cost is small** (−0.015 on average), larger for interaction
   than linear signals (−0.023 vs −0.008; difference p = 0.006), concentrated at amplitudes ≤ 2. No
   individual cell is significant after correction (the "9 of 54 vs 2 of 54" in earlier versions
   were uncorrected counts; §13).
7. **v1's "zero power at k = 10, q = 0.05" is arithmetic.** Knockoff+ needs ≥ 1/q = 20
   discoveries; with the +1 offset removed the same cells have power 0.985.
8. **Marginal screening fails the conditional target; knockoffs do not.** A Westfall–Young
   marginal scan finds 150–210 latents with FDR 0.78–0.93 against the planted conditional truth;
   Knockoff+ finds ≈ k with FDR 0.02–0.04 (fixed λ).

**What this establishes:** Gaussian knockoffs do not control FDR on SAE latents under standard
practice (cross-validated penalty), in planted designs matched to real labels, with the violation
concentrated in interaction signals; a conservative fixed penalty hides it. The bias comes from null
latents that co-vary with the signal beyond linear correlation. **What it does not:** behaviour on
real (unplanted) labels; which non-Gaussian feature (atom, co-firing, tails) drives the bias; the
large-k CV results rest on 10 replicates and 26% of CV choices at the smallest grid value (§16); and
block-diagonal MVR is an approximation to MVR at p = 2,048. Planted signals are random latents with
equal weights; a teammate's separate experiment plants them on clusters of related latents.

## 1. Design

| | |
|---|---|
| Latents / rows | 512 of the 2,048 retained (seeded random; median Pr(X = 0) = 0.891, so the atom is representative) · 20,000 rows (seeded random) · n/p = 39 · sample covariance (cond 151) |
| S matrices | MVR: mean s = 0.50, mean corr(X_j, X̃_j) = 0.50 · equicorrelated: s = 0.177, corr 0.82 · Gaussian control reuses the MVR S (corr 0.499) |
| Signal | linear and pairwise-interaction, k ∈ {10, 20, 30}, amplitudes {0.5, 1, 2, 3, 5, 8}, q ∈ {0.05, 0.10, 0.20} → 108 cells per dataset |
| Replicates | 30 per cell, each with a **fresh knockoff draw**, so they are independent given X; signal sets, weights and label noise are shared across datasets (paired) |
| Statistic | L1-logistic lasso (FISTA, λ = 0.02, step 1/L, ≤ 1000 iterations), W = \|w\| − \|w̃\|, Knockoff+ |
| Cost | 3,240 lasso fits + 360 for the λ check; **100% converged** (median residual 7.6e-5) |

Decision rule C1 (control must hold FDR ≤ q): **holds** — 0 of 108 control cells inflated,
worst −0.034 below q. The k = 0 harness is uninformative for Knockoff+ (it never discovers
anything when Y is independent of X); the Westfall–Young scan behaves at its nominal level
there (0.06 / 0.14 / 0.24 at α = 0.05 / 0.10 / 0.20, 50 replicates).

## 2. FDR

At q = 0.10, mean over forms and k (`fig13_stage3_v2_fdr.png`):

| amplitude | real, MVR | real, equicorrelated | Gaussian control |
|---|---|---|---|
| 0.5 | 0.007 | 0.005 | 0.008 |
| 1 | 0.012 | 0.011 | 0.008 |
| 2 | 0.017 | 0.016 | 0.007 |
| 3 | 0.024 | 0.025 | 0.004 |
| 5 | 0.041 | 0.041 | 0.004 |
| 8 | 0.060 | 0.058 | 0.004 |

Strictest target, q = 0.05: real 0.001 → 0.024, control 0.003–0.004. Real-minus-control FDR:
significantly positive in 48 of 108 cells, significantly negative in 1. Cells nearest
nominal (amplitude 8, linear, mean ± SE, real vs control): k = 10, q = 0.10: 0.099 ± 0.023
vs 0.003; k = 10, q = 0.20: 0.115 ± 0.025 vs 0.003; k = 20, q = 0.05: 0.053 ± 0.010 vs 0.006.
Only the k = 20, q = 0.05 cell has a mean above q (+0.003, not significant).

## 3. Power

At q = 0.10, mean over forms and k, real (MVR) / control: 0.415 / 0.497 (amplitude 0.5),
0.951 / 0.974 (1), 0.997 / 1.000 (2), 0.999 / 0.999 (3, 5, 8). Linear signals at amplitude
0.5: k = 10 0.900 vs 0.967, k = 20 0.820 vs 0.850, k = 30 0.503 vs 0.547. Interaction at
amplitude 0.5: k = 10 0.233 vs 0.497, k = 20 0.033 vs 0.108, k = 30 0.000 vs 0.011.
11 of 108 cells are significantly lower for real data (largest: amplitude 1, interaction,
k = 20, q = 0.05: −0.302 ± 0.123), none higher (`fig14_stage3_v2_power.png`).

**S matrix:** equicorrelated minus MVR power averages −0.004 (1 of 108 cells significant:
amplitude 0.5, linear, k = 30: 0.352 vs 0.503); FDR difference −0.0007. At p = 512 the
near-copy problem (corr 0.82) is much milder than at p = 2,048 (0.977 at Stage 2's full n;
0.946 on the 20,000 rows used in §9) and barely matters here.

## 4. Knockoff+ floor and the marginal baseline

- k = 10, q = 0.05 (12 cells): Knockoff+ power **0.000**, offset-0 power **0.985**. Over all
  cells (real, MVR) power is 0.782 with Knockoff+ vs 0.926 with offset 0; FDR 0.023 vs 0.029.
- Westfall–Young marginal scan, scored against the conditional truth (q = 0.10, amplitude ≥ 1):
  FDR 0.78–0.93 with 150–210 discoveries, power ≈ 1, versus Knockoff+ FDR 0.02–0.04 with
  ≈ k discoveries. This is the proposal's §6 "gap the standard pipeline ignores": correcting
  for multiplicity does not fix testing marginal rather than conditional association.

## 5. Sensitivity to the penalty (post-hoc, two values, amplitude 2, linear, q = 0.10)

| λ | FDR real (k = 10/20/30) | FDR control | power real | power control |
|---|---|---|---|---|
| 0.01 | 0.070 / 0.063 / 0.066 | 0.024 / 0.045 / 0.055 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| 0.02 (primary) | 0.015 / 0.008 / 0.021 | 0.006 / 0.006 / 0.007 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| 0.05 | 0.000 / 0.009 / 0.015 | 0.000 / 0.008 / 0.010 | 0.40 / 0.69 / 0.31 | 0.93 / 0.93 / 0.66 |

A weaker penalty raises real-data FDR toward nominal; a stronger one cuts real-data power to
about half the control's. So the "power is not reduced" statement in the Summary holds at
the pre-registered λ = 0.02 and not at λ = 0.05. λ was fixed in advance and not tuned.

## 6. Interpretation and limits

- The pre-registered conclusion for Stage 3 is the proposal's first outcome — "robustness
  despite misspecification" — **with an erosion caveat**: FDR control is empirical, the
  margin the valid construction has is largely consumed at high SNR, and the sweep did not
  reach a violation but did not leave the margin either. **Superseded (§16):** that conclusion
  holds only at the fixed λ = 0.02. Under the cross-validated penalty the outcome is the
  proposal's second one, FDR inflation, in both the strong-signal and the realistic designs.
- A plausible mechanism is that null real latents depend on the signal latents in ways
  (co-firing, sparsity) that Gaussian knockoffs cannot copy, so their W statistics are
  biased positive as the signal sharpens. This was untested when v2 was written; §11 tests
  it.
- Planted labels are a semi-synthetic stand-in; the data and cache are shared with Stage 2,
  so this is not independent of it; FDR is a Monte Carlo estimate over independent replicates
  conditional on this X.

## 7. Corrections to v1 (verified against the committed v1 JSON)

1. The findings claimed ASDP knockoffs; the code used `s_method: equicorrelated` (`asdp`
   appears nowhere). Stage 2 had measured this S as near-copy (0.977 at p = 2048) and said
   Stage 3 should use MVR.
2. "2,160 independent Lasso fits": 720 fits (2,160 = fits × 3 values of q).
3. "CI control YES in 100%": one of 72 conditions fails (amplitude 3.0, linear, k = 20,
   q = 0.05: 0.045 ± 0.011, upper bound 0.067). The code's CI rule is the reverse of the
   pre-registered one (mean ≤ q + 1.96·SE).
4. Executive-summary ranges did not match the tables: maximum FDR is 6.2% at q = 0.10 and
   7.1% at q = 0.20 (not 6.7% / 7.4%); "0–20.3% power for k ≥ 20 at amplitude 0.5" is up to
   55.3%; "0–3.8% at q = 0.10 for k ∈ {20, 30}" is 17.8% for linear k = 20.
5. Power 0 at k = 10, q = 0.05 was presented as a Gaussian-knockoff failure; it is the
   Knockoff+ offset (§4).
6. SEs used ddof = 0 and treated 10 replicates sharing one knockoff draw as independent.
7. The three "mechanisms for robustness" were asserted without a test.
8. The pre-registered Westfall–Young baseline was not implemented, and the design changed
   after the first result (single amplitude 3.0 → sweep; 20 → 3 × 10 replicates; first-20,000
   rows) without an amendment or a deviations section.

## 8. Deviations from v1 and from the pre-registration

p = 512 (v1: 2,048), because MVR does not converge at 2,048 (Stage 2) and the proposal
(§7) asks for knockoffs on the top few hundred latents; random rows and latents; MVR S;
a knockoff draw per replicate; amplitudes extended to 8; 30 replicates per cell; step size
1/L with convergence reporting; a Gaussian control; offset-0 and WY columns. All post-hoc,
declared in `stage3_amendment_1`, whose design commit (`13c81c5`, 03:42 IST) precedes the run.
Why v1 nevertheless saw a power collapse is answered in §9.

## 9. Follow-up at p = 2,048: where v1's power collapse comes from

Design fixed in `stage3_amendment_2` and committed (`01f1a4a`, 08:20 UTC, 27 Sep) before the
run. A teammate ran it on their GPU machine (68 min) and committed the results at 08:40 UTC
(diagnose, `fd6b528`) and 10:57 UTC (full, `5e7b0cb`).

**Setup.** Everything stays at v1's settings: p = 2,048 (all retained latents), n = 20,000,
n/p = 9.8, Ledoit–Wolf covariance, and equicorrelated S (s = 0.055, mean corr(X_j, X̃_j) =
0.946). λ = 0.02. Amplitudes are 0.5 and 1 (where v1 reported the collapse), with both
forms, k ∈ {10, 20, 30}, q ∈ {0.05, 0.10, 0.20} and 30 replicates. One knockoff draw costs
about 35 s at this size, so replicate *r* uses draw *r* of its dataset in every cell;
replicates within a cell stay independent. Five arms change one thing at a time:

| arm | rows | solver | isolates |
|---|---|---|---|
| `v1_setup` | first 20,000 | v1's (step 0.1, 500 iterations) | reproduces v1 (rule D1) |
| `first_fixed` | first 20,000 | fixed (step 1/L, convergence checked) | the solver |
| `random_fixed` | random 20,000 | fixed | the row choice |
| `gauss_fixed` | Gaussian, real covariance | fixed | the zero atom |
| `gauss_v1` | Gaussian, real covariance | v1's | the solver without the atom |

All fits converged (v1 solver residual ≤ 2e-5, below the 1e-4 tolerance), and none produced
non-finite weights.

**Power at q = 0.10** (2,048 columns: this run; 512 columns: §3 of this document):

| amplitude, form, k | v1 reported | `v1_setup` | `random_fixed` | `gauss_fixed` | real, MVR, p = 512 | control, p = 512 |
|---|---|---|---|---|---|---|
| 0.5, linear, 10 | 0.633 | 0.463 | 0.467 | 0.767 | 0.900 | 0.967 |
| 0.5, linear, 20 | 0.178 | 0.142 | 0.192 | 0.282 | 0.820 | 0.850 |
| 0.5, linear, 30 | 0.038 | 0.027 | 0.034 | 0.196 | 0.503 | 0.547 |
| 0.5, interaction, 10 | 0.067 | 0.033 | 0.100 | 0.190 | 0.233 | 0.497 |
| 0.5, interaction, 20 | 0.000 | 0.000 | 0.000 | 0.017 | 0.033 | 0.108 |
| 0.5, interaction, 30 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.011 |
| 1, linear, 10 | 1.000 | 0.960 | 0.867 | 1.000 | 1.000 | 1.000 |
| 1, linear, 20 | 0.935 | 0.938 | 0.902 | 0.975 | 1.000 | 1.000 |
| 1, linear, 30 | 0.824 | 0.814 | 0.879 | 0.953 | 0.976 | 0.984 |
| 1, interaction, 10 | 0.830 | 0.897 | 0.767 | 1.000 | 0.967 | 1.000 |
| 1, interaction, 20 | 0.583 | 0.667 | 0.623 | 0.720 | 0.930 | 0.980 |
| 1, interaction, 30 | 0.351 | 0.333 | 0.283 | 0.372 | 0.834 | 0.878 |

**Results against the pre-registered rules** (`fig16_stage3_p2048_power.png`):
- **D1, reproduction:** `v1_setup` matches v1's reported power in **36 of 36** cells, so v1's
  numbers are real.
- **The solver is not the cause.** For `first_fixed` − `v1_setup`, power differs by at most
  0.014 and in none of 36 cells significantly; the same holds without the atom
  (`gauss_fixed` − `gauss_v1`, 0 of 36). v1's step (0.1) is 3.0× the worst-case bound 1/L =
  0.033, but it converged anyway, because that bound assumes the maximum curvature of the
  logistic loss, which is not reached near the solution.
- **The row choice is not the cause.** `random_fixed` − `first_fixed` is significant in 1 of
  36 cells, about what chance produces at the 5% level.
- **The zero atom costs some power at p = 2,048.** `gauss_fixed` − `random_fixed` is
  significantly positive in **11 of 36** cells and negative in none, with mean +0.073. The
  largest gap is at amplitude 0.5, linear, k = 10, q = 0.10: +0.300 ± 0.119.
- **Most of the collapse is the setting itself (rule D3).** The Gaussian control, which has no
  atom, also loses most of its power at p = 2,048. For example, at amplitude 0.5, linear,
  k = 30, the control drops from 0.547 at p = 512 to 0.196 at p = 2,048 (−0.35). The real
  data then drops a further 0.16, to 0.034, which is the atom's share. The 512 → 2,048 drop
  combines fewer rows per latent (n/p 39 → 9.8) with near-copy knockoffs (corr 0.50 with MVR
  at 512, 0.946 with equicorrelated at 2,048). The two cannot be separated because MVR does
  not converge at p = 2,048.
- **FDR stays controlled:** no cell in any arm has mean − 1.96·SE > q; the maximum is 0.033.

**Interpretation.** v1's power collapse is a property of running Gaussian knockoffs on 2,048
latents with n = 20,000 and an equicorrelated S, rather than primarily of the zero atom. The
atom adds a loss whose size grows with the number of latents: it is negligible at p = 512 and
visible in about a third of the cells at p = 2,048. Practically, fewer latents (or a better S
matrix, if one can be computed at scale) recovers most of the power.

## 10. Follow-up: does FDR ever cross q? (amplitudes to 20)

> **Correction (§13):** the two amplitude-20 cells reported below as significantly above q do
> not survive correction for the 126 cells tested (0 under Holm, BH and BY). The rising trend does.

Design fixed in `stage3_amendment_3` and committed (`0a58f29`) before the run. The v2 data,
S matrices, solver and λ, with amplitudes 1, 2, 3, 5, 8, 12, 20 (both forms, k = 10/20/30,
q = 0.05/0.10/0.20, 30 replicates). The three v2 datasets reuse v2's seeds, so amplitudes 1–8
**reproduce v2 exactly** (rule R0: every replicate value identical). A fourth arm adds Stage 4's
SCIP hurdle knockoffs on the same labels (§11). Run locally, 96 min; all fits converged.

Mean FDR at q = 0.10 (over forms and k), and cells significantly above q (rule R4):

| amplitude | 1 | 2 | 3 | 5 | 8 | 12 | 20 | inflated cells |
|---|---|---|---|---|---|---|---|---|
| real, MVR | 0.012 | 0.017 | 0.024 | 0.041 | 0.060 | 0.078 | 0.086 | **2** (first at 20) |
| real, equicorrelated | 0.011 | 0.016 | 0.025 | 0.041 | 0.058 | 0.078 | 0.081 | 0 |
| real, hurdle knockoffs | 0.006 | 0.010 | 0.015 | 0.029 | 0.041 | 0.044 | 0.050 | 0 |
| Gaussian control | 0.008 | 0.007 | 0.004 | 0.004 | 0.004 | 0.002 | 0.004 | 0 |

- **The guarantee breaks at amplitude 20.** With MVR knockoffs, two cells exceed q = 0.05
  significantly: linear, k = 20: 0.083 ± 0.016; interaction, k = 20: 0.076 ± 0.011. Five cells
  at q = 0.05 and four at q = 0.10 have a mean above q. The largest FDR is 0.154 ± 0.032
  (interaction, k = 10, q = 0.20). Equicorrelated knockoffs behave the same (max 0.153) without
  a cell reaching significance.
- **The rise is steady, not a cliff.** Real data is significantly above the control in 0, 2,
  14, 16, 16, 16, 16 of 18 cells at amplitudes 1 → 20. Power at q = 0.10 is ≥ 0.93 in every
  arm from amplitude 1, so the extra FDR is not bought with power.
- The control's FDR stays at 0.002–0.008 (max 0.015) at every amplitude.

## 11. Follow-up: why FDR rises, and whether modelling the zero atom stops it

**What was measured.** For valid knockoffs every null latent's W is a fair coin:
P(W > 0 | W ≠ 0) = 0.5. Each fit recorded, per null latent, whether it beat its knockoff,
grouped by its *link* to the planted set (max |corr| with any signal latent). Gaussian
knockoffs copy correlations exactly, so if real nulls beat their knockoffs more than control
nulls *at the same link*, the bias comes from dependence beyond correlation (the zero atom,
co-firing). The lasso W is the pre-registered statistic. Because λ = 0.02 zeroes most nulls,
each fit also recorded a dense statistic that every null contributes to,
|corr(y, X_j)| − |corr(y, X̃_j)|, which is a fair coin under valid knockoffs too (secondary).

**Null P(W > 0), pooled over forms and k** (lasso W; SE over replicates):

| amplitude | 1 | 2 | 3 | 5 | 8 | 12 | 20 |
|---|---|---|---|---|---|---|---|
| real, MVR | 0.48 ± .07 | 0.54 ± .05 | 0.55 ± .04 | 0.62 ± .02 | 0.74 ± .02 | 0.76 ± .02 | 0.76 ± .02 |
| real, equicorrelated | 0.46 | 0.59 | 0.54 | 0.65 | 0.71 | 0.74 | 0.73 |
| real, hurdle | 0.23 | 0.27 | 0.27 | 0.40 | 0.41 | 0.44 | 0.45 |
| Gaussian control | 0.44 ± .08 | 0.56 | 0.38 | 0.49 | 0.64 | 0.46 | 0.51 ± .09 |

- **Only linked nulls become false discoveries.** No null with link < 0.05 (about 14,000 per
  amplitude) ever had a nonzero lasso W, in any arm. False discoveries per null at q = 0.10
  rise with link at every amplitude: on real/MVR at amplitude 20, 0 → 2·10⁻⁴ → 4·10⁻³ →
  1.3·10⁻² → 1.8·10⁻² → 7·10⁻² across the link bins (`fig20_stage3_mechanism.png`).
- **Rule R1 (diagnostic valid) holds:** the control's pooled coin is 0.474 ± 0.038 (lasso) and
  0.487 ± 0.001 (marginal), and no control cell is inflated. The marginal coin sits slightly
  *below* 0.5 on the control, i.e. Gaussian knockoffs fitted to the same sample are slightly
  conservative; the rules compare real with control bin by bin, so this offset cancels.
- **Rule R2 (mechanism), as pre-registered: not supported**, because one of its three checks
  cannot be computed. The other two hold clearly. Check 1: real nulls at amplitude ≥ 5 win
  0.731 ± 0.008 against 0.5. Check 3: in the high-link bins (link ≥ 0.3) they win 0.199 ± 0.058
  more often than control nulls. Check 2 compares the high-link bins with the lowest bin
  (link < 0.05), which has no nonzero lasso W, so it is undefined. On the secondary marginal
  statistic all three checks hold: 0.524 ± 0.001; high-minus-low link +0.025 ± 0.004;
  real-minus-control +0.033 ± 0.005. **Reading:** the lasso coin is strongly biased for real
  nulls and grows with amplitude, and only nulls tied to the signal are affected. That is the
  proposed mechanism, though the pre-registered test of the link gradient could not be run
  on the lasso statistic.
- **Rule R3 (zero atom is a cause): supported.** Hurdle knockoffs give significantly lower FDR
  than equicorrelated ones (paired, same labels) in **40 of 72** cells at amplitude ≥ 5, never
  significantly higher, with power different in 1 of 126 cells. At amplitude 20, mean FDR at
  q = 0.10 falls from 0.081 to 0.050. Their lasso coin is closer to 0.5 (0.44 vs 0.71). But
  they **overshoot**: hurdle knockoffs of strongly linked nulls beat their nulls (marginal coin
  0.25–0.36 in the top bin), and they are detectable (swap AUC 0.69 at |S| = 50, 0.86 at all
  512; Gaussian MVR 1.000/1.000; control 0.487/0.504). FDR with hurdle knockoffs still rises
  with amplitude (0.006 → 0.050). On the marginal statistic R3 is not supported, for the same
  overshoot (0.532 vs 0.514).

**Interpretation.** Gaussian knockoffs match second moments, so they fail in a specific place:
null latents whose dependence on the signal latents is more than linear. As the signal
sharpens, that mismatch turns into a systematic edge for the real latent, then into false
discoveries. Modelling the zero atom removes much of the edge, which ties it to the atom, but
the hurdle sampler's own misfit pushes the other way. The link measure is linear correlation,
so which non-Gaussian feature (atom, co-firing, tails) carries the bias is not isolated.

## 12. Follow-up: separating the causes of the power collapse at p = 2,048

**Design** (`stage3_amendment_3`, committed before the run). A 2 × 2 × 2 × 2 factorial:

- **latents:** 512 (the v2 set) or 2,048 (all);
- **rows per latent:** n/p = 9.77 or 32.88;
  - at 512 latents: n = 5,000 or 16,837 (seeded subsets);
  - at 2,048 latents: n = 20,000 (§9's rows) or 67,349 (every cached row);
- **S:** equicorrelated, or MVR. At 2,048 MVR uses knockpy's block-diagonal approximation
  (blocks ≤ 512, with its line search, so 2Σ − S stays PSD). This is the S §9 could not
  compute.
- **data:** real latents, or the Gaussian control with the matching covariance and n.

Other settings:

- Amplitudes 0.5 and 1, both forms, k = 10/20/30, q = 0.05/0.10/0.20, 30 replicates.
- Knockoffs come from a GPU sampler with the same law as knockpy's `GaussianSampler`. It was
  checked against knockpy before the run: corr(X, X̃) 0.878 vs 0.878 and 0.597 vs 0.596, with
  similar covariance errors.

Run on pkgpu in 100 min. All fits converged, and **no cell in any of the 16 datasets is
FDR-inflated**.

**Power at q = 0.10** (mean over forms, k and both amplitudes), with mean corr(X_j, X̃_j) in
brackets:

| latents, n | real, equicorr. | real, MVR | Gaussian, equicorr. | Gaussian, MVR |
|---|---|---|---|---|
| 512, n = 5,000 (n/p 9.8) | 0.269 (0.88) | 0.451 (0.60) | 0.305 (0.89) | 0.523 (0.68) |
| 512, n = 16,837 (n/p 32.9) | 0.629 (0.80) | 0.656 (0.50) | 0.710 (0.81) | 0.717 (0.54) |
| 2,048, n = 20,000 (n/p 9.8) | **0.449** (0.95) | 0.608 (0.75) | 0.636 (0.91) | 0.695 (0.76) |
| 2,048, n = 67,349 (n/p 32.9) | 0.546 (0.98) | **0.688** (0.76) | 0.671 (0.97) | 0.743 (0.79) |

The v1 setting is the 2,048 / 20,000 / real / equicorrelated cell (0.449). It matches §9's
`random_fixed` (0.43 over the same cells), which drew its knockoffs with knockpy.

**Effects at p = 2,048** (mean power gain; significant cells of 72 = 36 per comparison × 2):

| effect | real latents | Gaussian control |
|---|---|---|
| MVR instead of equicorrelated S | **+0.117 (39)** | +0.047 (23) |
| 3.4 × more rows (n/p 9.8 → 32.9) | +0.061 (16) | +0.039 (15) |
| no zero atom (control − real), equicorr. S | +0.143 / +0.111 at n/p 9.8 / 32.9 (21 / 19 of 36) | |
| no zero atom (control − real), MVR S | +0.063 / +0.052 (11 / 10 of 36) | |

- **Rule R5: the S matrix (near-copy knockoffs) is the main driver**, on real and Gaussian
  data alike. It is significant in more cells, with the larger mean.
- **Near-copy knockoffs and the zero atom compound each other.** The S effect is 2.5 times
  larger on real latents than on the control. Equivalently, the atom costs 0.11–0.14 power with
  equicorrelated knockoffs but 0.05–0.06 with MVR.
- **More rows helps less than expected, partly through the covariance estimator.** With more
  rows Ledoit–Wolf shrinks less, so equicorrelated knockoffs become *more* of a near-copy
  (0.946 → 0.977), offsetting part of the gain. Under MVR the rows effect is a clean
  +0.05 to +0.08.
- **Total rows matter more than rows per latent.** At the same n/p = 9.8, 512 latents with
  5,000 rows (0.27–0.45) do worse than 2,048 latents with 20,000 rows (0.45–0.61). The
  per-latent signal is fixed, so the lasso gains from total data. At similar n (17–20k),
  going from 512 to 2,048 latents costs 0.18 under equicorrelated S (0.629 → 0.449) but
  0.05 under MVR (0.656 → 0.608).
- At p = 512 the picture differs: rows dominate (+0.16 to +0.31), and the S matrix matters only
  at the smallest n (+0.14 at n = 5,000, +0.02 at 16,837). This is consistent with v2's
  "equicorrelated ≈ MVR" at n = 20,000.
- Weak interaction signals (amplitude 0.5) stay nearly undetectable everywhere (0.01–0.23).

**Practical reading.** At p = 2,048, block-diagonal MVR knockoffs plus all available rows
raise real-latent power at q = 0.10 from 0.449 to 0.688, with FDR controlled (max 0.029). The
zero atom's remaining cost is then about 0.05. Block MVR took about 40 s per S locally and
12–15 min on pkgpu's CPU. Its knockoffs (corr ~0.76) are less decorrelated than exact MVR's
at p = 512 (0.50), so there may be more power to gain.

## 13. Multiplicity reanalysis of the cell counts

Earlier sections count cells with mean − 1.96 SE beyond a target, across 36–126 cells, without
correction. `src/reanalysis_multiplicity.py` (committed with its output in `fef5773`) recomputes them
with Holm, Benjamini–Hochberg (BH) and Benjamini–Yekutieli (BY, valid under arbitrary dependence; the
cells share one activation matrix) within each family of cells, and replaces cell counts by one test
per claim where the claim is about an average or a trend: per-replicate means over the cells
(pooled test), or per-replicate slopes of FDR on log amplitude tested against 0.

| Claim | Uncorrected | Holm / BH / BY | One test per claim | Status |
|---|---|---|---|---|
| FDR above q at amplitude 20 (§10) | 2 of 126 | 0 / 0 / 0 | — | **withdrawn** |
| Real FDR above the control (v2) | 48 of 108 | 19 / 42 / 30 | — | holds |
| Real FDR above the control (amplitudes 1–20) | 80 of 126 | 47 / 76 / 63 | slope +0.025 per log-amplitude (real − control), t = 19.3 | holds |
| Zero atom costs power at p = 512 (§3) | 11 of 108 | 0 / 0 / 0 | −0.015; interaction −0.023 vs linear −0.008, difference p = 0.006 | average holds; cell counts withdrawn |
| Hurdle lowers FDR at amplitude ≥ 5 (§11, R3) | 40 of 72 | 4 / 27 / 0 | −0.019, p = 7 × 10⁻¹¹ | average holds; R3's cell criterion fails |
| S matrix drives power at p = 2,048 (§12) | 20, 19 of 36 | 10–12 / 18 / 15 | +0.125, p = 8 × 10⁻¹³ | holds |
| More rows raise power at p = 2,048 (§12) | 6, 10 of 36 | 0 / 0 / 0 | +0.053, p = 4 × 10⁻⁸ | average holds |

The control's FDR slope on log amplitude is −0.001 (p = 0.01): flat or slightly falling.

## 14. Calibration: where the real labels sit on the amplitude axis

Design `stage3_amendment_4` (committed `e0bb169` before the run; results `5a335e5`); script
`src/amplitude_calibration.py`; 2 min. A ridge-logistic probe (ridge chosen on validation log-loss)
is fitted to the real SST-2 labels and, with the same ridge and the same 60/20/20 split, to planted
labels at amplitudes 0.25–50 (3 draws each). Two placements: the amplitude at which the planted
labels give the same held-out probe AUC as the real labels, and the amplitude at which the probe's
top-k coefficient norm matches (the linear generator's coefficient vector has L2 norm = amplitude).
The ridge and amplitude grids were widened after a smoke test showed both choices at their edges
(recorded in the amendment).

Real labels: held-out AUC **0.972** (all 2,048 latents, 67,349 rows) and 0.933 (the Stage 3 subset).

| Planted design | AUC-matched amplitude (p = 2,048 / 512) | Norm-matched amplitude (p = 2,048 / 512) |
|---|---|---|
| linear, k = 10 / 20 / 30 | ≥ 50 / 37, 28 / 6.4, 11 / 5.2 | 1.2 / 1.6, 1.4 / 1.9, 1.5 / 2.0 |
| interaction, k = 10 / 20 / 30 | ≥ 50 / ≥ 50, ≥ 50 / 10.6, ≥ 50 / 8.0 | 2.0 / 2.7, 2.2 / 3.1, 2.4 / 3.2 |

The two placements differ by 5–30×: the real labels are as predictable as strong planted signals,
but no single latent carries much of it. The 10–30-signal benchmark of §1–§12 therefore never tested
the regime the real labels are in. The AUC match is imprecise near the top, where the planted curves
flatten (0.95–0.99); a linear probe cannot represent the interaction part of a label, so interaction
placements are lower bounds.

## 15. Many weak signals (k = 100–300), fixed penalty

Design `stage3_amendment_5` (committed `34ef131` before the run; results `f026ded`); run on pkgpu,
57 min. All 2,048 latents and 67,349 rows; seeded block-diagonal MVR knockoffs (mean s = 0.253,
corr(X_j, X̃_j) = 0.748) for the real latents and the Gaussian control; k = 100, 200, 300 ×
amplitudes 1–32 × both forms × q; 30 replicates; λ = 0.02. Amplitude 32 was added after a
diagnose run showed the interaction designs below the real labels' AUC at every amplitude up to 16
(recorded in the amendment).

- **K1 (control valid):** holds. **K3 (breach):** no cell, real or control, above q after BY.
- **K2 (erosion extends):** the real-minus-control FDR slope on log amplitude is positive for every
  k after Holm over the three k (p = 0.003, 0.047, 0.047), but about a quarter of the strong-signal
  slope (+0.005 to +0.007 against +0.025).
- Real-latent FDR exceeds the control's by +0.015, +0.014, +0.010 at k = 100, 200, 300 (pooled,
  p ≤ 3 × 10⁻⁴).
- With many signals the control itself uses much more of the budget than in §1–§12 (FDR 0.03–0.07
  at q = 0.10, against 0.004), and power falls with k at the fixed penalty (about 0.95, 0.7, 0.45 at
  k = 100, 200, 300 at strong signals).

FDR at q = 0.10 at the AUC-matched (realistic) amplitude:

| Design | k = 100 | k = 200 | k = 300 |
|---|---|---|---|
| linear (amplitude ≈ 8), real / control | 0.056 / 0.029 | 0.062 / 0.042 | 0.077 / 0.061 |
| interaction (amplitude ≥ 32, bound), real / control | 0.062 / 0.036 | 0.082 / 0.054 | 0.086 / 0.067 |

## 16. The cross-validated penalty

Design `stage3_amendment_6` (committed `4d9165e` before either full run; results `d06eaf7`); script
`src/stage3_followups.py --experiment cvlambda`; run on pkgpu (stress part 88 min, large-k part
292 min). Each saved cell is regenerated with the saved run's seeds (identical labels and knockoff
draws; reproduction check L0 exact for every arm) and refitted with λ chosen by 5-fold
cross-validation over λ_max × {0.5 … 0.0025} of [X, X̃] (lowest held-out log-loss), so cross-validated
and fixed λ are paired. The grid was extended down to 0.0025 after a smoke test put 17–28% of choices
at the old bottom edge (recorded in the amendment). Parts: the strong-signal cells of §10
(amplitudes 1, 5, 20, k = 10–30, 30 replicates) and the realistic designs of §15 (amplitudes 8 and
32, k = 100 and 300, 10 replicates, set from timing).

**Cross-validated λ:** median 0.0026 in the strong-signal part (0.02 × λ_max; 2–3% at a grid edge)
and 0.0005–0.0007 in the large-k part (0.005–0.01 × λ_max; 26% at the smallest value, just above the
pre-registered 25% limit and stated as a limitation, rule L3). If anything, a smaller λ would push
FDR higher.

**Strong-signal part (p = 512):**

| Amplitude | Real, fixed λ | Real, CV λ | Control, fixed λ | Control, CV λ |
|---|---|---|---|---|
| 1 | 0.012 | 0.094 | 0.008 | 0.077 |
| 5 | 0.041 | 0.115 | 0.004 | 0.081 |
| 20 | 0.086 | 0.143 | 0.004 | 0.083 |

(mean FDR at q = 0.10.) **L1:** 7 real-latent cells above q after BY, all at amplitude 20 with
interaction signals (largest 0.345 ± 0.024 at q = 0.20 and 0.207 ± 0.020 at q = 0.10); the control
none (largest cell 0.103 at q = 0.10). **L2:** real minus control at CV λ +0.035 (p = 3 × 10⁻⁹); the
amplitude trend persists (slope +0.013, p = 3 × 10⁻⁵; +0.023 at fixed λ). Power rises by +0.025.

**Realistic designs (p = 2,048, all rows):**

| Design (mean FDR at q = 0.10) | Real, fixed λ | Real, CV λ | Control, fixed λ | Control, CV λ |
|---|---|---|---|---|
| linear, amplitude 8, k = 100 / 300 | 0.067 / 0.059 | 0.098 / 0.111 | 0.025 / 0.060 | 0.095 / 0.088 |
| interaction, amplitude 8, k = 100 / 300 | 0.048 / 0.092 | **0.183 / 0.220** | 0.038 / 0.065 | 0.093 / 0.092 |
| linear, amplitude 32, k = 100 / 300 | 0.053 / 0.063 | 0.133 / 0.080 | 0.027 / 0.055 | 0.118 / 0.090 |
| interaction, amplitude 32, k = 100 / 300 | 0.059 / 0.098 | **0.357 / 0.277** | 0.035 / 0.069 | 0.090 / 0.105 |

**L1:** 12 of 24 real-latent cells above q after BY, every one an interaction design (largest 0.357
at q = 0.10, 0.531 at q = 0.20); linear designs sit at or slightly above q without a significant
breach; the control has none. **L2:** real minus control at CV λ +0.086 (p = 2 × 10⁻¹¹). Cross-validation
raises power from about 0.68 to about 1.0 for both arms.

**Interpretation.** With valid knockoffs a cross-validated penalty leaves FDR near but below q, as
the control shows. On real latents the invalid Gaussian knockoffs then produce real excess false
discoveries, concentrated where the label depends on products of latents. The fixed λ = 0.02 of the
earlier sections was legitimate but unusually conservative; it is why §1–§12 found only an eroding
margin. A follow-up testing the Stage 4 repairs under the cross-validated penalty in these realistic
designs (`stage4_amendment_4`) is running.

## 17. Reproduction

```bash
python src/planted_fdr_controls.py --config config/default.yaml --device cuda --stage diagnose  # ~4 min
python src/planted_fdr_controls.py --config config/default.yaml --device cuda --stage full      # ~76 min
```
Needs `knockpy==1.3.5`. Artefacts: `results/stage3_v2_fdr.json` (all cells, contrasts, harness,
convergence, λ check), `results/stage3_v2_records.npz` (per-replicate arrays),
`results/stage3_v2_diagnose.json`, `fig13_stage3_v2_fdr.png`, `fig14_stage3_v2_power.png`,
`fig15_stage3_v2_s_matrix.png`.

The p = 2,048 follow-up (§9):
```bash
python src/planted_fdr_controls.py --config config/default.yaml --device cuda --experiment p2048 --stage diagnose  # ~7 min
python src/planted_fdr_controls.py --config config/default.yaml --device cuda --experiment p2048 --stage full      # ~68 min
```
Artefacts: `results/stage3_p2048_diagnose.json`, `results/stage3_p2048_fdr.json` (all cells,
contrasts, the D1 check, convergence, draw diagnostics), `results/stage3_p2048_records.npz`
(per-replicate arrays), `fig16_stage3_p2048_power.png`.

The follow-ups (§10–§12):
```bash
python src/stage3_followups.py --config config/default.yaml --device cuda --experiment stress --stage full  # ~96 min (RTX 5050)
python src/stage3_followups.py --config config/default.yaml --device cuda --experiment dims --stage full    # ~100 min (pkgpu)
```
Both checkpoint after each replicate and resume when rerun. Artefacts: `results/stage3_stress.json`,
`results/stage3_stress_records.npz` (per-replicate metrics and per-link-bin mechanism counts),
`fig19_stage3_boundary.png`, `fig20_stage3_mechanism.png`; `results/stage3_dims.json`,
`results/stage3_dims_records.npz`, `fig21_stage3_dims.png`. A `--stage diagnose` run of each
gives timing, sampler checks and a one-cell pilot in minutes.

§13–§16:
```bash
python src/reanalysis_multiplicity.py --config config/default.yaml                     # seconds; reads saved records
python src/amplitude_calibration.py --config config/default.yaml --device cuda         # ~2 min
python src/stage3_followups.py --config config/default.yaml --device cuda --experiment largek --stage full              # ~1 h (pkgpu)
python src/stage3_followups.py --config config/default.yaml --device cuda --experiment cvlambda --part stress --stage full   # ~1.5 h
python src/stage3_followups.py --config config/default.yaml --device cuda --experiment cvlambda --part largek --stage full   # ~5 h
```
The cvlambda parts need the saved `stage3_stress_records.npz` and `stage3_largek_records.npz`, which they
pair with; largek needs `stage3_calibration.json`. Artefacts: `results/multiplicity_reanalysis.json`,
`stage3_calibration.json` + `fig23`, `stage3_largek.json` + records + `fig24`, `stage3_cvlambda_{stress,largek}.json`
+ records.
