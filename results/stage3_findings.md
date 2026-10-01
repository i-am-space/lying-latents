# Stage 3 findings — realised FDR of Gaussian knockoffs on real SAE latents (v2, with controls)

Config hash `d33d210c5acb` · master seed `20260904` · design fixed in
`config/preregistration.yaml` `stage3_amendment_1` (post-hoc, committed in `13c81c5` before
the full run) · script `src/planted_fdr_controls.py` · 76 min on an RTX 5050 laptop GPU.
The teammates' v1 (`src/planted_fdr.py`, `results/stage3_planted_fdr.json`) is kept as
history; §7 lists what was wrong with it. §9 reports a follow-up at v1's p = 2,048
(`stage3_amendment_2`) that explains v1's power collapse. §10–§12 report three further
follow-ups (`stage3_amendment_3`, script `src/stage3_followups.py`): the amplitude sweep
extended to 20, a direct test of why FDR rises, and a factorial that separates the causes of
the power collapse.

## Summary

The benchmark plants a known set S of latents in synthetic labels Y = f(X_S) + noise, keeps
the real activations, and runs Gaussian knockoff+ at q ∈ {0.05, 0.10, 0.20}. It is run on
three datasets with identical draws: **real latents with MVR knockoffs** (primary), **real
latents with equicorrelated knockoffs** (v1's choice), and a **Gaussian control**
(N(0, Σ) with the real covariance, same S) on which Gaussian knockoffs are exactly valid.
So real-vs-control isolates the zero atom, and equicorrelated-vs-MVR isolates the S matrix.

1. **No evidence that FDR exceeds q up to amplitude 8.** 0 of 108 cells have mean − 1.96·SE > q
   on either real dataset; 107 of 108 have mean ≤ q. The one exception is at nominal
   (amplitude 8, linear, k = 20, q = 0.05: 0.053 ± 0.010). **At amplitude 20 it does (§10):**
   two cells exceed q = 0.05 significantly (0.083 ± 0.016 and 0.076 ± 0.011).
2. **But the guarantee is not restored — the margin erodes with signal strength.** The
   valid-knockoff control has FDR ≈ 0.003–0.008 at q = 0.10 (Knockoff+ is very conservative
   here). Real-data FDR is significantly higher than the control in **48 of 108 cells**
   (mean +0.018) and rises with amplitude: at q = 0.10, 0.007 → 0.060 from amplitude 0.5
   to 8, while the control stays flat. In the highest-SNR small-k cells it reaches the
   nominal level (k = 10, amplitude 8: 0.099 ± 0.023 at q = 0.10, 0.115 ± 0.025 at
   q = 0.20; control 0.003). Control is empirical, not guaranteed. **Why (§11):** as the
   signal grows, null latents correlated with the signal latents beat their Gaussian knockoffs
   about 75% of the time instead of 50%; on the Gaussian control, with the same correlations,
   they do not. Hurdle knockoffs that model the zero atom cut FDR at amplitudes 5–20 by 30–45%
   but overshoot and are themselves detectable.
3. **v1's "power collapse" is real at p = 2,048, but it is mostly a high-dimension effect,
   not the zero atom.** At p = 512 (this run) power is within a few points of the Gaussian
   control for linear signals at every amplitude (11 of 108 cells significantly lower, none
   higher; mean −0.015), and equicorrelated ≈ MVR. The atom costs power only for weak
   interaction signals (amplitude 0.5, k = 10: 0.233 vs 0.497) and at a stronger penalty
   (λ = 0.05, §5). The follow-up at v1's p = 2,048 (§9) reproduces v1's low power in 36 of 36
   cells and rules out v1's solver and row choice. Most of the loss appears on the Gaussian
   control too, so it comes from p = 2,048 (n/p ≈ 10) together with near-copy equicorrelated
   knockoffs; these two cannot be separated there. The zero atom adds a smaller loss on top
   at p = 2,048 (11 of 36 cells, mean −0.073), so its cost grows with the number of latents.
   **The factorial in §12 separates them: the near-copy S matrix is the main driver.**
   Replacing equicorrelated with block-diagonal MVR knockoffs at p = 2,048 raises real-latent
   power by 0.12 (significant in 39 of 72 cells), against 0.06 for 3.4 times more rows;
   together they lift power at q = 0.10 from 0.45 to 0.69 with FDR still controlled.
4. **v1's "zero power at k = 10, q = 0.05" is arithmetic.** Knockoff+ needs ≥ 1/q = 20
   discoveries; with the +1 offset removed the same cells have power 0.985.
5. **Marginal screening with multiplicity correction fails the conditional target; knockoffs
   do not.** A Westfall–Young marginal scan at α = q finds 150–210 latents with FDR
   0.78–0.93 against the planted conditional truth; Knockoff+ finds ≈ k with FDR 0.02–0.04.

**What this establishes:** in the tested range Gaussian knockoffs do not visibly inflate
FDR on zero-inflated latents, yet they measurably lose the finite-sample conservatism a valid
construction has, and the loss grows with signal strength until, at amplitude 20, it is used
up (§10). The rise is tied to null latents that co-vary with the signal and is reduced by
modelling the zero atom (§11); the p = 2,048 power collapse is mostly the near-copy S matrix
(§12). **What it does not:** FDR at p = 2,048 above amplitude 1 (§9 and §12 test 0.5 and 1
only), other penalties (two tried), real rather than planted labels, or other statistics;
which non-Gaussian feature (the atom, co-firing, heavy tails) drives the bias in §11 is not
isolated; and block-diagonal MVR is an approximation to MVR at p = 2,048.

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
  reach a violation but did not leave the margin either.
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

## 13. Reproduction

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
