# Stage 3 findings — realised FDR of Gaussian knockoffs on real SAE latents (v2, with controls)

Config hash `d33d210c5acb` · master seed `20260904` · design fixed in
`config/preregistration.yaml` `stage3_amendment_1` (post-hoc, committed in `8f0babc` before
the full run) · script `src/planted_fdr_controls.py` · 76 min on an RTX 5050 laptop GPU.
The teammates' v1 (`src/planted_fdr.py`, `results/stage3_planted_fdr.json`) is kept as
history; §7 lists what was wrong with it.

## Summary

The benchmark plants a known set S of latents in synthetic labels Y = f(X_S) + noise, keeps
the real activations, and runs Gaussian knockoff+ at q ∈ {0.05, 0.10, 0.20}. It is run on
three datasets with identical draws: **real latents with MVR knockoffs** (primary), **real
latents with equicorrelated knockoffs** (v1's choice), and a **Gaussian control**
(N(0, Σ) with the real covariance, same S) on which Gaussian knockoffs are exactly valid.
So real-vs-control isolates the zero atom, and equicorrelated-vs-MVR isolates the S matrix.

1. **No evidence that FDR exceeds q anywhere.** 0 of 108 cells have mean − 1.96·SE > q on
   either real dataset; 107 of 108 have mean ≤ q. The one exception is at nominal
   (amplitude 8, linear, k = 20, q = 0.05: 0.053 ± 0.010).
2. **But the guarantee is not restored — the margin erodes with signal strength.** The
   valid-knockoff control has FDR ≈ 0.003–0.008 at q = 0.10 (Knockoff+ is very conservative
   here). Real-data FDR is significantly higher than the control in **48 of 108 cells**
   (mean +0.018) and rises with amplitude: at q = 0.10, 0.007 → 0.060 from amplitude 0.5
   to 8, while the control stays flat. In the highest-SNR small-k cells it reaches the
   nominal level (k = 10, amplitude 8: 0.099 ± 0.023 at q = 0.10, 0.115 ± 0.025 at
   q = 0.20; control 0.003). Control is empirical, not guaranteed; the sweep stops at
   amplitude 8, so behaviour beyond it is unknown.
3. **v1's "power collapse" is not reproduced and is not caused by the zero atom.**
   Power is within a few points of the Gaussian control for linear signals at every
   amplitude (11 of 108 cells significantly lower, none higher; mean −0.015), and
   equicorrelated ≈ MVR at p = 512. The atom does cost power for weak interaction signals
   (amplitude 0.5, k = 10: 0.233 vs 0.497) and at a stronger penalty (λ = 0.05, §5).
4. **v1's "zero power at k = 10, q = 0.05" is arithmetic.** Knockoff+ needs ≥ 1/q = 20
   discoveries; with the +1 offset removed the same cells have power 0.985.
5. **Marginal screening with multiplicity correction fails the conditional target; knockoffs
   do not.** A Westfall–Young marginal scan at α = q finds 150–210 latents with FDR
   0.78–0.93 against the planted conditional truth; Knockoff+ finds ≈ k with FDR 0.02–0.04.

**What this establishes:** in the tested range Gaussian knockoffs do not visibly inflate
FDR on zero-inflated latents, yet they measurably lose the finite-sample conservatism a valid
construction has, and the loss grows with signal strength. **What it does not:** behaviour
beyond amplitude 8, at p = 2048, at other penalties (two tried), on real rather than planted
labels, or with other statistics; the *mechanism* of the FDR rise is untested.

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
near-copy problem (corr 0.82) is much milder than at p = 2048 (0.977) and barely matters.

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
  biased positive as the signal sharpens. **This was not tested** and must be labelled a
  hypothesis in the paper.
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
declared in `stage3_amendment_1`, whose design commit (`8f0babc`, 03:42) precedes the run.
**Not tested:** p = 2048 with the control, so whether v1's power collapse came from p = 2048
(n/p ≈ 10), an unconverged solver (v1: step 0.1, 500 iterations, never checked) or the first-
20,000-row subsample is open.

## 9. Reproduction

```bash
python src/planted_fdr_controls.py --config config/default.yaml --device cuda --stage diagnose  # ~4 min
python src/planted_fdr_controls.py --config config/default.yaml --device cuda --stage full      # ~76 min
```
Needs `knockpy==1.3.5`. Artefacts: `results/stage3_v2_fdr.json` (all cells, contrasts, harness,
convergence, λ check), `results/stage3_v2_records.npz` (per-replicate arrays),
`results/stage3_v2_diagnose.json`, `fig13_stage3_v2_fdr.png`, `fig14_stage3_v2_power.png`,
`fig15_stage3_v2_s_matrix.png`.
