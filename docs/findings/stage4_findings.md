# Stage 4 findings — distribution-aware repairs vs Gaussian knockoffs

Config hash `d33d210c5acb` · master seed `20260904` · design fixed in
`config/preregistration.yaml` `stage4_amendment_2` (committed in `a7d884d` before any
real-data run; solver fix recorded and committed in `908de58` before the valid run) ·
script `src/stage4_repairs.py` · run on pkgpu, 1,065 min. §6–§9 add a block-MVR baseline
(`stage4_amendment_3`), corrected statistics, the repairs under the cross-validated penalty
(`stage4_amendment_4`) and hurdle stress tests (`stage4_amendment_5`).

## Summary

**Updated 7 October (§6–§9). The current conclusion:** under the lasso penalty chosen by
cross-validation (standard practice), Gaussian knockoffs, including the equicorrelated construction of
Enkhbayar (2025), exceed the FDR target on SAE latents, while **hurdle knockoffs keep FDR at the target
in the realistic designs** (0 breaches against 3 and 8 for the two Gaussian constructions; rule C3 met,
§8) at a power cost of 0.015. **Hurdle control is empirical, not guaranteed:** in stress tests (§9) it
breached once (1 of 42 conditions, against 6 and 16), in labels driven by activation magnitude, where
its log-normal size model is weakest. At the fixed penalty used in the original runs below, nothing
breaches, the hurdle trails block-MVR Gaussian knockoffs on power (§6), and most per-cell counts were
uncorrected (§7). **E-value splitting (§10)** also has no breach under CV λ but is very conservative
(FDR 0.001, power 0.952 against the hurdle's 0.974).

The proposal's three repairs were run on the Stage 3 planted-signal benchmark at p = 2,048 latents,
n = 20,000 rows, against Gaussian knockoffs and a Gaussian control with no zero atom, at the fixed
lasso penalty λ = 0.02. All arms share labels; 30 replicates; 54 conditions (amplitude 0.5 / 1 / 3
× linear / interaction × k = 10 / 20 / 30 × q = 0.05 / 0.10 / 0.20).

| arm | power (all cells) | power (excl. Knockoff+ floor) | mean FDR | max FDR | cells above q (uncorrected / BY) | corr(X_j, X̃_j) |
|---|---|---|---|---|---|---|
| Gaussian, equicorrelated (original baseline) | 0.551 | 0.620 | 0.020 | 0.068 | 0 / 0 | 0.946 |
| **Gaussian, block MVR (§6)** | **0.639** | **0.719** | 0.023 | 0.054 | 0 / 0 | 0.772 |
| Hurdle knockoffs (SCIP) | 0.594 | 0.668 | 0.013 | 0.043 | 0 / 0 | 0.864 |
| Binarised knockoffs | 0.398 | 0.447 | 0.065 | 0.248 | 2 / 0 | 0.797 |
| E-value sample splitting | 0.655 | 0.639 | 0.001 | 0.011 | 0 / 0 | — |
| Gaussian control, equicorrelated | 0.607 | 0.682 | 0.009 | 0.035 | 0 / 0 | 0.946 |
| Gaussian control, block MVR (§6) | 0.688 | 0.774 | 0.011 | 0.039 | 0 / 0 | 0.773 |

"Excl. Knockoff+ floor" drops the 6 cells with k = 10, q = 0.05, where every knockoff method
has power 0 by arithmetic (Knockoff+ needs ≥ 1/q = 20 discoveries) and e-values reach 0.787.
"BY" is the Benjamini–Yekutieli correction over each arm's 54 cells.

1. **Choosing the knockoff construction well matters more than modelling the zero atom.**
   Block-MVR Gaussian knockoffs beat equicorrelated ones by +0.100 power (excl. floor; pooled
   p < 10⁻¹³) and beat hurdle knockoffs by +0.045 (p = 5 × 10⁻⁷). By rule M1 the hurdle **trails**
   a well-chosen Gaussian baseline on power. With block MVR, the zero atom costs +0.048 power
   (control minus real).
2. **Hurdle knockoffs make fewer false discoveries.** Mean FDR 0.013 against 0.023 for block MVR
   (paired, −0.011, p = 10⁻⁹) and 0.020 for equicorrelated (−0.0075, p = 5 × 10⁻⁸). Their decoys are
   also harder to detect (full-swap AUC 0.935 against 1.000), but not exchangeable (rule F1), so the
   finite-sample guarantee is not restored. At the fixed penalty nothing breaches q, so lower FDR
   buys nothing here; whether it matters under the cross-validated penalty is §8.
3. **Binarised knockoffs pass the swap test and still inflate FDR.** Decoys indistinguishable from
   the binarised latents (full-swap AUC 0.504, MMD p = 0.92), yet FDR is consistently higher than
   the Gaussian baseline's (+0.044, p = 4 × 10⁻¹⁵; Holm-significant in 12 cells) and power is lower
   (−0.153; Holm 22 cells). Two cells exceed q uncorrected (amplitude 3, linear, k = 30: 0.170 ±
   0.020 at q = 0.10, 0.245 ± 0.017 at q = 0.20); one survives Holm and BH, none BY. A likely
   explanation, not tested, is that coarsening changes which latents are null (§3).
4. **E-value sample splitting is valid and conservative.** FDR ≤ 0.011 everywhere, even at
   q = 0.20. Its power advantage comes from the floor cells; excluding them it is below the
   hurdle's (−0.029, p = 7 × 10⁻⁵) and the block-MVR baseline's.

**Against the proposal's Stage 4 criterion** ("tighter adherence to the nominal FDR target or
strictly higher power than the Gaussian baseline"): at the fixed penalty, the hurdle met both against
the original equicorrelated baseline and the FDR half only against block-MVR Gaussian knockoffs. Under
the cross-validated penalty it meets the FDR half decisively, keeping FDR at the target where both
Gaussian constructions exceed it (§8), with an empirical, not guaranteed, margin (§9). E-values meet the
FDR half; binarised knockoffs meet neither.

## 1. Exchangeability (rules F0, F1)

Stage 2's swap tests (gradient-boosted classifier AUC, MMD permutation p-value), one decoy draw
per arm:

| arm | \|S\| = 0 | \|S\| = 50 | \|S\| = 2,048 | MMD p (full) | max zero-mass error |
|---|---|---|---|---|---|
| Gaussian knockoffs | 0.500 | 1.000 | 1.000 | 0.58 | — (no zeros) |
| Hurdle (SCIP) | 0.489 | 0.504 | **0.935** | 0.34 | 0.005 |
| Binarised | 0.510 | 0.492 | **0.504** | 0.92 | 0.005 |
| Gaussian control | 0.500 | 0.498 | 0.495 | 0.93 | — |

F0 holds: the test catches the known violation (1.000) and passes the valid control (0.495).
F1: binarised not detected as violated; hurdle violated. The hurdle and binarised samplers use
the same model for whether a latent fires, so the remaining mismatch is most likely the
hurdle's log-normal model of activation size (a hypothesis, not tested). For comparison, the teammate's Gaussian-copula hurdle
(`stage4_amendment_1`, `results/stage4_hurdle.json`) scored 0.648 at |S| = 50 and 0.999 at
|S| = 2,048: the proposal's sequential construction is clearly closer to exchangeable.

## 2. FDR and power

Power / FDR by signal (mean over k and q):

| amplitude, form | Gaussian | Hurdle | Binarised | E-values | Control |
|---|---|---|---|---|---|
| 0.5, linear | 0.25 / 0.010 | 0.33 / 0.007 | 0.10 / 0.029 | 0.31 / 0.003 | 0.31 / 0.006 |
| 0.5, interaction | 0.07 / 0.007 | 0.09 / 0.005 | 0.01 / 0.008 | 0.03 / 0.000 | 0.11 / 0.003 |
| 1, linear | 0.74 / 0.022 | 0.78 / 0.013 | 0.48 / 0.062 | 0.96 / 0.001 | 0.80 / 0.013 |
| 1, interaction | 0.48 / 0.012 | 0.60 / 0.009 | 0.24 / 0.044 | 0.64 / 0.000 | 0.64 / 0.014 |
| 3, linear | 0.88 / 0.039 | 0.88 / 0.023 | 0.80 / 0.137 | 0.99 / 0.000 | 0.89 / 0.010 |
| 3, interaction | 0.88 / 0.031 | 0.88 / 0.020 | 0.76 / 0.108 | 0.99 / 0.001 | 0.88 / 0.007 |

Paired contrasts with Gaussian knockoffs (cells with |difference| > 1.96 SE, of 54):

| contrast | mean power | higher | lower | mean FDR | FDR higher |
|---|---|---|---|---|---|
| hurdle − Gaussian | +0.043 | 9 | 0 | −0.008 | 0 |
| binarised − Gaussian | −0.153 | 0 | 36 | +0.044 | 25 |
| e-values − Gaussian | +0.105 | 15 | 9 | −0.019 | 0 |
| control − Gaussian (the atom's cost) | +0.056 | 16 | 1 | −0.011 | 1 |

Gaussian knockoffs again kept FDR below q in every cell (max 0.068), replicating Stage 3 at
p = 2,048, and the atom again cost power (control higher in 16 cells).
`fig18_stage4_repairs.png` shows power and FDR against amplitude at q = 0.10.

## 3. Why binarised knockoffs inflate FDR (interpretation)

The knockoff filter on binarised latents tests whether each on/off indicator Z_j is needed to
predict Y given the other indicators. The planted labels depend on the real-valued activations
of the signal latents. Once those are coarsened to on/off, a latent outside the planted set can
still carry information about how strongly the signal latents fired, so it is no longer null
in the binarised problem even though it is null in the real one. Its discoveries are then
correct for the binarised question and false for the real one. This fits the pattern that FDR
grows with signal amplitude (0.03 → 0.14 for linear signals), but it was not tested directly.

## 4. What this establishes, and what it does not

- ~~Modelling the zero atom conditionally recovers most of the power Gaussian knockoffs lose
  to it~~ (withdrawn, §6): the hurdle's power gain over equicorrelated knockoffs disappears, and
  reverses, against block-MVR Gaussian knockoffs. Its value is FDR: under the cross-validated
  penalty it keeps FDR at the target where Gaussian knockoffs do not (§8), except for labels driven
  by activation magnitude (§9).
- It does not restore the finite-sample guarantee: the hurdle decoys remain detectable under a
  full swap. Its FDR control here is empirical, like the Gaussian baseline's in Stage 3.
- Exchangeable decoys are not sufficient: the binarised arm is the only one with exchangeable
  decoys and the only one that violated FDR, because coarsening changes the null hypothesis.
- **The Barber et al. (2020) KL bound is not estimated.** It requires the KL divergence between
  the true conditional law of each latent and the working model, and the true law is unknown.
  The hurdle model's atom makes that divergence finite, which is what the proposal's argument
  needs, but finite is not the same as known.
- Limits: one dataset (SST-2), one SAE, planted labels; the e-value arm's validity rests on its
  logistic refit being correctly specified (true for the linear planted signals, not for the
  interaction ones, where its FDR nonetheless stayed near 0); amplitudes above 3 untested here.

## 5. Process and deviations

- **Discarded runs.** The first diagnose runs (pkgpu and local) used plain Newton steps for the
  SCIP logistic fits. Most fits diverged (20–28% converged; decoy zero mass off by up to 0.94),
  and their swap tests and one-replicate pilot were seen. All of it was discarded. The fix,
  a backtracking line search with a Newton-decrement stopping rule, makes the solver compute
  the pre-registered estimator; in the valid run 100% of fits converged in all 30 draws and the
  decoys' zero mass matched to within 0.009. Models, penalties, grid, arms, seeds and decision
  rules were unchanged (`stage4_amendment_2.solver_fix_after_first_runs`).
- **Other code fixes before the valid run:** a GPU memory-fragmentation fix (no numerical
  effect) and checkpoint/resume (checked to reproduce an uninterrupted run exactly).
- **Deviations from the proposal** (all declared in the amendment): SCIP uses fitted
  conditional models, so it is exact only if they are correct; log-normal positive part (the
  proposal allows log-normal or Gamma); sampled log-activations clipped to the observed range
  ± 1; the e-value arm is not assumption-free; p = 2,048 run on a server.
- Rule outcomes: F0 holds; F1 hurdle violated, binarised not detected; F2 hurdle and e-values
  valid, binarised not; F3 hurdle gains in 9 cells, e-values mixed (15 / 9); F4 hurdle gain
  attributed jointly to the atom model and less near-copy decoys; F5 no setting changed after
  results.

## 6. Block-MVR Gaussian baseline (stage4_amendment_3)

Design committed before the run (`e9fd8b7`; seed fix `22882ff`, recorded in the amendment after an
unseeded first attempt was stopped before any replicate finished); results `fef5773`; run locally,
16 min. Two arms were added on Stage 4's labels (same streams), with new knockoff seeds: block-MVR
Gaussian knockoffs for the real latents (knockpy blockdiag, blocks ≤ 512, line search; seeded solve;
mean s = 0.231, corr(X_j, X̃_j) = 0.772) and the same for the Gaussian control. Rule M0 held exactly:
rerunning the original Gaussian arm reproduced every saved replicate value, so the new arms are paired
with the saved ones replicate by replicate.

| Paired contrast (power, 54 cells) | Mean | Cells significant, uncorrected (+ / −) | Holm (+ / −) | Mean FDR difference |
|---|---|---|---|---|
| hurdle − block MVR | −0.045 | 0 / 13 | 0 / 2 | −0.011 |
| block MVR − equicorrelated | +0.089 | 22 / 0 | 16 / 0 | +0.003 |
| control (block MVR) − block MVR | +0.048 | 14 / 0 | 7 / 0 | −0.013 |
| e-values − block MVR | +0.016 | 7 / 25 | 6 / 14 | −0.023 |

The hurdle trails most for weak linear signals (−0.18 at amplitude 0.5, q = 0.10) and is level at
amplitude 3. Its share of the block-MVR atom gap is negative (−94%): it is below the baseline it
would need to improve on. Swap tests: block-MVR knockoffs on real latents AUC 1.000 at |S| = 50 and
all; on the control 0.498 / 0.502.

## 7. Corrected statistics

All cell counts above were uncorrected (mean − 1.96 SE beyond a value, over 54 cells).
`src/reanalysis_multiplicity.py` (results `fef5773`) recomputes them with Holm, Benjamini–Hochberg
and Benjamini–Yekutieli corrections and gives one test per average claim (per-replicate means over
cells, paired over replicates):

| Claim | Uncorrected | Holm / BY | One test |
|---|---|---|---|
| hurdle gains power over equicorrelated (9 of 54) | 9 | 0 / 0 | +0.043 (all cells), p = 3 × 10⁻⁵ |
| hurdle lowers FDR vs equicorrelated | 13 | 1 / 1 | −0.0075, p = 5 × 10⁻⁸ |
| binarised raises FDR | 25 | 12 / 16 | +0.044, p = 4 × 10⁻¹⁵ |
| binarised breaches q | 2 | 1 / 0 | — |
| zero atom costs power (equicorrelated) | 16 | 4 / 4 | +0.063 (excl. floor), p = 2 × 10⁻⁸ |
| e-values vs equicorrelated, power | 15 + / 9 − | 11 + / 4 − (Holm) | +0.019 (excl. floor), p = 0.008 |

Averages hold throughout; most per-cell claims about the hurdle do not. FDR trends in amplitude
(per-replicate slopes on log amplitude): equicorrelated +0.015 (p = 6 × 10⁻⁸), hurdle +0.009
(p = 5 × 10⁻⁷), binarised +0.059 (p = 10⁻¹⁶), control +0.002 (p = 0.15).

## 8. Under the cross-validated penalty (stage4_amendment_4)

Stage 3 §16 found that with the lasso penalty chosen by cross-validation, Gaussian knockoffs on real
latents exceed q in the designs matched to the real labels (100–300 weak signals), while the control
does not. At the fixed penalty nothing breaches q, so FDR could not separate the repairs. This
follow-up (design committed `a022006` before the run) refits the hurdle, equicorrelated, block-MVR and
control arms on Stage 4's data (p = 2,048, n = 20,000) in those designs (k = 100, 300; amplitudes 8,
32; both forms; 10 replicates), each at the fixed and the cross-validated penalty on the same knockoff
draw and labels. Run on pkgpu in two arm groups; four duplicate copies of the Gaussian group were
accidentally started and three were stopped mid-run (all used the same seeds and wrote identical
values; the final records have no missing values and 10 replicates).

| Arm | Mean FDR at q = 0.10, fixed λ | CV λ | Worst cell at q = 0.10, CV λ | Breaches after BY (of 24), CV λ | Power, fixed / CV |
|---|---|---|---|---|---|
| Gaussian, equicorrelated (Enkhbayar's construction) | 0.047 | 0.100 | 0.141 | **3** | 0.508 / 0.965 |
| Gaussian, block MVR | 0.074 | 0.125 | 0.164 | **8** | 0.616 / 0.990 |
| **Hurdle** | 0.040 | **0.096** | 0.117 | **0** | 0.549 / 0.974 |
| Gaussian control, block MVR | 0.058 | 0.099 | 0.108 | 0 | 0.634 / 0.991 |

- **C1:** the control has no breach and sits at the target (0.099).
- **C2:** both Gaussian constructions breach, all in interaction designs (block MVR up to 0.164 at
  q = 0.10 and 0.302 at q = 0.20; equicorrelated up to 0.141 and 0.262). So the construction used in the
  existing SAE application also fails under a weak penalty.
- **C3 (met):** the hurdle has no breach, and its FDR is below block MVR's (paired, −0.028, p = 0.001), at a
  power cost of −0.015 (p = 4 × 10⁻⁶). Its FDR matches the control's.
- **C4:** 5–10% of CV choices at a grid edge.

## 9. Hurdle stress tests (stage4_amendment_5)

Hurdle knockoffs remain detectable (full-swap AUC 0.935), so their control is empirical. Design
committed `e8816e7` before the run: 14 cells in three designs, same data, arms and CV procedure as §8,
10 replicates. **few_strong:** amplitudes 8 and 20 × linear/interaction × k = 10, 30. **many_weak_extreme:**
amplitude 64, interaction, k = 100, 300. **size:** a new label form that depends only on how strongly the
planted latents fire when they fire (per latent, log1p(x) standardised over its firing rows, 0 elsewhere,
then the linear generator), amplitudes 8 and 32, k = 30, 100; it targets the hurdle's log-normal size
model. Rule S2: the hurdle fails in a design if any of its cells breaches after Benjamini–Yekutieli (over
its 42 cell-q combinations).

| Design | Hurdle | Equicorrelated | Block MVR | Control | Hurdle − block MVR, FDR / power |
|---|---|---|---|---|---|
| few_strong | 0 | 0 | 1 | 0 | −0.013 (p = 0.29) / 0 |
| many_weak_extreme | 0 | 0 | 5 | 0 | −0.051 (p = 0.0004) / −0.020 |
| size | **1** | 6 | 10 | 0 | −0.085 (p = 0.0002) / −0.091 |
| **total** | **1** | **6** | **16** | **0** | |

(Breaches after BY at CV λ; at the fixed penalty no arm breaches anywhere.)

- **S1:** control valid.
- **S2: the hurdle fails in the size design:** amplitude 32, k = 100, q = 0.20: FDR 0.292 ± 0.016. Another
  size cell is elevated without significance (amplitude 32, k = 30: 0.186 at q = 0.10). It holds in the
  few_strong and extreme designs.
- **S3:** the hurdle's FDR is below block MVR's where Gaussian knockoffs break; the price is power,
  most in the size design (−0.09).
- **S4:** no CV choice at a grid edge.
- The equicorrelated construction breaches only in the size design (6 cells), a second label structure,
  besides interactions, where Enkhbayar's construction fails; block MVR now also breaches in a few-strong
  cell at p = 2,048 (amplitude 20, interaction, k = 30, q = 0.20: 0.378).

**Interpretation.** Modelling the zero atom removes nearly all the violations Gaussian knockoffs show, but
not all: the remaining weakness is the size model, which matters when labels depend on activation
magnitude. The size form is a construction made to stress the hurdle; it shows the method can fail, not
that real labels look like it. A better size model (e.g. Gamma) or testing concepts rather than single
latents are the natural next steps.

## 10. E-value splitting under the cross-validated penalty (stage4_amendment_6)

Design committed `4c616a3` before the run: the §8 cells (same labels, 10 replicates), paired with the §8
records. The rows are split in half; the lasso penalty is fixed (0.02) or chosen by 5-fold CV **within half A
only**; at most 500 selected latents are tested on half B (e-values, BY). Run locally, 8.7 min.

| arm | fixed λ: FDR (q = .10) / power | CV λ: FDR / power | breaches after BY (CV) |
|---|---|---|---|
| **E-value splitting** | 0.009 / 0.567 | **0.001** / 0.952 | **0** |
| Hurdle | 0.040 / 0.549 | 0.096 / 0.974 | 0 |
| Gaussian, block MVR | 0.074 / 0.616 | 0.125 / 0.990 | 8 |
| Gaussian, equicorrelated | 0.047 / 0.508 | 0.100 / 0.965 | 3 |

- **E1:** no e-value breach under CV λ.
- **E2:** e-value minus hurdle at CV λ: FDR −0.114 (p = 4 × 10⁻¹⁰), power −0.023 (p = 2 × 10⁻⁷); at fixed λ,
  power +0.019 (p = 0.015). Against block MVR at CV λ: FDR −0.142, power −0.038.
- **E3: the 500 cap binds** under CV λ (median 891 latents selected on half A; 121 at fixed λ), so the
  e-value power is that of the capped procedure.

**Interpretation.** E-value splitting is valid but very conservative (FDR ≈ 0.001 at a 0.10 target); the
hurdle uses the error budget and keeps more power under the standard penalty. It is the
guarantee-backed fallback; the hurdle remains the more powerful repair.

## 11. Reproduction

```bash
python src/stage4_repairs.py --config config/default.yaml --device cuda --stage diagnose
python src/stage4_repairs.py --config config/default.yaml --device cuda --stage full       # ~18 h on pkgpu; resumable
```
Needs the activation cache and `knockpy==1.3.5`.
Artefacts: `results/stage4_repairs.json` (all cells, contrasts, swap tests, convergence),
`results/stage4_repairs_records.npz` (per-replicate arrays), `results/stage4_repairs_diagnose.json`,
`results/fig18_stage4_repairs.png`.

§6–§8:
```bash
python src/stage4_repairs.py --config config/default.yaml --device cuda --experiment mvr --stage full        # ~16 min; pairs with stage4_repairs_records.npz
python src/reanalysis_multiplicity.py --config config/default.yaml
python src/stage4_repairs.py --config config/default.yaml --device cuda --experiment cv --arm-group hurdle --stage full   # ~3 h (pkgpu)
python src/stage4_repairs.py --config config/default.yaml --device cuda --experiment cv --arm-group gauss --stage full    # ~2-3 h
python src/stage4_repairs.py --config config/default.yaml --experiment cv --stage analyse
python src/stage4_repairs.py --config config/default.yaml --device cuda --experiment stress --arm-group hurdle --stage full   # ~3.5 h
python src/stage4_repairs.py --config config/default.yaml --device cuda --experiment stress --arm-group gauss --stage full    # ~3.5 h
python src/stage4_repairs.py --config config/default.yaml --experiment stress --stage analyse
```
`scripts/run_when_gpu_free.sh -- <command>` starts a command once a GPU is completely free (no processes,
3 checks 30 s apart).
Artefacts: `results/stage4_mvr.json`, `stage4_mvr_records.npz`, `fig22_stage4_mvr.png`,
`results/multiplicity_reanalysis.json`; `stage4_cv_{hurdle,gauss}_{records.npz,info.json}`, `stage4_cv.json`;
`stage4_stress_{hurdle,gauss}_{records.npz,info.json}`, `stage4_stress.json`;
`stage4_cv_evalue_{records.npz,info.json}` (§10: `--experiment cv --arm-group evalue --stage full`, ~9 min).
