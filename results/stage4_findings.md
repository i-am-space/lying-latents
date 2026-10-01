# Stage 4 findings — distribution-aware repairs vs Gaussian knockoffs

Config hash `d33d210c5acb` · master seed `20260904` · design fixed in
`config/preregistration.yaml` `stage4_amendment_2` (committed in `a7d884d` before any
real-data run; solver fix recorded and committed in `908de58` before the valid run) ·
script `src/stage4_repairs.py` · run on pkgpu, 1,065 min.

## Summary

The proposal's three repairs were run on the Stage 3 planted-signal benchmark in the setting
where the zero atom costs power (p = 2,048 latents, n = 20,000 rows), against Gaussian
knockoffs and a Gaussian control that has no zero atom. All arms share labels; 30 replicates;
54 conditions (amplitude 0.5 / 1 / 3 × linear / interaction × k = 10 / 20 / 30 × q = 0.05 /
0.10 / 0.20).

| arm | power (all cells) | power (excl. Knockoff+ floor) | mean FDR | max FDR | cells with FDR > q |
|---|---|---|---|---|---|
| Gaussian knockoffs (baseline) | 0.551 | 0.620 | 0.020 | 0.068 | 0 |
| **Hurdle knockoffs (SCIP)** | **0.594** | **0.668** | **0.013** | **0.043** | **0** |
| Binarised knockoffs | 0.398 | 0.447 | 0.065 | 0.248 | **2** |
| E-value sample splitting | 0.655 | 0.639 | 0.001 | 0.011 | 0 |
| Gaussian control (no atom) | 0.607 | 0.682 | 0.009 | 0.035 | 0 |

"Excl. Knockoff+ floor" drops the 6 cells with k = 10, q = 0.05, where every knockoff method
has power 0 by arithmetic (Knockoff+ needs ≥ 1/q = 20 discoveries) and e-values reach 0.787.

1. **Hurdle knockoffs are the best repair in practice.** They have more power than Gaussian
   knockoffs (significantly higher in 9 of 54 cells, lower in none; mean +0.043), lower FDR
   on average (−0.008; never significantly higher), and close 77% of the power gap between
   Gaussian knockoffs and the no-atom control. Excluding the floor cells they come within
   0.014 of that control. 5 of the 9 gains are interaction signals at amplitude 1 (+0.12 to
   +0.24); the rest are linear signals at amplitude 0.5–1 (+0.02 to +0.23). No gains at
   amplitude 3, where every knockoff arm is already near 0.88.
2. **But they are not exactly exchangeable,** so the finite-sample guarantee is still not
   restored (rule F1). Swapping 50 latents is undetectable (AUC 0.504), but swapping all
   2,048 is detected (AUC 0.935; MMD p = 0.34). Their decoys are also less of a near-copy
   than Gaussian ones (mean corr(X_j, X̃_j) 0.864 vs 0.946), which by itself raises power, so
   per rule F4 the gain is attributed jointly to modelling the atom and to that difference.
3. **Binarised knockoffs pass the swap test and still fail on FDR.** Their decoys are
   indistinguishable from the binarised latents (full-swap AUC 0.504, MMD p = 0.92), yet
   FDR exceeds q in 2 cells (amplitude 3, linear, k = 30: 0.170 ± 0.020 at q = 0.10 and
   0.245 ± 0.017 at q = 0.20), is significantly higher than Gaussian in 25 of 54 cells, and
   power drops by 0.153 (lower in 36 cells). Valid decoys do not guarantee valid inference
   if the variables they stand in for are not the ones the truth is defined on (§3).
4. **E-value sample splitting is valid and conservative, not uniformly better.** FDR is
   below 0.011 everywhere, even at q = 0.20, where it leaves almost the whole error budget
   unused. It wins where Knockoff+ cannot discover anything (the floor cells) and with strict
   targets, and loses at q = 0.20 (up to −0.34 power). Excluding the floor cells its power
   (0.639) is below the hurdle's (0.668).

**Against the proposal's Stage 4 criterion** ("tighter adherence to the nominal FDR target or
strictly higher power than the Gaussian baseline"): the hurdle meets both; e-values meet the
FDR half and are mixed on power; binarised knockoffs meet neither.

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

- Modelling the zero atom conditionally (the proposal's hurdle) recovers most of the power
  Gaussian knockoffs lose to it at p = 2,048, while lowering FDR, on this benchmark.
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

## 6. Reproduction

```bash
python src/stage4_repairs.py --config config/default.yaml --device cuda --stage diagnose
python src/stage4_repairs.py --config config/default.yaml --device cuda --stage full       # ~18 h on pkgpu; resumable
```
Needs the activation cache and `knockpy==1.3.5`.
Artefacts: `results/stage4_repairs.json` (all cells, contrasts, swap tests, convergence),
`results/stage4_repairs_records.npz` (per-replicate arrays), `results/stage4_repairs_diagnose.json`,
`results/fig18_stage4_repairs.png`.
