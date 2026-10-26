# More powerful e-values (no cap, 5-split average) — findings

Pre-registered in `config/preregistration.yaml` `stage4_amendment_7` (commit `70913ce`, before any run).

| Item | Setting |
|---|---|
| Baseline (E0) | The existing e-values (`saebench_amendment_5`, `stage4_amendment_6`): one split, CV lasso on half A, Wald p on half B, cap 500, κ = 0.5, e-BH over all p |
| E1 `evalue_nocap` | E0 with no cap: every latent CV-selected on A is tested on B. Same split as E0, so E1 − E0 isolates the cap |
| E2 `evalue_k5` (primary) | E1's e-values from 5 independent splits, averaged per latent, then e-BH (Vovk & Wang 2021; e-BH valid under any dependence) |
| E3 split likelihood ratio | Dropped before any run: 0 power in the smoke test (the in-sample null MLE on ~750 dimensions dominates) and ~24 min per replicate |
| Cells, labels, replicates | Same as E0 and the knockoff arms, so every record is paired: SST-2 `stage4_cv` (8 cells), SAEBench 8 realistic cells; q ∈ {0.05, 0.10, 0.20}; 10 replicates; CV λ only |
| Datasets | SST-2; Gemma-2 8 SAEBench datasets; Gemma-3 and Pythia: EuroParl, Bias in Bios 1, Amazon category (15 settings) |
| Script | `src/stage4_repairs.py --experiment {cv,saebench} --arm-group evalue_plus` |
| Results | `results/stage4_cv_evalue_plus_*`, `results/saebench/stage34/*_evalue_plus_*`, `results/models/{gemma3_1b,pythia70m}/saebench/stage34/*_evalue_plus_*`; rules V1–V4 in `results/stage4_cv.json` (`evalue_plus`) and the per-dataset and `*_summary.json` files (`V_evalue_plus`) |

## Summary

1. **Neither variant closes the gap (V3).** No variant comes within 0.10 of the hurdle's power on any
   SAEBench dataset. The pre-registered success bar (a majority of the 15) is not met.
2. **Removing the cap loses power (E1).**
   - E1 − E0 is −0.01 to −0.11 on SAEBench and −0.23 on SST-2.
   - Significant (Holm) on 13 of 15 settings.
   - Testing ~700 latents on ~6k rows instead of 500 makes the half-B Wald fit noisier, so fewer
     e-values clear the e-BH threshold.
3. **Averaging 5 splits helps a little on SAEBench and hurts on SST-2 (E2).**
   - SAEBench: +0.01 to +0.10 (median about +0.04). It recovers 2–17% of the E0-to-hurdle gap.
   - Significant gains on 8 of 14 SAEBench settings; a significant loss on Pythia EuroParl (−0.035).
   - SST-2: −0.16. E2 inherits E1's no-cap loss, and averaging cannot make up for it.
4. **E2 breaches on Gemma-2 AG News (V1).**
   - 3 BY breaches, all in one cell (amplitude 8, linear, k = 100): FDR 0.30, 0.36 and 0.45 at
     q = 0.05, 0.10 and 0.20.
   - E1 and E0 have 0 breaches everywhere, including this cell.
5. **Cause: the half-B Wald p-values are anti-conservative (diagnostic, not pre-registered).**
   - Known-null latents in that AG News cell, 10 replicates, half-B refit.
   - With the 500 cap (E0): null z variance 1.69 instead of 1; P(|z| > 3.29) = 0.0115, about 11× the
     nominal 0.001.
   - Without the cap (~700 latents): variance 5.17; P(|z| > 3.29) = 0.156.
   - E0 stays within target only because the κ = 0.5 calibrator and e-BH are conservative. E2's
     averaging lets more of these inflated e-values through.
   - Not yet separated: too many latents per row against signal latents left out of the refit.
6. **Cost (V4).** E1 takes 27–76 s per replicate, E2 137–382 s (pkgpu, RTX 5000 Ada).

## Per dataset (CV λ; power is the mean over cells × q × replicates)

V2 is the paired difference from E0 (Holm over the 2 variants). The gap is the share of the
E0-to-hurdle power gap recovered.

| Model | Dataset | E0 power | Hurdle power | E1 − E0 (p Holm) | E2 − E0 (p Holm) | E2 gap recovered | E2 breaches |
|---|---|---|---|---|---|---|---|
| Gemma-2 | SST-2 | 0.952 | 0.974 | −0.232 (2e-7) | −0.162 (2e-6) | — | 0 |
| Gemma-2 | EuroParl | 0.631 | 0.957 | −0.058 (0.007) | +0.022 (0.22) | 0.07 | 0 |
| Gemma-2 | AG News | 0.479 | 0.940 | −0.048 (0.007) | +0.047 (0.014) | 0.10 | **3** |
| Gemma-2 | Bias in Bios 1 | 0.536 | 0.944 | −0.035 (0.002) | +0.051 (5e-4) | 0.13 | 0 |
| Gemma-2 | Bias in Bios 2 | 0.541 | 0.931 | −0.057 (0.003) | +0.036 (0.013) | 0.09 | 0 |
| Gemma-2 | Bias in Bios 3 | 0.546 | 0.949 | −0.046 (0.09) | +0.053 (0.024) | 0.13 | 0 |
| Gemma-2 | GitHub code | 0.460 | 0.893 | −0.051 (0.034) | +0.056 (0.034) | 0.13 | 0 |
| Gemma-2 | Amazon sentiment | 0.130 | 0.734 | −0.010 (0.24) | +0.103 (0.001) | 0.17 | 0 |
| Gemma-2 | Amazon category | 0.560 | 0.969 | −0.078 (6e-4) | +0.024 (0.13) | 0.06 | 0 |
| Gemma-3 | EuroParl | 0.650 | 0.959 | −0.087 (0.019) | +0.007 (0.72) | 0.02 | 0 |
| Gemma-3 | Bias in Bios 1 | 0.558 | 0.946 | −0.075 (0.002) | +0.030 (0.052) | 0.08 | 0 |
| Gemma-3 | Amazon category | 0.562 | 0.961 | −0.081 (0.015) | +0.013 (0.55) | 0.03 | 0 |
| Pythia | EuroParl | 0.705 | 0.975 | −0.108 (6e-5) | −0.035 (0.044) | −0.13 | 0 |
| Pythia | Bias in Bios 1 | 0.574 | 0.969 | −0.073 (0.008) | +0.048 (0.008) | 0.12 | 0 |
| Pythia | Amazon category | 0.615 | 0.982 | −0.074 (0.005) | +0.031 (0.034) | 0.08 | 0 |

On SST-2, E0 is already within 0.03 of the hurdle, so the gap fraction is not meaningful there.

## What this establishes

- The cap was not the bottleneck. Lifting it costs power, because the half-B refit gets worse with
  more latents per row.
- Averaging over splits is valid in theory but recovers little of the gap, and it is only as valid
  as the per-split p-values. With anti-conservative Wald p-values it breached in one AG News cell.
- **E0's validity is empirical, not guaranteed.** Its null z-scores are miscalibrated (variance 1.69);
  it held on all 15 settings because the calibration step is conservative. The paper should say
  "never breached in our runs", not "valid by construction".
- A real fix needs a half-B test that does not rely on large-sample Wald theory at these
  latents-per-row ratios (for example a much smaller cap, or a permutation or conditional test).
  That would be a new pre-registered experiment.

## Notes

- No rule was changed after results. E3 was dropped after the smoke test and before any run, and the
  drop is documented in the amendment.
- The one-line fix in `analyse_cv` (skip arms with no fixed-λ records) changes no reported value; the
  regenerated analysis JSONs only gain fields.
- The Wald calibration check (point 5) is an ad hoc diagnostic on one cell of one dataset.
