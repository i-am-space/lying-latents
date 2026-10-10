# E-value splitting on SAEBench, Gemma-3 and Pythia — findings

Pre-registered in `config/preregistration.yaml` `saebench_amendment_5` (commit `3220cdd`, before any run).

| Item | Setting |
|---|---|
| Cells, labels and replicates | Those of the knockoff arms already run: `stage4_saebench`, 8 realistic cells (amplitudes 8 and 32 × linear / interaction × k = 100, 300), q ∈ {0.05, 0.10, 0.20}, 10 replicates. |
| Pairing | Labels are regenerated from the same planted stream, so every e-value record is paired with the hurdle and Gaussian records. |
| E-value settings | Unchanged from SST-2 (`stage4_amendment_6`): split 0.5, at most 500 selected, κ = 0.5, e-BH over all p. |
| Selection penalty | Fixed λ = 0.02, or chosen by 5-fold CV within half A only. |
| Script | `src/stage4_repairs.py --experiment saebench --arm-group evalue` |
| Results | `results/saebench/stage34/`, `results/models/{gemma3_1b,pythia70m}/saebench/stage34/` (`*_evalue_records.npz`, `*_evalue_info.json`); rules in the per-dataset JSONs (`EV1`–`EV3`) and the `*_summary.json` aggregates |

## Summary

1. **E-values never breach (EV1).**
   - 0 Benjamini–Yekutieli breaches on all 14 datasets: Gemma-2 8, Gemma-3 3, Pythia 3.
   - That holds at both penalties.
   - Realised FDR at q = 0.10 is 0.000–0.008.
   - This includes Pythia, where hurdle knockoffs breach 4, 7 and 16 times.
2. **The power cost is large (EV2).**
   - Under the CV penalty, power is 0.27–0.46 below the hurdle on 13 datasets and 0.60 below on Amazon
     sentiment (n = 5,000).
   - The cost against block MVR is about the same.
   - Every difference is significant (paired, p < 10⁻⁷).
   - On SST-2 the cost was only −0.02 (`stage4_findings.md` §10).
3. **The 500 cap binds (EV3).**
   - 81–91% of CV fits select more than 500 latents on half A (median about 700–760). Amazon
     sentiment: 41%, median 475.
   - So these powers are those of the capped procedure.
4. **Likely reasons for the larger cost on SAEBench** (not tested):
   - Half the rows: about 6,250 per half on SAEBench against 10,000 on SST-2, and 2,500 on Amazon
     sentiment.
   - The conservative p-to-e calibration with many weak signals.
   - The cap dropping true signals when k = 300.

## Per dataset (CV λ, q = 0.10; mean over 8 cells and 10 replicates)

| Model | Dataset | Pr(X=0) | E-value FDR / power | Hurdle FDR / power (BY breaches) | Block MVR FDR / power | E-value − hurdle power | Share above cap |
|---|---|---|---|---|---|---|---|
| Gemma-2 | EuroParl | 0.73 | 0.002 / 0.631 | 0.094 / 0.957 (0) | 0.104 / 0.969 | −0.326 | 0.90 |
| Gemma-2 | AG News | 0.67 | 0.001 / 0.479 | 0.087 / 0.940 (0) | 0.095 / 0.947 | −0.461 | 0.86 |
| Gemma-2 | Bias in Bios 1 | 0.59 | 0.001 / 0.536 | 0.094 / 0.944 (0) | 0.109 / 0.943 | −0.409 | 0.91 |
| Gemma-2 | Bias in Bios 2 | 0.59 | 0.001 / 0.541 | 0.091 / 0.931 (0) | 0.095 / 0.920 | −0.391 | 0.81 |
| Gemma-2 | Bias in Bios 3 | 0.58 | 0.001 / 0.546 | 0.099 / 0.949 (0) | 0.102 / 0.948 | −0.403 | 0.91 |
| Gemma-2 | GitHub code | 0.49 | 0.003 / 0.460 | 0.103 / 0.893 (0) | 0.097 / 0.858 | −0.433 | 0.89 |
| Gemma-2 | Amazon sentiment | 0.32 | 0.008 / 0.130 | 0.069 / 0.734 (0) | 0.075 / 0.736 | −0.604 | 0.41 |
| Gemma-2 | Amazon category | 0.30 | 0.001 / 0.560 | 0.113 / 0.969 (1) | 0.095 / 0.955 | −0.409 | 0.85 |
| Gemma-3 | EuroParl | 0.71 | 0.001 / 0.650 | 0.097 / 0.959 (0) | 0.111 / 0.963 | −0.309 | 0.89 |
| Gemma-3 | Bias in Bios 1 | 0.62 | 0.001 / 0.558 | 0.105 / 0.946 (0) | 0.099 / 0.937 | −0.388 | 0.85 |
| Gemma-3 | Amazon category | 0.40 | 0.000 / 0.562 | 0.111 / 0.961 (0) | 0.098 / 0.950 | −0.399 | 0.82 |
| Pythia | EuroParl | 0.72 | 0.001 / 0.705 | 0.124 / 0.975 (4) | 0.106 / 0.977 | −0.270 | 0.81 |
| Pythia | Bias in Bios 1 | 0.51 | 0.001 / 0.574 | 0.134 / 0.969 (7) | 0.098 / 0.963 | −0.395 | 0.89 |
| Pythia | Amazon category | 0.31 | 0.001 / 0.615 | 0.158 / 0.982 (16) | 0.093 / 0.972 | −0.367 | 0.85 |

## What this establishes

Across three models and 14 datasets, no method is both valid and powerful on SAE latents:

| Method | Validity | Power |
|---|---|---|
| Gaussian knockoffs | Most powerful, but not exchangeable. They breach in interaction designs (SST-2; Bias in Bios and GitHub code in the matched sweep). | Highest |
| Hurdle knockoffs | Nearly as powerful and mostly within target on JumpReLU SAEs, but not exchangeable either. They fail on the ReLU SAE. | Nearly as high |
| E-value splitting | Valid in every setting run. | Gives up roughly 40% of the true discoveries at SAEBench sample sizes. |

## Constraints

- The e-values' validity rests on the half-B logistic refit being correctly specified, which it is not
  for the interaction cells; they hold there empirically.
- The powers are with the pre-registered 500 cap.
- 3 datasets per new model.
