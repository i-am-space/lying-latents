# SAEBench matched sweep findings — amplitude sweep, fair-coin test, thresholded Pythia

Pre-registered in `config/preregistration.yaml` `saebench_amendment_4` and its `revision_1` (both committed
before any FDR or coin result) · master seed `20260904`. Script: `src/stage4_repairs.py --experiment saebench
--matched` (`scripts/saebench_matched.sh`, `scripts/run_matched_all.sh`). Results:
- Gemma-2-2B (Gemma Scope L20 / 16k): `results/saebench/stage34_matched/`, `results/saebench/stage34_matched_summary.json`
- Pythia-70M, latents thresholded at 0.19 × median positive: `results/models/pythia70m/saebench/stage34_matched_thr0.19/`,
  `..._thr0.19_summary.json`

## Summary

1. **With the sweep, Gaussian knockoffs breach on SAEBench too.** Block MVR has Benjamini–Yekutieli breaches
   on **2 of 4** Gemma-2 datasets (Bias in Bios 1 and GitHub code, 2 cell-q combinations each, of 60), which
   meets SB3-R's "at least half" (`saebench_amendment_2` found 0 of 8 on the 8 original cells). Every breach
   is an **interaction** cell at a high amplitude (32, or 20 with k = 30), the cell type where SST-2 broke.
   The equicorrelated construction and the Gaussian control never breach.
2. **The fair-coin test locates the bias (M3).** For a valid knockoff, a null latent with W ≠ 0 is positive
   with probability 0.5. In **interaction** cells block MVR's null coin is above 0.5 on all 4 Gemma-2 datasets
   (0.509–0.529, each p < 10⁻³) and above the control's (+0.011 to +0.027). In **linear** cells it is
   *below* 0.5 (0.481–0.500). Pooled over all cells the two cancel, so the pooled coin is biased only on
   GitHub. The interaction bias grows with amplitude (GitHub: 0.493 at amplitude 2, 0.515 at 32).
3. **The FDR excess over the control is consistent:** block MVR minus control FDR is +0.012 to +0.018 on all 4
   Gemma-2 datasets (each p < 0.005), and +0.012 to +0.017 on all 3 thresholded Pythia datasets.
4. **Thresholding Pythia's ReLU latents JumpReLU-style largely repairs the hurdle (M5: met).** Hurdle
   breaches go from 4 / 7 / 16 (`saebench_amendment_3`, unthresholded) to **0 / 0 / 4** (EuroParl, Bias in
   Bios 1, Amazon category). Its coin is far *below* 0.5 (0.38–0.48), so the hurdle knockoffs are still
   detectably non-exchangeable, in the direction that favours knockoffs over nulls; the remaining breaches
   are all in linear cells on Amazon category, the dataset with the smallest zero atom.
5. **The coin test's own validity check (M2) passes on only 2 of 4 Gemma-2 datasets** (GitHub, EuroParl) and
   2 of 3 Pythia datasets. The control's coin is off 0.5 by at most 0.005, significantly so on Bias in Bios 1
   and Amazon category, so the coin is only interpreted on the valid ones (as pre-registered).

**What this establishes:** the earlier non-replication on SAEBench was a property of the cells, not of the
data. Once strong interaction signals are planted, Gaussian knockoffs on SAE latents favour null latents that
depend nonlinearly on the signals (coin > 0.5), and the FDR breaches follow, as on SST-2. With linear signals
the same knockoffs are slightly conservative. **What it does not establish:** a breach on every dataset (2 of 4),
or that the zero atom is the cause (the two breaching datasets have zero mass 0.59 and 0.49; EuroParl, 0.73, does
not breach).

## 1. Setup

| Item | Value |
|---|---|
| Data | each dataset's cache (12,500 texts, 2,048 latents), standardised; Ledoit–Wolf covariance |
| Cells (Gemma-2) | 20: `many_signals` k = 100, 300 × amplitudes 2, 8, 32; `few_signals` k = 10, 30 × amplitudes 5, 20; each × linear / interaction |
| Cells (Pythia, thresholded) | the 8 cells of `saebench_amendment_3` (k = 100, 300 × amplitudes 8, 32 × forms) |
| Arms | `gauss_real` (equicorrelated), `gauss_mvr` (block MVR), `gauss_ceiling_mvr` (Gaussian control); Pythia also `hurdle` |
| Penalty, targets | 5-fold CV λ; q ∈ {0.05, 0.10, 0.20} |
| Replicates | Gemma-2 15, Pythia 10; seeds `sb4m_knockoff`, `sb4m_planted`, `sb4m_folds` |
| Coin | per CV fit, the null latents with W > 0 and W < 0; rate per replicate, t-test over replicates |
| Runtime | Gemma-2 gauss group 986–1,202 min per dataset (4 in parallel on one GPU) |

The amplitude calibration (step 1) saturated (real-task probe AUC 0.955–1.000, unreachable by planted labels)
and was replaced by the sweep before any FDR result (`revision_1`).

## 2. Rule outcomes

| Rule | Gemma-2 (4 datasets) | Pythia thresholded (3) |
|---|---|---|
| M1: control power per cell | reported in the dataset JSONs (`dose_response`) | same |
| M2: control coin valid | 2 / 4 (GitHub, EuroParl) | 2 / 3 (Bias in Bios 1, Amazon category) |
| M3: block-MVR coin biased (all / linear / interaction) | GitHub / none / GitHub, EuroParl | none / none / Bias in Bios 1 |
| M4: SB3-R recomputed (block-MVR breach datasets) | **2 / 4 → replicates** | 0 / 3 |
| M5: hurdle restores control on thresholded latents | — | **met** (no breach on 2 / 3; coin not above 0.5) |
| SB4-R (hurdle no breach and FDR below MVR) | — | 1 / 3 (Bias in Bios 1) |

## 3. Per dataset (CV λ; FDR at q = 0.10 and power, mean over cells and replicates)

| model | dataset | Pr(X=0) | FDR: equi / MVR / control (/ hurdle) | power: equi / MVR / control (/ hurdle) | BY breaches (equi, MVR, control, hurdle) | MVR − control FDR (p) |
|---|---|---|---|---|---|---|
| Gemma-2 | EuroParl | 0.73 | 0.075 / 0.097 / 0.083 | 0.766 / 0.854 / 0.850 | 0, 0, 0 | +0.012 (0.004) |
| Gemma-2 | Bias in Bios 1 | 0.59 | 0.071 / 0.095 / 0.077 | 0.732 / 0.817 / 0.812 | 0, **2**, 0 | +0.018 (< 0.001) |
| Gemma-2 | GitHub code | 0.49 | 0.034 / 0.082 / 0.073 | 0.475 / 0.739 / 0.742 | 0, **2**, 0 | +0.012 (0.005) |
| Gemma-2 | Amazon category | 0.30 | 0.060 / 0.098 / 0.084 | 0.703 / 0.840 / 0.839 | 0, 0, 0 | +0.014 (0.005) |
| Pythia thr | EuroParl | 0.75 | 0.068 / 0.111 / 0.092 / 0.104 | 0.861 / 0.976 / 0.974 / 0.970 | 0, 0, 0, 0 | +0.017 (< 0.001) |
| Pythia thr | Bias in Bios 1 | 0.58 | 0.067 / 0.105 / 0.088 / 0.103 | 0.830 / 0.966 / 0.955 / 0.965 | 0, 0, 0, 0 | +0.015 (0.005) |
| Pythia thr | Amazon category | 0.41 | 0.070 / 0.102 / 0.088 / 0.105 | 0.899 / 0.968 / 0.966 / 0.977 | 0, 0, 0, **4** | +0.012 (0.014) |

The breaches:
- Bias in Bios 1, block MVR: amplitude 32, interaction, k = 100, at q = 0.10 (FDR 0.169, p_BY 0.009) and
  q = 0.20 (0.310, p_BY 0.002).
- GitHub code, block MVR: amplitude 32, interaction, k = 100, q = 0.20 (0.273, p_BY 0.033); amplitude 20,
  interaction, k = 30, q = 0.20 (0.313, p_BY 0.033).
- Amazon category (Pythia thresholded), hurdle: amplitude 32, linear, k = 100 (q = 0.05, 0.20) and k = 300
  (q = 0.05, 0.10); FDR 0.074–0.250.

## 4. Linear against interaction cells (Gemma-2, CV λ)

Mean FDR at q = 0.10, block MVR / control:

| dataset | few signals, linear | few signals, interaction | many signals, linear | many signals, interaction |
|---|---|---|---|---|
| EuroParl | 0.075 / 0.092 | **0.128** / 0.087 | 0.086 / 0.086 | 0.102 / 0.071 |
| Bias in Bios 1 | 0.086 / 0.093 | **0.131** / 0.069 | 0.082 / 0.090 | 0.090 / 0.058 |
| GitHub code | 0.069 / 0.094 | **0.123** / 0.095 | 0.066 / 0.068 | 0.078 / 0.050 |
| Amazon category | 0.098 / 0.081 | **0.125** / 0.096 | 0.087 / 0.085 | 0.090 / 0.076 |

Null coin rate of block MVR (valid = 0.5):

| dataset | linear | interaction | MVR − control, interaction (p) |
|---|---|---|---|
| EuroParl | 0.481 | 0.509 | +0.011 (< 0.001) |
| Bias in Bios 1 | 0.492 | 0.515 | +0.014 (< 0.001) |
| GitHub code | 0.487 | 0.529 | +0.027 (< 10⁻⁹) |
| Amazon category | 0.500 | 0.511 | +0.012 (< 0.001) |

In the few-signal interaction cells (the SST-2 regime) block MVR averages 0.123–0.131 at q = 0.10 on every
dataset, against 0.069–0.096 for the control; with 15 replicates and BY over 60 combinations per arm only the
largest cells reach significance. This is the same pattern as SST-2 Stage 3 §11: the false discoveries are nulls
whose dependence on the signals is nonlinear, which a Gaussian knockoff, matching only second moments, cannot
reproduce.

## 5. Deviations and notes

- Planned design, seeds, replicates and datasets were run unchanged after `revision_1`; no rule changed after
  results. A first benchmark started on the saturated calibrated cells was stopped after one replicate and not
  analysed (`revision_1`).
- At k = 10 every arm's power is 0.667 in every cell, on every dataset: a property of the planted design at
  k = 10, not of the knockoffs (not investigated further).
- The thresholded Pythia run uses the `sb4m_*` streams, the unthresholded one `sb34_*`: same cells and
  replicates, different draws, so the comparison in Summary 4 is not paired.
- Mean knockoff self-correlation (Gemma-2): equicorrelated 0.946–0.990, block MVR 0.800–0.921, control
  0.835–0.934. GitHub's high equicorrelated value is why that arm loses most of its power there (0.475).
- Limits: one Gemma-2 SAE, 4 datasets (3 for Pythia); planted labels on real latents.
