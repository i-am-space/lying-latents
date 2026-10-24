# Deep knockoffs on SAEBench — v1 and v2 findings

Pre-registered in `config/preregistration.yaml`:
- v1: `saebench_amendment_6` (and its `revision_1`, `revision_2`)
- v2: `saebench_amendment_7` (and its `revision_1`, `revision_2`)

Master seed `20260904`. Code: `src/deep_knockoffs.py`, `src/stage4_repairs.py --experiment saebench` (arm groups `deep`, `deep2`). Script: `scripts/deep_knockoffs.sh`.

Results:
- Gemma-2 matched sweep: `results/saebench/stage34_matched/*_deep{,2}_*`
- Gemma-2 8 datasets, original cells: `results/saebench/stage34/*_deep{,2}_*`, `results/saebench/stage34_summary.json` (keys `DK`, `DK_v2`)
- Pythia-70M: `results/models/pythia70m/saebench/stage34/*_deep{,2}_*`

## Summary

1. **Deep knockoffs repair nothing.** "Repairs" is rule DK2: no BY breach under CV, FDR not significantly above the Gaussian control, and (matched sweep) interaction coin not significantly above 0.5.
   - v1 repairs 0 of 13 dataset settings.
   - v2 repairs 2 of 13 (Gemma-2 AG News and EuroParl, original cells). Block MVR has no breach on either, so there was nothing to repair.
   - On the 3 pre-registered decision datasets, the ones where block MVR or the hurdle breached, v2 repairs **0 of 3**. The revision_1 bar of at least 2 of 3 is not met.
2. **v2 fixes v1's own failure, then stops at block MVR.** v2's FDR is lower than v1's on every Gemma-2 dataset (significant on 6 of 8, by 0.01–0.03). Its linear-cell coin is back to 0.48–0.49, from v1's 0.58–0.62. But v2 minus block MVR is not significant on any of the 8 Gemma-2 datasets (FDR difference −0.005 to +0.003).
3. **The learned part does nothing measurable.**
   - The network's share of the knockoff noise is 0.7–1.2%.
   - Early stopping ends after the validation loss improves by only 2–5%.
   - The third-order moment error equals block MVR's.
   - v2 minus `deep2_zero` (the same sampler with the network output set to 0) is not significant on the decision datasets.
   - All of v2's gain over v1 comes from the exact second-order construction, which the Gaussian already has.
4. **The interaction-cell excess remains.** The v2 interaction coin is 0.51–0.53 on the matched sweep, the same as block MVR (0.515–0.529). Learning nonlinear structure was meant to fix exactly this.
5. **Training does not damage a valid starting point (DK7).** `deep2_ctrl` (v2 trained on the Gaussian control data) has 0 breaches everywhere. Its FDR differs from the control's only on Gemma-2 Bias in Bios 3 (+0.008, p = 0.004).

**What this establishes:**
- A learned residual on the block-MVR Gaussian knockoff, trained on ~6k rows per fold, cannot learn the nonlinear dependence that makes Gaussian knockoffs fail in interaction cells.
- Per-latent diagnostics mislead. v1 looked best on single-latent swap AUC and zero mass, and had the highest FDR.
- Together with the hurdle and Gaussian results, three sampler families fail. That is the motivation for Stage 5, which needs no model of the latents.

**What it does not establish:** that no learned sampler can work. Larger unlabeled pools and different objectives were not tried.

## 1. The two variants

| | v1 (`deep`) | v2 (`deep2`) |
|---|---|---|
| Base | block-MVR Gaussian knockoff + learned residual | same, with exact second-moment correction |
| Zero atom | hard floor (no gradient) | learned fire / no-fire gate (straight-through) |
| Network | 2 × 1024 | trunk width 256, weight decay 1e-4 |
| Training | 4,000 steps, no held-out check | early stopping on 15% held out |
| Diagnostic arms | — | `deep2_ctrl` (DK7), `deep2_zero` (DK9: network output 0) |

## 2. Decision datasets (v2)

| Dataset | deep2 breaches | block MVR | v1 deep | Repairs? | Why not |
|---|---|---|---|---|---|
| Gemma-2 matched Bias in Bios 1 | 0 | 2 | 2 | No | FDR +0.020 above the control (p = 1e-4); interaction coin 0.512 (p = 1e-5) |
| Gemma-2 matched GitHub code | 5 | 2 | 1 | No | more breaches than block MVR |
| Pythia EuroParl | 3 | 0 | 12 | No | breaches where block MVR has none |

On Pythia Bias in Bios 1 and Amazon category, deep2 has 0 breaches, but so does block MVR. There deep2's FDR is +0.012 to +0.014 above the control, a small but significant gap.

DK9 ablation on the decision datasets:
- `deep2_zero` has FDR +0.013 to +0.022 above the control on all 5 main-set datasets (all significant).
- It breaches 1, 3 and 2 times on the 3 decision datasets.
- The exact second-moment correction changes the Gaussian noise by about 46%. The pre-registration expected about 17%.
- Compared with plain block MVR, deep2 is no better, and is worse on GitHub (+0.010 FDR, p = 0.05), EuroParl (p = 0.03) and Amazon (p = 0.03).

## 3. Gemma-2, 8 datasets, original cells (CV λ, FDR at q = 0.10)

| Dataset | deep | deep2 | deep2_zero | block MVR | control | Breaches (deep, deep2, deep2_zero, MVR) | DK2 (v2) |
|---|---|---|---|---|---|---|---|
| Bias in Bios 1 | 0.108 | 0.106 | 0.104 | 0.109 | 0.093 | 3, 1, 0, 0 | no |
| Bias in Bios 2 | 0.108 | 0.099 | 0.100 | 0.095 | 0.083 | 0, 1, 0, 0 | no |
| Bias in Bios 3 | 0.117 | 0.103 | 0.108 | 0.102 | 0.088 | 5, 0, 3, 0 | no |
| Amazon category | 0.099 | 0.097 | 0.098 | 0.095 | 0.083 | 0, 0, 0, 0 | no |
| Amazon sentiment | 0.078 | 0.076 | 0.084 | 0.075 | 0.064 | 0, 0, 0, 0 | no |
| GitHub code | 0.092 | 0.096 | 0.096 | 0.097 | 0.070 | 0, 0, 0, 0 | no |
| AG News | 0.103 | 0.088 | 0.099 | 0.095 | 0.087 | 1, 0, 0, 0 | **yes** |
| EuroParl | 0.114 | 0.107 | 0.111 | 0.104 | 0.094 | 3, 0, 0, 0 | **yes** |

Paired differences (DK3, over cells and replicates):

| Comparison | FDR | Power |
|---|---|---|
| deep2 − block MVR | −0.005 to +0.003 (none significant) | −0.001 to +0.013 |
| deep2 − v1 deep | −0.009 to −0.031 (significant on 6 of 8) | +0.011 to +0.030 |
| deep2 − control | +0.002 to +0.024 (not significant on AG News and EuroParl only) | — |

The coin is computed only on the matched sweep.

## 4. v1 recap (amendment 6)

- Repairs 0 of 13. Breaches on 9 of 13 (Pythia EuroParl 12).
- FDR above the control on 13 of 13 (+0.023 to +0.067), and above block MVR on 12 of 13.
- Coin 0.59–0.63 in both linear and interaction cells.
- Single-latent swap AUC 0.60–0.86, against block MVR's 0.79–0.98.
- Zero-mass swap rule at chance.
- Self-correlation 0.89–0.95, above the block-MVR target.

## 5. Notes

- No rule was changed after results. v2's revision_2 ran all settings at once instead of gating SET=main and gemma8 on the decision datasets. The 2-of-3 count is reported as pre-registered.
- The SET=main run (decision datasets and Pythia) was on the L40S machine; gemma8 on the RTX 5000 Ada machine. Same code and seeds.
- Limits:
  - one Gemma-2 SAE;
  - planted labels on real latents;
  - generators trained on ~6k rows per fold (no unlabeled pool).
