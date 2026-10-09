# Three models: what holds, what does not (summary for the team, 9 October)

Pre-registered in `config/preregistration.yaml` `saebench_amendment_3` (Pythia-70M with the
`pythia-70m-deduped-res-sm` ReLU SAE, layer 4; Gemma-3-1B with Gemma Scope 2, layer 22, 16k, medium
L0), alongside Gemma-2-2B (`saebench_amendment_1`/`_2`, `docs/findings/stage34_sae_findings.md`).
Stages 1–2 cover all 8 SAEBench datasets per model. Stages 3–4 cover 3 per model (EuroParl, Bias in
Bios 1, Amazon category, fixed in advance), a compute limit.

Results:
- `results/models/{gemma3_1b,pythia70m}/saebench/`
- the positive-part check, `results/models/positive_shape_check.json` (`scripts/diag_positive_shape.py`)

## Summary

1. **Stage 1 holds on every model.**
   - When search and validation reuse the same data, the standard pipeline certifies features under
     the global null on all 35 tasks.
   - Median FWER for the mean-difference score: Gemma-3 0.69, Pythia 0.94, against 0.05.
   - With held-out validation it falls to about α: 0.066 and 0.078.
   - So the failure is data reuse, as on Gemma-2.
2. **Stage 2 holds on every model.**
   - Gaussian knockoffs are detectably non-exchangeable on 8 of 8 datasets for each model.
   - Single-latent swap AUC is 0.84–0.95; the full swap gives 1.0.
   - The pre-registered validity check SB2-V passes on only 2 of 8 datasets for Gemma-3 and 5 of 8 for
     Pythia.
     - The control exceeds the 97.5% point of a 10-permutation null by at most about 0.015, the
       too-narrow-band issue described in `stage2_sae_findings.md` §4.
     - So Gemma-3 falls below the 3 valid datasets SBM-2 needs, and its generalisation verdict is
       formally not made.
3. **Gaussian knockoffs breach the FDR target only on SST-2.**
   - Under the cross-validated penalty, block MVR breaches on 1 of 3 Gemma-3 datasets (EuroParl, 2
     cells) and on no Pythia dataset.
   - The equicorrelated construction (Enkhbayar's) breaches on none.
   - With Gemma-2's 0 of 8 on SAEBench, SST-2 is the only setting where Gaussian knockoffs clearly
     fail.
   - The consistent small upward bias over the valid Gaussian control remains:
     - Gemma-3: +0.007 to +0.015 (significant on 2 of 3);
     - Pythia: +0.002 to +0.011 (significant on 1 of 3).
4. **Hurdle knockoffs fail on Pythia.**
   - They breach on all 3 Pythia datasets (4, 7 and 16 of 24 cell-q combinations; FDR at q = 0.10
     0.124 / 0.134 / 0.158; worst cell 0.221), mostly in linear designs.
   - They hold on Gemma-3 (0 breaches).
   - Gaussian arms and the control do not breach on Pythia, so this is the hurdle's own failure.
5. **The likely reason is the SAE type.**
   - Pythia's SAE is a plain ReLU SAE; the Gemma SAEs are JumpReLU, which has a threshold.
   - On Pythia, about 20% of each latent's positive values lie below 1% of its maximum (Gemma: 0–1.5%).
   - The smallest positive value is 10⁻⁴ of the median (Gemma: 0.13–0.36).
   - log x is left-skewed (−0.86 against +0.2 to +0.5).
   - The hurdle's log-normal size model cannot produce that tail of tiny activations. This matches its
     one Stage 4 stress-test failure (size-driven labels).
   - This is a descriptive match, not a causal test: the overall KS distance to a log-normal is similar
     across models.

## Per dataset (Stages 3–4, CV λ, q = 0.10)

| model | dataset | Pr(X=0) | FDR: equi / MVR / hurdle / control | power: equi / MVR / hurdle / control | BY breaches (equi, MVR, hurdle, control) |
|---|---|---|---|---|---|
| Gemma-3 | EuroParl | 0.71 | 0.075 / 0.111 / 0.097 / 0.096 | 0.890 / 0.963 / 0.959 / 0.964 | 0, **2**, 0, 0 |
| Gemma-3 | Bias in Bios 1 | 0.62 | 0.070 / 0.099 / 0.105 / 0.085 | 0.820 / 0.937 / 0.946 / 0.934 | 0, 0, 0, 0 |
| Gemma-3 | Amazon category | 0.40 | 0.062 / 0.098 / 0.111 / 0.094 | 0.823 / 0.950 / 0.961 / 0.951 | 0, 0, 0, 0 |
| Pythia | EuroParl | 0.72 | 0.070 / 0.106 / **0.124** / 0.098 | 0.866 / 0.977 / 0.975 / 0.974 | 0, 0, **4**, 0 |
| Pythia | Bias in Bios 1 | 0.51 | 0.057 / 0.098 / **0.134** / 0.089 | 0.817 / 0.963 / 0.969 / 0.954 | 0, 0, **7**, 0 |
| Pythia | Amazon category | 0.31 | 0.071 / 0.093 / **0.158** / 0.090 | 0.900 / 0.972 / 0.982 / 0.973 | 0, 0, **16**, 0 |

At the fixed λ = 0.02, no arm breaches on either model.

Pre-registered rule outcomes:

| Rule | Gemma-3 | Pythia |
|---|---|---|
| SB3-R (Gaussian breaches replicate) | not met (1 of 3 datasets) | not met (0 of 3) |
| SB4-R (hurdle restores control) | 1 of 3 (EuroParl) | 0 of 3 |
| SB-Z (zero mass vs excess) | ρ = 0.5, p = 0.67 | ρ = 0.5, p = 0.67 |

## Positive-part shape (same 3 datasets, cached latents)

| | Gemma-2 (JumpReLU) | Gemma-3 (JumpReLU) | Pythia (ReLU) |
|---|---|---|---|
| share of positives below 1% of the latent's max (median) | 0–0.015 | 0–0.010 | **0.17–0.22** |
| min positive / median positive | 0.13–0.25 | 0.15–0.36 | **0.0001–0.0002** |
| skew of log x (log-normal: 0) | +0.21 to +0.25 | +0.28 to +0.48 | **−0.83 to −0.90** |
| KS distance of standardised log x to N(0, 1) | 0.03–0.07 | 0.03–0.09 | 0.06 |

## What this means for the paper

- **Still strong.**
  - Stage 1: data reuse in search-then-validate, on three models and 35 tasks each.
  - Stage 2: Gaussian knockoffs are never exchangeable on SAE latents, so their FDR guarantee does not
    apply on any SAE we tested.
- **Narrower than before.**
  - Realised FDR inflation from Gaussian knockoffs is shown on SST-2 (Gemma-2, the heaviest zero atom,
    0.89) and weakly on one Gemma-3 dataset. Elsewhere there is only a small, consistent bias.
  - On Pythia-70M, Enkhbayar's model, his construction stayed at or below the target in our benchmark.
    Our claim must be "the guarantee does not hold and the method can fail", not "his results are
    wrong". Our Pythia run uses SAEBench tasks, not his exact SST-2 setup.
- **Withdrawn.** The hurdle as a general repair.
  - It works when its model of the activations fits the SAE (JumpReLU) and fails when it does not
    (ReLU).
- **Suggested framing.**
  - Whether knockoff FDR control works on SAE latents depends on the SAE's activation distribution.
    Neither Gaussian nor hurdle knockoffs are safe without checking.
  - The swap tests and the positive-part check are how to check.
- **Open (not run).** A causal test of the size-tail explanation, e.g. Pythia's hurdle with a size
  model that has mass near zero, or with tiny activations thresholded.
