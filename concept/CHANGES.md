# concept/ — changes

## 2026-10-05 — concept dilution, Steps 0–8 (`16a0b86` … this commit)

- Scaffold, seed streams (spawn key 9000) and config overlay; nothing outside `concept/` edited.
- Caches: content-token residuals (layers 12, 20; fp32) and sparse per-sentence latents for 9 Gemma
  Scope SAEs (layer 12 16k–1M, layer 20 16k/65k). The layer-20 16k reproduction check fails the
  plan's 1e-4 criterion, explained by 14 JumpReLU threshold flips plus fp32 rounding.
- Library: block-S Gaussian sampler, group MVR, batched FISTA, group W, MKF(c)+, group-sum
  statistic; 46 tests.
- Pre-registration `concept_amendment_1` (before any labelled run) and `concept_amendment_2`
  (group-sum arm, before Steps 7–8).
- Results: toy proposition (T1 pass; group knockoffs rescue redundancy, not dilution, with the lasso
  statistic); census (median family 1 → 6 children from 16k to 1M, 99% dilution); gate G1 pass;
  Step 7 (per-latent certifies 90% → 18% of a concept's latents; P4 on concept power fails;
  group-sum beats per-latent in design A); Step 8 (real-label concept sets unstable across widths
  for every method). Write-up: `results/concept_findings.md`.
