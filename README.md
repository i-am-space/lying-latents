# Lying Latents: Auditing Distribution-Free Inference for SAE Feature Discovery

Auditing whether the Model-X knockoff framework, as applied to Sparse Autoencoder (SAE) latents by
Enkhbayar (2025), satisfies the exchangeability property its FDR guarantee requires, and evaluating
its realised error control and statistical power against known ground truth.

**Scope of this repository is Stages 1–3** — activation cache, distributional
summary, Gaussian knockoff generation, exchangeability diagnostics with a
validated null harness, FWER calibration of the standard pipeline, and
semi-synthetic planted-signal FDR benchmark. Stages 4 and 5 are not built.

## What this does and does not establish

These are **necessary-condition** tests. Exchangeability is required only on null
latents, and the null set is unknown on real data, so the diagnostics are sufficient
to *falsify* exchangeability, never to confirm it. A violation **voids the FDR
guarantee**; it does not demonstrate that realised FDR exceeds the nominal target.
That is Stage 3 and is out of scope here.

## Layout

```
config/preregistration.yaml   frozen thresholds, seeds and decision rules
config/default.yaml           operational config (bound by the above)
src/cache_activations.py      SST-2 -> Gemma Scope SAE latents -> data/cache/<hash>.npz
src/describe_latents.py       distributional summary + premise check
src/knockoff_audit.py         harness validation, then the exchangeability diagnostics
src/calibrate_pipeline.py     Stage 1: FWER calibration under the global null
src/planted_fdr.py            Stage 3: planted-signal FDR benchmark with amplitude sweep
src/common.py                 config loading, seed derivation, cache hashing
scripts/                      one-off verification of reference-code behaviour
NOTES_reference.md            what the audited pipeline actually does
results/                      figures, JSON summaries, per-stage findings
```

## Reproducing

```bash
pip install -r requirements.txt
python src/cache_activations.py --config config/default.yaml   # ~5 min, 1 GPU
python src/describe_latents.py  --config config/default.yaml   # ~2 min, CPU
python src/knockoff_audit.py    --config config/default.yaml   # ~85 min, CPU
python src/calibrate_pipeline.py --config config/default.yaml --device cuda  # ~3 min, GPU
python src/planted_fdr.py       --config config/default.yaml --device cuda  # ~15 min, GPU
```

Every run is determined by `master_seed` in the config. Per-component RNG streams are
derived from it via `numpy.random.SeedSequence(...).spawn()` in a fixed order
(`src/common.py:STREAMS`). The cache filename is a hash of the cache-determining
config subset, so a cache built under a different aggregator or layer can never be
read by accident.

The activation cache (1.7 GB) is gitignored. Results are committed.

## Configuration

Gemma-2-2b, residual stream layer 20 (`blocks.20.hook_resid_post`), Gemma Scope
`gemma-scope-2b-pt-res-canonical` / `layer_20/width_16k/canonical` (JumpReLU, 16384
latents). SST-2 train, all 67,349 sentences, one row per sentence. Latents are
aggregated over non-special tokens only; **BOS must be excluded** — it is an
attention sink with residual norm ~2900 against ~350 for content tokens, and the SAE
does not model it (L0 ~7000 there against ~69 on real tokens). Retained latents are
those firing on ≥1% of sentences, capped at the top 2048 by firing rate.

## Result

Exchangeability is violated. Median accuracy of the trivial classifier `1{x = 0}` at
separating a real column from its knockoff is **0.945** (0.50 under exchangeability),
and swapping a **single** latent out of 2048 is detected by a held-out classifier at
**AUC 0.9966**, 65 standard deviations above a label-permutation null — against a
harness that returns AUC 0.4973 (CI containing 0.50) on Gaussian data where the same
knockoffs are provably valid.

This **voids the FDR guarantee**. It does not show that realised FDR exceeds the
nominal target; that is Stage 3 and was not run. Full writeup with the interpretation
constraints in [results/stage2_findings.md](results/stage2_findings.md).
