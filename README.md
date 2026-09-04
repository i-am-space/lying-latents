# Lying Latents — Stage 2: Model-X exchangeability audit

Auditing whether the Model-X knockoff framework, as applied to SAE latents by
Enkhbayar (2025), satisfies the exchangeability property its FDR guarantee requires.

**Scope of this repository is Stage 2 only** — activation cache, distributional
summary, Gaussian knockoff generation, and exchangeability diagnostics with a
validated null harness. Stages 1, 3, 4 and 5 are deliberately not built.

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
src/common.py                 config loading, seed derivation, cache hashing
scripts/                      one-off verification of reference-code behaviour
NOTES_reference.md            what the audited pipeline actually does
results/                      figures, JSON summaries, stage2_findings.md
```

## Reproducing

```bash
pip install -r requirements.txt
python src/cache_activations.py --config config/default.yaml   # ~5 min, 1 GPU
python src/describe_latents.py  --config config/default.yaml   # ~2 min, CPU
python src/knockoff_audit.py    --config config/default.yaml   # ~30 min, CPU
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
