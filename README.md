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
That is Stage 3 (`results/stage3_findings.md`): no inflation beyond q was found in the tested
range, but the margin a valid construction has erodes as signal strength grows.

## Layout

```
config/preregistration.yaml   frozen thresholds, seeds and decision rules
config/default.yaml           operational config (bound by the above)
src/cache_activations.py      SST-2 -> Gemma Scope SAE latents -> data/cache/<hash>.npz
src/describe_latents.py       distributional summary + premise check
src/knockoff_audit.py         harness validation, then the exchangeability diagnostics
src/calibrate_pipeline.py     Stage 1: FWER calibration under the global null
src/calibrate_validation.py   Stage 1: validate-step FWER (same-data vs held-out) + power gate
src/planted_fdr.py            Stage 3 v1: planted-signal FDR benchmark (superseded, kept as history)
src/planted_fdr_controls.py   Stage 3 v2: same benchmark with a Gaussian control, MVR S, WY baseline
src/stage3_followups.py       Stage 3 follow-ups: mechanism of the FDR rise, failure boundary, p = 2048 factorial
src/stage4_repairs.py         Stage 4: hurdle (SCIP), binarised and e-value repairs vs Gaussian knockoffs
src/saebench/                 SAEBench sparse-probing tasks: build_tasks, cache, stage1, stage2 (saebench_amendment_1)
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
python src/calibrate_pipeline.py --config config/default.yaml --device cuda  # ~6 min, GPU (fits in 8 GB)
python src/calibrate_validation.py --config config/default.yaml --device cuda  # ~5 min, GPU (Stage 1 validate step)
python src/planted_fdr.py       --config config/default.yaml --device cuda  # ~15 min, GPU (v1)
python src/planted_fdr_controls.py --config config/default.yaml --device cuda --stage full  # ~76 min, GPU; needs knockpy
python src/planted_fdr_controls.py --config config/default.yaml --device cuda --experiment p2048 --stage full  # p = 2048 follow-up
python src/stage3_followups.py --config config/default.yaml --device cuda --experiment stress --stage full  # mechanism + boundary
python src/stage3_followups.py --config config/default.yaml --device cuda --experiment dims --stage full    # power-collapse factorial
python src/stage4_repairs.py --config config/default.yaml --device cuda --stage full  # ~18 h on a server GPU; resumable
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

## SAEBench (Stages 1-2 on the 35 sparse-probing tasks)

Design: `config/preregistration.yaml` `saebench_amendment_1`; settings: `config/default.yaml` `saebench`.
Same model, layer, SAE and pooling as the SST-2 work; only the texts change.

```bash
# 1. texts, with SAEBench's own code, in a small separate env (its GitHub-code loader needs datasets<4)
conda create -n saebench-data python=3.12 -y && conda activate saebench-data
pip install "datasets>=3,<4" pandas tqdm transformers einops pyyaml && pip install sae-bench==0.6.0 --no-deps
python src/saebench/build_tasks.py --config config/default.yaml          # writes data/saebench/tasks/, results/saebench/tasks_manifest.json

# 2-4. main env
conda activate lying-latents
python src/saebench/cache.py  --config config/default.yaml               # GPU, one cache per dataset
python src/saebench/stage1.py --config config/default.yaml --device cuda  # GPU, per task, resumable
python src/saebench/stage2.py --config config/default.yaml               # CPU, per dataset, resumable
python src/saebench/stage1.py --config config/default.yaml --aggregate
python src/saebench/stage2.py --config config/default.yaml --aggregate
```
