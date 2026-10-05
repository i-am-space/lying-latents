# concept/ — concept dilution: does per-latent discovery degrade with SAE width?

**Thesis.** As an SAE dictionary widens, a concept is carried by more latents — split children that
fire on disjoint inputs (*dilution*) or near-duplicates (*redundancy*). Per-latent feature discovery
should therefore lose power with width, while concept-level (group) discovery should not. This branch
tests that and builds a two-level procedure (group knockoffs and the multilayer knockoff filter, MKF+).

Plan: `../CONCEPT_DILUTION_PLAN.md`. Pre-registration: `config/preregistration_concept.yaml`
(`concept_amendment_1`, committed before any labelled run). Findings: `results/concept_findings.md`.

Nothing outside `concept/` is edited on this branch, so the merge is a pure addition. Code imports the
main repo's `src/` (standardisation, Ledoit–Wolf, the FISTA lasso, Knockoff+ thresholds, planted
labels, multiplicity corrections) rather than copying it.

## Layout

```
config/concept.yaml                 overlay on ../config/default.yaml (master_seed unchanged)
config/preregistration_concept.yaml amendments for this branch
src/cseeds.py                       seed streams: SeedSequence(master_seed, spawn_key=(9000, i, ...))
src/cache_resid.py                  Step 2a  content-token residual cache (layers 12, 20), fp32
src/cache_sparse_latents.py         Step 2b  sparse per-sentence latents for one (layer, width)
src/check_reproduction.py           Step 2b  L20 16k vs the main cache
src/toy_proposition.py, toy_report.py   Step 1  synthetic proposition and library FDR test (T1)
src/family_census.py, freeze_tau.py     Step 3  parent mapping, family sizes, mechanism, tau rule
src/build_candidates.py             Step 4  fixed-p* candidate sets, groups, S solves (never reads labels)
src/group_knockoffs.py              Step 5  block-S sampler, group MVR, batched lasso, group W, MKF+
src/early_gate.py                   Step 6  gate G1
src/planted_concept.py, report_concept.py   Step 7  planted-concept benchmark and contrasts
src/width_sweep_real.py             Step 8  real SST-2 sweep and cross-width stability
src/bench.py, gpu.py                shared engine; CUDA start-up with retries
scripts/                            fetch_saes.sh, cache_queue.py, run_toy.sh, run_all.sh
tests/                              pytest -q concept/tests   (45 tests)
status.sh                           progress at a glance
```

Caches live in `data/cache/concept` (a symlink to `/scratch`; `/home` is full).

## Reproduce

```bash
pytest -q concept/tests                                         # library tests, ~10 s
concept/scripts/fetch_saes.sh data/cache/concept/sae 16 <paths>  # ~40 GB; link-bound (~2 h at 5 MB/s)
python concept/src/cache_resid.py --layers 12 20                 # ~4 min on one L40S
python concept/scripts/cache_queue.py                            # encodes each SAE when its download lands (<1 min each, 1M ~3 min)
concept/scripts/run_toy.sh                                       # Step 1 (~60 min: S solves on CPU, fits on one GPU)
concept/scripts/run_all.sh                                       # Steps 3-8, both GPUs, resumable (~90 min after caches)
bash concept/status.sh                                           # progress
```

Every long job is detached (`setsid nohup`) and checkpointed per width and replicate; rerunning a
script resumes it.
