# Concept dilution experiment — implementation plan for Claude Code

**Repo:** `i-am-space/lying-latents` (main @ `34ef131`)
**Branch:** `concept-dilution` (off `main`)
**Directory:** everything new lives in `concept/`. Do **not** edit anything under `src/`, `config/`, `scripts/` or `results/` on this branch. That keeps the merge a pure addition.

---

## 0. Context you need before writing code

### What the thesis is

Per-latent feature discovery in SAEs tests the wrong unit. As the dictionary gets wider, a concept is carried by more latents, through two mechanisms:

- **Dilution (feature splitting).** Children fire on mostly disjoint inputs. Each child is still conditionally non-null, but its standardised signal shrinks roughly like 1/√m (SNR like 1/m) for m children ([Bricken et al., 2023](https://transformer-circuits.pub/2023/monosemantic-features); [Leask et al., 2025](https://arxiv.org/abs/2502.04878)).
- **Redundancy (duplicates, absorption).** Children co-fire with near-identical roles, so each is conditionally null given the others even though the concept matters ([Chanin et al., 2024](https://arxiv.org/abs/2409.14507)).

**Prediction:** per-latent discovery degrades with width; concept-level (group) discovery does not. This branch tests that prediction and builds the two-level procedure.

### What already exists and must be reused (import, do not copy)

| Need | Existing code |
|---|---|
| Config + master seed | `src/common.py` (`load_config`, `STREAMS`, `rng_for`) |
| Standardise, covariance | `src/knockoff_audit.py:30` `standardise`, `:37` `estimate_cov` (Ledoit–Wolf) |
| FISTA L1-logistic, converged-checked | `src/planted_fdr_controls.py:144` `fit_lasso_checked` |
| Knockoff(+) threshold | `src/planted_fdr_controls.py:173` `knockoff_threshold(W, q, offset)` |
| Planted labels | `src/planted_fdr.py:39` `generate_planted_labels(X, S, form, amp, rng)` |
| FDR/power of a discovery set | `src/planted_fdr.py` `evaluate_discoveries` |
| Gaussian knockoffs on GPU | `src/stage3_followups.py:585` `GaussKnockoffGPU` (**diagonal S only**, see Step 5) |
| Seeded block-MVR S solve | `src/stage3_followups.py:629` `s_matrix(Sigma, "mvr", sec, seed)` |
| Full-data setting template | `src/stage3_followups.py:969` `build_largek` (all 2048 latents, all 67,349 rows, LW cov, block MVR) |
| Activation caching template | `src/cache_activations.py` (BOS/EOS/pad excluded, mean over content tokens) |

### Facts already established that constrain the design

- **n/p drives power.** At p = 2048 the power collapse came mostly from near-copy knockoffs and n/p, not the zero atom; block-diagonal MVR lifted power 0.45 → 0.69 (`CHANGES.md`, Stage 3 follow-ups; `results/stage3_dims.json`). **So p must be held fixed across widths, or width is confounded with n/p.**
- **Knockoff+ floor.** Knockoff+ needs ≥ 1/q discoveries, so planting fewer than 1/q concepts gives zero power by arithmetic (`CHANGES.md` §1). Plant ≥ 20 concepts, or report offset-0 knockoffs alongside (the existing `k0_*` metrics).
- **Real SST-2 signal is many weak latents** (probe AUC 0.972; amplitude calibration, `config/preregistration.yaml` `stage3_amendment_4/5`).
- **The project's culture is pre-register → diagnose → run → findings.** Each run gets an amendment written *before* the full run, a one-replicate diagnose pass, and a `*_findings.md`. Keep that.

### Gemma Scope widths actually available (checked against `sae_lens==6.37.6` `pretrained_saes.yaml`)

| Layer | Widths |
|---|---|
| **20** (current project layer) | 16k, 65k **only** |
| 19 | 16k, 65k, 1M |
| **12** | 16k, 32k, 65k, 131k, 262k, 524k, 1M |

131k at layer 12 is only in `gemma-scope-2b-pt-res` (pick the `average_l0_*` closest to the canonical L0); every other entry is in `gemma-scope-2b-pt-res-canonical` as `layer_{L}/width_{W}/canonical`.

**Decision:** the width sweep runs at **layer 12** (7 widths). Layer 20 at 16k vs 65k is a **bridge** to all existing results. A 1M width does not exist at layer 20, so do not try to load it.

---

## Step 0 — Branch and scaffold

```bash
git checkout main && git pull
git checkout -b concept-dilution
```

Create:

```
concept/
  README.md                       what this directory is, how to run it, how it merges
  config/concept.yaml             overlays config/default.yaml (see below)
  config/preregistration_concept.yaml   amendments for this branch only
  src/_paths.py                   inserts <repo>/src on sys.path; nothing else
  src/cseeds.py                   seed streams for this branch (see below)
  src/toy_proposition.py          Step 1
  src/cache_resid.py              Step 2a
  src/cache_sparse_latents.py     Step 2b
  src/family_census.py            Step 3
  src/build_candidates.py         Step 4
  src/group_knockoffs.py          Step 5 (library: block-S sampler, group S, group W, MKF+)
  src/early_gate.py               Step 6
  src/planted_concept.py          Step 7
  src/width_sweep_real.py         Step 8
  tests/                          unit tests (pytest)
  results/                        figures, JSON, findings
```

**Config overlay.** `concept/config/concept.yaml` holds only new keys plus `base: ../../config/default.yaml`. Write a `load_concept_config()` that loads the base with `common.load_config`, deep-merges the overlay, and asserts `master_seed` is unchanged.

**Seeds.** Do **not** append to `src/common.py:STREAMS`; that would edit main-line code. In `concept/src/cseeds.py`, define a separate ordered list `CSTREAMS` and derive each stream as `SeedSequence(entropy=master_seed, spawn_key=(9000, CSTREAMS.index(name), *cell_idx))`. Spawn key 9000 cannot collide with the main repo's `spawn(len(STREAMS))` children (keys 0–34). Mirror `planted_fdr_controls.cell_rng` for per-cell streams. Same rule as main: append only, never reorder.

**Caches** go under `data/cache/concept/` (already gitignored via `data/cache/`).

Commit: `Concept: scaffold directory, config overlay and seed streams`.

---

## Step 1 — Toy proposition, numerically (CPU, ~1 h)

**Purpose:** pin down the precise claim, and validate every testing procedure where truth is exact before touching SAEs.

`concept/src/toy_proposition.py`. A synthetic generator with one concept carried by a group G of m latents, plus nulls:

- **(a) Dilution:** a concept indicator z ~ Bernoulli(π); when z = 1, exactly one of m children fires (uniform), with a log-normal magnitude and a zero atom matching real latents (median zero mass ≈ 0.95, `results/stage2_findings.md`). Y = logistic(amp · standardise(Σ_{j∈G} X_j)).
- **(b) Redundancy:** m near-duplicate children (corr ≥ 0.95) carrying the same concept.

Sweep m ∈ {1, 2, 4, 8, 16, 32}, with p fixed (pad with null latents) and n fixed.

Report, per m:
1. Per-child standardised effect and its theoretical value (expect ∝ 1/√m under (a)).
2. Per-latent knockoff power (latent level) and **concept recovery** (≥ 1 child selected).
3. Group knockoff concept power.
4. MKF+ power at both layers.
5. Realised FDR at both levels against exact truth.

Deliverables: `results/toy_proposition.json`, `fig_c1_toy.png`, and a half-page LaTeX statement of the proposition in `results/proposition.md` (two parts matching (a) and (b)). The prediction is per-latent decay in m and flat group power. Use these toy runs as the **unit tests for Step 5** (FDR ≤ q at both layers across 200 null replicates).

---

## Step 2 — Caches for several widths

### 2a. Residual cache (GPU, once per layer)

`cache_resid.py`. Run Gemma-2-2B once per layer (12 and 20) over SST-2 train. Store **only content tokens**, using the same masking rules as `src/cache_activations.py` (exclude BOS, EOS, pad; BOS is an attention sink with norm ≈ 2900). Write the activations as an fp16 memmap of shape (total_tokens, 2304) plus an offsets array per sentence. This decouples the model from the SAEs, so each width needs only an encode pass.

### 2b. Sparse latent cache per (layer, width) (GPU)

`cache_sparse_latents.py`. For each SAE:

- Encode residual chunks of about 2,048 tokens. A 1M-wide fp32 SAE is about 19 GB of weights, and a dense (2048 × 1M) encode is about 8 GB, so this fits on an L40 with chunking. **Never materialise (batch, T, d_sae) at 1M.**
- Mean-aggregate per sentence over content tokens, as in the existing cache, and store a `scipy.sparse.csr_matrix` (n × d_sae, float32). Expect ≤ ~1,500 nnz per sentence.
- Store `firing_rate_all`, `labels`, `sentence_ids`, the SAE id, mean L0 on content tokens, and explained variance. **Log mean L0 per width:** canonical SAEs differ in L0, which is a confound to report.
- **Sanity check:** layer 20, 16k, restricted to the existing retained 2048 latents, must reproduce the existing cache's `X_mean` (max abs diff < 1e-4). This proves the new pipeline is the old one.

Also save each SAE's decoder `W_dec` (row-normalised, fp16) to `data/cache/concept/dec_L{layer}_{width}.npy` for Step 3.

Commit: `Concept: residual and sparse multi-width latent caches; 16k L20 reproduces the main cache`.

---

## Step 3 — Family census (decides whether there is anything to test)

`family_census.py`. All label-free.

1. **Reference parents:** at each layer, the 16k latents with firing ≥ 1%, top 2048 by firing rate. This is the existing filter (`config/default.yaml` `latent_filter`); at layer 20 it is exactly the existing retained set.
2. **Parent mapping:** for each wider SAE, assign each latent to `argmax` decoder cosine over the 16k decoder, keeping its max cosine. Compute this in GPU chunks (1M × 2304 by 2304 × 16k).
3. **Per parent, per width:** number of children with cos ≥ τ, for τ ∈ {0.3, 0.4, 0.5, 0.6, 0.7}; children's firing rates; sum of children's firing vs parent firing.
4. **Mechanism split per family:** mean pairwise co-firing Jaccard and activation correlation between children. Low co-firing means **dilution**; correlation ≥ 0.9 means **redundancy**. Report the fraction of families in each regime per width.

Figure `fig_c2_census.png`: family size vs width (median, IQR), and mechanism fractions vs width.

**Gate C0:** if the median family size at the widest width is 1 for every τ, splitting is too rare at this layer for the thesis; stop and report.

**Freeze τ** from the diagnose output (the smallest τ where children's summed firing tracks parent firing) and record it in the amendment before Step 6.

---

## Step 4 — Candidate sets and groups (label-free, fixed p)

`build_candidates.py`. Writes `results/candidates_L{layer}.json`.

**Hold p fixed at p\* = 2048 at every width.** Otherwise width is confounded with n/p.

- **Planted pool:** a seeded random set of K_pool 16k parents whose families exist at every width (cos ≥ τ) and whose union at the widest width fits within about half of p\*. K_pool must be ≥ 40 so that 30 planted concepts can be drawn.
- **Filler:** add whole families of other parents, in seeded random order, until p = p\*. Truncate the last family rather than exceeding p\*.
- **Child firing floor:** 0.1%, not 1%. The 1% filter would delete split children by construction, because dilution lowers firing rates. This is the most important design detail here.
- **Groups:** primary = parent family (decoder-only, so it uses no X and no Y). Sensitivity = agglomerative clustering on decoder cosine within the width (average linkage, distance 1 − cos, same τ). **Cap groups at 64 latents**: knockpy's `merge_groups` fails on groups larger than `max_block`.
- At 16k each group is a singleton, so group and per-latent testing coincide. That is the m = 1 anchor.

**Never use Y to form groups:** a label-dependent grouping voids FDR control. Add an assertion that `build_candidates.py` never loads `labels`.

---

## Step 5 — Group knockoff library

`group_knockoffs.py`, with tests in `concept/tests/`.

1. **`BlockGaussKnockoffGPU`.** `GaussKnockoffGPU` takes `np.diag(S)` and silently discards off-diagonal blocks, which is wrong for group S. Re-implement it with full block-diagonal S: A = I − Σ⁻¹S, V = 2S − SΣ⁻¹S, then sample as before. **Test:** with singleton groups it must match `GaussKnockoffGPU` exactly given the same seed.
2. **Group S:** `knockpy.smatrix.compute_smatrix(Sigma, groups=g1, method="mvr", how_approx="blockdiag", max_block=512)`, where `g1` is 1-indexed. **Pass `method="mvr"` explicitly**: with groups and no method, knockpy defaults to `"sdp"` (`knockpy/smatrix.py:11` `parse_method`). Seed `np.random` first, because `merge_groups` shuffles. Check that the minimum eigenvalue of (2Σ − S) is > 0 and log it.
3. **Group statistic:** fit `fit_lasso_checked` on [Z, Z̃], then W_g = Σ_{j∈g}|β_j| − Σ_{j∈g}|β̃_j|. This is antisymmetric under group swaps, the same construction as knockpy's `combine_Z_stats(group_agg="sum")`. Threshold with the existing `knockoff_threshold` at offset 1 and offset 0.
4. **MKF+** ([Katsevich & Sabatti, 2019](https://arxiv.org/abs/1706.09375)), two layers. Layer 1 is individual latents with per-latent knockoffs; layer 2 is concepts with group knockoffs. **Each layer needs its own knockoff draw.** A latent j is selected iff W¹_j ≥ t₁ and W²_{g(j)} ≥ t₂. FDP̂_m(t) = (1 + #{W^m ≤ −t_m}) / max(1, |S_m(t)|), where S_m is the set of layer-m units containing a selected latent. Take the lower-left corner of the feasible set {t : FDP̂_m ≤ q_m ∀m}. Implement from the paper's Section 2 and **verify against a brute-force grid search on small p**. The theorem gives FDR_m ≤ (1.93/c)·q_m for MKF(c)+, and the authors report nominal control in simulation with c = 1. Run c = 1 as primary and c = 1.93 as the guaranteed variant.
5. **Tests:** singleton equivalence; group exchangeability (swap a random set of whole groups: the first two moments must be invariant up to Monte Carlo error); and toy FDR ≤ q at both layers (Step 1).

Commit: `Concept: block-S Gaussian sampler, group MVR knockoffs, MKF+ with tests`.

---

## Step 6 — Early gate: per-latent only, real and planted (run before anything else at scale)

`early_gate.py`. Uses only per-latent block-MVR knockoffs, the best existing baseline.

- **Planted (design B, below), per-latent only:** concept recovery vs width at layer 12, amplitudes {1, 3, 8}, 10 replicates.
- **Real SST-2:** number of per-latent discoveries at q = 0.1 vs width (median over 10 knockoff draws), and Jaccard of discovered *parent* sets vs 16k.

**Gate G1 (pre-register it):** go ahead only if per-latent concept recovery falls with log₂(width). Test this with a paired slope over replicates, one-sided p < 0.05 after Holm across amplitudes. **If it doesn't fall, stop and rethink**: the thesis is wrong at this layer.

---

## Step 7 — Planted-concept benchmark (the core result)

`planted_concept.py`. Layer 12, all widths, plus a layer-20 16k/65k bridge. All 67,349 rows, p = p\*, Ledoit–Wolf covariance.

**Two label designs:**

- **A — exact truth (primary for FDR claims).** For each width, take the planted concepts' children G_g^w and set c_g = standardise(Σ_{j∈G_g^w} X_j). Generate Y with `generate_planted_labels(C, arange(K_c), form, amp, rng)`. Every child of a planted family is truly non-null; everything else is null. Latent-level and concept-level FDR are both exact.
- **B — realistic (primary for width trends).** Y is generated from the **16k parent's own activation**, identical across widths for a given replicate, so widths are paired. Concept-level truth is the parent's family. State plainly that latent-level FDR is approximate here, because the parent's activation is not exactly a function of its wide-SAE children.

**Grid:** K_c ∈ {20, 30} planted concepts (both ≥ 1/q at q = 0.05), amplitudes {0.5, 1, 2, 3, 5, 8, 20}, linear and interaction, q ∈ {0.05, 0.10, 0.20}, 30 replicates. Each fit gets a fresh knockoff draw; seeds come from per-cell streams, so arms within a replicate see identical labels.

**Arms (same labels per replicate):**
1. Per-latent Knockoff+ (block MVR)
2. Group Knockoff+ (group MVR)
3. MKF+ (c = 1) and MKF+ (c = 1.93)
4. Group knockoffs with clustering groups (sensitivity)

**Metrics per (width, cell):** latent-level FDR and power; concept-level FDR and power. For the per-latent arm, a concept counts as discovered if ≥ 1 child is selected. Also report the number of discoveries and the lasso convergence rate. Metrics must be in the existing format (`METRICS`-style arrays + JSON rows) so `src/reanalysis_multiplicity.py`-style corrected statistics apply.

**Pre-registered contrasts:** (i) slope of concept power vs log₂(width) per arm; (ii) group − per-latent concept power, paired, per width; (iii) FDR breaches claimed only after Benjamini–Yekutieli, as in `stage3_amendment_5` rule K3.

**Stratify** every result by family mechanism (dilution vs redundancy, from Step 3). The redundancy stratum is where per-latent methods should fail completely.

Figures: `fig_c3_power_vs_width.png` (headline: concept power vs width, per-latent vs group vs MKF), `fig_c4_fdr_vs_width.png`, `fig_c5_mechanism.png`.

**Resumable checkpointing** per replicate, copying `stage3_followups.save_checkpoint` / `load_checkpoint`.

---

## Step 8 — Real SST-2 width sweep and cross-width stability

`width_sweep_real.py`. Layer 12, all widths, plus the layer-20 bridge, using the Step 4 candidate sets.

- Run per-latent, group and MKF+ at q ∈ {0.05, 0.10, 0.20}, with 30 knockoff redraws.
- **Per width:** discoveries; selection frequency of each latent and concept across redraws.
- **Stability:** map discoveries to 16k parents and compute the Jaccard of parent sets (selection frequency ≥ 0.5) between every width pair. **Prediction:** per-latent Jaccard decays with width, while group Jaccard stays flat.
- **Caveat to state:** filler families differ across widths. Compute stability on the fixed planted pool's families as the clean comparison, and on all families as secondary.

Figure: `fig_c6_real_sweep.png` (discoveries vs width; stability heatmaps).

---

## Step 9 — Pre-registration, findings, merge

For each of Steps 6–8:

1. Write an amendment in `concept/config/preregistration_concept.yaml` in the existing format (`meta` / `design` / `decision_rules` / `interpretation_constraints` / `changes_after_smoke_test`). Commit it **before** the full run.
2. Run a one-replicate diagnose pass and disclose anything changed afterwards.
3. Do the full run, then write `concept/results/concept_findings.md`.

Write `concept/README.md` with reproduction commands and timings, and add an entry to a new `concept/CHANGES.md`.

**On merge to main:** append the concept amendments to `config/preregistration.yaml` verbatim, and add one line to the main README layout. Nothing else touches main-line files.

---

## Order of work and rough cost (1–2 L40s)

| Step | Compute | Time |
|---|---|---|
| 0 scaffold | — | 1 h |
| 1 toy | CPU | 0.5 day |
| 2 caches (2 layers, 8 SAEs) | GPU | ~1 day, mostly the 524k and 1M encodes |
| 3 census | GPU/CPU | 0.5 day |
| 4 candidates | CPU | 2 h |
| 5 library + tests | CPU/GPU | 1–2 days |
| **6 early gate** | GPU | **0.5 day; stop here if G1 fails** |
| 7 planted benchmark | GPU | 2–3 days (8 widths × 4 arms × 30 reps × grid; trim amplitudes if needed, via an amendment) |
| 8 real sweep | GPU | 0.5 day |

**Out of scope for this branch** (later branches): additional SAEBench tasks, a second model (Llama Scope on Llama-3.1-8B has 32k and 128k residual SAEs in `sae_lens` as `llama_scope_lxr_8x` / `_32x`, which gives a cross-model width pair), steering with concept directions, and the hurdle-dCRT repair.

**Check before starting:** the preregistration mentions "a teammate's separate cluster experiment" (`stage3_amendment_5`) that is not in the repo. Confirm with them, so Step 4's clustering doesn't duplicate it.

---

## References

- Barber & Candès (2015), *Controlling the false discovery rate via knockoffs*, Ann. Stat. — [arXiv:1404.5609](https://arxiv.org/abs/1404.5609)
- Candès, Fan, Janson & Lv (2018), *Panning for gold: Model-X knockoffs* — [arXiv:1610.02351](https://arxiv.org/abs/1610.02351)
- Dai & Barber (2016), *The knockoff filter for FDR control in group-sparse and multitask regression*, ICML — [arXiv:1602.03589](https://arxiv.org/abs/1602.03589)
- Katsevich & Sabatti (2019), *Multilayer knockoff filter: controlled variable selection at multiple resolutions*, Ann. Appl. Stat. 13(1) — [arXiv:1706.09375](https://arxiv.org/abs/1706.09375), [AoAS](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-13/issue-1/Multilayer-knockoff-filter-Controlled-variable-selection-at-multiple-resolutions/10.1214/18-AOAS1185.full)
- Spector & Janson (2022), *Powerful knockoffs via minimizing reconstructability* (MVR) — [arXiv:2011.14625](https://arxiv.org/abs/2011.14625)
- Bricken et al. (2023), *Towards monosemanticity* (feature splitting) — [transformer-circuits.pub](https://transformer-circuits.pub/2023/monosemantic-features)
- Chanin et al. (2024), *A is for absorption* — [arXiv:2409.14507](https://arxiv.org/abs/2409.14507)
- Leask et al. (2025), *Sparse autoencoders do not find canonical units of analysis*, ICLR 2025 — [arXiv:2502.04878](https://arxiv.org/abs/2502.04878). Closest prior work on cross-width structure; cite it as the motivation for the parent mapping.
- Bussmann et al. (2025), *Learning multi-level features with Matryoshka SAEs* — [arXiv:2503.17547](https://arxiv.org/abs/2503.17547). A related view of the concept hierarchy.
- Lieberum et al. (2024), *Gemma Scope* — [arXiv:2408.05147](https://arxiv.org/abs/2408.05147)
- Enkhbayar (2025), *Model-X knockoffs for SAE FDR control* — [arXiv:2511.11711](https://arxiv.org/abs/2511.11711)

Verify every reference before it goes into the paper.