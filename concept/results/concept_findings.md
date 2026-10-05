# Concept dilution — findings

Branch `concept-dilution`. Pre-registration: `concept/config/preregistration_concept.yaml`
(`concept_amendment_1`, committed 17:14 UTC on 2026-10-05, before any labelled run;
`concept_amendment_2`, committed before the Step 7 and 8 runs). Full pipeline run 19:44–21:14 UTC.
Headline figure: `fig_c0_headline.png`.

## Summary

**The thesis holds in a sharper form than predicted, and fails in the form it was pre-registered.**

1. **Concepts do split as the SAE widens.** At layer 12 the median 16k concept has 1 child at
   16k and 6 at 1M (IQR 2–12). **99% of multi-child families are dilution** (children fire on
   disjoint inputs); redundancy is essentially absent (< 0.1%).
2. **Per-latent discovery keeps the concept but loses its extent.** With exact ground truth
   (design A, q = 0.1), per-latent Knockoff+ certifies **90% of a planted concept's latents at
   16k and 18% at 1M**. It still flags the concept through at least one latent 82% of the time,
   because one strong child usually survives. Group methods certify 80% (lasso statistic) and 86%
   (group-sum statistic) of the concept's latents at 1M.
3. **The pre-registered success criterion (P4), stated on concept-level power, fails.** Group
   knockoffs' concept power falls with width exactly as per-latent does (BY-significant negative
   slope in 76 of 84 design-B cells, vs 78 for per-latent). It is never significantly higher than
   per-latent at any width. MKF+ is bounded by per-latent power by construction and tracks it
   exactly.
4. **The failure is in the importance statistic, not in group knockoffs.** The toy showed this
   first, and the registered follow-up confirmed it on real activations. A statistic that sums
   each family before the lasso (the group-sum arm) beats per-latent concept power at every width
   above 16k in design A (+0.009 to +0.029; up to 16 of 84 cells BY-significant, none
   significantly worse). For weak signals in large families (8+ children) it recovers **0.92 vs
   0.65**. In the realistic design B the effect is mixed: worse at 65k–262k, better at 524k and 1M.
5. **On real SST-2 labels no method finds the same concepts across widths.** The stable discovered
   parents at 16k and at 1M overlap with Jaccard 0.12–0.14 for every arm, even restricted to
   parents present at both widths.
6. **FDR is controlled where it is guaranteed and measurable** (design A). Per-latent latent-level
   FDR: 0 BY breaches in 588 cells. MKF+ (c = 1.93): 0 at both levels. Group arms: 5–9 breaches,
   all at q = 0.05 and amplitude 8–20 (worst 0.138), the same high-amplitude erosion Stage 3 found.

## 1. Setup

| | |
|---|---|
| Model / SAEs | Gemma-2-2B residual stream. **Layer 12**: Gemma Scope 16k, 32k, 65k, 131k, 262k, 524k, 1M. **Layer 20 bridge**: 16k, 65k |
| Data | SST-2 train, all 67,349 sentences; latents mean-pooled over content tokens (BOS/EOS/pad excluded) |
| Candidates | p\* = 2,048 at every width (no n/p confound); 60 planted-pool 16k parents present at every width; filler families; child floor 0.1%; τ = 0.3 |
| Knockoffs | Gaussian second order, Ledoit–Wolf Σ, block MVR (per-latent) / group MVR (families, clusters); FISTA L1-logistic, λ = 0.02; 100% convergence in every fit |
| Grid (Step 7) | designs A and B × K_c ∈ {20, 30} × 7 amplitudes (0.5–20) × 2 forms × q ∈ {0.05, 0.10, 0.20} × 30 replicates × 9 widths |

SAE fidelity on SST-2 content tokens (explained variance / mean L0) runs from 0.920 / 76.8 at
16k to 0.952 / 99.2 at 1M. **L0 is not monotone in width (69–121)**; it is logged as a confound
(`census_L12.json`).

## 2. Step 1 — toy proposition (`proposition.md`, `fig_c1_toy.png`)

The per-child standardised effect follows theory to three decimals: m^(−1/2) under dilution
(0.712, 0.504, 0.355, 0.250, 0.174 for m = 2–32) and m^(−1) under redundancy. Under redundancy,
per-latent MVR gives near-copy knockoffs (mean s 0.089 → 0.017) and per-latent concept power
collapses (1.00 → 0.73 → 0.01 at m = 2, 4, 8). **Group knockoffs hold 1.00 to m = 32.** Under
dilution, group knockoffs with the lasso statistic collapse alongside per-latent (0.89 at m = 8,
0.00 at m = 16). The library test **T1 passed**: no BY-significant breach in 288 global-null and
864 planted Gaussian cells.

## 3. Steps 2–3 — caches and family census (`fig_c2_census_L12.png`)

| layer 12, τ = 0.3 | 16k | 32k | 65k | 131k | 262k | 524k | 1M |
|---|---|---|---|---|---|---|---|
| median children per parent (IQR) | 1 | 1 (1–2) | 2 (1–4) | 3 (2–6) | 4 (2–8) | 5 (2–10) | 6 (2–12) |
| Σ child firing / parent firing | 1.00 | 0.83 | 0.76 | 1.14 | 1.06 | 0.95 | 0.86 |
| dilution share of multi-child families | – | 0.89 | 0.96 | 0.97 | 0.99 | 0.99 | 0.99 |
| planted-pool latents (of 2,048) | 60 | 107 | 163 | 227 | 308 | 411 | 543 |

Gate C0 passed. τ was frozen by rule TAU at **0.3**, the lowest value in the grid. At that value
children cover their parent's firing within ×0.76–1.14 at every width, so the grid edge is not
binding. Layer 20 shows the same picture (65k: median 3 children, 93% dilution).

## 4. Step 6 — gate G1 (`fig_c_gate.png`, `early_gate.json`)

Per-latent concept recovery (design B, 10 replicates) falls with log₂(width) at every amplitude:
slopes −0.075, −0.027, −0.016 per doubling at amplitudes 1, 3, 8. Holm-adjusted p = 2×10⁻⁷,
7×10⁻⁸, 4×10⁻⁵. **G1 passed**, so the full benchmark ran.

## 5. Step 7 — planted-concept benchmark

### Power (`fig_c0_headline.png` panels 2–3, `fig_c3_power_vs_width.png`)

Mean over cells at q = 0.1, layer 12:

| | 16k | 32k | 65k | 131k | 262k | 524k | 1M |
|---|---|---|---|---|---|---|---|
| **A** per-latent: share of concept's latents certified | 0.90 | 0.74 | 0.52 | 0.44 | 0.33 | 0.25 | **0.18** |
| **A** group (lasso W): share of latents | 0.90 | 0.87 | 0.85 | 0.84 | 0.83 | 0.81 | 0.80 |
| **A** group-sum: share of latents | 0.90 | 0.89 | 0.88 | 0.88 | 0.88 | 0.87 | **0.86** |
| **A** concept power, per-latent / group / group-sum | 0.90 / 0.90 / 0.90 | 0.88 / 0.88 / 0.90 | 0.87 / 0.86 / 0.88 | 0.86 / 0.86 / 0.88 | 0.85 / 0.85 / 0.88 | 0.83 / 0.83 / 0.87 | 0.82 / 0.81 / 0.85 |
| **B** per-latent: share of latents | 0.90 | 0.52 | 0.32 | 0.25 | 0.19 | 0.13 | **0.10** |
| **B** concept power, per-latent / group / group-sum | 0.90 / 0.90 / 0.90 | 0.78 / 0.77 / 0.78 | 0.70 / 0.70 / 0.69 | 0.74 / 0.73 / 0.71 | 0.72 / 0.72 / 0.70 | 0.68 / 0.68 / 0.70 | 0.65 / 0.65 / 0.67 |

Per-latent Knockoff+ selects 23–43 latents at every width (mean over cells), while the planted
concepts' latents grow from 20–30 at 16k to about 180–270 at 1M. It reports a fragment of each concept, and the fragment shrinks
with width.

Design B loses about 25 points of concept power for **every** arm between 16k and 65k, then
plateaus. That pattern is the coverage loss pre-registered as an interpretation constraint: the
16k parent's signal is partly carried by wide latents outside its family. Group testing cannot
recover signal that its groups do not contain.

### Pre-registered contrasts (layer 12; BY across cells)

| rule | result |
|---|---|
| P1 slope of concept power vs log₂(width) | negative for every arm. Design B: per-latent 78/84 cells BY-negative, group 76/84, group-sum 71/84, MKF+ (1) 76/84. Design A: 67–76/84 per arm (MKF+ 1.93: 42/84), median −0.002 to −0.004 per doubling |
| P2 group − per-latent concept power | never BY-positive at any width; mean −0.008 to −0.021; 3 of 1,176 cells BY-negative |
| P3 FDR breaches (design A only) | latent arm, latent level: 0/588. MKF+ (1.93): 0/588 at both levels. group: 9/588 concept level (worst 0.138 at q = 0.05, amp 20). group-sum: 5/588 (worst 0.097). MKF+ (1): 5/588 (worst 0.136) |
| P4 thesis criterion | **not met**: group concept power falls with width as per-latent does, and is never higher |
| S2 group-sum − per-latent (amendment 2) | design A: +0.009, +0.011, +0.014, +0.023, +0.029, +0.007 at 32k–1M; 7, 6, 11, 12, 16, 15 of 84 cells BY-positive, none negative. Design B: −0.009, −0.026, −0.043, −0.043, +0.001, +0.006 (up to 10 cells BY-negative at 131k; 39 BY-positive at 1M) |

### Stratification (`fig_c5_mechanism.png`)

All planted-pool families are single-child or dilution; redundancy never occurs, so the planned
mechanism split reduces to family size. At amplitude 1 in design A, concept recovery falls with
family size for per-latent, lasso-group and MKF+ (0.93 → 0.65 from 1 to 8+ children). Group-sum
holds at 0.92.

### FDR (`fig_c4_fdr_vs_width.png`)

Design A concept-level FDR averages 0.02–0.05 for every family-based arm at q = 0.1. Two features
of the figure are not inference failures:

- **The cluster arm** exceeds q (up to 0.26). With τ = 0.3, decoder clusters merge several 16k
  parents (1,834 clusters for 2,048 parents even at 16k), so a correctly selected cluster brings in
  other families that family-level truth counts as false. This is a unit mismatch.
- **Design B** FDR averages 0.13–0.34 for every arm. Its truth is approximate at both levels (disclosed
  after the smoke test), so it is not an error rate.

### Layer-20 bridge

The same pattern as layer 12. Design A per-latent latent power is 0.90 → 0.46 from 16k to 65k;
group 0.84; group-sum 0.87. Concept power is 0.90 → 0.86 for all three.

## 6. Step 8 — real SST-2 labels (`fig_c6_real_sweep.png`, `real_sweep.json`)

At q = 0.1, per-latent Knockoff+ selects 49–68 latents at every width (median of 30 draws). These
map to 42–57 distinct 16k parents, peaking at 65k. Group arms select whole families (49 → 610
latents) but about the same number of concepts. Group-sum finds the most concepts at 32k–65k
(63–68) and the fewest at 262k (34).

**Stability.** The pre-registered clean comparison used the 60 planted-pool families. It is
uninformative: the pool is random parents, and only 0–12 of them are ever stably selected on
sentiment labels. Post hoc, restricting to parents present in both widths' candidate sets, the
stable parent set at 16k and at a wider width has Jaccard 0.26 (32k) falling to 0.12 (1M) for
per-latent. Group, MKF+ and cluster arms are the same. Group-sum is marginally higher at 0.14,
with adjacent widths at 0.26–0.58. The prediction that group stability stays flat **fails**: the
identity of the concepts discovered on real labels changes with width for every method. Likely
causes are selection among many weakly predictive correlated latents (real sentiment signal is
diffuse, probe AUC 0.972) and the changing candidate composition. This run cannot separate the two.

## 7. Verdicts on pre-registered rules

| rule | verdict |
|---|---|
| T1 library FDR test | PASS |
| C0 splitting exists | PASS |
| TAU | τ = 0.3 |
| G1 per-latent recovery falls with width | PASS (Holm p ≤ 4×10⁻⁵) |
| P1 | per-latent slopes negative; group slopes also negative |
| P2 | group never beats per-latent on concept power |
| P3 | no latent-level breach for per-latent; no breach at either level for MKF+ (1.93); 5–9 high-amplitude q = 0.05 breaches for group arms |
| P4 thesis on concept power | **FAIL** |
| S1 group-sum valid on the toy | PASS |
| S2 group-sum vs per-latent | positive in design A at every width > 16k; mixed in design B |

## 8. What this establishes, and what it does not

**Establishes.**
- Wider Gemma Scope SAEs split concepts by dilution.
- Per-latent knockoff discovery certifies a shrinking fraction of the latents that carry a concept
  while FDR stays controlled. At 1M a "discovered feature" is about a fifth of the concept's
  representation.
- Group knockoffs fix this only when the statistic pools within the group. With the standard lasso
  statistic they fail under dilution, in the toy and on real activations. MKF+ cannot fix it,
  because a latent must pass the per-latent layer.

**Does not establish.**
- That concepts become undiscoverable. Concept-level recovery declines only modestly when the
  signal is exactly defined (design A, 0.90 → 0.82).
- That group-sum is a general fix. It was designed after the toy result, its advantage is small
  (≤ 0.03 on average, 0.27 for weak signals in large families) and it reverses at middle widths in
  design B. It also assumes children act in the same direction.
- Anything about redundancy on real SAEs, which this census did not find.
- Causal attribution to width alone. L0 varies non-monotonically across widths (69–121), and
  layer 12 is not the main project's layer 20 (the bridge agrees in direction).

## 9. Deviations and disclosures

All recorded in the pre-registration file before the runs they affect:

- **Residuals stored fp32, not fp16.** fp16 would fail the plan's own reproduction check.
- **The reproduction check fails the plan's 1e-4 criterion.** Max |diff| 1.705; 14 JumpReLU
  threshold flips in 137.9M entries; everything else is fp32 rounding (max relative difference
  6.6e-5). `check_reproduction.json`.
- **Design B FDR is approximate at both levels** (found in the smoke test). FDR claims rest on
  design A.
- **Label columns within a replicate share one knockoff draw per arm,** as in the main repo's
  p = 2048 runs. Replicates use independent draws.
- **The group-sum arm** (amendment 2) was motivated by the toy result. It is exploratory on the
  toy and confirmatory on Steps 7–8. It reuses the group arm's draws, and the other arms were
  verified unchanged.
- **Post hoc:**
  - the shared-parent stability scope (the registered pool scope was empty on real labels);
  - display choices in `fig_c0` (new summary) and `fig_c5` (amplitude 1 instead of 3, where every
    arm saturated);
  - the latent-level power headline (latent power was a registered metric, but P4 was stated on
    concept power).
- **Infrastructure:**
  - the host's GPU driver intermittently fails CUDA initialisation, so jobs retry;
  - the encode for the 524k and 1M SAEs uses smaller chunks and an fp16 decoder for the
    explained-variance diagnostic only (tested to reproduce the 32k cache), because the GPUs were
    shared.
- **Not checked:** overlap of the clustering arm with a teammate's separate cluster experiment
  (`stage3_amendment_5`).

## 10. Suggested next steps

1. **Pre-register the group-sum statistic as primary** and replicate on a second layer (19, which
   has 16k/65k/1M) or model (Llama Scope 32k vs 128k).
2. **Signed pooling.** Replace the plain within-family sum with a group lasso or a
   sign-aware pooled statistic, so cancelling children are handled.
3. **Separate coverage loss from dilution in design B.** Include orphan wide latents most
   correlated with each parent in its group, chosen label-free from decoder cosine or co-firing.
4. **Report "concept extent"** (share of a concept's latents certified) alongside discovery
   counts in any SAE feature-discovery claim.

## Reproduction

`concept/README.md`. Full pipeline: `concept/scripts/run_all.sh` (≈ 90 min on two L40S after
caches). Caches: ≈ 2 h, bound by the ~40 GB SAE download on this host's ≈ 5 MB/s link.
