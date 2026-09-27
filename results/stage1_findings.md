# Stage 1 findings — FWER calibration of the standard search-then-validate pipeline

Config hash `d33d210c5acb` · master seed `20260904` · v1 pre-registered in
`config/preregistration.yaml` (`stage1`); v2 additions registered *after* v1 results, in
`stage1_amendment_1` — every v2 item is post-hoc and is labelled as such below.

## Summary

1. **An uncorrected per-latent scan is unusable.** Under permuted labels a naive |t| > 1.96
   test passes **104.2 latents per experiment (5.1% of p = 2,048; expected 102.4)** and
   the family-wise error rate is **1.000**. Bonferroni and Westfall–Young hold it at
   **0.054 / 0.050**. On the real labels 1,764 latents pass naively, 1,408 pass Bonferroni,
   1,407 pass WY. This is the proposal's §3 claim, measured directly.
2. **The certification step v1 relied on has no demonstrated power, so v1's FWER of
   0.2–1.6% says nothing about the pipeline.** It certifies 0/10 latents on the real
   labels for every method, and on planted signals (k = 10, amplitude 3.0) it fires in only
   10% / 50% / 50% of replicates (needed: ≥ 80%). Per pre-registered rule A1 the
   accuracy-drop FWERs are reported as **uninformative**. The v1 conclusion "the pipeline
   does not inflate FWER" is withdrawn.
3. **v1's numbers reproduce exactly** on different hardware (FWER counts 8/6/1 of 500,
   WY p = 0.002, survivors, top-10 sets), so the arithmetic was right; the interpretation
   was not.
4. **Exploratory, post-hoc:** ablating the top-10 *jointly* does separate real from null
   (joint accuracy drop, WY p = 0.004–0.048) and fires on planted signals (90–100% for
   AUROC / probe weight), unlike single-latent zeroing. Not pre-registered; see §5.
5. **A powerful validate step (Amendment 2, pre-registered before the run, §6) certifies
   noise.** A paired test that ablating the top-10 raises the probe's log-loss passes the
   power gate in both regimes, and under permuted labels certifies in **98–100% of
   experiments when validated on the same data** (the conventional workflow) and in
   **9.6–27% when validated on held-out rows** (nominal 5%). Even ablating 10 *random*
   latents is significant in 84% of same-data experiments (exploratory diagnostic), so
   the certification says almost nothing about the selected latents. The held-out excess
   over 5% triggered rule B3; the cause is that the test's null ("ablation changes
   nothing") is false even under the global null, not a coding error.

**What this establishes:** naive scanning has FWER ≈ 1 and ~5% spurious passes;
Bonferroni/WY control it; the single-latent zeroing criterion is powerless here; and a
same-data significance test of ablation certifies essentially every noise experiment.
**What it does not establish:** a calibrated false-certification rate (no test we have
built has a valid null: it needs a null-intervention reference, e.g. matched random
latents, which is a further amendment); anything about the model's generated behaviour
(ablation is a change in a linear probe's predictions on cached latents); anything about
FDR (Stage 3).

## 1. Setup

| | |
|---|---|
| Cache | `d33d210c5acb`, n = 67,349 sentences, p = 2,048 retained latents, mean-pooled |
| Class balance | 55.8% positive (n₁ = 37,569, n₀ = 29,780) |
| Scoring | mean activation difference (on **raw** activations, hence scale-dependent), AUROC (rank-sum, ties by stable sort order), \|probe weight\| |
| Probe | **L2** logistic, C = 1.0, **200** gradient-descent steps, lr 0.1, on GPU, fit on a fresh random 70% split per permutation |
| Ablation | zero the latent's raw activation, re-predict with the same fitted probe (no refit) |
| Certification (v1) | ≥ 1 of the top-10 whose zeroing drops held-out accuracy by ≥ δ = 0.02 |
| Permutations | B = 500 |
| Runtime | teammates' v1 run: 3.0 min (reported, NVIDIA L40S). This re-run incl. v2 additions: 6.1 min on an RTX 5050 laptop GPU, peak 2.4 GB GPU / 3.0 GB host RAM |

## 2. v1 results (reproduced)

Every v1 quantity below matches the committed v1 run exactly (FWER counts, WY p, survivor
counts, top-10 index sets); null max-scores agree to ≤ 6e-5 relative (GPU float noise).

| method | real max score | certified / top-10 (real labels) | null 95th | WY p (max) |
|---|---|---|---|---|
| mean_diff | 2.977347 | 0 / 10 | 0.228073 | 0.0020 |
| auroc | 0.257274 | 0 / 10 | 0.010329 | 0.0020 |
| probe_weight | 0.392511 | 0 / 10 | 0.047340 | 0.0020 |

WY p = 0.002 is the minimum attainable at B = 500. Latents surviving per-latent WY
correction: mean_diff 181 / 141, auroc 1,145 / 1,052, probe_weight 215 / 189 (α = 0.05 / 0.01);
**mean_diff has the fewest**, not probe_weight as v1's summary said.

v1 certification FWER (held-out): mean_diff 0.016, auroc 0.012, probe_weight 0.002
(8 / 6 / 1 of 500). **See §3: these cannot be interpreted.**

## 3. Power check — the certification step fails it

Labels generated from 10 known latents (Stage 3's linear generator, amplitude 3.0), 10
replicates, same pipeline:

| method | recall@10 of the planted latents | v1 criterion fires | same-data variant fires |
|---|---|---|---|
| mean_diff | 0.08 | 10% | 0% |
| auroc | 0.35 | 50% | 30% |
| probe_weight | 0.99 | 50% | 30% |

Pre-registered criterion: ≥ 80% for at least one method → **`criterion_has_power = False`**.
Per rule A1 the accuracy-drop FWERs (held-out 1.6/1.2/0.2%; same-data 0.4/0.4/0.6%) are
uninformative, and per rule A2 the criterion is not adjusted to make it pass.

Why: zeroing one latent out of 2,048 correlated, 89%-zero latents rarely moves a probe's
accuracy by 2%, even when that latent is a true signal (probe_weight finds 99% of them, yet
certifies in half the replicates). mean_diff on raw activations ranks high-variance latents,
not the planted (standardised) ones, hence recall 0.08.

The same-data (conventional) variant does **not** show inflated FWER relative to the
held-out split. Under a criterion this weak that is not evidence that data reuse is
harmless — only that this test cannot see it.

## 4. Naive per-latent testing under the null (post-hoc addition)

Welch |t| per latent on the training split, B = 500 permutations:

| procedure | threshold | mean false passes per null experiment | FWER | passes on real labels |
|---|---|---|---|---|
| naive | \|t\| > 1.96 | 104.2 (expected 102.4) | **1.000** | 1,764 |
| Bonferroni | \|t\| > 4.22 | 0.056 | 0.054 | 1,408 |
| Westfall–Young | \|t\| > 4.24 (95th pct of max\|t\|) | — | 0.050 (in-sample, by construction) | 1,407 |

(`fig11_stage1_naive_vs_corrected.png`.) Passing a marginal test is not a causal claim: it
says the latent is associated with the label, not that it carries it.

## 5. Behavioural effects of ablation (post-hoc, exploratory)

Effect of ablating the top-10 (jointly) or the best single latent on the fitted probe,
held-out split, real-label value vs the permutation null (WY-style p = (1 + #{null ≥ real}) / 501).
`fig10_stage1_behaviour_curve.png` gives the FWER(τ) curves.

| method | effect | real | null 95th | p |
|---|---|---|---|---|
| mean_diff | joint accuracy drop | 0.0135 | 0.0124 | 0.048 |
| | joint mean \|Δp\| | 0.0441 | 0.0379 | 0.044 |
| | joint flip rate | 0.0405 | 0.1410 | 0.549 |
| | best-single flip rate | 0.0210 | 0.1301 | 0.721 |
| auroc | joint accuracy drop | 0.0319 | 0.0077 | 0.006 |
| | joint mean \|Δp\| | 0.0678 | 0.0238 | 0.010 |
| | joint flip rate | 0.0618 | 0.0811 | 0.138 |
| | best-single flip rate | 0.0082 | 0.0584 | 0.834 |
| probe_weight | joint accuracy drop | 0.0297 | 0.0084 | 0.004 |
| | joint mean \|Δp\| | 0.0649 | 0.0241 | 0.008 |
| | joint flip rate | 0.0571 | 0.0888 | 0.693 |
| | best-single flip rate | 0.0082 | 0.0613 | 1.000 |

- **Flip rate is not informative:** the real-label value sits inside the null and the null
  ranges up to 0.35–0.45, so any fixed flip-rate threshold is arbitrary (this is why v2
  reports curves, not one threshold).
- **Joint accuracy drop and joint mean |Δp| do separate real from null** for AUROC and
  probe weight, and on planted signals joint accuracy drop exceeds the null 95th percentile
  in 90% (auroc) and 100% (probe_weight) of replicates (mean_diff: 30%).
- Choosing among these effects after seeing results is a forking path. Treat this as a
  hypothesis for a second, pre-registered amendment, not a finding.

## 6. The validate step (Amendment 2 — pre-registered before the run)

Design fixed in `stage1_amendment_2` and committed (`a17be07`) before this script ran on
real data. Search is unchanged (top-10 by score). Validate: one-sided paired test that
ablating the candidates raises the probe's log-loss, certify at p ≤ 0.05. Primary variant:
all ten ablated jointly. Two regimes: **same-data** (validate on the rows the probe was fit
on: the conventional workflow) and **held-out** (the 30% never used). B = 500 fresh
permutations (`validate_permutation` stream). Script: `src/calibrate_validation.py`.

**FWER of the primary variant** (fraction of permuted-label experiments with a
certification; Wilson 95% interval):

| method | same-data | held-out |
|---|---|---|
| mean_diff | 0.984 [0.969, 0.992] | 0.274 [0.237, 0.315] |
| auroc | 1.000 [0.992, 1.000] | 0.096 [0.073, 0.125] |
| probe_weight | 1.000 [0.992, 1.000] | 0.106 [0.082, 0.136] |

Secondary variants (any of the 10 latents tested alone; same-data / held-out):

| method | any-of-10, uncorrected | any-of-10, Bonferroni |
|---|---|---|
| mean_diff | 0.852 / 0.656 | 0.388 / 0.278 |
| auroc | 0.758 / 0.400 | 0.116 / 0.118 |
| probe_weight | 1.000 / 0.592 | 0.614 / 0.158 |

On the **real labels** every method certifies in both regimes (p underflows to 0).

**Power gate (B1)** — labels from 10 planted latents, 20 replicates, primary variant
certifies (same-data / held-out): amplitude 3.0: mean_diff 90% / 85%, auroc 100% / 100%,
probe_weight 100% / 100%; amplitude 1.0 (descriptive): 100% / 85%, 100% / 95%, 100% / 100%.
**Gate passed in both regimes.** But recall@10 of the planted latents is 0.05 (mean_diff),
0.35 (auroc), 0.99 (probe_weight): mean_diff certifies in 90% of replicates while picking the
*wrong* latents 95% of the time. The gate shows the test detects *an* ablation effect, not
that it identifies the right latents; it is not specific.

**Rule B3 triggered.** A valid held-out test should have FWER ≤ ~0.05; ours is 0.096–0.274
with intervals excluding 0.05, so it was investigated before interpretation. Findings
(exploratory diagnostics on the saved nulls and a 100-permutation rerun with `probe_weight`,
seed not pre-registered):
- The ablation arithmetic is not the problem (same identity that reproduced v1's held-out
  certification 0/1500). The hypothesis that ablation shifts the probe's mean logit
  (miscalibrating it) is **refuted**: the mean shift is 0.000 (sd 0.041).
- Held-out t-statistics across permutations are centred near zero (mean −0.24 to −0.06 for
  auroc / probe_weight, +0.82 for mean_diff) but **over-dispersed: sd 1.31, 1.53, 2.07
  instead of 1**. Under the global null, ablating latents still changes the probe's
  expected loss by a small systematic amount whose sign and size depend on the random
  probe; at ~20k rows that is detectable. The test rejects "ablation changes nothing",
  which is false under the null, so it is not a level-α test that the latents carry signal.
- **Null-intervention control:** ablating 10 *random* latents instead of the top-10 gives
  held-out 0.06 (top-10: 0.08; B = 100, interval about ±0.05) and same-data **0.84** (top-10:
  1.00). In-sample, almost any ablation is "significant"; selection adds little.

**Interpretation.**
1. The conventional same-data validate step certifies essentially every experiment on pure
   noise (0.98–1.00), and mostly because any in-sample ablation registers, not because the
   selected latents matter. This is the proposal's §3 point (data reuse, and no null
   intervention to compare against), now measured.
2. Sample-splitting cuts the rate to 0.10–0.27 but does not reach nominal, for the reason
   above. No procedure here has a calibrated FWER.
3. The natural repair is a **null-intervention reference**: certify only if the effect
   exceeds that of matched random latents (as Stage 5 proposes). That is a new design and
   needs its own amendment; per B2 it is not applied to these data now.

Caveats: the joint-ablation idea came from exploratory v2 results on this same dataset, so
this is not independent confirmatory evidence; behaviour is the probe's log-loss on cached
latents, not the model's output. A plotting bug (negative error bar when FWER = 1.0) crashed
the run's final step after results were saved; it was fixed and `fig12` regenerated from the
saved JSON, with no recomputation.

## 7. Deviations and corrections

**Deviations from the pre-registered v1 (both in the original commit):**
1. L2 gradient-descent probe on GPU instead of L1/SAGA on CPU (SAGA projected ~49 h). The
   v1 claim that this "cannot bias the calibration" is **withdrawn**: it is true for the
   WY max-score null, but the probe sets how much one latent's ablation moves predictions,
   so it bears on any certification FWER. L2 weights are dense, not sparse.
2. `--device` flag; no numerical effect.

**Corrections to the v1 writeup:** 300 → 200 gradient steps; "probe_weight has the fewest
WY-surviving latents" → mean_diff does; "any excess of certified discoveries is attributable
to signal" → vacuous (none certified on real labels); "FWER ≤ 5% so the calibration is
sound" → unsupported (§3). `stage1.meta.date_registered: 2026-09-10` is not corroborated by
git history (first appears 2026-09-17, in the same commit as the code).

**Post-hoc additions (after v1 results were seen):** naive/Bonferroni/WY testing (§4),
label-free and joint behavioural effects and their null curves (§5), same-data variant and
planted-signal power check (§3). None of these is confirmatory.

## 8. Reproduction

```bash
# cache must exist first (GPU machine, ~5 min):
python src/cache_activations.py --config config/default.yaml

# Stage 1 (v1 numbers + v2 additions; ~6 min on an 8 GB laptop GPU, or --device cpu):
python src/calibrate_pipeline.py --config config/default.yaml --device cuda
# quick check: add --limit-perms 20 --skip-planted

# Stage 1 validate step (Amendment 2; ~5 min on an 8 GB laptop GPU):
python src/calibrate_validation.py --config config/default.yaml --device cuda
```
Validate-step artefacts: `results/stage1_validation.json`, `results/stage1_validation_null.npz`
(per-permutation p-values), `results/fig12_stage1_validation.png`.

Artefacts: `results/stage1_calibration.json` (v1 keys unchanged; new `v2` block),
`results/stage1_adjusted_pvalues.npz` (per-latent WY-adjusted p), `results/stage1_behaviour_null.npz`
(per-permutation null arrays), `fig5_permutation_null.png`, `fig6_fwer_by_method.png`,
`fig10_stage1_behaviour_curve.png`, `fig11_stage1_naive_vs_corrected.png`.
