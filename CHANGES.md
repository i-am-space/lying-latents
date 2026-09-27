# Changes on branch `stage1-power-fix`

This branch reviews and repairs the teammates' Stage 1 and Stage 3 work on `main`
(commits `e378e1c`…`23e3500`). Commit messages are kept to one line; this file records
what each change did, why, and what came out of it. Stage 2 is untouched.

All numbers below come from the committed result files and can be re-derived from them.

## At a glance

| Commit | What |
|---|---|
| `15f44eb` | Stage 1: extend the calibration script; show the v1 certification step has no power |
| `9622b80` | Stage 1: pre-register (Amendment 2) and add a validate step with a power gate |
| `70d0a11` | Stage 3: pre-register (Amendment 1) and add a controlled realised-FDR benchmark |
| `aaef35d` | Stage 1: validate-step results |
| `8eb134d` | Stage 3: controlled-benchmark results and rewritten findings |

**Headline results**

- **Stage 1, search:** an uncorrected per-latent scan passes about 5% of latents on pure
  noise (104 of 2,048) and has FWER 1.000. Bonferroni / Westfall–Young bring it to about 0.05.
- **Stage 1, validate:** the v1 criterion had no power, so its 0.2–1.6% FWER meant nothing.
  A powerful replacement certifies 98–100% of noise experiments when validated on the same
  data. Even ablating *random* latents is "significant" 84% of the time. No calibrated
  false-certification rate exists yet.
- **Stage 3:** with a proper control, Gaussian knockoffs showed no FDR inflation beyond q in
  any of 108 cells. But their safety margin erodes as signal grows, reaching nominal in the
  highest-signal cells. v1's "power collapse" is not caused by the zero atom.

## 1. Review of the teammates' work (why this branch exists)

The arithmetic in both stages reproduced from the committed JSON. The conclusions did not
hold up.

**Stage 1 (v1, `calibrate_pipeline.py`)**
- The certification step (zero one latent, need a ≥ 2% held-out accuracy drop) certified
  **0 of 10** latents on the real labels for every method, although 141–1,052 latents are
  Westfall–Young significant. A test that never fires cannot show a low false-alarm rate.
- Validation used a held-out split. That is sample-splitting — the *fix* for the data-reuse
  problem the proposal (§3) attributes to the conventional workflow, not the workflow itself.
- The proposal's motivating number (how many latents pass a naive 5% test on a random
  label) was never measured.
- The pre-registration said an L1/SAGA probe on CPU. The code, committed in the same commit
  and five minutes before the results, used an L2 gradient-descent probe on GPU. The
  registration date (2026-09-10) is not supported by git history.
- Write-up errors: "300 gradient steps" (config: 200); "probe_weight has the fewest WY
  survivors" (mean_diff does); "L2 gives sparse-ish weights" (false); "the L1→L2 change
  cannot bias the calibration" (false for FWER).

**Stage 3 (v1, `planted_fdr.py`)**
- The findings claimed ASDP knockoffs; the code used equicorrelated. Stage 2 had already
  shown this gives near-copy knockoffs (corr ≈ 0.977 at p = 2,048) and recommended MVR.
- No control for which Gaussian knockoffs are exactly valid, so nothing could be attributed
  to the zero atom.
- Power exactly 0 at k = 10, q = 0.05 was reported as a finding. Knockoff+ needs ≥ 1/q = 20
  discoveries, so that cell is zero by arithmetic.
- The amplitude sweep stopped at 3.0 while FDR was still rising.
- "2,160 Lasso fits" was 720 (2,160 = fits × 3 values of q).
- "CI control in 100% of conditions" was false for one of 72.
- The code's CI rule was the reverse of the pre-registered one.
- Summary ranges did not match the tables.
- SEs used ddof = 0 and treated replicates that shared one knockoff draw as independent.
- The pre-registered Westfall–Young baseline was never implemented, and the design changed
  after the first result without an amendment.

## 2. Stage 1 — search step and power check (`15f44eb`)

`src/calibrate_pipeline.py` keeps the v1 computation and RNG order untouched, so v1 is a
built-in regression check. The re-run reproduced all v1 numbers exactly on different
hardware: FWER counts 8/6/1 of 500, WY p = 0.002, survivor counts and top-10 sets. Added:

- **Naive per-latent testing** (Welch |t|): uncorrected, Bonferroni and Westfall–Young, under
  500 label permutations.
- **Behavioural effects of ablation** (flip rate, mean |Δp|, accuracy drop; best single
  latent and all top-10 jointly), on the held-out and the training (same-data) split, as
  null curves FWER(τ) rather than one threshold.
- **Planted-signal power check:** 10 known latents, amplitude 3.0, 10 replicates; the
  criterion must fire in ≥ 80% of replicates.
- Chunked AUROC so the run fits in 8 GB of GPU memory. Fixed `--cache <path>` always
  exiting with "Cache not found". Added `--skip-planted`.

`config/preregistration.yaml` gained `stage1_amendment_1`, disclosed as post-hoc because it
was written after v1's results were seen. It records the registration-date and probe issues
above, and adds rule A1 (FWER is only interpreted if the power check passes) and rule A2 (no
tuning to pass).

**Results**

| | mean false passes per null experiment | FWER | passes on real labels |
|---|---|---|---|
| naive \|t\| > 1.96 | 104.2 (expected 102.4) | **1.000** | 1,764 |
| Bonferroni | 0.056 | 0.054 | 1,408 |
| Westfall–Young | — | 0.050 | 1,407 |

- The v1 criterion fired in only **10% / 50% / 50%** of planted replicates (mean_diff /
  auroc / probe_weight), so `criterion_has_power = False`.
- Under rule A1, v1's FWER is reported as uninformative and "the pipeline does not inflate
  FWER" is withdrawn.

## 3. Stage 1 — a validate step that can detect effects (`9622b80`, `aaef35d`)

`src/calibrate_validation.py` (new). The search step is unchanged: top-10 by score.

- **Validate step:** a one-sided paired test that ablating the candidates raises the probe's
  log-loss, certifying at p ≤ 0.05. The primary variant ablates all 10 jointly.
- **Two regimes:** *same-data* (the conventional workflow) and *held-out*.
- **Power gate:** at least 80% certification on planted signals, in both regimes.

The design was fixed in `stage1_amendment_2` and committed before the real run. Rule B3
says a held-out FWER clearly above 0.05 means something is wrong and must be investigated
first.

**Results** (FWER, B = 500, Wilson 95% interval)

| method | same-data | held-out |
|---|---|---|
| mean_diff | 0.984 [0.969, 0.992] | 0.274 [0.237, 0.315] |
| auroc | 1.000 [0.992, 1.000] | 0.096 [0.073, 0.125] |
| probe_weight | 1.000 [0.992, 1.000] | 0.106 [0.082, 0.136] |

- **The power gate passed in both regimes** (85–100%). mean_diff certified in 90% of planted
  replicates while picking the right latents only 5% of the time, so the gate shows an
  effect is detected, not that the right latents were found.
- **Rule B3 fired.**
  - Not a code error: the same ablation code reproduced v1 with 0 mismatches in 1,500 cases.
  - Not a mean shift in the probe's predictions (measured ≈ 0.000).
  - Actual cause: the test statistics are over-dispersed under the null (sd 1.3–2.1 instead
    of 1). Ablating any latents changes the probe's loss a little, and with ~20k rows that
    is detectable, so "ablation changes nothing" is false even on noise.
- **Random-latent control** (exploratory, B = 100): ablating 10 random latents was
  significant in 84% of same-data experiments (top-10: 100%). Passing this check says little
  about the selected latents.

**Conclusion:**
- Same-data validation certifies almost anything, which matches the proposal's §3.
- There is still no calibrated false-certification rate.
- The next step is a null-intervention reference: compare against matched random latents.
  That needs its own amendment and has not been done.
- A plotting bug crashed the last step after the results were saved. It was fixed and fig12
  was regenerated from the saved JSON, with no recomputation.

## 4. Stage 3 — controlled benchmark (`70d0a11`, `8eb134d`)

`src/planted_fdr_controls.py` (new). The same planted-signal benchmark is run on three
datasets with identical draws:

| dataset | role | mean corr(X_j, knockoff_j) |
|---|---|---|
| real latents, MVR S | primary | 0.50 |
| real latents, equicorrelated S | v1's choice; isolates S | 0.82 |
| Gaussian data, real covariance, MVR S | knockoffs exactly valid: **no zero atom** | 0.50 |

**Design** (fixed in `stage3_amendment_1`, committed before the full run):
- **Size:** p = 512 of 2,048 latents, chosen at random (MVR does not converge at 2,048), and
  n = 20,000 random rows.
- **Replicates:** 30 independent replicates per cell, each with a fresh knockoff draw.
- **Signals:** amplitudes 0.5–8, linear and interaction forms, k ∈ {10, 20, 30},
  q ∈ {0.05, 0.10, 0.20}.
- **Solver:** step size 1/L with a convergence check (100% converged).
- **Baselines:** Knockoff+ with offset 0, and a Westfall–Young marginal scan.

Rules: C1 (the Gaussian control must hold FDR); C2 (a power loss is blamed on the zero atom
only where real is significantly below the control); C3 (no tuning). Runtime: 76 minutes.

**Results**

- **C1 holds:** the control's FDR was at or below q in all 108 cells.
- **No inflation beyond q:** 0 of 108 real-data cells had FDR significantly above q, and 107
  of 108 had a mean at or below q.
- **The margin erodes.** At q = 0.10 the control's FDR sits at about 0.004–0.008. Real-data
  FDR rises from 0.007 to 0.060 as amplitude goes from 0.5 to 8. It is significantly above
  the control in 48 of 108 cells, and reaches nominal in the highest-signal small-k cells
  (k = 10, amplitude 8: 0.099 ± 0.023 at q = 0.10).
- **The "power collapse" is not the zero atom.**
  - Linear-signal power is within a few points of the control.
  - 11 of 108 cells are significantly lower and none higher (mean −0.015).
  - The atom does cost power for weak interaction signals (amplitude 0.5, k = 10: 0.23 vs
    0.50) and at a stronger penalty (λ = 0.05).
  - The equicorrelated S made almost no difference at p = 512.
- **The Knockoff+ floor:** k = 10, q = 0.05 has power 0.000, versus 0.985 with offset 0.
- **Marginal vs conditional:** the Westfall–Young marginal scan finds 150–210 latents with
  FDR 0.78–0.93 against the planted conditional truth. Knockoffs find about k with FDR
  0.02–0.04.

**Not tested:**
- p = 2,048 with the control, so what caused v1's power collapse (dimension, v1's
  never-checked solver, or its first-20,000-row subsample) is open.
- The mechanism behind the rising FDR.
- Amplitudes above 8.

## 4b. Stage 3 — follow-up at p = 2,048 (built, not yet run on real data)

**Why.** v2 at 512 latents did not reproduce v1's power collapse. At q = 0.10, v1 had linear
power 0.633 / 0.178 / 0.038 at amplitude 0.5 for k = 10 / 20 / 30; v2 had 0.900 / 0.820 / 0.503.
The two runs differ in four ways at once:
- the number of latents (2,048 vs 512);
- the solver;
- the rows used;
- the S matrix.

**A lead found while designing it.** On the real data at 2,048 latents, a safe lasso step is
about 0.033. v1 used a fixed step of 0.1, three times that, for 500 iterations, and never
checked convergence.

**What it runs.** `planted_fdr_controls.py --experiment p2048`, designed in
`stage3_amendment_2`. Everything stays at v1's settings (2,048 latents, Ledoit–Wolf covariance,
equicorrelated S), and five arms change one thing at a time:

| arm | rows | solver | purpose |
|---|---|---|---|
| `v1_setup` | first 20,000 | v1's, verbatim | should reproduce v1 (rule D1) |
| `first_fixed` | first 20,000 | fixed | effect of the solver |
| `random_fixed` | random 20,000 | fixed | effect of the rows |
| `gauss_fixed` | Gaussian control | fixed | effect of the zero atom at 2,048 |
| `gauss_v1` | Gaussian control | v1's | solver effect without the atom |

It covers the conditions where v1 reported the collapse: amplitudes 0.5 and 1, both forms,
k = 10/20/30, and 30 replicates.
- **Draws.** One knockoff draw takes about 21 s at this size, so each dataset gets a bank of 30
  draws and replicate *r* uses draw *r* in every condition. Replicates within a condition stay
  independent.
- **Pairing.** Arms on the same dataset share draws and labels, so their comparisons are paired.
- **Testing so far.** Smoke-tested on synthetic data only. Peak memory was 0.3 GB GPU and
  5.6 GB host RAM.
- **Not separable here.** Whatever gap remains between 2,048 and 512 latents is blamed jointly
  on the number of latents and the near-copy S matrix (rule D3).

**To run it** (on a GPU machine with the activation cache and `knockpy==1.3.5`):
```bash
git fetch && git switch stage1-power-fix
python src/planted_fdr_controls.py --config config/default.yaml --device cuda --experiment p2048 --stage diagnose  # ~5 min, prints projected runtime
python src/planted_fdr_controls.py --config config/default.yaml --device cuda --experiment p2048 --stage full      # est. ~1-1.5 h on a data-centre GPU
```
Send back `results/stage3_p2048_diagnose.json`, `results/stage3_p2048_fdr.json`,
`results/stage3_p2048_records.npz` and `results/fig16_stage3_p2048_power.png`.

## 5. Other files

- `config/default.yaml`: new `stage1` keys, plus `stage1_validate` and `stage3_v2` blocks.
- `src/common.py`: RNG streams 9–13 appended (`validate_permutation`, `validate_planted`,
  `s3v2_data`, `s3v2_planted`, `s3v2_knockoff`). Earlier streams are untouched, so earlier
  results stay reproducible.
- `results/stage1_findings.md` and `results/stage3_findings.md` were rewritten from the
  actual numbers, with sections listing the corrections to v1.
- `README.md`: the new scripts, runtimes and Stage 3 status.
- The v1 scripts and results are kept unchanged as history.

## 6. Reproducing

```bash
python src/calibrate_pipeline.py   --config config/default.yaml --device cuda               # ~6 min
python src/calibrate_validation.py --config config/default.yaml --device cuda               # ~5 min
python src/planted_fdr_controls.py --config config/default.yaml --device cuda --stage full  # ~76 min, needs knockpy
```

All three ran on an 8 GB laptop GPU (RTX 5050), with a peak of about 2.4 GB of GPU memory
for Stage 1. They need the activation cache `data/cache/d33d210c5acb.npz`.

## 7. Open items

1. Stage 1: add a null-intervention reference (matched random latents) to get a calibrated
   false-certification rate. This needs Amendment 3.
2. Stage 3: rerun v1's setting (p = 2,048, its solver and its row subsample) against the
   control, to explain v1's power collapse. Consider amplitudes above 8.
3. Paper: a "deviations from the proposal" section. Ablation is on a probe, not the model;
   the scope is SST-2 / Gemma only; thresholds were fixed through post-hoc amendments; and
   the Stage 1 false-certification rate is incomplete.
4. Merge into `main` (via a PR) so the corrected findings replace the v1 claims there.

## Note on history

The first three commits on this branch were reworded to one-line messages and their
trailers removed. Their content is byte-identical to the originals. Both their author and
committer timestamps were restored to the original times (02:16, 02:40, 03:42), which show
each design commit preceding its results (02:45 for Stage 1, 04:58 for Stage 3). The branch
was force-pushed once to apply this, shortly after it was first pushed.
