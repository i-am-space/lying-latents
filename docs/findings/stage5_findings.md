# Stage 5 findings — interventional FDR control for SAE latents

Pre-registered in `config/preregistration.yaml`:
- `stage5_amendment_1`: method, tasks, splits, δ rule, baselines
- `stage5_amendment_2`: re-split evaluation, stress test, v2 specification, fairer baselines. Written after reading the Gemma-2 results and before any Gemma-3 result was opened.

Plan: `docs/planning/stage5_interventional_fdr.pdf`. Code: `src/stage5_interventional.py` (`--stage all`, then `--stage v2`). Script: `scripts/stage5.sh`.

Results:
- Gemma-2-2B (Gemma Scope L20 16k): `results/stage5/<task>/`
- Gemma-3-1B (Gemma Scope 2 L22 16k): `results/models/gemma3_1b/stage5/<task>/`

The headline numbers are in `evaluate2_summary.json`. The amendment-1 analysis is kept in `evaluate.json`.

## Summary

1. **FDR is controlled everywhere.** The grid is 2 models × 4 tasks × 2 specifications × 3 interventions × the δ grid × n = 100–2,000 × α ∈ {0.05, 0.1, 0.2}.
   - No breach anywhere.
   - Realised FDR is at most 0.002, and 0.000 almost everywhere.
   - In the boundary stress test, every latent's Δ is shifted so its mean is exactly δ, which makes every rejection false. P(any rejection) is at most 0.005 in every setting.
2. **Power is usable, and the v2 specification (K = 50, bet cap 0.9) is confirmed on Gemma-3.** v2 was chosen after reading Gemma-2. On Gemma-3, where it is the pre-registered primary analysis, v2's power is higher than the original specification's on all 4 tasks at n = 1,000 and 2,000, with no loss of validity. At n = 2,000 (resampling, α = 0.1), power is 0.62–1.00 on Gemma-2 and 0.34–0.66 on Gemma-3.
3. **Few latents matter alone, most matter a little.**
   - Of the 250 attribution-ranked candidates, only 1–12 per task have an effect above δ (0.05 SD of the metric, about 1 percentage point of correct-class probability).
   - Yet 59–91% have a real positive effect (τ > 2 SE).
   - All above-δ latents are in the top 50 by attribution.
4. **A zero-effect null certifies trivial effects.** The zero-null baseline is the same e-values with δ = 0, the closest prior work (Khera 2026).
   - At n = 2,000 its FDR against the minimum-effect truth is 0.66–0.96 on Gemma-2 and 0.60–0.84 on Gemma-3.
   - At small n it is 0, and it rises with n and with power.
   - Against its own target (τ > 2 SE) it stays at about 0. So it is valid, but it answers a question that labels nearly every latent causal once there is enough data.
5. **Observational selection does not find the latents that drive the model.** Baselines over 10 draws, each class at α / C: 4 Gemma-2 tasks, and 3 Gemma-3 tasks (no Gemma-3 SST-2 cache).
   - Gaussian knockoffs select nothing on 6 of 7 tasks. On Gemma-2 SST-2 at q = 0.1 they select 34 latents, none above δ; 42% have no detectable effect and 31% have τ ≤ 0.
   - Observational e-values select 0–7 latents per task. Where they select, 70–100% of their picks are below δ, with one exception: Gemma-3 AG News, where they pick one above-δ latent in about half the draws. On Gemma-3 Bias in Bios, about half their picks have no detectable effect.
   - Power against the oracle (q = 0.1) shows the gap. Knockoffs find 0 of the above-δ latents on every task and both models. Observational e-values find 0–0.10 of them, with one exception: on Gemma-2 Bias in Bios, the single above-δ latent is selected in every draw. Ours finds 0.34–1.00 at n = 2,000.

**What this establishes:** within-model causal claims about individual SAE latents can be made with exact, finite-sample FDR control, without a model of the latents' distribution. The procedure is valid in a stress test on real models, and confirmed on a second model with a pre-registered specification. Both widely used alternatives fail on our real data:
- observational knockoffs answer an associational question, and their samplers are themselves invalid (Stages 3–4);
- zero-effect nulls certify negligible effects.

**What it does not establish:**
- Claims about the model beyond this prompt format, these tasks and these interventions.
- Claims about latents never screened. Power is relative to the 250 candidates.
- Claims about redundant latents that matter only jointly; group testing is deferred.
- Results at scale: the models are 1–2B parameters.

## 1. Setup

| Item | Value |
|---|---|
| Models | Gemma-2-2B, Gemma Scope L20 16k (canonical); Gemma-3-1B, Gemma Scope 2 L22 16k (medium L0); float32 |
| Tasks | SST-2, AG News, Bias in Bios (class set 1, 5 professions), EuroParl (5 languages); 1-shot-per-class prompts |
| Metric m(x) | probability of the correct class among the label tokens; Δ_j(x) = m(x) − m(x; intervene on j) |
| Hypotheses | H_j: τ_j ≤ δ, with δ = 0.05 × SD(m) on the pilot split (0.008–0.016); sensitivity 0.02 and 0.10 × SD |
| Interventions | primary: conditional resampling from a token-level hurdle sampler of a_j given its 64 most co-active latents; sensitivity: zero and mean ablation |
| Splits | pilot 300, screen A 1,000, B pool 2,500, oracle O 2,500 (disjoint) |
| Screening | attribution patching on A; candidates = top 250 positive scores (v2: top 50) |
| Test | betting e-values (aGRAPA, prior variance 0.25; bet cap 0.5 prereg, 0.9 v2) and e-BH, α ∈ {0.05, 0.1, 0.2} |
| Evaluation (v2) | 200 replicates per n, each re-splitting B ∪ O into n test inputs and an oracle from the rest; stress test with 200 bootstrap replicates at n = 250, 1,000, 2,000 |
| Gate | prompted balanced accuracy ≥ 0.70 and replay equivalence ≤ 1e-3; all 8 pairs passed (accuracy 0.77–0.95, max replay error 4e-6) |

## 2. Validity

| Model | Breaches (S5-1, S5-5) | Max realised FDR | Max stress P(any rejection) |
|---|---|---|---|
| Gemma-2 (4 tasks) | 0 | 0.000 | 0.005 (EuroParl, v2, zero / mean ablation) |
| Gemma-3 (4 tasks) | 0 | 0.002 (Bias in Bios, v2, mean ablation, n = 1,000) | 0.005 (SST-2 and EuroParl, v2, mean ablation) |

The re-split replicates still resample one finite pool. Their SE is often 0, because the same strong latents are selected in every replicate. The stress test is therefore the main validity evidence. The e-values are conservative by construction (Markov's inequality), which matches a near-zero rejection rate at the boundary.

## 3. Power (resampling, primary δ, α = 0.1, prereg → v2)

| Model | Task | Pilot acc. | τ > δ | n = 250 | n = 500 | n = 1,000 | n = 2,000 |
|---|---|---|---|---|---|---|---|
| Gemma-2 | SST-2 | 0.95 | 3 | 0.00 → 0.54 | 0.40 → 0.67 | 0.67 → 0.67 | 0.66 → 0.99 |
| Gemma-2 | AG News | 0.84 | 4 | 0.01 → 0.25 | 0.25 → 0.26 | 0.25 → 0.46 | 0.47 → 0.75 |
| Gemma-2 | Bias in Bios | 0.85 | 1 | 0.00 → 0.00 | 0.00 → 0.04 | 0.01 → 0.93 | 1.00 → 1.00 |
| Gemma-2 | EuroParl | 0.95 | 5 | 0.00 → 0.39 | 0.39 → 0.39 | 0.40 → 0.56 | 0.56 → 0.62 |
| Gemma-3 | SST-2 | 0.87 | 4 | 0.00 → 0.01 | 0.00 → 0.25 | 0.25 → 0.27 | 0.26 → 0.51 |
| Gemma-3 | AG News | 0.77 | 8 | 0.06 → 0.21 | 0.20 → 0.26 | 0.25 → 0.48 | 0.42 → 0.61 |
| Gemma-3 | Bias in Bios | 0.83 | 3 | 0.00 → 0.00 | 0.00 → 0.00 | 0.00 → 0.21 | 0.28 → 0.34 |
| Gemma-3 | EuroParl | 0.82 | 9 | 0.02 → 0.28 | 0.31 → 0.58 | 0.54 → 0.65 | 0.66 → 0.66 |

Power is against the oracle's τ > δ, so latents only just above δ are nearly undetectable at any n. Zero and mean ablation give similar power, measured on 1,000-input pools, so n ≤ 1,000.

## 4. Effect sizes (resampling, all 5,000 B ∪ O inputs, 250 candidates)

| Model | Task | τ > 0 | τ > 2 SE | τ > δ | Largest τ |
|---|---|---|---|---|---|
| Gemma-2 | SST-2 | 233 | 195 | 3 | 0.046 |
| Gemma-2 | AG News | 235 | 227 | 4 | 0.066 |
| Gemma-2 | Bias in Bios | 232 | 216 | 1 | 0.027 |
| Gemma-2 | EuroParl | 219 | 196 | 5 | 0.064 |
| Gemma-3 | SST-2 | 205 | 147 | 4 | — |
| Gemma-3 | AG News | 234 | 210 | 8 | — |
| Gemma-3 | Bias in Bios | 218 | 172 | 3 | — |
| Gemma-3 | EuroParl | 221 | 196 | 9 | — |

## 5. Zero-null baseline (v2 specification, resampling, α = 0.1)

FDR against the minimum-effect truth (τ > δ) by n; against τ > 2 SE it is ≤ 0.002 everywhere.

| Model | Task | n = 500 | n = 1,000 | n = 2,000 |
|---|---|---|---|---|
| Gemma-2 | SST-2 | 0.00 | 0.40 | 0.66 |
| Gemma-2 | AG News | 0.01 | 0.50 | 0.80 |
| Gemma-2 | Bias in Bios | 0.00 | 0.78 | 0.96 |
| Gemma-2 | EuroParl | 0.03 | 0.59 | 0.78 |
| Gemma-3 | SST-2 | 0.00 | 0.34 | 0.67 |
| Gemma-3 | AG News | 0.02 | 0.60 | 0.79 |
| Gemma-3 | Bias in Bios | 0.16 | 0.56 | 0.84 |
| Gemma-3 | EuroParl | 0.05 | 0.44 | 0.60 |

## 6. Observational baselines (baselines2: 10 draws, each class at α / C)

### Gemma-2

Columns are FDP measured three ways: against τ > δ, against τ > 2 SE, and against τ > 0. "Selected" is the mean number of latents per draw.

| Task | Method, q | Selected | Draws selecting | FDP: τ > δ / τ > 2 SE / τ > 0 |
|---|---|---|---|---|
| SST-2 | knockoffs 0.05 / 0.1 / 0.2 | 11 / 34 / 53 | 0.4 / 1 / 1 | 0.40 / 1.00 / 1.00 · 0.16 / 0.42 / 0.51 · 0.11 / 0.31 / 0.32 |
| SST-2 | e-values 0.05 / 0.1 / 0.2 | 4 / 5 / 7 | 1 / 1 / 1 | 1.00 everywhere · 0.07 / 0.08 / 0.09 · 0.07 / 0.08 / 0.06 |
| AG News | knockoffs | 0 | 0 | — |
| AG News | e-values | 1 | 0.9–1 | 0.8–0.9 · 0 · 0 |
| Bias in Bios | knockoffs | 0 | 0 | — |
| Bias in Bios | e-values | 4 | 1 | 0.70–0.73 · 0 · 0 |
| EuroParl | both | 0 | 0 | — |

### Gemma-3

No SST-2 run: there is no Gemma-3 SST-2 cache. Columns as above.

| Task | Method, q | Selected | Draws selecting | FDP: τ > δ / τ > 2 SE / τ > 0 |
|---|---|---|---|---|
| AG News | knockoffs | 0 | 0 | — |
| AG News | e-values 0.05 / 0.1 / 0.2 | 0.4 / 0.5 / 0.6 | 0.4 / 0.5 / 0.6 | 0 · 0 · 0 (the selected latent is above δ) |
| Bias in Bios | knockoffs | 0 | 0 | — |
| Bias in Bios | e-values 0.05 / 0.1 / 0.2 | 3.8 / 3.9 / 4.2 | 1 | 0.93 / 0.93 / 0.91 · 0.48 / 0.50 / 0.51 · 0.27 / 0.26 / 0.25 |
| EuroParl | both | 0 | 0 | — |

### Power against the oracle (q = 0.1; ours: v2, resampling, n = 2,000)

| Model | Task | τ > δ | Knockoffs | Obs. e-values | Ours |
|---|---|---|---|---|---|
| Gemma-2 | SST-2 | 3 | 0 | 0 | 0.99 |
| Gemma-2 | AG News | 4 | 0 | 0.03 | 0.75 |
| Gemma-2 | Bias in Bios | 1 | 0 | 1.00 | 1.00 |
| Gemma-2 | EuroParl | 5 | 0 | 0 | 0.62 |
| Gemma-3 | AG News | 8 | 0 | 0.06 | 0.61 |
| Gemma-3 | Bias in Bios | 3 | 0 | 0.10 | 0.34 |
| Gemma-3 | EuroParl | 9 | 0 | 0 | 0.66 |

On Gemma-2 SST-2, knockoffs selected 93 latents outside U, which delta2 evaluated. The amendment-1 single-draw numbers (classes merged at q) are in `evaluate.json`. They were less favourable to the baselines (FDP 0.6–1.0), which is why the comparison was redone.

Differences that remain:
- The baselines see mean-pooled activations of the raw text and only the 2,048 retained latents; our method sees the latents inside the prompt.
- They target association with the label, not effect on the model.

These differences are stated, not corrected.

## 7. Deviations and notes

- **Gemma-3 runs:** a co-tenant took GPU 1, so Gemma-3 AG News and EuroParl ran out of memory at load. They were rerun on GPU 0 with nothing changed.
- **Gemma-3 observational baselines:** `baselines2` was run on a teammate's machine, which has the Gemma-3 caches (commit 895a1cc).
  - There is no Gemma-3 SST-2 cache, so SST-2 has none.
  - Knockoffs select nothing on AG News, Bias in Bios or EuroParl.
  - Observational e-values select on average 0.4–0.6 latents on AG News, about 4 on Bias in Bios, and none on EuroParl.
  - The 3 Bias in Bios latents outside U were measured by `delta2` on srijam (8 min), followed by `evaluate2` (about 3 min per task).
- **v2 is exploratory on Gemma-2:** it was chosen after reading Gemma-2's results (the `stage5_amendment_2` trigger records the CPU checks behind the choice), and Gemma-2's primary analysis stays prereg. On Gemma-3, v2 is confirmatory.
- **Runtime** on an RTX 5000 Ada:
  - Gemma-2 `delta`: 2–9 h per task;
  - Gemma-3 `delta`: about 0.5–2 h;
  - `baselines2` + `delta2`: under 1 h per task;
  - `evaluate2`: about 1 min per task on a CPU.
- **Limits:**
  - 2 models (1–2B) and one SAE each;
  - 4 tasks, one prompt format;
  - candidates limited to attribution-screened latents;
  - no group testing.
