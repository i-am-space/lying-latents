# Lying Latents: paper story (status as of 29 October 2026)

Team SujayFans: Vishal Rao, Ayush Daga, Krithik Kambhampati, Mohammed Sami.

Sections 1–9 are the paper's story, written so they can be turned into the draft directly. Section 10 is not part of the draft: it lists what is still running and still to do.

---

## 1. One-paragraph pitch (abstract draft)

Interpretability work routinely reports that particular sparse-autoencoder (SAE) latents "drive" a model's behaviour. Two things are needed for such claims to be valid: the selection must control false discoveries, and it must be causal. We audit the standard ways of selecting SAE latents on three models (Gemma-2-2B, Gemma-3-1B, Pythia-70M) and nine datasets.

- Reusing data for search and validation certifies pure noise.
- Model-X knockoffs, the principled alternative, rest on an exchangeability assumption that SAE latents violate on every dataset we test. Three knockoff samplers, from Gaussian to a learned deep generator, each exceed the false discovery rate (FDR) target in some regime.
- Diagnostics that look at one latent at a time point the wrong way: the deep sampler looks best on them and is the worst on FDR.
- Even a valid observational method answers the wrong question: whether a latent predicts the label, not whether it drives the model.

We therefore propose interventional FDR control. Each candidate latent is intervened on inside the model, and we test whether its effect on the model's answer exceeds a pre-specified minimum effect δ, using betting e-values and the e-BH procedure. The guarantee is exact in finite samples, valid under any dependence between latents, and needs no model of the latents' distribution.

On two models and four prompted classification tasks:
- realised FDR never exceeds the target, including in a boundary stress test;
- power reaches 0.34–1.00 at 2,000 inputs;
- a zero-effect null, the closest prior approach, reaches an FDR of 0.60–0.96 against the minimum-effect truth;
- observational knockoffs find none of the latents that actually move the model.

## 2. Introduction: the argument in order

1. **Claims about SAE features are statistical claims made without statistics.** Dozens of latents are screened and the "top" ones are reported as meaningful, with no control of false discoveries.
2. **The naive fixes fail on real SAE latents.**
   - Validating on the data used for selection certifies noise.
   - Held-out validation reduces but does not remove the problem, because the usual ablation test's null hypothesis ("ablation changes nothing") is false even under random labels.
3. **The principled observational fix, Model-X knockoffs, does not hold up either.**
   - SAE latents are sparse and non-negative, with a point mass at zero and nonlinear co-firing. Knockoff samplers do not reproduce this.
   - We show the failure is detectable on every dataset and translates into FDR breaches in realistic regimes.
   - A better sampler (hurdle, deep generative) moves the failure around but does not remove it.
4. **Even a valid observational method answers an associational question.** Interpretability claims are causal: latent j matters for the model's decision.
5. **Contribution:** interventional e-BH with minimum-effect nulls. Interventions on our own model are free, exact and repeatable, so we can test the causal question directly, with finite-sample FDR control that does not depend on any distributional model.
6. **Contributions list:**
   - a pre-registered audit of SAE feature selection: three models, nine datasets, three knockoff samplers plus e-values;
   - evidence that per-latent diagnostics mislead (the deep knockoff result);
   - the interventional method;
   - its validation against oracle effects on real LLMs, including a stress test and a held-out-model confirmation;
   - a scientific finding: causal effects are spread thinly over many latents, and few matter individually.

## 3. Background (short)

- **SAEs.** Gemma Scope (JumpReLU) for Gemma-2-2B (layer 20, 16k) and Gemma-3-1B (layer 22, 16k); a ReLU SAE for Pythia-70M.
- **Multiple testing.** FWER and FDR; Benjamini–Hochberg; Benjamini–Yekutieli for dependent tests.
- **Model-X knockoffs (Candès et al. 2018).** The guarantee requires exchangeability of (X, X̃). The robustness bound (Barber, Candès & Samworth 2020) ties FDR inflation to how far the sampler's conditionals are from the true ones.
- **E-values and e-BH (Wang & Ramdas 2022).** e-BH controls FDR under arbitrary dependence. Betting e-values for bounded means (Waudby-Smith & Ramdas 2024).
- **Prior SAE-specific work.**
  - Enkhbayar (2025): Gaussian knockoffs on SAE latents.
  - Khera (ICML 2026 workshop): ablation e-processes with a zero-effect null.
  - Marks et al. (2025): attribution patching / sparse feature circuits.
  - Bal (2026): many SAE features are causally inert.

## 4. Experimental setup

- **Models and SAEs:** as above.
- **Datasets:**
  - SST-2 (sentiment);
  - eight SAEBench datasets: Bias in Bios sets 1–3, Amazon reviews (category and sentiment), GitHub code, AG News, EuroParl.
  - Activations are mean-pooled over each text (12,500 texts per SAEBench dataset; 2,048 most frequent latents retained).
- **Planted-signal benchmark (Part I):** real activations, synthetic labels Y = f(X_S) + noise from a known set S.
  - Linear and interaction designs;
  - 10–300 signal latents;
  - amplitudes matched to real labels.
  - Every method runs next to a **Gaussian control**: synthetic Gaussian data with the real covariance, on which Gaussian knockoffs are provably valid. The difference real-minus-control isolates the effect of the latents' non-Gaussian structure.
  - The penalty is chosen by 5-fold cross-validation (standard practice). A fixed conservative penalty hides the failures.
- **Pre-registration.** Every analysis, decision rule and seed was written down before the corresponding run. Changes made after seeing results are logged as amendments and labelled exploratory. In particular, the improved Stage 5 specification was chosen after seeing Gemma-2 and confirmed on Gemma-3, whose results had not been seen.

## 5. Part I: auditing SAE feature selection

### 5.1 Data reuse certifies noise

| Finding | SST-2 (Gemma-2) | Across 35 SAEBench tasks / models |
|---|---|---|
| Naive per-latent scan under the null | FWER 1.000; 104 false passes of 2,048 (5%) | FWER ≥ 0.95 everywhere; Bonferroni / Westfall–Young control it |
| Same-data validation of the top-10 | certifies noise in 98–100% of null experiments | median 92–100% (Gemma-2); Gemma-3 median 0.69, Pythia 0.94 |
| Held-out validation | 9.6–27% (nominal 5%) | median about 0.15 on Gemma-2 (3× α); about α on Gemma-3 / Pythia |

The held-out excess comes from the test's null hypothesis, not from a bug: ablating 10 *random* latents is "significant" in 84% of same-data experiments.

### 5.2 Knockoffs on SAE latents are not exchangeable

- **A typical latent is exactly zero on 30–89% of texts** (SST-2 0.89; SAEBench 0.30–0.73, falling with text length). A Gaussian knockoff never produces an exact zero. Even the trivial rule "is this value exactly 0?" separates real from knockoff at accuracy 0.65–0.87, matching (1 + Pr(X = 0)) / 2 to three decimals.
- **A classifier detects one swapped latent out of 2,048** at AUC 0.84–0.95 on all three models and all datasets. Swapping all of them gives AUC 1.0. The Gaussian control stays at 0.49–0.51, which validates the test.
- **The theory explains why:** any atom of positive mass makes the KL term in the robustness bound infinite.

### 5.3 Non-exchangeability becomes FDR breaches in realistic regimes

| Setting | Result (CV penalty) |
|---|---|
| SST-2, strong signals | Gaussian knockoffs breach in 7 of 54 cells after BY (all interaction designs); control: 0 |
| SST-2, many weak signals matched to the real labels | 12 of 24 cells breach (all interaction); control: 0 |
| SAEBench, original cells | small, consistent excess over the control (+0.007 to +0.026 FDR on 8 of 8 datasets), but no breach |
| SAEBench, amplitude sweep with strong interaction signals | block-MVR Gaussian knockoffs breach on 2 of 4 Gemma-2 datasets, all in interaction cells |

**Mechanism.** Null latents that co-fire with signal latents beat their Gaussian knockoffs.
- A "fair-coin" test makes this concrete. For valid knockoffs, a null latent's statistic is positive with probability 0.5. Under block-MVR Gaussian knockoffs it is 0.509–0.529 in interaction cells on all four datasets (each p < 10⁻³), and below 0.5 in linear cells.
- Gaussian knockoffs match second moments only, so they cannot reproduce nonlinear dependence.

### 5.4 Better samplers move the failure, they do not remove it

| Sampler | What it fixes | Where it fails |
|---|---|---|
| Gaussian, equicorrelated / block MVR | baseline; block MVR has much higher power | SST-2; strong interaction signals |
| Hurdle (models the zero atom and the size of positive values) | 0 breaches on the SST-2 realistic designs; holds on Gemma-3 | breaches on all 3 Pythia datasets (4, 7, 16 cells): the ReLU SAE has a tail of tiny activations its log-normal size model cannot produce. Thresholding the latents (JumpReLU-style) reduces this to 0 / 0 / 4. |
| Deep knockoffs v1 (learned residual on the Gaussian knockoff) | per-latent fit improves: single-latent swap AUC 0.60–0.86 against 0.79–0.98 for Gaussian; the zero atom is matched | repairs 0 of 13 settings; FDR above the control on 13 of 13 (+0.02 to +0.07); breaches on 9 of 13 |
| Deep knockoffs v2 (second-order exact, regularised, early stopping) | fixes v1's own failure (the linear-cell coin returns to 0.48–0.49) | repairs 0 of 3 pre-registered decision datasets; the learned part changes the knockoff by about 1%, so v2 behaves like block MVR |

**Key observation for the paper: diagnostics that check one latent at a time mislead.** The deep sampler looks best on them and is the worst on FDR. The guarantee is a joint property, and no sampler we built can be certified jointly on 2,048 dependent, zero-inflated latents.

### 5.5 The valid observational fallback is costly

E-value sample splitting never breaches: 0 BY breaches on 14 datasets, with FDR 0.000–0.008. But it gives up 0.27–0.46 power relative to knockoffs on SAEBench, and 0.60 on the smallest dataset. Its validity there is also empirical rather than guaranteed: the half-sample refit's null z-scores are over-dispersed.

**Part I conclusion:** no observational method is both valid and powerful on SAE latents. And even a valid one targets association with the label, not effect on the model.

## 6. Part II: interventional FDR control

### 6.1 Estimand and hypotheses

- **Metric:** m(x) = the model's probability of the correct class (among the label tokens) at the end of a few-shot prompt.
- **Effect:** Δ_j(x) = m(x) − m(x with latent j intervened on), and τ_j = E_x[Δ_j(x)].
- **Hypothesis:** H_j: τ_j ≤ δ, with δ = 0.05 × SD of m on a pilot split (about 1 percentage point).
- **Why not δ = 0:** a zero-effect null is useless at scale. Almost every latent moves the output slightly, so with enough data everything is rejected (we show this, §7.4).
- **The claim:** causality inside the model's computation, for this prompt format and these tasks. It is *not* causal structure among latents, real-world causes of the label, or uniqueness (redundant latents can be missed).

### 6.2 Intervention

- **What is changed:** latent j's activation is changed on every query token at the SAE layer, and (new − old) × its decoder vector is added to the residual stream. The SAE's reconstruction error is kept, and the few-shot demonstrations are never touched.
- **Primary intervention: conditional resampling.** A_j is redrawn from a token-level hurdle model of a_j given its 64 most co-active latents, which keeps it near the data manifold.
- **Sensitivity:** zero ablation and mean ablation.
- **Robustness argument:** the test's validity does not depend on the sampler, which only affects how meaningful the intervention is. So the imperfect samplers that break observational knockoffs are harmless here.
- **Efficiency:** an intervention at layer L reruns only the layers after L, on the cached residual stream. This is checked against the full forward pass (maximum difference 4 × 10⁻⁶ in class probability).

### 6.3 Procedure

1. **Splits:** pilot (300 inputs), screening (1,000), test pool (2,500), oracle (2,500), all disjoint.
2. **Screening:** attribution patching (gradient × activation) ranks all 16k latents. The top K become hypotheses (K = 250 pre-registered; K = 50 in the improved specification).
3. **Testing:** for each hypothesis, a betting e-value for a bounded mean.
   - Wealth = ∏(1 + λ_t(Δ_t − δ)), with λ_t predictable, i.e. computed only from earlier inputs (aGRAPA bets).
   - The bet cap is 0.5 pre-registered and 0.9 improved; the wealth factor never falls below 1 − cap.
   - Under H_j the wealth is a nonnegative supermartingale, so it is an e-value.
4. **Selection:** e-BH over the K hypotheses at α ∈ {0.05, 0.1, 0.2}.
5. **Guarantee:** FDR ≤ α in finite samples, under any dependence between latents. The assumptions are random input sampling and independence of the screening and test splits. No model of P(latents) is needed.

### 6.4 Validation design (real model, real ground truth)

- **Oracle:** τ_j for every latent measured on held-out inputs, so the FDR and power of each run can be computed against it.
- **Replicates:** each replicate re-splits the 5,000 test and oracle inputs at random into n test inputs and an oracle built from the rest; n = 100 to 2,000, 200 replicates.
- **Boundary stress test:** every latent's Δ is shifted so its mean is exactly δ, and inputs are bootstrapped. Every hypothesis is then null, so any rejection is false: the hardest case for an FDR procedure.
- **Baselines, scored against the same oracle:**
  - zero-effect null e-values (Khera-style);
  - attribution top-k (Marks et al.);
  - Gaussian knockoffs (Enkhbayar);
  - observational e-value splitting.

  The observational baselines are averaged over 10 draws, with each class at α / C so that the union over classes is FDR-controlled.

## 7. Part II results

### 7.1 FDR is controlled everywhere

- No breach in any setting: 2 models × 4 tasks × 2 specifications × 3 interventions × the δ grid × all n and α.
- Maximum realised FDR is 0.003.
- In the boundary stress test, P(any rejection) is at most 0.010, against α up to 0.2.

*(Table: validity per model and task.)*

### 7.2 Power is usable, and the improved specification is confirmed on a held-out model

Power at α = 0.1, resampling intervention, at n = 2,000 (pre-registered → improved):
- **Gemma-2:** 0.47–1.00 → 0.62–1.00.
- **Gemma-3** (confirmatory): 0.26–0.66 → 0.34–0.66. The improved specification has higher power on all four tasks at n = 1,000 and on three of four at n = 2,000 (tied on EuroParl).
- **Small n:** at n = 250 the improved specification gets 0–0.54 on Gemma-2. Power at small n is the main limitation.

*(Figure: power vs n, both specifications, 2 × 4 panels.)*

### 7.3 Few latents matter individually; most matter a little

- Of the 250 screened candidates, only 1–12 per task have τ > δ.
- Yet 59–91% have a reliably positive effect (τ > 2 SE).
- Every above-δ latent ranks in the top 50 by attribution, which is why K = 50 loses nothing.

*(Figure: sorted effects relative to δ.)*

### 7.4 A zero-effect null certifies negligible effects

- Same e-values with δ = 0.
- Its FDR against the minimum-effect truth is 0 at small n, then rises steeply. At n = 2,000 it is **0.66–0.96 on Gemma-2 and 0.60–0.84 on Gemma-3**.
- Against its own target (τ > 2 SE) it stays at about 0.
- So it is valid, but it answers a question that labels nearly every latent causal once there is enough data. This is the main difference from Khera.

*(Figure: FDR vs n, ours vs zero null.)*

### 7.5 Observational methods do not find the latents that drive the model

Power against the oracle, q = 0.1:

| Model | Task | # with τ > δ | Gaussian knockoffs | Obs. e-values | Ours (n = 2,000) |
|---|---|---|---|---|---|
| Gemma-2 | SST-2 | 3 | 0 | 0 | 0.99 |
| Gemma-2 | AG News | 4 | 0 | 0.03 | 0.75 |
| Gemma-2 | Bias in Bios | 1 | 0 | 1.00 | 1.00 |
| Gemma-2 | EuroParl | 5 | 0 | 0 | 0.62 |
| Gemma-3 | AG News | 8 | 0 | 0.06 | 0.61 |
| Gemma-3 | Bias in Bios | 3 | 0 | 0.10 | 0.34 |
| Gemma-3 | EuroParl | 9 | 0 | 0 | 0.66 |

- **Knockoffs:** with the per-class correction they select nothing on 6 of 7 tasks. On SST-2 they select about 34 latents, none above δ, and 31% with τ ≤ 0.
- **Observational e-values:** select a handful of latents. Where they select, 70–100% are below δ, with two exceptions: the single above-δ latent on Gemma-2 Bias in Bios, and one above-δ latent on Gemma-3 AG News in about half the draws.
- **Honest framing:** observational methods target association, use pooled raw-text activations and see a smaller latent pool. The comparison shows an estimand mismatch, not an FDR breach on their own target.

### 7.6 (Pending) Do better observational samplers pick causally real latents?

Paired comparison of Gaussian, block-MVR and hurdle knockoffs against the same oracle, with both fixed and CV penalties:
- On Gemma-2 Bias in Bios and EuroParl, all three samplers select nothing.
- The decisive task (SST-2) is still running; see §10.

Expected message, either way: a better sampler can at most remove statistical false positives; it cannot turn association into effect.

## 8. Discussion and limitations

- **What we claim:**
  - observational SAE feature selection is unreliable in both senses, statistical validity and causal relevance;
  - interventional e-BH gives exact, assumption-light FDR control for within-model causal claims.
- **Limitations:**
  - Models are 1–2B parameters.
  - One SAE per model.
  - Four tasks, one prompt format.
  - Power is relative to screened candidates; latents never screened are not counted.
  - Single-latent tests miss latents that matter only jointly (redundancy). Group testing of correlated clusters is future work.
  - Power at small n is limited by e-BH over many hypotheses.
- **Why the stress test matters:** e-values are conservative by construction. The stress test shows they are not just trivially silent, since power is substantial at moderate n.
- **Future work (the ARR version):**
  - larger models (e.g. Gemma-2-9B);
  - group / hierarchical testing;
  - learned samplers that are second-order exact and trained on large unlabeled pools;
  - other intervention families (steering, patching).

## 9. Figures and tables for the draft

1. Pipeline schematic: screen → intervene → betting e-values → e-BH, with the oracle validation (to be drawn).
2. Part I summary table: method × validity × power (from §5.4–5.5).
3. Exchangeability: single-latent swap AUC per dataset and sampler, real vs control.
4. FDR breaches: SST-2 and amplitude-sweep cells, real vs control; the coin test by design type.
5. Stage 5 power vs n (2 × 4 panels). *Exists.*
6. Zero null FDR vs n. *Exists.*
7. Sorted effect sizes. *Exists.*
8. Table: validity per model and task. *Exists.*
9. Table: power against the oracle, ours vs observational. *Exists.*

---

## 10. Not part of the draft: what's left to do

**Running now / waiting on GPUs**
- **Sampler comparison on Gemma-2 SST-2 and AG News** (§7.6): running on the L40S, about 1 h. It decides the pre-registered hurdle-vs-block-MVR contrast; this is currently undefined because only SST-2 has selections from both.
- **Exchangeability tests that crashed** (hurdle on 3 more Gemma-2 datasets; Pythia unthresholded and thresholded): rerunning on the L40S, about 2–3 h. Appendix material.
- **Gemma-3 sampler comparison and exchangeability tests:** Ayush, on his machine (it has the Gemma-3 caches), when its GPUs are free. Afterwards we measure any new latents on srijam (minutes).

**Writing (critical path, due 31 Oct)**
- Draft sections 1–8. Krithik starts from this document; the per-stage findings docs in `docs/findings/` have every number with its source.
- Draw the pipeline schematic (Figure 1).
- Make the Part I figures (exchangeability AUC plot; breach / coin plot). Older figures exist in `results/figures/` (fig10–fig19) and may be reusable.
- Related work: read Khera (ICML 2026 workshop) carefully before fixing the novelty framing.
- One pass to make sure every number in the draft matches the findings docs; update §7.6 when the sampler comparison finishes.

**Where things are (for the drafters only)**
- Stage 5 figures and tables:
  - figures: `results/figures/fig20_stage5_power.png`, `fig21_stage5_zero_null.png`, `fig22_stage5_effects.png`;
  - tables: `results/figures/table_stage5_validity.tex`, `table_stage5_baselines.tex`;
  - regenerated by `python src/stage5_figures.py`.
- Findings docs:
  - Part I: `stage1*_findings.md`, `stage2*`, `stage3_findings.md`, `stage4_findings.md`, `stage34_*`, `models_sae_findings.md`, `evalue_*`, `deep_knockoffs_sae_findings.md`;
  - Part II: `stage5_findings.md`.
- Planning doc shared with the team on 22 Oct: `docs/planning/stage5_interventional_fdr.pdf`.
- Pre-registration and amendments: `config/preregistration.yaml`.

**Decisions to make as a team**
- Title (working: *"Which SAE latents drive the model? Valid feature discovery needs interventions, not knockoffs"*).
- How much of Part I goes in the main text vs the appendix. Suggestion: §5.1–5.3 condensed to one page plus one table in the main text; the samplers (§5.4) in a table, with details in the appendix.
- Whether to attempt Gemma-2-9B for the ARR version, using the Jarvis Labs credits. Not for 31 Oct.
