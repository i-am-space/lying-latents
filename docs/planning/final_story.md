# Lying Latents — the final story (start to finish)

**Second review (31 Oct): `final_report.tex` wins over this file.** "Knockoffs select nothing" is largely the Knockoff+ floor of
1/(q/C) picks per class; "at most 5%, whichever sampler" and "repair the statistics, not the selections" are withdrawn.

**Status (31 Oct, after the team's review of 1c45a69):** corrected for the Khera comparator error and other factual points.
This text predates the team's same-day edits of `final_report.tex`; where the two differ on wording (e.g. observational
methods, binarised counts, describing resampling), **the team's edited report wins**. Line numbers in §10 are stale.

Written 31 Oct 2026 as the source for the final paper. Every number here is in `results/FINAL_RESULTS.json`
(Khera comparison: section `6_khera_comparison`), `results/stage5_baselines3_summary.json` or
`results/stage5_khera_baseline.json`. Supersedes the framing in
`paper_story.md` where they disagree.

---

## 0. The message in one paragraph

People find "the sentiment latent" in a sparse autoencoder by scanning thousands of latents and checking the winners,
which certifies noise. The statistical fix that has been proposed, Model-X knockoffs, carries a guarantee that does not
apply to SAE latents, because they are exactly zero most of the time and fire together in ways no decoy we built
reproduces. We repaired the decoys as far as we could; the best repair (hurdle) fixes the statistics but not the
answer: whichever sampler is used, the latents knockoffs pick matter about as often as a random screened candidate
(descriptive; few hits, no test). Knockoffs answer "does this latent predict the label?", while interpretability claims mean "does the model
use it?". So we test that question directly, with interventions, betting e-values, e-BH and a minimum effect size,
and verify against measured effects on two real models that it keeps its error guarantee.

**Slogan:** *we could repair the statistics of knockoffs, but not the question they answer.*

### Draft abstract (numbers checked)

> Interpretability work often claims that a particular sparse autoencoder (SAE) latent drives a model's behaviour,
> after selecting it from thousands of candidates. We audit how such latents are selected on three models and nine
> datasets. Searching and validating on the same data certifies pure noise in up to 100% of experiments. Model-X
> knockoffs, the proposed statistical fix, compare each latent with a matched fake copy, but SAE latents are exactly
> zero most of the time and fire together in ways no fake we built reproduces; the guarantee fails and FDR is exceeded
> in interaction settings. A hurdle sampler that reproduces the zeros repairs the statistics but not the selections:
> measured by intervening on the model, few knockoff picks from any sampler have a meaningful effect, about as
> many as among attribution-screened candidates. We therefore test the causal question directly, with interventions, betting e-values, e-BH and a
> minimum meaningful effect. On two models and four tasks its realised FDR never exceeds 0.003, and power reaches
> 0.34–0.66 on the held-out model at 2,000 inputs, while a zero-effect null certifies negligible latents as data grows.
> Only 1–9 of 250 screened latents per task have a meaningful individual effect.

### Contributions (for the intro)

1. An audit of SAE feature selection: the standard pipeline under a permutation null, and the exchangeability and FDR
   of Gaussian, hurdle, deep and binarised knockoffs on 3 models and 9 datasets (§4).
2. A downstream check of knockoff selections against measured causal effects: whichever sampler, few picks matter, about as
   often as a random screened candidate (§5).
3. Interventional e-BH with a minimum-effect null, and evidence that the minimum effect is necessary: a zero null certifies
   negligible latents as n grows, and a relative null's target depends on the candidate set (§6).
4. Validation of realised FDR against measured effects on two real models, with a stress test and a held-out model (§7).
5. The finding that few latents matter individually (§7).

---

## 1. How this maps to what we promised

| Proposal / mid-review promise | Where it is in the paper | Delivered? |
|---|---|---|
| Stage 1: the standard pipeline under a permutation null | §4.1 | Yes, 3 models, 9 datasets |
| Stage 2: audit Model-X exchangeability (zero mass, swap tests) | §4.2 | Yes, plus hurdle and deep samplers |
| Stage 3: realised FDR on planted signals | §4.3 | Yes, CV penalty, SST-2 + SAEBench, 3 models |
| Stage 4: repairs (hurdle, binarised, e-value splitting) | §4.4–4.5 | Yes, plus two deep samplers |
| Stage 5: "do filter survivors actually matter?" (proposal: steering; mid-review: "clamp latents selected by hurdle and by Gaussian knockoffs") | §5 | Yes, as interventions on the model's own task (see §7 for why not steering) |
| Interim plan item 3: "supply a procedure that holds" | §6 | Yes: interventional e-BH with a minimum effect |
| Gemma-2-9B, Gamma hurdle | Limitations | Dropped (compute) |

---

## 2. The problem (Introduction)

- An SAE rewrites a model's activations as sparse combinations of ~16k learned directions (latents). Papers then claim
  "latent j encodes concept X and the model uses it".
- The standard recipe is **search-then-validate**: rank latents by association with a label, keep the top few, ablate
  them and see the model change.
- Two statistical problems: (1) thousands of tests and no multiplicity correction; (2) the same data selects and
  validates (double dipping).
- One proposed fix exists for SAEs: Model-X knockoffs (Enkhbayar 2025), which promise false discovery rate (FDR)
  control. Its guarantee needs the decoys to be exchangeable with the real latents. We test whether that holds, whether
  it matters, whether it can be repaired, and whether knockoff picks are the latents the model uses.

---

## 3. Background in plain words (§2)

- **FDR:** the expected share of wrong picks in the list you report.
- **Knockoffs:** for each latent, build a fake copy without looking at the label. Fit a sparse model on real and fake
  latents together. A latent is picked if it predicts the label clearly better than its own fake. For a useless latent,
  real and fake are interchangeable, so each wins half the time; counting how often fakes win estimates the number of
  false picks, and the threshold is set to keep FDR ≤ q. This requires the fakes to be **exchangeable**: swapping any
  set of latents with their fakes must leave the joint distribution unchanged. If not exact, FDR inflation is bounded by
  a KL divergence (Barber et al. 2020).
- **Samplers:** Gaussian second-order (equicorrelated, MVR, block MVR), sequential conditional (SCIP), deep generative.
- **E-values:** a score that averages ≤ 1 when there is no effect, so a large value is evidence. **e-BH** turns e-values
  into a list with FDR ≤ α under any dependence. **Betting** builds an e-value as a gambler's wealth when betting
  against the null.
- **Related SAE work:** Enkhbayar (Gaussian knockoffs on SAEs); Khera (ICML 2026 Hypothesis Testing workshop:
  zero-ablation e-processes against a random candidate, e-BH; 6 of 89 features on GPT-2 small IOI; ground truth only on
  a synthetic 2-layer transformer; Gaussian knockoffs select nothing on GPT-2 and breach on their synthetic SAE at
  α = 0.2–0.3, FDP 0.27 and 0.45); Shi et al. (NeurIPS 2024, hypothesis testing of a whole circuit); Bal (many SAE features are causally inert); Guo et al.
  (arXiv 2610.15850, 11 Oct 2026: observational discovery scores do not predict interventional effects; no FDR control);
  Marks et al. (attribution patching to pick latents).

---

## 4. The audit (§4) — Stages 1–4

### 4.1 Search-then-validate certifies noise (Stage 1)

Shuffle the labels 500 times so no latent can carry signal, then run the standard pipeline.

| | SST-2 (G-2) | SAEBench G-2 | G-3 | Pythia |
|---|---|---|---|---|
| Validate on the same data (should be 0.05) | 0.98–1.00 | 0.92–1.00 | 0.69 | 0.94 |
| Validate on held-out data | 0.10–0.27 | 0.15 | 0.07 | 0.08 |
| Uncorrected scan, false passes (expected 102) | 104 | 102 | – | – |

Held-out validation stays above 0.05 on Gemma-2 because "ablation changes nothing" is false even under random labels:
ablating 10 random latents is significant 84% of the time. **First hint that "has some effect" is a useless bar.**

### 4.2 Knockoffs are not exchangeable on SAE latents (Stage 2)

- A typical latent is **exactly zero** on 89% of SST-2 sentences (30–73% on SAEBench). Gaussian fakes are never zero.
- **Zero rule:** "the one that is exactly 0 is real" detects the fake with accuracy (1 + Pr(X=0))/2: 0.95 on SST-2,
  0.65–0.87 on SAEBench (valid fakes: 0.5).
- Classifier swap test: one swapped Gaussian latent is detected (AUC₁ 0.997 SST-2, 0.84–0.95 SAEBench); all swapped,
  AUC 1.00. Gaussian control data: 0.49–0.51; synthetic Gaussian: AUC 0.497, MMD p = 0.96 (the tests are calibrated).
- Any atom at zero makes the KL term infinite, so the robustness bound gives nothing.

### 4.3 Non-exchangeability becomes FDR breaches (Stage 3)

Planted labels on real latents (CV penalty), BY-corrected breach counts:

| Data | Design | Breaches |
|---|---|---|
| SST-2 | strong signals | 7 / 54 cells |
| SST-2 | weak, matched to real-label AUC | 12 / 24 cells |
| Gemma-2, 8 datasets | realistic | 0 / 8 (FDR +.007 to +.026 over control) |
| Gemma-2, 4 datasets | interaction sweep | 2 / 4 |
| Gemma-3, 3 datasets | realistic | 1 / 3 |
| Pythia, 3 datasets | realistic | 0 / 3 |

- Every breach is in an interaction design; the Gaussian control never breaches.
- Mechanism (fair-coin test): a null latent should beat its fake half the time; under block MVR it wins 0.509–0.529 in
  interaction cells (each p < 10⁻³). Null latents that co-fire with signal latents are biased, a nonlinear dependence
  second-moment fakes cannot reproduce.
- Power cost: k = 10, amplitude 0.5 interactions: 0.23 found vs 0.50 on control.
- **Honest size of the effect:** breaches are real but modest (excess FDR 0.01–0.03), concentrated in interaction designs.

### 4.4 Better samplers move the failure (Stage 4)

| Sampler | Single-swap AUC₁ | Settings breaching | Notes |
|---|---|---|---|
| Gaussian block MVR | 0.79–0.98 | 2 / 13 | |
| **Hurdle (SCIP)** | **0.49–0.60** | 4 / 11 | zero rule 0.50; all-swap still 0.94–1.00 |
| Deep (residual) | 0.60–0.86 | 9 / 13 | null coin 0.59–0.63 even in linear cells |
| Deep (2nd-order exact) | 0.78–0.96 | 4 / 13 | learned part ≈ 1% of the noise |
| Binarised (SST-2) | 0.50 (all swapped) | 2 / 54 cells | FDR mean .065, up to .17 |

**Hurdle in detail (our main proposed repair), being fair:**
- Wins: removes the zero-atom signal everywhere; recovers 77% of the power gap on SST-2 (.668 vs .620, control .682)
  with lower FDR (.013 vs .020); removes the Gaussian breach on Gemma-3 EuroParl (2 → 0 cells); power on SAEBench is
  within ±0.03 of Gaussian.
- Losses: still detected when all latents are swapped; breaks on Pythia's ReLU SAE (7, 16, 4 breaching cells vs 0 for
  Gaussian) because its log-normal size model cannot fit the tail of tiny activations (0, 4, 0 after a JumpReLU-style
  floor); one breach on Gemma-2 Amazon category.

**Per-latent diagnostics mislead:** the residual deep sampler is the best on single-swap tests and the worst on FDR.
FDR is a joint property of 2,048 dependent latents and none of our samplers can be certified jointly.

### 4.5 The valid fallback is costly

E-value splitting (lasso on one half, refit on the other, e-BH) never breached (14 settings, FDR 0.000–0.008) but has
0.27–0.46 less power than hurdle knockoffs, and its validity is empirical (null z-scores have variance 1.69).

**Section conclusion:** no observational method is both valid and powerful on SAE latents.

---

## 5. The downstream check: do knockoff picks matter to the model? (Stage 5 as promised)

This is the experiment the proposal and mid-review promised ("clamp latents selected by hurdle and by Gaussian
knockoffs"). It is the bridge to the method: put it **before** §6 in the paper.

**Setup.**
- Four prompted classification tasks (SST-2, AG News, Bias in Bios, EuroParl) on Gemma-2-2B and Gemma-3-1B, one
  demonstration per class; a task runs only if prompted balanced accuracy ≥ 0.70 (all 8 pass, 0.77–0.95).
- For a latent j, the effect on input x is how much the model's probability of the correct label drops when j is
  replaced by a resampled value. τⱼ is its average over 2,500 held-out inputs: this is the **oracle**, the
  measured truth.
- A latent **matters** if τⱼ > δ = 0.05 × SD of that probability (0.008–0.016, about one percentage point).
- Knockoffs (Gaussian equicorrelated = Enkhbayar's, Gaussian block MVR, hurdle) are run on each task's real labels,
  10 paired draws, fixed and CV penalties, q ∈ {0.05, 0.1, 0.2}. **Every latent any sampler picks is intervened on and
  measured.**

**Results.**
- At q = 0.1, knockoffs select nothing on 6 of 7 tasks. On Gemma-2 SST-2 they select 19–71 latents per draw; **none**
  of them matters (0% above δ), and none of the 3 latents that matter is found.
- At q = 0.2: still 0% on SST-2 (36–137 picks); elsewhere at most 2–5% of picks matter, recall ≤ 6%. A random
  latent from the screened, measured set matters 0.4–3.6% of the time, so knockoff picks look **about as good as a random
  screened candidate**. This is descriptive: there are very few hits (1 of 20 on AG News) and no test behind it.
- Hurdle vs Gaussian on "reliably positive effect" (τ > 2 SE): hurdle beats block MVR (+0.071, p = 0.015, 10 paired
  draws, CV) but loses to equicorrelated Gaussian (−0.11). The ranking follows how many latents each sampler picks,
  not how exchangeable it is.
- Observational e-values: mostly miss too, except Gemma-2 Bias in Bios (they find the only latent above δ, plus ~3
  below it).

**Takeaway:** better exchangeability does not produce better picks. The problem is the question (association with the
label), not only the sampler. Independent support: Khera found knockoffs select nothing on GPT-2 IOI and breach on a
synthetic SAE (FDP 0.27 / 0.45 at α = 0.2 / 0.3); Guo et al. find the attribution of observationally selected directions is a
median 2.8–4.6% of the probe coefficient (a ratio of magnitudes, not measured interventional effects).

**Caveat to state:** recall is measured among latents we intervened on (250 attribution candidates + every pick), not
all 16k. Precision ("0% of picks matter") has no such caveat.

**Pool check (Gemma-2, run 31 Oct on Sami's machine):** of the 13 latents that matter, 8 are inside the 2,048-latent
pool knockoffs choose from (SST-2 1/3, AG News 2/4, Bias in Bios 1/1, EuroParl 4/5). Knockoffs pick none of those 8 at
q = 0.1, so the miss is not caused by the pool. The other 5 on Gemma-2 fire too rarely to enter the pool, a second,
independent limit of observational selection on the most frequently firing latents; attribution screening sees all
16k. **Gemma-3 (team's run): 17 of 20 in the pool; both models: 25 of 33.**

---

## 6. Interventional FDR control (§6) — the method

### 6.1 The method in plain English

1. **Decide what "matters" means.** Track the model's probability of the correct answer. A latent matters if changing
   it lowers that probability by more than δ on average. δ is fixed from a separate pilot split (300 inputs) before any
   testing.
2. **Shortlist cheaply.** Attribution patching (one gradient pass) on a separate screening split (1,000 inputs) ranks
   all 16k latents; keep the top K = 50. Separate data means the shortlist cannot bias the test.
3. **Intervene.** On fresh test inputs, replace the latent's value with one resampled from a fitted model of that
   latent given its 64 most co-active latents, only on the query tokens, and record the drop in correct-class
   probability. Zero and mean ablation are sensitivity checks.
4. **Bet.** For each latent, a gambler starts with 1 coin and, input by input, bets a fraction of it that the drop
   exceeds δ (aGRAPA, capped so one input can never wipe out more than 90%). If the latent does not matter, the game
   is fair or losing and the money does not grow in expectation; if it matters, it grows. The final wealth is the e-value.
5. **Select.** e-BH over the 50 e-values: at most α of the selected latents are false, under any dependence.
6. **Verify.** Measure every candidate's true effect on held-out inputs; 200 re-splits; a stress test where every
   latent sits exactly at δ so every selection is wrong.

Assumptions: inputs drawn at random; screening split independent of the test split. Nothing about how latents are
distributed.

### 6.2 Results (resampling, α = 0.1)

| Model | Task | Acc. | δ | τ > 2 SE (of 250) | τ > δ | Max FDR | Stress | Power n = 2,000 |
|---|---|---|---|---|---|---|---|---|
| G-2 | SST-2 | .95 | .008 | 195 | 3 | .000 | .000 | .99 |
| G-2 | AG News | .84 | .011 | 227 | 4 | .000 | .005 | .75 |
| G-2 | Bias in Bios | .85 | .015 | 216 | 1 | .000 | .000 | 1.00 |
| G-2 | EuroParl | .95 | .010 | 196 | 5 | .000 | .010 | .62 |
| G-3 | SST-2 | .87 | .008 | 147 | 4 | .000 | .005 | .51 |
| G-3 | AG News | .77 | .009 | 210 | 8 | .000 | .000 | .61 |
| G-3 | Bias in Bios | .83 | .016 | 172 | 3 | .003 | .000 | .34 |
| G-3 | EuroParl | .82 | .009 | 196 | 9 | .001 | .005 | .66 |

- **Validity:** realised FDR ≤ 0.003 everywhere (both K, three interventions, δ grid, n = 100–2,000, all α); stress
  test at most 0.010.
- **Power:** 0.34–0.66 on Gemma-3, the held-out model; weak at n ≤ 250. K = 50 / cap 0.9 were chosen after seeing Gemma-2's
  results (keep this disclosure; Gemma-2's 0.62–1.00 is on tuned settings) and applied unchanged to Gemma-3, where they beat K = 250 / cap 0.5 on 4/4 tasks at n = 1,000 and 3/4 at n = 2,000.
- **Few latents matter individually:** 59–91% of candidates have a reliably positive effect, but only 1–9 (up to 12
  under ablation) exceed δ; all of those are in the attribution top 50.

### 6.3 Comparison with the alternatives (all scored against the same oracle)

| Method | What it asks | Result |
|---|---|---|
| Attribution top-k (Marks et al.) | "rank by gradient, keep k" | power ≈ 1 but FDR 0.10–0.90 (k = 10), 0.64–0.96 (k = 25), 0.82–0.98 (k = 50); no way to choose k |
| Zero-effect null (ours with δ = 0) | "any effect > 0" | FDR vs τ > δ grows with n: 0.60–0.96 at n = 2,000 |
| **Khera (relative to a random candidate)** | "hurts more than a random latent" | agrees with ours at n ≤ 2,000; task-specific drift at n = 4,000 (6.4) |
| Gaussian / hurdle knockoffs | "predicts the label" | 0% of picks matter; chance-level at q = 0.2 |
| Observational e-values | "predicts the label" | 70–100% of picks below δ, except G-2 Bias in Bios |
| **Ours** | "moves the answer by more than δ" | FDR ≤ 0.003, power 0.34–0.66 on the held-out model |

### 6.4 Comparison with Khera (`scripts/stage5_khera_baseline.py`, `results/stage5_khera_baseline.json`)

**Corrected 31 Oct.** The first version tested 50 latents but drew the comparator from 250, which is not Khera's design
(Appendix A: uniform over the tested set minus j). Its numbers (FDR 0.21–0.76, "6–23× below δ", "certifies negligible
latents") are withdrawn; that setup survives only as a labelled sensitivity column.

This is a **reimplementation on our data, not a reproduction**: same comparator rule, cap 0.5/B and α = 0.05, but our
metric (correct-class probability, not a logit difference), our attribution candidates (not features active on ≥ 20% of
prompts). 200 re-splits. Limits: re-splits share one finite pool, and n = 4,000 leaves only 1,000 inputs for the oracle.

**Zero ablation (Khera's intervention), n = 1,000, same K and cap for both:** no false picks for either in any setup.
With K = 50, comparator from the 50, ours has higher power on 6 tasks, ties on 1, and is lower on 1 (Khera 0.16–0.73,
ours 0.32–0.74). The pool is 2,000 inputs, so larger n cannot be tested under zero ablation.

**Resampling (not Khera's intervention), growing n.** Khera FDR against τ > δ, α = 0.05 (mean picks):

| Model | Task | Test 50/pool 50: n = 2,000 | n = 3,000 | n = 4,000 | Test 250/pool 250: n = 4,000 | Sensitivity, test 50/pool 250: n = 4,000 | Ours n = 4,000 FDR (picks) / power Khera vs ours |
|---|---|---|---|---|---|---|---|
| G-2 | SST-2 | .00 | .00 | **.00** (3.0) | .00 (3.0) | .37 | .00 (3.0) / .96 vs .96 |
| G-2 | AG News | .00 | .00 | **.17** (4.9) | .35 (6.1) | .43 | .00 (3.0) / 1.00 vs .75 |
| G-2 | Bias in Bios | .00 | .00 | **.00** (1.0) | .00 (1.0) | .75 | .00 (1.0) / 1.00 vs 1.00 |
| G-2 | EuroParl | .00 | .00 | **.01** (4.0) | .27 (6.9) | .44 | .00 (4.0) / .81 vs .81 |
| G-3 | SST-2 | .00 | .00 | **.00** (4.0) | .00 (4.0) | .20 | .00 (4.0) / .99 vs .98 |
| G-3 | AG News | .00 | .00 | **.00** (5.0) | .08 (9.0) | .28 | .00 (5.4) / .60 vs .66 |
| G-3 | Bias in Bios | .01 | .28 | **.35** (4.0) | .35 (4.0) | .56 | .00 (1.0) / 1.00 vs .44 |
| G-3 | EuroParl | .00 | .00 | **.00** (6.0) | .19 (11.0) | .30 | .00 (6.0) / .69 vs .69 |

How to read it:
- **Both are valid for their own target:** Khera's FDR against "beats the comparator" is 0.00 in every cell.
- **At n ≤ 2,000 the two agree:** Khera's FDR is ≤ .01 in the test-50 setup.
- **Drift is real but task-specific.** At n = 4,000 in the test-50 setup, 2 of 8 tasks exceed α (G-2 AG News .17, G-3 Bias
  in Bios .35); 6 are ≤ .01. In the test-250 setup, 5 of 8 are above .05 (.08–.35).
- **Those false picks are near δ, not negligible:** in the top 50, the first latent below δ sits at 0.86 δ on G-2 AG News
  and 0.79 δ on G-3 Bias in Bios. Do not call them negligible.
- **At large n Khera has more power:** 1.00 vs 0.44 on G-3 Bias in Bios, for example, because that task's true latents sit
  at about 1.07 δ. δ is a chosen cut-off, and ours is conservative near it.
- **Khera's bar depends on the candidate set:** the comparator's mean effect is 1.2–5.7× below δ for the top 50, but
  6–23× below for the 250. The same latent can pass or fail depending on who else was screened. The FDR differences across
  the columns show this. Ours does not depend on the set, since δ is absolute.

**Paragraph for the paper (test-250 is the reviewer's suggested setup; test-50 matches our K; team to choose, see §10):**
"We reran Khera's relative null on our candidates and metric. Both procedures are valid for their own targets and agree
at n ≤ 2,000. Khera's target depends on the candidate set, because its comparator is drawn from it: a random latent's mean
effect is 1.2–5.7× below δ among the top 50 and 6–23× below among 250. At n = 4,000 under resampling, its FDR against the
minimum-effect truth is at most .01 on six of eight tasks with K = 50 and reaches .17 and .35 on two, where it selects
latents just below δ; with K = 250 it reaches .35. Ours stays at or below .002 at α = .05, with lower power near δ."

---

## 7. What is new, and what is not (be explicit in the paper)

| | New? |
|---|---|
| Audit of knockoff exchangeability on SAE latents, three models | Yes |
| Planted-signal FDR measurement, repairs (hurdle, deep, binarised), "per-latent diagnostics mislead" | Yes |
| Knockoff picks scored against measured causal effects (chance level, any sampler) | Yes |
| Interventional e-values + e-BH on SAE latents | **No** — concurrent with Khera (2026) |
| Minimum-effect null with evidence on a zero null (certifies negligible latents) and a relative null (target depends on the candidate set) | Yes for the evidence; minimum-effect testing itself is classical |
| Realised FDR checked against measured effects on real 1–2B models + stress test + held-out model | Yes |
| "Only 1–9 of 250 latents matter individually" | Yes |

Wording: "We adapt interventional e-BH (concurrent with Khera, 2026) with a minimum-effect null and resampling,
and verify its realised FDR against measured effects on two real models."

**Why interventions instead of steering (one sentence in §6, one in Limitations):** on classification prompts every
latent's effect on the answer can be measured directly, which gives a ground truth for every candidate; steering has no
such ground truth.

---

## 8. Discussion and limitations (paper text guidance)

Discussion:
- Observational selection fails twice: its guarantee rests on a joint distributional assumption we could not satisfy,
  and its target is association. Interventional e-BH needs neither.
- The minimum effect is the central design choice: "most latents have some effect" and "1–9 have a meaningful one" are
  both true, and a zero null erases the difference as data grows.
- Few latents matter individually: computation is spread across many latents; single-latent editing usually does little.

Limitations (results-free, before references):
- Models of 1–2B parameters, one SAE each; four tasks, one prompt format.
- Single-latent tests miss jointly redundant latents; power is relative to the screened candidates; weak power at small n.
- K and the bet cap were tuned on Gemma-2 (Gemma-3 is the held-out confirmation).
- Recall of knockoffs is measured among intervened latents only.
- Khera's procedure reimplemented on our candidates and metric; the growing-n comparison uses resampling, not zero ablation,
  and n = 4,000 leaves only 1,000 oracle inputs.
- Steering, Gemma-2-9B and a Gamma hurdle size model were planned and not run.

---

## 9. Viva questions and answers

- **"Why did you switch to interventions?"** It is Stage 5 of the proposal: we checked whether knockoff survivors
  matter to the model, found they do not, and built a test for the question that matters.
- **"Did the hurdle repair work?"** It fixed what it targeted (zeros, single-latent exchangeability, power, one Gemma-3
  breach) but not joint exchangeability, and it breaks on ReLU SAEs. Its picks are no better than Gaussian ones causally.
- **"Comparing knockoffs to an interventional oracle is unfair."** Yes, different targets; it is the downstream check we
  promised, not a contest, and we say so ("estimand mismatch, not a breach").
- **"Isn't your method just Khera?"** Same machinery, different target. Khera's is relative, so it depends on the
  candidate set; ours is an absolute minimum effect. On our data they agree at n ≤ 2,000, and at n = 4,000 Khera drifts on
  some tasks, picking latents just below δ. We also check FDR against measured effects on real models.
- **"Your power is lower than Khera's."** At large n near δ, yes (e.g. 0.44 vs 1.00 on Gemma-3 Bias in Bios). That is
  the cost of a hard cut-off, and δ is a chosen value.
- **"Why should I care that only 1–9 latents matter?"** Because single-latent claims and edits dominate SAE
  interpretability; most latents' individual effects are too small to support them.

---

## 10. Paper edits and open items

Structure:
1. Title leads with the audit, e.g. "Lying Latents: Auditing Knockoff-Based Discovery of Sparse Autoencoder Features".
2. Order: Intro → Background → Setup → Audit (§4) → **Downstream check (§5, new position)** → Method (§6) → Results →
   Discussion → Limitations.
3. Add attribution top-k and Khera to the baselines table; add the Khera paragraph and a small table (6.4,
   n = 1,000 / 2,000 / 4,000), or a figure: FDR against "matters" vs n for ours, Khera and the zero null.
4. Cite Guo et al. (arXiv 2610.15850) as concurrent support for §5; use Khera's "knockoffs found none" in §4.

Decisions for the team:
- Which Khera setup goes in the paper (test 50/pool 50 or test 250/pool 250; both now at 200 re-splits).
- Whether to restructure, or keep the current structure and add a short downstream-check paragraph and a Khera paragraph.

Text fixes:
- Words like "pre-registered" may go, but keep the disclosure that K and the cap were chosen after seeing Gemma-2's results.
- The 73% vs 47% null-coin number is fixed-penalty, placed beside CV results and described as block MVR; drop or relabel it.
- Fill the Khera bib entry (ICML 2026 Workshop on Hypothesis Testing, Seoul). Add Shi et al. (NeurIPS 2024) and Guo et al.

Open checks: none. Pool check done on both models (§5); Khera's paper read (`16_Anytime_Valid_Hypothesis_Te.pdf`).
