# Final submission report: checklist

Review of `ANLP_FinalSubmission.pdf` (9 pages, 30 Oct 2026). Numbers checked against `docs/findings/` and the result files.

## A. Must fix (incomplete or wrong)

- [x] **Red TBD under Table 3 (hurdle rows):** fill these from the exchange tests and delete the box.
  - Gemma-3 hurdle and Gaussian exchange tests are done (`results/models/gemma3_1b/saebench/stage34/*_exchange.json`).
  - Gemma-2 and Pythia files are in `results/saebench/stage34/` and `results/models/pythia70m/saebench/`.
- [x] **Red TBD in §6 (better knockoff samplers vs. the oracle):** needs Gemma-3 baselines3.
  - Gemma-3 knockoffs do select latents: Gaussian, fixed penalty, picked 30–63 on EuroParl at q = 0.2, and hurdle, CV penalty, picked 20 in one AG News draw.
  - So "select nothing" doesn't carry over, and these picks need oracle scoring before they go in the text.
- [x] **"1–12 of 250 latents have a meaningful effect"** (abstract, §6, Discussion): Table 6's τ > δ column is 1–**9**. Fix the number or give its source.
- [x] **Table 10 caption** says "FDR ≤ 0.002 for every intervention", but Table 6 has Max FDR **0.003** (Gemma-3 Bias in Bios, mean ablation, n = 1,000).
- [x] **§6, "zero and mean ablation give similar power":** this is false on Gemma-3.
  - SST-2: resampling 0.27 vs. zero 0.77 / mean 0.76.
  - Bias in Bios: 0.21 vs. 0.67 / 0.49.
- [x] **§3, "12,500 texts each":** Amazon sentiment has 5,000 and AG News 10,000; now "5,000–12,500 texts".
- [x] **References:** the Bal (2026) and Khera (2026) entries still say "[title to be added]" and need full author names.

## B. Clarity and consistency

- [x] **Table 6 caption:** it says "resampling intervention", but Max FDR is taken over all three interventions. Say which column covers what.
- [x] **Table 4 vs. Table 8 on SST-2:** 7/54 breaches vs. 0. They use different samplers (block MVR vs. equicorrelated) and penalties (CV vs. fixed), so name both in each caption.
- [x] **Table 3's Gaussian AUC₁ (0.84–0.95) vs. Table 5's block MVR (0.79–0.98):** say which Gaussian sampler Table 3 uses.
- [x] **§4.4, binarised FDR "0.17 at q = 0.1":** that is the worst single cell (amplitude 3, linear, k = 30). The mean is 0.065 (Table 8), so write "up to 0.17 in one cell".
- [x] **§4.5, "14 datasets":** make it "14 model–dataset settings" (9 distinct datasets). Also say the 0.27–0.46 power loss is against hurdle knockoffs with the CV penalty.
- [x] **Appendix D:**
  - The "breached once" case was one cell at all three q levels (FDR 0.30, 0.36, 0.45).
  - Mention the five-split version's −0.16 on SST-2.
  - The "11× nominal" figure applies to the capped version only.
- [x] **Table 7:** add a caption note that there is no Gemma-3 SST-2 row because that pooled cache doesn't exist.
- [x] **§3:** "most frequent" → "most frequently firing" latents (the code keeps the 2,048 with the highest firing rate).
- [ ] **Figure 1 label (deferred, part of the image):** "(2,048 most active latents; 89% zeros)" → "most frequently firing" when the figure is regenerated.
- [x] **Table 1:** Gemma Scope 2 needs its own citation, not Lieberum et al. (2024), which is Gemma Scope 1.
- [x] **Contribution 1 says "3 knockoff samplers"** but Table 5 has 5 rows. Align the wording, e.g. "three sampler families".

## C. Layout

- [ ] **Figures 2 and 3:** the fonts are unreadable at column width. Enlarge them or go full-width, and drop the titles inside the plots that repeat the captions.
- [ ] **Page 3:** there's a large blank gap above §4.2. Move Tables 3 and 4 to the top of the page.
- [x] **Table 9:** change "1.0→1.0" to "1.00→1.00".

## D. Team decisions

- [x] **"Observational methods find none of the latents that move the model":** this is true for knockoffs only. Observational e-values reach 1.00 power on Gemma-2 Bias in Bios. Re-check the wording after the Gemma-3 knockoff picks are scored.
- [x] **Gemma-2 settings were tuned on Gemma-2:** say explicitly that Gemma-2's Stage 5 numbers use settings tuned on Gemma-2 itself, so they're exploratory, and that Gemma-3 is the confirmation.

## E. Readability

- [x] **Abstract has too many numbers; make it more intuitive, and don't use terms it doesn't explain.**
  - Undefined terms there: "exchangeability", "swapped latent", "AUC", "minimum effect δ", "betting e-values", "e-BH", "zero-effect null".
  - It has about 12 numbers. Keep two or three headline ones, e.g. "pure noise is certified in up to 100% of null experiments", "realised FDR never above 0.003", "only a handful of 250 latents matter".
- [x] **Related work, Knockoffs: explain in words rather than mathematical symbols.**
  - Replace "Y ⊥ X_j | X_−j", the swap-equality formula and Eq. (1) with plain sentences.
  - For example: a knockoff is a fake copy of each latent that looks statistically identical but is built without the label. A real latent matters if it predicts the label better than its fake. The threshold is chosen so the expected share of false picks stays under q.
  - Move Eq. (1) to an appendix if it's needed at all.
- [x] **Related work: explain what e-values are.** One sentence on what an e-value is, one on what e-BH does with them, one on betting.
  - An e-value is a score that, when there's no effect, averages at most 1, so a large value is evidence against the null.
  - e-BH turns a set of e-values into a list of discoveries with controlled FDR.
  - Betting builds an e-value as the winnings of a gambler who bets against the null.
- [x] **Elaborate the SAE-specific work section; it makes no sense right now.**
  - Currently it's two sentences split around Table 2, citing Enkhbayar, Khera and Bal with no titles.
  - Write a short paragraph for each: what each paper did, and how ours differs.
    - Enkhbayar: Gaussian knockoffs on SAE latents; we show that assumption fails.
    - Khera: ablation e-values with a zero-effect null; we add a minimum effect and oracle checks.
    - Bal: many SAE features have no causal effect; we quantify this with error control.
- [x] **Explain the zero rule in §4.2.**
  - The rule: guess "real" whenever a value is exactly 0, because Gaussian knockoffs are never exactly 0.
  - Say why its accuracy is (1 + Pr(X = 0))/2, and that 0.5 means it can't tell real from fake.
- [x] **§5 Target should be explained more clearly.**
  - Define m(x) with an example, e.g. "the model's probability of 'positive' for a positive review".
  - Define Δⱼ(x) as how much that probability drops when we change latent j.
  - τⱼ is the average drop. δ is the smallest drop we count as meaningful, about 1 percentage point.
  - Explain why we test τⱼ > δ rather than τⱼ > 0.
- [x] **§5 Intervention uses terms like h that aren't defined; every term in an equation should be defined.**
  - Undefined: h (the residual-stream activation), aⱼ (latent j's activation), ãⱼ (its replacement value), dⱼ (decoder direction), L (the SAE layer, which appears suddenly in "layers after L"), "SAE error".
  - Also explain "query token" and "replayed".
- [x] **The same applies elsewhere, e.g. aGRAPA.** (Terms well known to NLP readers, such as AUC, lasso and z-scores, were left unexplained on purpose.) Places with undefined or unexplained terms:
  - **§5, Eq. 3:** aGRAPA, "predictable" bets, λₜ, the cap 0.9/(1+δ), "wealth", "nonnegative supermartingale".
  - **§5 procedure:** K, E₍ₖ₎ (the k-th largest e-value), k*.
  - **§5 baselines:** C in "α/C per class" (the number of classes).
  - **§2:** W_j, β̂ⱼ, "MVR", "equicorrelated", "SCIP", "Metropolized", and "KL gap".
  - **§4.2:** AUC₁ / AUC_all (only in the Table 3 caption) and MMD.
  - **§4.3:** "fair-coin test" and "null coin" (defined only in Appendix B), "amplitude", and k.
  - **§4.4:** "single-latent swaps" and "second-order-exact".
  - **§4.5:** "lasso", "logistic refit", "z-scores", and κ (Appendix D).
  - **Table 6:** "SD(m)", "pilot split", "Stress".
  - **Tables 2 and 4:** "BY correction" is expanded in §2 but never explained.
