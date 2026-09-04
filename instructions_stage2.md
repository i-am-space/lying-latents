# Build Instructions — Stage 0 and Stage 2 Only

Context document for an implementation session. Read this in full before writing code.
The attached proposal PDF gives the research framing; this document defines the scope,
the decisions that are already made, and the acceptance criteria.

---

## 0. Scope

Build **only** the activation cache, the distributional summary, and the knockoff
exchangeability audit. This is a go/no-go gate: it determines whether the paper's
central empirical claim holds before any further work is committed.

**In scope**
1. Activation caching pipeline (SST-2 → Gemma Scope SAE latents → on-disk matrix)
2. Distributional summary of cached latents
3. Gaussian second-order knockoff generation
4. Exchangeability diagnostics with a validated null harness
5. A short results writeup with figures

**Explicitly out of scope — do not build, do not stub, do not scaffold**
- Shuffled-label permutation null (Stage 1)
- Planted-signal FDR study (Stage 3)
- Hurdle / binarised / e-value procedures (Stage 4)
- Steering evaluation (Stage 5)
- Any Streamlit/dashboard/CLI surface beyond plain scripts

If a design choice would only matter for an out-of-scope stage, pick the simple option
and move on. Do not generalise in advance.

---

## 1. Environment

Single NVIDIA L40S (Ada, sm_89), CUDA 12.x. Gemma-2-2b is already downloaded and
HF/WandB auth is done.

Pin exact versions in `requirements.txt` and commit the lockfile. `sae_lens` and
`transformer_lens` have both shipped breaking API changes; an unpinned environment
that silently changes SAE loading behaviour mid-project is the worst failure mode here.

```
torch, transformers, datasets, accelerate
sae_lens, transformer_lens
knockpy
numpy, scipy, scikit-learn, statsmodels, pandas
matplotlib, pyyaml, tqdm
```

**Read the `knockpy` documentation before implementing any knockoff sampling.** It
provides second-order Gaussian knockoffs (`knockpy.knockoffs.GaussianSampler`) and
`knockpy.metro` for Metropolized sampling. Do not hand-roll a Gaussian knockoff
sampler — use the library, and verify against its own tests.

---

## 2. Reference material

Clone as read-only reference, do not import from it:

```
git clone https://github.com/WesternDundrey/Model-X-for-SAEs vendor/enkhbayar_reference
```

This is the paper under audit. Our configuration should match theirs wherever a choice
is otherwise arbitrary, because Stage 2 is an audit *of that specific pipeline*, not of
knockoffs in general. Before finalising the config, extract from their code and record
in `NOTES_reference.md`:

- Which Pythia SAE they used (exact repo/release ID)
- How they aggregated token-level activations to a per-example vector
- How they selected the 512 "high-activity" latents (threshold, on what statistic)
- How they estimated the covariance matrix (shrinkage? regularisation? normalisation?)
- Whether padding and BOS tokens were excluded from aggregation
- Whether they cite Barber, Candès & Samworth (2020) for robustness

Two known issues to check for specifically, as they affect what we replicate:
- **Padding handling.** If pad tokens are included in a mean aggregation, every vector
  is diluted toward zero by a factor of the padding ratio, which inflates the apparent
  zero-mass and varies with sentence length.
- **Covariance scaling.** A scalar `s` applied to an unnormalised covariance produces
  near-useless knockoffs whose quality is confounded with raw activation energy. If
  this is what their code does, note it — it is a separate finding from the atom
  argument and should be reported separately, not merged into it.

---

## 3. Decisions already made — do not revisit

These are fixed. Implement them as specified.

**Unit of analysis: one row per SST-2 sentence.** Model-X knockoffs assume i.i.d. rows.
Tokens within a sentence are not independent, so per-token rows would invalidate the
framework before the analysis starts. Aggregate token-level latent activations to a
single vector per sentence.

**Aggregator: mean over non-special tokens, with max-pooling computed alongside.**
Exclude padding, BOS, and EOS from the aggregation. The aggregator materially changes
$\Pr(X_j = 0)$ — max-pooling over ~25 tokens drives the zero-mass down sharply versus
mean-pooling — and the zero-atom is the paper's central argument, so both must be
recorded. Mean is primary unless the reference code uses max, in which case switch the
primary to match and say so in the config.

**Dataset: SST-2 train split**, full (~67k sentences). No subsampling unless memory
forces it. n ≫ p is important for covariance estimation.

**Model/SAE: Gemma-2-2b, residual stream, layer 20, width 16k, canonical L0.** Pin the
exact SAELens release string and SAE ID in the config; verify they resolve before the
long run. Pythia-70M replication is deferred — do not build it now, but keep the
loading path parameterised by model name so it costs nothing later.

**High-activity filter: retain latents with firing rate ≥ 1% across the corpus**, then
take the top 2048 by firing rate if more survive. Record the resulting $p$ and the
firing-rate distribution of the retained set.

**Precision: float32 on disk.** ~67k × 2048 × 4B ≈ 550 MB. Fits in memory; everything
downstream is numpy.

---

## 4. Pre-registration

Before running any experiment, write `config/preregistration.yaml` containing every
threshold, seed, and decision rule in §7, and **commit it as its own commit** so the
git timestamp precedes the first results commit.

This is not ceremony. The paper's thesis is that the field selects on its own results;
being caught doing the same thing would be fatal to it. The commit history is the
evidence that we did not.

---

## 5. Deliverable 1 — Activation cache

`src/cache_activations.py`

- Load Gemma-2-2b and the pinned Gemma Scope SAE
- Stream SST-2 train through in batches, hook the residual stream at the target layer,
  encode with the SAE
- Aggregate to one vector per sentence (mean and max, excluding special tokens)
- Apply the high-activity filter, retaining latent indices
- Write to `data/cache/{config_hash}.npz` containing: the latent matrix, the retained
  latent indices, the labels, the sentence IDs, and the full config dict

The config hash in the filename is mandatory. Silently reading a cache built under a
different aggregator or layer is the single most likely way to waste a week.

Estimated runtime: under an hour on one L40S. This is the only GPU-bound step in scope.

---

## 6. Deliverable 2 — Distributional summary

`src/describe_latents.py`

**Run this immediately after caching, before anything else.** It is ~50 lines and it
either confirms or destroys the paper's central premise. Per retained latent, report:

- $\Pr(X_j = 0)$ — exact zeros, not near-zeros
- Firing rate, mean and median of the nonzero part
- Skewness and kurtosis of the nonzero part
- Upper tail quantiles (0.9, 0.99, 0.999) and max
- Correlation structure summary: eigenvalue spectrum of the sample covariance,
  condition number, fraction of variance in the top 10 components

Figures: histogram of $\Pr(X_j = 0)$ across latents; for 6 representative latents (a
spread of firing rates), the marginal distribution with the zero-mass shown explicitly
as a spike alongside the continuous part.

**Premise check.** The argument requires substantial mass at exactly zero. If the median
$\Pr(X_j = 0)$ over retained latents is above 0.8 under the primary aggregator, the
premise holds and the project proceeds. If it is below 0.5, stop and report — the atom
argument would need rewriting around the aggregator, and it is far better to learn this
on day one. Report the number either way, for both mean and max aggregation.

Also flag here whether the covariance is near-singular. Sparse non-negative data with
many rarely-co-firing latents often produces an ill-conditioned covariance, which
affects knockoff construction independently of the atom issue and needs shrinkage
(Ledoit-Wolf) if so.

---

## 7. Deliverable 3 — Exchangeability audit

`src/knockoff_audit.py`

### 7.1 Knockoff generation

Generate second-order Gaussian knockoffs $\tilde{X}$ from the cached matrix using
`knockpy`, with the covariance estimated as the reference code does (and, separately,
with Ledoit-Wolf shrinkage if the raw estimate is ill-conditioned — report both).
Standardise columns before construction and record whether you did, since this is
exactly the confound noted in §2.

### 7.2 Validate the harness before trusting it

**This step is not optional and comes before the real diagnostics.** Every test below
is designed to detect a violation of exchangeability. A buggy test detects violations
everywhere, including where none exist, which would produce a result that looks like a
finding and is not.

Run the entire diagnostic suite on synthetic data drawn from a multivariate Gaussian
with the *same* covariance as the real latents. There, second-order knockoffs are
exactly valid and exchangeability genuinely holds. Every diagnostic must come back null:

- Classifier two-sample AUC within noise of 0.50
- MMD test p-value uniform under repeated draws
- Moment comparisons showing no systematic gap

If any diagnostic fires on Gaussian data, it is broken. Fix it before running on real
latents. Record the synthetic-null results in the writeup as a calibration figure —
this is what makes the real result credible rather than assertable.

### 7.3 Diagnostics

Exchangeability requires that for any subset $S$ of nulls, swapping
$X_j \leftrightarrow \tilde{X}_j$ for $j \in S$ leaves the joint law of
$(X, \tilde{X})$ unchanged. Test three necessary consequences:

**(a) Zero-mass discrepancy.** Per latent, the accuracy of the trivial classifier
$\mathbb{1}\{x = 0\}$ in separating real from knockoff. Under exchangeability this must
be 0.5; here it will be near $\Pr(X_j = 0)$. Report the distribution across latents.

Note in the writeup that this is deliberately the simplest possible falsification —
its force is that it requires no test at all, only the definition. Do not oversell it
as a sophisticated finding; its value is that it is unarguable.

**(b) Swap two-sample tests.** For a range of swap-subset sizes $|S|$ (e.g. 1, 10, 50,
all), construct the swapped and unswapped samples, then:
- Classifier two-sample test: train a gradient-boosted classifier to distinguish them,
  report held-out AUC with a permutation-derived null band
- MMD test with a Gaussian kernel, bandwidth by the median heuristic, p-value by
  permutation

Report AUC as a function of $|S|$.

**(c) Higher moments and tails.** Second-order construction matches moments 1 and 2 by
design, so compare skewness, kurtosis, and upper tail quantiles between real and
knockoff columns. Verify moments 1 and 2 *do* match — if they don't, the construction is
misconfigured and nothing else is interpretable.

### 7.4 Interpretation constraints

Write these into the results file so they survive into the paper:

- The null set is unknown on real data, so exchangeability is only *required* on nulls.
  These are therefore necessary-condition tests: sufficient to falsify, not to confirm.
  State this explicitly; do not claim to have "confirmed the violation on nulls".
- A violation of exchangeability voids the guarantee. It does not demonstrate that
  realised FDR exceeds the target — that is Stage 3, and it is not in scope here. Do
  not write conclusions that outrun the evidence; the distinction between a void
  guarantee and a demonstrated failure is the one the whole paper rests on.

---

## 8. Determinism

One seed policy, threaded everywhere: a single master seed in the config, from which
per-component seeds are derived (numpy `default_rng`, torch, and the knockoff sampler
each get their own stream). Every output file records the seed and the config hash.
Any run must be exactly reproducible from the config alone.

---

## 9. Repository layout

```
config/          preregistration.yaml, model/sae config
src/             cache_activations.py, describe_latents.py, knockoff_audit.py
data/cache/      gitignored
results/         figures + json/csv summaries, committed
vendor/          reference clone, gitignored
NOTES_reference.md
requirements.txt
README.md
```

Commit results (figures, summary tables) but never the activation cache.

---

## 10. Acceptance criteria

The session is done when:

1. `python src/cache_activations.py --config config/default.yaml` produces a cache
   reproducibly from a clean checkout
2. The distributional summary reports median $\Pr(X_j = 0)$ for both aggregators, with
   figures, and the premise check has an explicit verdict
3. The diagnostic suite returns null on synthetic Gaussian data (calibration figure)
4. The diagnostic suite runs on real latents with results recorded as JSON plus figures
5. `results/stage2_findings.md` states, in plain prose: whether exchangeability is
   violated, by how much, on what evidence, and what that does and does not establish

Nothing beyond this. The next stage is decided by these numbers, not by anticipating
them.