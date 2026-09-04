# NOTES_reference.md — what the audited pipeline actually does

Source: <https://github.com/WesternDundrey/Model-X-for-SAEs>, commit `d7beffa`,
single file `modelx_knockoffs_experiment.py` (627 lines). An identical copy sits at
`enkhbayar_model_x.py` in this repo root. Read-only reference; nothing is imported from it.

Everything below is extracted from that file. Line numbers refer to it.

---

## 1. Model and SAE

| Field | Value | Line |
|---|---|---|
| `model_name` | `EleutherAI/pythia-70m-deduped` | 85 |
| `sae_release` | `pythia-70m-deduped-res-sm` | 86 |
| `sae_id` | `None` → auto-resolved | 87, 148–183 |

The SAE id is **not pinned**. `resolve_sae_id_for_release` filters the release's
`saes_map` to ids containing `hook_resid_pre`, sorts by layer index, and takes
`target[len(target) // 2]` — the median layer. The exact SAE therefore depends on the
contents of the SAELens pretrained-SAE directory at run time, which is a moving target.
There is no recorded release/ID pair to replicate against.

## 2. Token → example aggregation

**There is none. The reference takes a single token position:**

```python
acts = cache[hook_name][:, -1, :]        # line 283
```

after

```python
tokens = model.to_tokens(texts, prepend_bos=get_sae_prepend_bos(sae))   # line 271
```

with `activation_batch_size = 16` (line 90).

This is neither mean- nor max-pooling. It is "last position of the padded batch".

### 2.1 Padding is included, and it dominates

`HookedTransformer` defaults to `default_padding_side = "right"`
(`transformer_lens/HookedTransformer.py:121`). `to_tokens` pads each batch to the
longest sequence *in that batch*. Position `-1` is therefore a **pad token** for every
sentence that is not the longest of its 16.

Verified directly on their exact model (`scripts/verify_reference_padding.py`):

```
padding_side           : right
tokens shape           : (2, 15)
  row 0: last id=0 -> '<|endoftext|>'   # "a stirring film ."  -> PAD
  row 1: last id=964 -> ' .'            # the longest row      -> real final token
rows whose [-1] is PAD : 1/2
```

So for most examples the "activation vector" is the SAE encoding of the residual stream
at a padding position, not at any content token. The per-example feature vector is
largely a function of *how long the other 15 sentences in the batch happened to be*.

This is worse than the dilution failure mode anticipated in the build instructions §2:
it is not that pad tokens dilute a mean, it is that pad tokens **are** the signal for
most rows. Batch composition depends on `dataset.shuffle(seed)`, so the design matrix is
not even a deterministic function of the corpus at fixed model/SAE — it depends on batch
packing.

`prepend_bos` is taken from SAE metadata (default `True`), so a BOS token is present at
position 0; it only enters the aggregate for length-1 inputs.

## 3. High-activity latent selection

```python
energy = np.mean(np.abs(activations.matrix), axis=0)   # line 301
order  = np.argsort(-energy)[:top_k]                    # lines 302–304
```

- Statistic: **mean absolute activation** ("energy"), *not* firing rate. Under JumpReLU
  activations are non-negative, so this is just the mean activation. A latent that fires
  rarely but enormously outranks one that fires often but weakly.
- Selection rule: **top-k**, not a threshold. `top_k_features = 1024` by default (line 92).
- The paper reports p = 512; the committed default is 1024. The value actually used is
  not recoverable from the code.
- Firing rate *is* computed (`activation_rate`, line 308) but only reported, never used
  for selection.

## 4. Covariance estimation and knockoff construction

`GaussianKnockoffSampler`, lines 321–372. Hand-rolled; `knockpy` is not used.

```python
cov = (centered.T @ centered) / (n - 1)
cov += self.ridge * np.eye(p)                  # ridge = 1e-3  (line 97)
lambda_min = max(eigvals.min(), 1e-6)
s_val = min(2 * lambda_min, self.smax)         # smax = 0.99   (line 98)
S = s_val * np.eye(p)                          # equicorrelated
```

- **No standardisation, no normalisation, no shrinkage** before the covariance is formed.
  `X` enters on its raw activation scale. The docstring says "simple shrinkage"; the only
  regularisation present is the `1e-3` ridge, which is a numerical guard, not shrinkage.
- **S is a single scalar times the identity**, capped at `smax = 0.99` in *absolute*
  units. This is the confound flagged in build instructions §2, and it bites in both
  directions:
  - If activation variances are ≫ 1 (they are — SAE latents are heavy-tailed and
    unnormalised), then `S = 0.99·I` is negligible relative to `Σ`. The knockoff
    transform `I − Σ⁻¹S` is then ≈ `I`, so `X̃ ≈ X`: knockoffs are near-copies of the
    originals, and power collapses.
  - If `Σ` is near-singular (likely: sparse, rarely co-firing latents), `2·λ_min ≈ 0`,
    `S ≈ 0`, and `X̃ ≈ X` exactly.
  Either way knockoff quality is confounded with raw activation energy. **Report this
  separately from the zero-atom argument** — it is an independent defect.
- `n ≤ p` raises (line 335). With the committed defaults (`max_samples = 512`,
  `top_k_features = 1024`, `split = "validation"`) the script **cannot run as shipped**.
  SST-2 validation is 872 rows, so the reported p = 512 implies n = 872, p = 512:
  **n/p ≈ 1.7**. A 512×512 covariance estimated from 872 samples is severely
  under-determined; the sample covariance is nearly rank-deficient at that ratio.

## 5. Knockoff statistic

Lines 375–411. L1-logistic regression on `[X ‖ X̃]`, `W_j = |β_j| − |β̃_j|`, knockoff+
threshold (lines 414–426, standard form with `offset = 1`).

One further issue: `StandardScaler` is fitted on the concatenated design **after**
knockoff construction (lines 387–388). Real and knockoff columns have different empirical
SDs, so each is divided by a *different* constant. That rescaling is not a swap-equivariant
operation on `(X, X̃)`, so even a perfectly exchangeable pair would be made
non-exchangeable by this step before `W` is computed.

## 6. Robustness citation

`Barber, Candès & Samworth (2020)` is **not cited anywhere** in the code or repository.
Grep for `barber|samworth|robust` returns exactly one hit: the word "shrinkage" in the
`GaussianKnockoffSampler` docstring. There is no comment, docstring, or README
acknowledging that the Gaussian surrogate is an approximation, and no reference to the
KL-divergence robustness bound that would be needed to justify it.

There is no README in the repository at all.

---

## 7. Consequences for our configuration

Build instructions §3 say to match the reference wherever a choice is otherwise
arbitrary, and to make max-pooling primary *if the reference uses max*. It does not — it
uses last-token-of-padded-batch, which is not a defensible aggregator and which we are
not going to reproduce as our primary.

Decisions taken, recorded here and in `config/preregistration.yaml`:

1. **Primary aggregator: mean over non-special tokens**, as specified. Unchanged, because
   the escape clause ("unless the reference uses max") is not triggered.
2. **Max-pooling cached alongside**, as specified.
3. **Last-real-token cached as a third aggregator.** Not in the build instructions, added
   because it is the closest *defensible* analogue of what the reference does (their
   position `-1` with padding removed). It costs one extra column-slice at cache time and
   lets us report whether the atom argument survives under their intended aggregator,
   separately from their padding bug. Their literal `[:, -1, :]` is not reproduced.
4. **Latent filter: firing rate ≥ 1%, then top 2048 by firing rate**, as specified — not
   their top-k-by-energy. Their statistic is recorded per latent so the two selections can
   be compared.
5. **Covariance: report both the raw estimate (their choice) and Ledoit–Wolf**, on
   standardised columns, with the standardisation flag recorded. Their unstandardised
   scalar-`s` construction is a confound we must not silently inherit.
6. **n/p:** we use the full 67,349-row SST-2 train split with p ≤ 2048, so n/p ≥ 33
   against their ≈ 1.7. Covariance estimation is not the bottleneck for us; it plausibly
   is for them.
