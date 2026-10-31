# Lying Latents

**Auditing knockoff-based discovery of sparse autoencoder features.**

Final project for Advanced NLP, IIIT Hyderabad, Monsoon 2026.
Team SujayFans: Vishal Rao, Ayush Daga, Krithik Kambhampati, Mohammed Sami.

The report is [`final_report.tex`](final_report.tex).

## The idea in one minute

Sparse autoencoders (SAEs) break a language model's activations into thousands of *latents*. Papers often say
something like "latent 1234 encodes sentiment and the model uses it". That is a statistical claim, and it rarely
comes with an error rate.

We asked two questions:

1. **Do the current selection methods keep their promises?** We audited the common search-then-validate recipe and
   Model-X knockoffs, which promise a controlled false discovery rate (FDR).
2. **Do the latents they pick matter to the model?** We intervened on each latent inside the model and measured how
   much the answer changed.

Then we turned that measurement into a test: **interventional FDR control**. We change one latent, measure the drop
in the probability of the correct answer, and test whether the average drop exceeds a minimum effect δ, using betting
e-values and e-BH. The FDR guarantee needs no assumption about how latents are distributed. It does need randomly
drawn inputs and a screening split independent of the test split.

## Main results

| Question | Result |
|---|---|
| Does reusing data certify noise? | Yes: up to 100% of pure-noise experiments with a linear probe on pooled latents |
| Are knockoffs exchangeable with SAE latents? | No, on all 9 datasets. A typical SST-2 latent is exactly zero 89% of the time; Gaussian knockoffs never are |
| Does that cause FDR breaches? | In some regimes, mainly when labels depend on interactions between latents. The excess is real but modest (about 0.01–0.03) |
| Do better samplers fix it? | Partly. Hurdle knockoffs are close to chance one latent at a time, but are still detected jointly and breach on Pythia's ReLU SAE |
| Do knockoff picks matter to the model? | On Gemma-2 SST-2, none of the 34 picks at q = 0.1 is above δ. Other tasks select nothing at q/C per class, largely because of the Knockoff+ selection floor |
| Does interventional e-BH work? | Realised FDR ≤ 0.003 up to 2,000 inputs, on 2 models and 4 tasks. Power on the held-out model is 0.34–0.66 at 2,000 inputs |
| How many latents matter? | 59–91% of the 250 screened latents have a reliable effect, but only 1–9 per task exceed δ (about one percentage point) |

These results come with limits; see "Limitations and Future Work" in the report. In short: models of 1–2B parameters,
one SAE per model, four tasks, single-latent tests, and planted labels in the audit. `K` and the bet cap were tuned on
Gemma-2, so only Gemma-3's power is held out.

## How the project maps to the code

Work went in stages. The name "stage" appears throughout the code and results.

| Stage | Question | Main code | Findings |
|---|---|---|---|
| 1 | Does the standard pipeline certify noise? | `src/calibrate_pipeline.py`, `src/calibrate_validation.py`, `src/saebench/stage1.py` | `stage1_findings.md`, `stage1_sae_findings.md` |
| 2 | Are knockoffs exchangeable with the latents? | `src/knockoff_audit.py`, `src/saebench/stage2.py` | `stage2_findings.md`, `stage2_sae_findings.md` |
| 3 | Is the FDR exceeded on planted labels? | `src/planted_fdr_controls.py`, `src/stage3_followups.py` | `stage3_findings.md` |
| 4 | Can better samplers repair it? | `src/stage4_repairs.py`, `src/deep_knockoffs.py` | `stage4_findings.md`, `stage34_sae_findings.md`, `stage34_matched_sae_findings.md`, `models_sae_findings.md`, `evalue_sae_findings.md`, `evalue_variants_findings.md`, `deep_knockoffs_sae_findings.md` |
| 5 | Do picks matter, and can we test that directly? | `src/stage5_interventional.py`, `scripts/stage5_khera_baseline.py` | `stage5_findings.md` |

All findings are in [`docs/findings/`](docs/findings/). Each one starts with a summary and states what the stage does
and does not establish.

**Where the report's Stage 5 sections come from:**

| Report section | Run |
|---|---|
| §5 Do knockoff picks matter? | `baselines2`, `baselines3`, `delta2`, `delta3`, `evaluate3` (stage5 amendments 2 and 3) |
| §6–7 Interventional FDR control | `pilot`, `screen`, `delta`, `evaluate2` (stage5 amendments 1 and 2) |
| §7 Comparison with Khera (2026) | `scripts/stage5_khera_baseline.py` |

## Repository layout

```
final_report.tex        the report (ACL format; needs acl.sty and acl_natbib.bst from the ACL template)
config/
  default.yaml          settings for all experiments (Gemma-2-2B)
  models/               overlays for Gemma-3-1B and Pythia-70M; each replaces only what differs
  preregistration.yaml  decision rules and seeds, written before each run, with logged amendments
src/
  common.py             config loading, seeds, cache hashing
  cache_activations.py  text -> SAE latents -> cached matrix
  stage*, knockoff_audit.py, calibrate_*.py, planted_fdr*.py, deep_knockoffs.py   the experiments above
  stage5_figures.py     Stage 5 figures and tables
  saebench/             the same stages on the SAEBench datasets
scripts/                run scripts for long jobs, plus one-off diagnostics (diag_*)
docs/
  findings/             one write-up per stage (start here to understand a result)
  planning/             plans and the paper story; final_story.md lists superseded claims
  CHANGES.md            what changed, why, and what came out
  NOTES_reference.md    what the audited reference pipeline does
reference/              proposal, interim report and audited reference code (read-only)
results/                every output the report uses; FINAL_RESULTS.json collects the final numbers
```

Not in git: activation caches (`data/`), `screen.npz` files (rebuilt by `--stage screen`), logs and trained generators.

## Which results are current?

The project went through many versions, so not everything in `results/` is the final word.

- **Start with [`results/FINAL_RESULTS.json`](results/FINAL_RESULTS.json).** It collects the numbers of the final
  design, and each section names its source files.
- Other final files: `results/stage5_baselines3_summary.json`, `results/stage5_khera_baseline.json`, and each model's
  `stage5/<task>/evaluate2_summary.json`.
- **Superseded, kept for history:** fixed-penalty Stage 3–4 runs (replaced by the cross-validated penalty), and the
  first Stage 5 evaluation (`evaluate.json`, replaced by `evaluate2.json`). `scripts/collect_final_results.py` lists these.
- If the report and an older document disagree, the report wins.

## Setup

We used Python 3.13 and one or two GPUs with 32 GB or more.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Versions are pinned because `sae-lens` and `transformer-lens` have shipped breaking changes.

Gemma-2 and Gemma-3 are gated on Hugging Face. Accept their licences, then set `HF_TOKEN`. All models and SAEs are
public releases; this project trained and uploaded no models to Hugging Face or WandB.

| Model | SAE | Layer |
|---|---|---|
| `google/gemma-2-2b` | Gemma Scope, JumpReLU, 16k | 20 |
| `google/gemma-3-1b-pt` | Gemma Scope 2, JumpReLU, 16k | 22 |
| `EleutherAI/pythia-70m-deduped` | ReLU (`pythia-70m-deduped-res-sm`) | 4 |

## Reproducing the results

Long scripts resume if restarted. Run them from the repo root. Times are rough and depend on the GPU.

**1. SST-2 audit (Stages 1–4)**

```bash
python src/cache_activations.py   --config config/default.yaml                 # about 5 min, GPU
python src/knockoff_audit.py      --config config/default.yaml                 # about 85 min, CPU
python src/calibrate_pipeline.py  --config config/default.yaml --device cuda
python src/calibrate_validation.py --config config/default.yaml --device cuda
python src/planted_fdr_controls.py --config config/default.yaml --device cuda --stage full
python src/stage3_followups.py    --config config/default.yaml --device cuda --experiment stress --stage full
python src/stage4_repairs.py      --config config/default.yaml --device cuda --stage full   # about 18 h
```

**2. SAEBench audit**

The task texts need SAEBench's own code in a separate environment, because it requires `datasets<4`:

```bash
pip install "datasets>=3,<4" pandas tqdm transformers einops pyyaml && pip install sae-bench==0.6.0 --no-deps
python src/saebench/build_tasks.py --config config/default.yaml
```

Then, in the main environment:

```bash
scripts/models_pipeline.sh config/models/gemma3_1b.yaml all    # caches, then Stages 1-4 (also pythia70m.yaml)
scripts/saebench_stage34.sh gauss                              # Gemma-2 Stages 3-4 (also: hurdle)
scripts/run_matched_all.sh                                     # amplitude sweep with the fair-coin test
GROUP=deep2 SET=all nohup scripts/deep_knockoffs.sh > logs/deep2_all.log 2>&1 &      # deep knockoffs
```

**3. Interventional FDR control (Stage 5)**

```bash
mkdir -p logs
GPU_A=0 GPU_B=1 nohup scripts/stage5.sh > logs/stage5.log 2>&1 &   # main run, 2-9 h per Gemma-2 task
nohup scripts/stage5.sh v2 > logs/stage5_v2.log 2>&1 &             # final evaluation and fairer baselines
nohup scripts/stage5.sh v3 > logs/stage5_v3.log 2>&1 &             # sampler comparison and exchangeability tests
scripts/stage5.sh status                                           # progress
python scripts/stage5_khera_baseline.py                            # relative null, CPU
```

`delta2` and `delta3` need `screen.npz`, so run them on the machine that ran `screen`. The observational baselines
need the activation caches.

**4. Figures and the results file**

```bash
python src/stage5_figures.py               # power, zero-null and effect figures, plus LaTeX tables
python scripts/generate_pipeline.py        # pipeline figure
python scripts/collect_final_results.py    # results/FINAL_RESULTS.json
```

## Reproducibility

- **Seeds:** every random stream comes from `master_seed` in the config (`STREAMS` in `src/common.py`).
- **Caches:** file names hash the settings that determine their contents, so a cache built with other settings cannot
  be read by accident.
- **Decisions:** rules were fixed in `config/preregistration.yaml` before runs. Changes made after seeing results are
  logged as amendments and labelled in the findings. For example, the Stage 5 `K` and bet cap were chosen after seeing
  Gemma-2 and then held fixed for Gemma-3.
