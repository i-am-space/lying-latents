#!/usr/bin/env python3
"""
Model-X Knockoffs for SAE Features
==================================

This module implements a reproducible pipeline for declaring SAE latents
"real" while controlling the false discovery rate (FDR) via the Model-X
knockoff framework:

1. Load a causal LM and pretrained sparse autoencoder (SAE) using SAELens.
2. Collect SAE activations for a labeled text dataset (default: GLUE SST-2).
3. Score and retain the most active latents to keep covariance estimation
   tractable.
4. Fit equi-correlated Gaussian knockoffs that match the latent covariance
   but break causal links to the labels.
5. Train an L1-regularized logistic classifier on the concatenated design
   `[X || X~]` and compute knockoff statistics `W_j = |β_j| - |β̃_j|`.
6. Use the knockoff+ threshold to select latents with provable FDR control.
7. Persist a JSON summary plus CSV/NumPy artefacts for downstream analysis.

The default configuration targets `gpt2-small` with the public
`gpt2-small-res-jb` SAE release, providing a light-weight example that can run
on CPU. Larger SAEs or alternate datasets can be plugged in through CLI flags
or YAML configs.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from difflib import get_close_matches
from datasets import Dataset, load_dataset
from scipy import linalg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import yaml

try:  # pragma: no cover - informative error for missing deps
    from sae_lens import SAE, HookedSAETransformer
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "sae-lens is required. Install with `uv pip install sae-lens transformer-lens`."
    ) from exc

try:  # pragma: no cover - optional helper for auto SAE selection
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
except Exception:  # pragma: no cover
    try:
        from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
    except Exception:
        get_pretrained_saes_directory = None


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2025
SAE_RELEASE_SUGGESTION_COUNT = 5


@dataclass
class DatasetConfig:
    """Configuration for loading labeled text data."""

    dataset_name: str = "glue"
    dataset_config: Optional[str] = "sst2"
    split: str = "validation"
    text_column: str = "sentence"
    label_column: str = "label"
    trust_remote_code: bool = False


@dataclass
class ExperimentConfig:
    """Top-level configuration for the knockoff experiment."""

    model_name: str = "EleutherAI/pythia-70m-deduped"
    sae_release: str = "pythia-70m-deduped-res-sm"
    sae_id: Optional[str] = None
    device: str = DEFAULT_DEVICE
    max_samples: int = 512
    activation_batch_size: int = 16
    max_seq_len: Optional[int] = None
    top_k_features: int = 1024
    fdr: float = 0.1
    logistic_c: float = 0.5
    logistic_penalty: str = "l1"
    logistic_max_iter: int = 3000
    knockoff_ridge: float = 1e-3
    knockoff_smax: float = 0.99
    random_seed: int = SEED
    output_dir: str = "experiments/modelx_knockoffs/results"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


@dataclass
class ActivationTable:
    """Container for SAE activations aligned with dataset labels."""

    matrix: np.ndarray
    labels: np.ndarray
    prompts: List[str]


@dataclass
class ReducedActivations:
    """Subset of SAE latents retained for knockoff inference."""

    matrix: np.ndarray
    labels: np.ndarray
    prompts: List[str]
    latent_indices: np.ndarray
    activation_mean: np.ndarray
    activation_rate: np.ndarray
    energy: np.ndarray


@dataclass
class KnockoffSummary:
    """Serializable summary of the knockoff run."""

    config: ExperimentConfig
    n_samples: int
    n_features: int
    threshold: float
    fdr_level: float
    discoveries: int
    train_accuracy: float
    train_log_loss: float


def _layer_index_from_sae_id(sae_id: str) -> int:
    """Best-effort extraction of SAE layer index from its identifier."""
    for part in sae_id.split("."):
        if part.isdigit():
            return int(part)
    return 0


def resolve_sae_id_for_release(release_name: str, override: Optional[str]) -> str:
    """Pick a representative SAE id (defaulting to mid-layer resid_pre) if none provided."""
    if override:
        return override
    if get_pretrained_saes_directory is None:
        raise ValueError(
            "SAE id was not supplied and pretrained directory metadata is unavailable. "
            "Install sae-lens>=3.2 or pass --sae-id explicitly."
        )
    directory = get_pretrained_saes_directory()
    if release_name not in directory:
        suggestions = get_close_matches(
            release_name,
            list(directory.keys()),
            n=SAE_RELEASE_SUGGESTION_COUNT,
            cutoff=0.2,
        )
        suggestion_text = (
            f" Closest matches: {suggestions}"
            if suggestions
            else " Call get_pretrained_saes_directory() to inspect available releases."
        )
        raise ValueError(
            f"SAE release '{release_name}' not found.{suggestion_text}"
        )
    release = directory[release_name]
    saes_map = getattr(release, "saes_map", None)
    if saes_map is None:
        saes_map = release.get("saes_map") if isinstance(release, dict) else None
    if not saes_map:
        raise ValueError(f"SAE release '{release_name}' does not expose a saes_map.")
    candidates = list(saes_map.keys())
    resid_pre = [name for name in candidates if "hook_resid_pre" in name]
    target = resid_pre or candidates
    target.sort(key=_layer_index_from_sae_id)
    return target[len(target) // 2]


def get_sae_hook_name(sae: SAE) -> str:
    """Extract the hook name recorded in the SAE metadata."""
    metadata = getattr(sae.cfg, "metadata", None)
    hook_name = getattr(metadata, "hook_name", None) if metadata else None
    if hook_name:
        return hook_name
    hook_name = getattr(sae, "hook_name", None)
    if hook_name:
        return hook_name
    raise ValueError(
        "Unable to determine SAE hook name from metadata. Please ensure the release "
        "stores `metadata.hook_name` or pass an SAE trained on a specific hook."
    )


def get_sae_prepend_bos(sae: SAE) -> bool:
    """Return prepend_bos flag from the SAE config/metadata (defaults True)."""
    metadata = getattr(sae.cfg, "metadata", None)
    if metadata is not None:
        prepend = getattr(metadata, "prepend_bos", None)
        if prepend is not None:
            return bool(prepend)
    return bool(getattr(sae.cfg, "prepend_bos", True))

def set_seeds(seed: int) -> None:
    """Set RNG seeds across numpy/torch for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_text_dataset(cfg: DatasetConfig, max_samples: Optional[int], seed: int) -> Dataset:
    """Load and (optionally) subsample a Hugging Face dataset."""
    dataset = load_dataset(
        cfg.dataset_name,
        cfg.dataset_config,
        split=cfg.split,
        trust_remote_code=cfg.trust_remote_code,
    )
    if max_samples is not None:
        dataset = dataset.shuffle(seed=seed)
        max_count = min(max_samples, len(dataset))
        dataset = dataset.select(range(max_count))
    return dataset


def batched(iterable: Iterable[dict], batch_size: int) -> Iterable[List[dict]]:
    """Yield lists of size <= batch_size from an iterable of dicts."""
    batch: List[dict] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def collect_sae_activations(
    model: HookedSAETransformer,
    sae: SAE,
    dataset: Dataset,
    text_column: str,
    label_column: str,
    batch_size: int,
    max_seq_len: Optional[int],
) -> ActivationTable:
    """
    Run the model with SAEs attached and capture per-example latent activations.
    """

    sae_hook = get_sae_hook_name(sae)
    hook_name = f"{sae_hook}.hook_sae_acts_post"
    latents: List[np.ndarray] = []
    labels: List[int] = []
    prompts: List[str] = []

    formatted = dataset.with_format("python")
    total_batches = math.ceil(len(dataset) / batch_size)
    iterator = batched(formatted, batch_size)
    for batch in tqdm(iterator, total=total_batches, desc="Collecting SAE activations"):
        texts = [example[text_column] for example in batch]
        raw_labels = [example[label_column] for example in batch]
        tensor_labels = [int(label) for label in raw_labels]
        tokens = model.to_tokens(texts, prepend_bos=get_sae_prepend_bos(sae))
        if max_seq_len is not None and tokens.shape[1] > max_seq_len:
            tokens = tokens[:, -max_seq_len:]
        tokens = tokens.to(model.cfg.device)

        with torch.no_grad():
            _, cache = model.run_with_cache_with_saes(
                tokens,
                saes=[sae],
                names_filter=lambda name: name == hook_name,
            )

        acts = cache[hook_name][:, -1, :]
        latents.append(acts.detach().cpu().float().numpy())
        labels.extend(tensor_labels)
        prompts.extend(texts)

    matrix = np.concatenate(latents, axis=0)
    label_array = np.asarray(labels, dtype=np.int64)
    return ActivationTable(matrix=matrix, labels=label_array, prompts=prompts)


def score_and_reduce_features(
    activations: ActivationTable,
    top_k: int,
) -> ReducedActivations:
    """
    Keep the top-k most energetic latents to stabilize covariance estimation.
    """

    energy = np.mean(np.abs(activations.matrix), axis=0)
    order = np.argsort(-energy)
    if top_k is not None and top_k < len(order):
        order = order[:top_k]

    reduced_matrix = activations.matrix[:, order]
    activation_mean = reduced_matrix.mean(axis=0)
    activation_rate = (reduced_matrix > 0).mean(axis=0)

    return ReducedActivations(
        matrix=reduced_matrix,
        labels=activations.labels,
        prompts=activations.prompts,
        latent_indices=order,
        activation_mean=activation_mean,
        activation_rate=activation_rate,
        energy=energy[order],
    )


class GaussianKnockoffSampler:
    """Equi-correlated Gaussian knockoffs with simple shrinkage."""

    def __init__(self, ridge: float = 1e-3, smax: float = 0.99):
        self.ridge = ridge
        self.smax = smax
        self.mean_: Optional[np.ndarray] = None
        self.transform_: Optional[np.ndarray] = None
        self.chol_: Optional[np.ndarray] = None
        self.s_value_: Optional[float] = None

    def fit(self, X: np.ndarray) -> "GaussianKnockoffSampler":
        n, p = X.shape
        if n <= p:
            raise ValueError(
                f"Need more samples than features for covariance estimation (n={n}, p={p})."
            )
        mean = X.mean(axis=0, keepdims=True)
        centered = X - mean
        cov = (centered.T @ centered) / (n - 1)
        cov += self.ridge * np.eye(p)

        eigvals = np.linalg.eigvalsh(cov)
        lambda_min = max(eigvals.min(), 1e-6)
        s_val = min(2 * lambda_min, self.smax)
        if s_val <= 0:
            raise ValueError("Failed to construct PSD knockoff matrix; increase ridge or reduce p.")
        S = s_val * np.eye(p)

        sigma_inv = linalg.inv(cov)
        cov_term = 2 * S - S @ sigma_inv @ S
        cov_term = (cov_term + cov_term.T) / 2
        min_eig = np.linalg.eigvalsh(cov_term).min()
        if min_eig <= 0:
            cov_term += np.eye(p) * (abs(min_eig) + 1e-6)
        chol = linalg.cholesky(cov_term, lower=True)

        transform = np.eye(p) - sigma_inv @ S

        self.mean_ = mean
        self.transform_ = transform
        self.chol_ = chol
        self.s_value_ = s_val
        return self

    def sample(self, X: np.ndarray) -> np.ndarray:
        if any(value is None for value in (self.mean_, self.transform_, self.chol_)):
            raise RuntimeError("Sampler must be fitted before sampling knockoffs.")
        centered = X - self.mean_
        noise = np.random.randn(*centered.shape) @ self.chol_.T
        knockoffs = centered @ self.transform_ + noise
        return knockoffs + self.mean_


def fit_knockoff_classifier(
    X: np.ndarray,
    y: np.ndarray,
    sampler: GaussianKnockoffSampler,
    C: float,
    penalty: str,
    max_iter: int,
) -> Tuple[np.ndarray, float, float, LogisticRegression]:
    """Fit logistic regression on [X, X~] and return knockoff statistics."""

    knockoffs = sampler.sample(X)
    design = np.concatenate([X, knockoffs], axis=1)
    scaler = StandardScaler()
    design_scaled = scaler.fit_transform(design)

    solver = "saga" if penalty == "l1" else "lbfgs"
    n_jobs = -1 if solver == "saga" else None
    clf = LogisticRegression(
        penalty=penalty,
        solver=solver,
        C=C,
        max_iter=max_iter,
        n_jobs=n_jobs,
    )
    clf.fit(design_scaled, y)

    probs = clf.predict_proba(design_scaled)[:, 1]
    preds = (probs >= 0.5).astype(np.int64)
    acc = accuracy_score(y, preds)
    ce = log_loss(y, probs, labels=[0, 1])

    coef = clf.coef_.reshape(-1)
    p = X.shape[1]
    original = np.abs(coef[:p])
    knock = np.abs(coef[p:])
    stats = original - knock
    return stats, acc, ce, clf


def knockoff_threshold(stats: np.ndarray, fdr: float, offset: int = 1) -> float:
    """Compute the knockoff+ threshold."""
    if np.allclose(stats, 0):
        return math.inf
    t_candidates = np.sort(np.abs(stats[stats != 0]))
    if t_candidates.size == 0:
        return math.inf
    for t in t_candidates:
        num = offset + np.sum(stats <= -t)
        denom = max(np.sum(stats >= t), 1)
        if num / denom <= fdr:
            return t
    return math.inf


def serialize_results(
    summary: KnockoffSummary,
    stats: np.ndarray,
    reduced: ReducedActivations,
) -> Path:
    """Write summary + feature-level outputs to disk and return the artefact dir."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(summary.config.output_dir) / timestamp
    result_dir.mkdir(parents=True, exist_ok=True)

    summary_path = result_dir / "knockoff_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                **asdict(summary),
                "config": asdict(summary.config),
            },
            fp,
            indent=2,
        )

    np.save(result_dir / "knockoff_stats.npy", stats)
    np.save(result_dir / "latent_indices.npy", reduced.latent_indices)

    selected = np.where(stats >= summary.threshold)[0] if math.isfinite(summary.threshold) else []
    feature_rows = []
    for idx in selected:
        feature_rows.append(
            {
                "latent_index": int(reduced.latent_indices[idx]),
                "knockoff_stat": float(stats[idx]),
                "mean_activation": float(reduced.activation_mean[idx]),
                "activation_rate": float(reduced.activation_rate[idx]),
                "energy": float(reduced.energy[idx]),
            }
        )

    feature_path = result_dir / "selected_features.json"
    with feature_path.open("w", encoding="utf-8") as fp:
        json.dump(feature_rows, fp, indent=2)

    print(f"Wrote artefacts to {result_dir}")
    return result_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model-X knockoffs for SAE features.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config.")
    parser.add_argument("--model-name", default=argparse.SUPPRESS, help="TransformerLens model id (default: pythia-1.4b-deduped).")
    parser.add_argument("--sae-release", default=argparse.SUPPRESS, help="SAELens release repo (default: pythia-1.4b-deduped-res-jb).")
    parser.add_argument("--sae-id", default=argparse.SUPPRESS, help="SAE identifier within the release; defaults to the median resid_pre layer.")
    parser.add_argument("--dataset-name", default=argparse.SUPPRESS, help="Hugging Face dataset name (default: glue).")
    parser.add_argument("--dataset-config", default=argparse.SUPPRESS, help="Dataset config/subset (default: sst2).")
    parser.add_argument("--split", default=argparse.SUPPRESS, help="Dataset split (default: validation).")
    parser.add_argument("--text-column", default=argparse.SUPPRESS, help="Text column name (default: sentence).")
    parser.add_argument("--label-column", default=argparse.SUPPRESS, help="Label column name (default: label).")
    parser.add_argument("--max-samples", type=int, default=argparse.SUPPRESS, help="Dataset cap (default: 512).")
    parser.add_argument("--activation-batch-size", type=int, default=argparse.SUPPRESS, help="Batch size for activation capture (default: 16).")
    parser.add_argument("--max-seq-len", type=int, default=argparse.SUPPRESS, help="Optional context truncation.")
    parser.add_argument("--top-k-features", type=int, default=argparse.SUPPRESS, help="Number of latents to keep (default: 1024).")
    parser.add_argument("--fdr", type=float, default=argparse.SUPPRESS, help="Target false discovery rate (default: 0.1).")
    parser.add_argument("--logistic-c", type=float, default=argparse.SUPPRESS, help="Inverse regularisation strength (default: 0.5).")
    parser.add_argument("--logistic-penalty", choices=("l1", "l2"), default=argparse.SUPPRESS, help="Penalty for the classifier (default: l1).")
    parser.add_argument("--logistic-max-iter", type=int, default=argparse.SUPPRESS, help="Max iterations for the classifier (default: 3000).")
    parser.add_argument("--knockoff-ridge", type=float, default=argparse.SUPPRESS, help="Covariance ridge term (default: 1e-3).")
    parser.add_argument("--knockoff-smax", type=float, default=argparse.SUPPRESS, help="Upper bound on equi-correlated S (default: 0.99).")
    parser.add_argument("--device", default=argparse.SUPPRESS, help=f"Computation device (default: {DEFAULT_DEVICE}).")
    parser.add_argument("--output-dir", default=argparse.SUPPRESS, help="Result directory root (default: experiments/modelx_knockoffs/results).")
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS, help=f"Random seed (default: {SEED}).")
    parser.add_argument("--trust-remote-code", action="store_true", default=argparse.SUPPRESS, help="Forward to datasets.load_dataset.")
    return parser


def load_config_from_yaml(path: str) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text())
    dataset_cfg = DatasetConfig(**data.get("dataset", {}))
    exp_kwargs = {k: v for k, v in data.items() if k != "dataset"}
    return ExperimentConfig(dataset=dataset_cfg, **exp_kwargs)


def apply_override(obj, attr: str, value) -> None:
    setattr(obj, attr, value)


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if getattr(args, "config", None):
        cfg = load_config_from_yaml(args.config)

    top_level_fields = {
        "model_name": "model_name",
        "sae_release": "sae_release",
        "sae_id": "sae_id",
        "device": "device",
        "max_samples": "max_samples",
        "activation_batch_size": "activation_batch_size",
        "max_seq_len": "max_seq_len",
        "top_k_features": "top_k_features",
        "fdr": "fdr",
        "logistic_c": "logistic_c",
        "logistic_penalty": "logistic_penalty",
        "logistic_max_iter": "logistic_max_iter",
        "knockoff_ridge": "knockoff_ridge",
        "knockoff_smax": "knockoff_smax",
        "output_dir": "output_dir",
        "seed": "random_seed",
    }

    for arg_name, field_name in top_level_fields.items():
        if hasattr(args, arg_name):
            apply_override(cfg, field_name, getattr(args, arg_name))

    dataset_fields = {
        "dataset_name": "dataset_name",
        "dataset_config": "dataset_config",
        "split": "split",
        "text_column": "text_column",
        "label_column": "label_column",
        "trust_remote_code": "trust_remote_code",
    }

    for arg_name, field_name in dataset_fields.items():
        if hasattr(args, arg_name):
            apply_override(cfg.dataset, field_name, getattr(args, arg_name))

    return cfg


def run_modelx_knockoffs(cfg: ExperimentConfig) -> Path:
    """Run the full Model-X knockoff pipeline and return the artefact directory."""
    resolved_sae_id = resolve_sae_id_for_release(cfg.sae_release, cfg.sae_id)
    cfg = replace(cfg, sae_id=resolved_sae_id)

    set_seeds(cfg.random_seed)
    torch.set_grad_enabled(False)

    dataset = load_text_dataset(cfg.dataset, cfg.max_samples, cfg.random_seed)
    print(f"Loaded {len(dataset)} samples from {cfg.dataset.dataset_name}/{cfg.dataset.split}.")

    model = HookedSAETransformer.from_pretrained(cfg.model_name, device=cfg.device)
    sae, _, _ = SAE.from_pretrained(
        release=cfg.sae_release,
        sae_id=cfg.sae_id,
        device=str(cfg.device),
    )
    model.eval()

    activations = collect_sae_activations(
        model=model,
        sae=sae,
        dataset=dataset,
        text_column=cfg.dataset.text_column,
        label_column=cfg.dataset.label_column,
        batch_size=cfg.activation_batch_size,
        max_seq_len=cfg.max_seq_len,
    )

    reduced = score_and_reduce_features(activations, cfg.top_k_features)

    sampler = GaussianKnockoffSampler(ridge=cfg.knockoff_ridge, smax=cfg.knockoff_smax)
    sampler.fit(reduced.matrix)

    stats, acc, ce, _ = fit_knockoff_classifier(
        reduced.matrix,
        reduced.labels,
        sampler,
        cfg.logistic_c,
        cfg.logistic_penalty,
        cfg.logistic_max_iter,
    )

    threshold = knockoff_threshold(stats, cfg.fdr)
    discoveries = int(np.sum(stats >= threshold)) if math.isfinite(threshold) else 0

    summary = KnockoffSummary(
        config=cfg,
        n_samples=reduced.matrix.shape[0],
        n_features=reduced.matrix.shape[1],
        threshold=threshold,
        fdr_level=cfg.fdr,
        discoveries=discoveries,
        train_accuracy=acc,
        train_log_loss=ce,
    )

    return serialize_results(summary, stats, reduced)


def main() -> None:
    parser = build_arg_parser()
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unrecognized arguments (likely from Jupyter): {unknown}")
    cfg = build_config(args)
    run_modelx_knockoffs(cfg)


if __name__ == "__main__":
    main()