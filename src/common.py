"""Shared config loading, seeding and hashing. Kept deliberately small."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent

# Fixed order — must match config/preregistration.yaml seeds.streams. Do not reorder.
STREAMS = [
    "dataset_order",
    "torch_global",
    "knockoff_sampler",
    "row_partition",
    "classifier",
    "mmd_permutation",
    "synthetic_null",
    "label_permutation",   # Stage 1: FWER calibration
    "planted_signal",      # Stage 3: Planted signal FDR benchmark
    "validate_permutation",  # Stage 1 validate step: label permutations + splits
    "validate_planted",      # Stage 1 validate step: planted-signal power check
    "s3v2_data",             # Stage 3 v2: latent/row subsets and the Gaussian control data
    "s3v2_planted",          # Stage 3 v2: planted signals and labels (per-cell streams)
    "s3v2_knockoff",         # Stage 3 v2: knockoff draws and WY permutations (per-cell streams)
    "s3p2048_data",          # Stage 3 p2048: row subset and the Gaussian control data
    "s3p2048_planted",       # Stage 3 p2048: planted signals and labels (per-cell streams)
    "s3p2048_knockoff",      # Stage 3 p2048: knockoff draw bank (per dataset and replicate)
    "s4_hurdle_fit",         # Stage 4: randomised probability-integral transform of the zero atom
    "s4_planted",            # Stage 4: planted signals and labels (per-cell streams)
    "s4_knockoff",           # Stage 4: knockoff draw bank (per arm and replicate)
    "s4_diagnostics",        # Stage 4: exchangeability swap tests
    "s4r_data",              # Stage 4 repairs: latent subset and the Gaussian control data
    "s4r_planted",           # Stage 4 repairs: planted signals and labels (per-cell streams)
    "s4r_knockoff",          # Stage 4 repairs: knockoff draw seeds (per arm and replicate)
    "s4r_split",             # Stage 4 repairs: e-value sample splits (per replicate)
    "s4r_diagnostics",       # Stage 4 repairs: exchangeability swap tests
    "s3s_hurdle",            # Stage 3 stress: hurdle knockoff draw bank (per replicate)
    "s3s_diagnostics",       # Stage 3 stress: exchangeability swap tests
    "s3d_data",              # Stage 3 dims: row subsets and the Gaussian control data
    "s3d_planted",           # Stage 3 dims: planted signals and labels (per-cell streams)
    "s3d_knockoff",          # Stage 3 dims: knockoff draws (per-cell streams)
    "s3c_calibration",       # Stage 3 amplitude calibration: data split and planted labels
    "s3k_data",              # Stage 3 largek: Gaussian control data and the block-MVR solve seed
    "s3k_planted",           # Stage 3 largek: planted signals and labels (per-cell streams)
    "s3k_knockoff",          # Stage 3 largek: knockoff draws (per-cell streams)
    "s3cv_folds",            # Stage 3 cvlambda: cross-validation folds (per-cell streams)
]


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def rng_for(cfg: dict, stream: str) -> np.random.Generator:
    """Derive a component's RNG from the single master seed, reproducibly."""
    if stream not in STREAMS:
        raise KeyError(f"{stream!r} is not a pre-registered seed stream: {STREAMS}")
    children = np.random.SeedSequence(cfg["master_seed"]).spawn(len(STREAMS))
    return np.random.default_rng(children[STREAMS.index(stream)])


def cache_config(cfg: dict) -> dict:
    """The subset of config that determines cache contents. Anything here changing
    must produce a different file, or we risk silently reading a cache built under a
    different aggregator or layer."""
    return {
        "data": cfg["data"],
        "model": {k: cfg["model"][k] for k in ("name", "dtype", "layer", "max_seq_len")},
        "sae": cfg["sae"],
        "aggregation": cfg["aggregation"],
        "latent_filter": cfg["latent_filter"],
        "master_seed": cfg["master_seed"],
    }


def config_hash(d: dict) -> str:
    blob = json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def cache_path(cfg: dict) -> Path:
    sub = cache_config(cfg)
    return ROOT / cfg["paths"]["cache_dir"] / f"{config_hash(sub)}.npz"


def results_dir(cfg: dict) -> Path:
    d = ROOT / cfg["paths"]["results_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d
