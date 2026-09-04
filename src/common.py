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
