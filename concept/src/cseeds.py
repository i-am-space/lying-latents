"""Seed streams for the concept-dilution branch.

Kept separate from src/common.py:STREAMS so main-line code is untouched. Each stream is
SeedSequence(entropy=master_seed, spawn_key=(9000, CSTREAMS.index(name), *cell_idx)); the
main repo's streams are spawn(len(STREAMS)) children with keys 0..34, so key 9000 cannot
collide. Append only, never reorder.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

import _paths  # noqa: F401
from common import load_config

CSTREAMS = [
    "toy_data",          # Step 1: synthetic designs
    "toy_knockoff",      # Step 1: knockoff draws
    "candidates",        # Step 4: planted pool and filler order
    "s_solve",           # Step 5: seeds for knockpy's MVR solver (merge_groups shuffles)
    "gate_planted",      # Step 6: planted labels
    "gate_knockoff",     # Step 6: knockoff draws
    "planted",           # Step 7: planted concepts and labels
    "planted_knockoff",  # Step 7: knockoff draws
    "real_knockoff",     # Step 8: knockoff redraws on real labels
    "tests",             # unit tests
]
ROOT = Path(__file__).resolve().parents[2]
CONCEPT_CFG = ROOT / "concept" / "config" / "concept.yaml"


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_concept_config(path: str | Path = CONCEPT_CFG) -> dict:
    path = Path(path)
    over = yaml.safe_load(path.read_text())
    base = load_config((path.parent / over.pop("base")).resolve())
    cfg = _merge(base, over)
    assert cfg["master_seed"] == base["master_seed"], "the overlay must not change master_seed"
    return cfg


def cseed(cfg: dict, stream: str, *cell_idx: int) -> np.random.SeedSequence:
    if stream not in CSTREAMS:
        raise KeyError(f"{stream!r} is not a concept seed stream: {CSTREAMS}")
    return np.random.SeedSequence(entropy=cfg["master_seed"],
                                  spawn_key=(9000, CSTREAMS.index(stream), *(int(i) for i in cell_idx)))


def crng(cfg: dict, stream: str, *cell_idx: int) -> np.random.Generator:
    return np.random.default_rng(cseed(cfg, stream, *cell_idx))


def cint(cfg: dict, stream: str, *cell_idx: int) -> int:
    return int(crng(cfg, stream, *cell_idx).integers(2**31))
