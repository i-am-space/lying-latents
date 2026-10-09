"""Shared helpers for the SAEBench scripts: paths, cache loading and the one-vs-rest tasks.

The binary tasks follow SAEBench's sparse_probing.probe_training.prepare_probe_data, applied to the
train and test pools separately: the positives are every text of the class; the negatives take
ceil(n_pos / n_other_classes) texts from each other class, then n_pos of those at random. For a
two-class dataset (Amazon sentiment) the two one-vs-rest tasks are mirror images, so only the first
class is used, giving SAEBench's 35 tasks. Sampling uses our seed stream (sb_tasks), not torch's
global RNG, so it is reproducible here; the procedure is SAEBench's.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/, for `common`

from common import ROOT, config_hash, stream_rng   # noqa: E402


def slug(name: str) -> str:
    return name.replace("/", "__")


def sb_cache_config(cfg: dict, dataset: str) -> dict:
    """Everything that determines a dataset's cache contents (mirrors common.cache_config, with
    the SAEBench text source in place of the SST-2 data block)."""
    sb = cfg["saebench"]
    return {
        "saebench": {"dataset": dataset, "sae_bench_version": sb["sae_bench_version"],
                     "text_seed": sb["text_seed"], "probe_train_set_size": sb["probe_train_set_size"],
                     "probe_test_set_size": sb["probe_test_set_size"], "aggregation": sb["aggregation"]},
        "model": {k: cfg["model"][k] for k in ("name", "dtype", "layer", "max_seq_len")},
        "sae": cfg["sae"],
        "latent_filter": cfg["latent_filter"],
        "master_seed": cfg["master_seed"],
    }


def tasks_path(cfg: dict, dataset: str) -> Path:
    return ROOT / cfg["saebench"]["data_dir"] / "tasks" / f"{slug(dataset)}.json"


def cache_path(cfg: dict, dataset: str) -> Path:
    h = config_hash(sb_cache_config(cfg, dataset))
    return ROOT / cfg["saebench"]["data_dir"] / "cache" / f"{h}__{slug(dataset)}.npz"


def results_dir(cfg: dict, sub: str = "") -> Path:
    d = ROOT / cfg["paths"]["results_dir"] / "saebench" / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def pooled_rows(task_json: dict):
    """Row order of a dataset cache: every train text class by class, then every test text.
    Returns (texts, class_idx, split) with split 0 = SAEBench train, 1 = SAEBench test."""
    texts, cls, split = [], [], []
    for s, key in enumerate(("train", "test")):
        for ci, c in enumerate(task_json["classes"]):
            block = task_json[key][c]
            texts += block
            cls += [ci] * len(block)
            split += [s] * len(block)
    return texts, np.asarray(cls, dtype=np.int16), np.asarray(split, dtype=np.int8)


def load_cache(cfg: dict, dataset: str) -> dict:
    p = cache_path(cfg, dataset)
    if not p.exists():
        raise SystemExit(f"No cache for {dataset} at {p}. Run src/saebench/cache.py first.")
    d = np.load(p, allow_pickle=True)
    return {"X": d["X"].astype(np.float32, copy=False), "class_idx": d["class_idx"], "split": d["split"],
            "classes": [str(c) for c in d["classes"]], "config_hash": str(d["config_hash"]),
            "retained_idx": d["retained_idx"], "path": str(p)}


def positive_threshold(X: np.ndarray, frac: float) -> tuple[np.ndarray, float]:
    """JumpReLU-like thresholding of a ReLU SAE's latents: every latent's positive values below
    frac * (its median positive value) are set to exactly 0. A JumpReLU SAE has no activations
    just above 0 (Gemma Scope: minimum positive / median positive 0.13-0.36); a ReLU SAE has
    (Pythia: ~1e-4), which the hurdle's log-normal size model cannot reproduce (saebench_amendment_4).
    Returns (thresholded copy, share of the positive entries that were zeroed)."""
    X = np.array(X, dtype=np.float32, copy=True)
    pos = X > 0
    med = np.nanmedian(np.where(pos, X, np.nan), axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    cut = (frac * med).astype(np.float32)
    below = pos & (X < cut[None, :])
    X[below] = 0.0
    return X, float(below.sum() / max(int(pos.sum()), 1))


def binary_tasks(cfg: dict, dataset: str, C: dict) -> list[dict]:
    """The dataset's one-vs-rest tasks as row indices into the cache. Each task: name, class,
    rows (train then test), y (1 = positive), saebench_split (0 train / 1 test)."""
    di = cfg["saebench"]["datasets"].index(dataset)
    classes = C["classes"]
    use = classes[:1] if len(classes) == 2 else classes
    tasks = []
    for ci, c in enumerate(use):
        rng = stream_rng(cfg, "sb_tasks", di, ci)
        rows, ys, sp = [], [], []
        for s in (0, 1):
            pos = np.flatnonzero((C["class_idx"] == ci) & (C["split"] == s))
            others = [k for k in range(len(classes)) if k != ci]
            per = math.ceil(len(pos) / len(others))
            neg = np.concatenate([rng.permutation(np.flatnonzero((C["class_idx"] == k) & (C["split"] == s)))[:per]
                                  for k in others])
            neg = rng.permutation(neg)[: len(pos)]
            if len(neg) != len(pos):
                raise RuntimeError(f"{dataset}/{c}: only {len(neg)} negatives for {len(pos)} positives")
            r = np.concatenate([pos, neg])
            rows.append(r)
            ys.append(np.r_[np.ones(len(pos)), np.zeros(len(neg))])
            sp.append(np.full(len(r), s))
        tasks.append({"name": f"{slug(dataset)}__{c}", "dataset": dataset, "class": c,
                      "rows": np.concatenate(rows), "y": np.concatenate(ys).astype(np.float32),
                      "saebench_split": np.concatenate(sp).astype(np.int8)})
    return tasks


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())
