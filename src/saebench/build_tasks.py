"""SAEBench step 1 — build the sparse-probing texts with SAEBench's own code.

Runs in a separate, small environment, because SAEBench's GitHub-code loader is a script-based
dataset that needs datasets<4 (the main environment pins datasets 4.0):

  conda create -n saebench-data python=3.12 -y && conda activate saebench-data
  pip install "datasets>=3,<4" pandas tqdm transformers einops pyyaml
  pip install sae-bench==0.6.0 --no-deps

For each dataset, SAEBench's get_multi_label_train_test_data + filter_dataset give, per chosen
class, the train and test texts its sparse-probing eval uses (2,000 train / 500 test per class at
the default 4,000 / 1,000 task sizes). They are written to <data_dir>/tasks/<slug>.json, and a
manifest with per-file SHA-256 and counts goes to results/saebench/tasks_manifest.json (the texts
themselves are gitignored; the manifest is committed).

Usage: python src/saebench/build_tasks.py --config config/default.yaml [--dataset fancyzhx/ag_news]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def slug(name: str) -> str:
    return name.replace("/", "__")


def main() -> None:
    ap = argparse.ArgumentParser(description="SAEBench: build sparse-probing task texts")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--dataset", default=None, help="build one dataset only")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    sb = cfg["saebench"]
    got = version("sae-bench")
    if got != sb["sae_bench_version"]:
        sys.exit(f"sae-bench {got} installed, config pins {sb['sae_bench_version']}")
    from sae_bench.sae_bench_utils import dataset_info, dataset_utils

    out_dir = ROOT / sb["data_dir"] / "tasks"
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = ROOT / cfg["paths"]["results_dir"] / "saebench" / "tasks_manifest.json"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}

    names = [args.dataset] if args.dataset else sb["datasets"]
    for name in names:
        if name not in sb["datasets"]:
            sys.exit(f"{name} is not in saebench.datasets")
        t = time.time()
        classes = list(dataset_info.chosen_classes_per_dataset[name])
        train, test = dataset_utils.get_multi_label_train_test_data(
            name, sb["probe_train_set_size"], sb["probe_test_set_size"], sb["text_seed"])
        train = dataset_utils.filter_dataset(train, classes)
        test = dataset_utils.filter_dataset(test, classes)
        payload = {"dataset": name, "classes": classes, "train": train, "test": test,
                   "sae_bench_version": got, "text_seed": sb["text_seed"],
                   "probe_train_set_size": sb["probe_train_set_size"],
                   "probe_test_set_size": sb["probe_test_set_size"]}
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        path = out_dir / f"{slug(name)}.json"
        path.write_bytes(blob)
        counts = {c: [len(train[c]), len(test[c])] for c in classes}
        manifest[name] = {"file": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(blob).hexdigest(),
                          "classes": classes, "train_test_counts": counts,
                          "sae_bench_version": got, "text_seed": sb["text_seed"]}
        print(f"{name}: {len(classes)} classes, counts {counts}  ({time.time() - t:.0f}s)", flush=True)
        man_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_dir} and {man_path}")


if __name__ == "__main__":
    main()
