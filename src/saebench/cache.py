"""SAEBench step 2 — task texts -> Gemma Scope SAE latents, one cache per dataset.

Encoding is cache_activations.encode_texts (same model, layer, SAE, special-token masking and mean
pooling as the SST-2 cache). Latents are retained per dataset by the latent_filter rule applied to
that dataset's rows (firing rate >= 1%, top 2,048 by firing rate), because which latents fire
depends on the domain (code vs biographies). Only the primary aggregator is stored.

Usage: python src/saebench/cache.py --config config/default.yaml [--dataset fancyzhx/ag_news] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/
sys.path.insert(0, str(Path(__file__).resolve().parent))       # src/saebench/

from cache_activations import encode_texts, load_model_and_sae   # noqa: E402
from common import config_hash, load_config                      # noqa: E402
from sbutil import (cache_path, pooled_rows, read_json, results_dir,   # noqa: E402
                    sb_cache_config, tasks_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="SAEBench: cache SAE latents per dataset")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--dataset", default=None, help="one dataset only")
    ap.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    ap.add_argument("--limit", type=int, default=None, help="debug: first N rows only (not a valid cache)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override model.batch_size to save GPU memory; pooling is masked per text, so values "
                         "do not depend on it (and it is not part of the cache hash)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.batch_size:
        cfg["model"]["batch_size"] = args.batch_size
    sb = cfg["saebench"]
    torch.manual_seed(np.random.SeedSequence(cfg["master_seed"]).spawn(7)[1].generate_state(1)[0])
    torch.set_grad_enabled(False)
    names = [args.dataset] if args.dataset else sb["datasets"]
    todo = [n for n in names if args.force or args.limit or not cache_path(cfg, n).exists()]
    if not todo:
        print("all requested caches exist (use --force to rebuild)")
        return
    tok, model, sae = load_model_and_sae(cfg)
    lf, agg = cfg["latent_filter"], sb["aggregation"]
    summary_path = results_dir(cfg) / "cache_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    for name in todo:
        tj = read_json(tasks_path(cfg, name))
        texts, cls, split = pooled_rows(tj)
        if args.limit:
            texts, cls, split = texts[: args.limit], cls[: args.limit], split[: args.limit]
        print(f"\n=== {name}: {len(texts)} texts, classes {tj['classes']} ===", flush=True)
        acc, n_tokens, ev, l0, secs = encode_texts(texts, tok, model, sae, cfg["model"])
        X = acc[agg]
        del acc
        firing = (X > 0).mean(axis=0)
        surv = np.flatnonzero(firing >= lf["min_firing_rate"])
        n_firing = int(surv.size)
        if surv.size > lf["max_retained"]:
            surv = surv[np.argsort(-firing[surv], kind="stable")[: lf["max_retained"]]]
        retained = np.sort(surv)
        Xr = X[:, retained]
        p0 = float(np.median((Xr == 0).mean(axis=0)))
        print(f"  {secs / 60:.1f} min | SAE explained variance {ev:.4f} | mean L0 {l0:.1f} | "
              f"latents firing >= {lf['min_firing_rate']}: {n_firing} | retained p = {retained.size} | "
              f"median Pr(X=0) {p0:.3f}", flush=True)
        if args.limit:
            print("  --limit given: debug run, cache not written")
            continue
        min_ev = sb.get("min_explained_variance", 0.3)
        if ev < min_ev:
            # Gemma Scope L20 gives 0.71-0.80 on these texts; an SAE fed the wrong layer or hook gives far less
            # (typically near zero or negative)
            raise SystemExit(f"  SAE explained variance {ev:.3f} < {min_ev}: hook / layer / SAE mismatch? cache NOT written")
        out = cache_path(cfg, name)
        out.parent.mkdir(parents=True, exist_ok=True)
        cc = sb_cache_config(cfg, name)
        np.savez(out, X=Xr, class_idx=cls, split=split, classes=np.asarray(tj["classes"]),
                 n_tokens=n_tokens, retained_idx=retained, firing_rate_all=firing.astype(np.float32),
                 config_json=np.asarray(json.dumps(cc)), config_hash=np.asarray(config_hash(cc)),
                 sae_explained_variance=np.asarray(ev), sae_mean_l0=np.asarray(l0),
                 master_seed=np.asarray(cfg["master_seed"]))
        summary[name] = {"cache": out.name, "config_hash": config_hash(cc), "n": len(texts),
                         "classes": tj["classes"], "sae_explained_variance": ev, "sae_mean_l0": l0,
                         "n_latents_firing_1pct": n_firing, "p_retained": int(retained.size),
                         "median_zero_mass_retained": p0, "mean_tokens": float(n_tokens.mean()),
                         "model": cfg["model"]["name"], "layer": cfg["model"]["layer"],
                         "sae": f"{cfg['sae']['release']}/{cfg['sae']['sae_id']}", "minutes": secs / 60}
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"  wrote {out} ({out.stat().st_size / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
