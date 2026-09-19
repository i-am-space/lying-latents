"""Stage 2, Deliverable 1 — SST-2 -> Gemma Scope SAE latents -> on-disk matrix.

One row per sentence. Token-level latents are aggregated three ways (mean, max,
last) over non-special tokens only; padding, BOS and EOS are excluded. See
NOTES_reference.md §2 for why the reference's [:, -1, :] is not reproduced.

Usage: python src/cache_activations.py --config config/default.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import cache_config, cache_path, config_hash, load_config

AGGS = ("mean", "max", "last")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--limit", type=int, default=None, help="debug: cap sentences")
    ap.add_argument("--out", default=None, help="debug: override output path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(np.random.SeedSequence(cfg["master_seed"]).spawn(7)[1].generate_state(1)[0])
    torch.set_grad_enabled(False)

    out = cache_path(cfg) if args.out is None else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"config hash : {config_hash(cache_config(cfg))}")
    print(f"output      : {out}")

    # ---- data -------------------------------------------------------------
    dc = cfg["data"]
    ds = load_dataset(dc["dataset"], dc["dataset_config"], split=dc["split"])
    if args.limit:
        ds = ds.select(range(args.limit))
    sents = list(ds[dc["text_column"]])
    labels = np.asarray(ds[dc["label_column"]], dtype=np.int8)
    sent_ids = np.asarray(ds["idx"], dtype=np.int64)
    n = len(sents)
    print(f"sentences   : {n}")

    # ---- model + SAE ------------------------------------------------------
    mc, sc = cfg["model"], cfg["sae"]
    dtype = getattr(torch, mc["dtype"])
    tok = AutoTokenizer.from_pretrained(mc["name"])
    model = AutoModelForCausalLM.from_pretrained(
        mc["name"], dtype=dtype, attn_implementation=mc["attn_implementation"]
    ).to("cuda").eval()

    from sae_lens import SAE

    loaded = SAE.from_pretrained(release=sc["release"], sae_id=sc["sae_id"], device="cuda")
    sae = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
    sae = sae.to(dtype).eval()
    d_sae = sae.cfg.d_sae
    hook_layer = mc["layer"]
    print(f"d_sae       : {d_sae}   layer {hook_layer} resid_post")

    special = {i for i in (tok.pad_token_id, tok.bos_token_id, tok.eos_token_id) if i is not None}
    print(f"special ids : {sorted(special)}")

    # ---- allocate ---------------------------------------------------------
    acc = {a: np.zeros((n, d_sae), dtype=np.float32) for a in AGGS}
    n_tokens = np.zeros(n, dtype=np.int32)

    # Sort by length so batches are tight. Aggregation is masked, so batch
    # composition cannot affect any row's value -- unlike the reference.
    enc_len = np.asarray([len(tok(s)["input_ids"]) for s in sents])
    order = np.argsort(enc_len, kind="stable")

    bs, msl = mc["batch_size"], mc["max_seq_len"]
    recon_num, recon_den = 0.0, 0.0
    l0_num, l0_den = 0.0, 0.0
    t0 = time.time()
    for start in tqdm(range(0, n, bs), desc="cache"):
        idx = order[start : start + bs]
        batch = tok([sents[i] for i in idx], return_tensors="pt", padding=True,
                    truncation=True, max_length=msl)
        ids = batch["input_ids"].to("cuda")
        am = batch["attention_mask"].to("cuda")

        hs = model(input_ids=ids, attention_mask=am, output_hidden_states=True
                   ).hidden_states[hook_layer + 1]          # resid_post of `layer`
        z = sae.encode(hs)                                   # (B, T, d_sae)

        keep = am.bool()
        for sid in special:
            keep &= ids != sid
        # every sentence must retain >= 1 token; fall back to attention mask if not
        empty = ~keep.any(dim=1)
        if empty.any():
            keep[empty] = am.bool()[empty]

        km = keep.unsqueeze(-1)
        cnt = keep.sum(dim=1, keepdim=True).clamp(min=1)
        zm = z * km
        mean = zm.sum(dim=1) / cnt
        mx = zm.max(dim=1).values                            # latents are >= 0
        last_pos = keep.float().cumsum(dim=1).argmax(dim=1)  # last True index
        lastv = z[torch.arange(z.shape[0], device=z.device), last_pos]

        acc["mean"][idx] = mean.float().cpu().numpy()
        acc["max"][idx] = mx.float().cpu().numpy()
        acc["last"][idx] = lastv.float().cpu().numpy()
        n_tokens[idx] = keep.sum(dim=1).cpu().numpy()

        # sanity: SAE reconstruction quality on the SAME tokens we aggregate over.
        # This must exclude BOS: it is an attention sink with residual norm ~2900
        # (vs ~350 for content tokens) and L0 ~7000, and Gemma Scope SAEs do not
        # model it. Including it drives explained variance to about -13 and tells
        # you nothing about whether the hook is correct.
        rec = sae.decode(z)
        sel = km
        recon_num += ((hs - rec) ** 2 * sel).sum().item()
        mu = (hs * sel).sum(dim=(0, 1), keepdim=True) / sel.sum().clamp(min=1)
        recon_den += ((hs - mu) ** 2 * sel).sum().item()
        l0_num += ((z > 0).float().sum(-1) * keep.float()).sum().item()
        l0_den += keep.sum().item()

    elapsed = time.time() - t0
    ev = 1.0 - recon_num / recon_den
    mean_l0 = l0_num / l0_den
    print(f"elapsed {elapsed/60:.1f} min | SAE explained variance {ev:.4f} | mean L0 {mean_l0:.1f}")

    # ---- high-activity filter --------------------------------------------
    prim = cfg["aggregation"]["primary"]
    firing_rate = (acc[prim] > 0).mean(axis=0)                       # Pr(X_j > 0)
    energy = np.abs(acc[prim]).mean(axis=0)                          # reference's statistic
    lf = cfg["latent_filter"]
    surv = np.flatnonzero(firing_rate >= lf["min_firing_rate"])
    print(f"latents with firing rate >= {lf['min_firing_rate']}: {surv.size}")
    if surv.size > lf["max_retained"]:
        surv = surv[np.argsort(-firing_rate[surv], kind="stable")[: lf["max_retained"]]]
    retained = np.sort(surv)
    print(f"retained p  : {retained.size}")

    payload = {
        "labels": labels,
        "sentence_ids": sent_ids,
        "n_tokens": n_tokens,
        "retained_idx": retained,
        "firing_rate_all": firing_rate.astype(np.float32),
        "energy_all": energy.astype(np.float32),
        "config_json": np.asarray(json.dumps(cache_config(cfg))),
        "config_hash": np.asarray(config_hash(cache_config(cfg))),
        "sae_explained_variance": np.asarray(ev),
        "sae_mean_l0": np.asarray(mean_l0),
        "master_seed": np.asarray(cfg["master_seed"]),
    }
    for a in AGGS:
        payload[f"X_{a}"] = acc[a][:, retained]

    np.savez(out, **payload)
    print(f"wrote {out}  ({out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
