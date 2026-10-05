"""Step 2a — residual-stream cache of CONTENT tokens only, for several layers in one pass.

Same tokenisation, batching and masking as src/cache_activations.py (BOS/EOS/pad excluded,
length-sorted batches of 64, max 128 tokens), so the hidden states are those the main cache
encoded. Stored fp32, not fp16: Step 2b must reproduce the main cache to 1e-4, and fp16
rounding of norm-350 residuals would flip JumpReLU threshold crossings.

Output per layer:  resid_L{layer}.f32  memmap (total_tokens, 2304), sentence i occupying rows
offsets[i]:offsets[i+1];  resid_meta.npz  offsets, labels, sentence_ids, n_tokens.

Usage: python concept/src/cache_resid.py --layers 12 20
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import _paths  # noqa: F401
from cseeds import ROOT, load_concept_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[12, 20])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=40)
    args = ap.parse_args()
    cfg = load_concept_config()
    out = ROOT / cfg["concept"]["cache_dir"]
    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    dev = args.device

    dc, mc = cfg["data"], cfg["model"]
    ds = load_dataset(dc["dataset"], dc["dataset_config"], split=dc["split"])
    if args.limit:
        ds = ds.select(range(args.limit))
    sents = list(ds[dc["text_column"]])
    n = len(sents)
    tok = AutoTokenizer.from_pretrained(mc["name"])
    special = {i for i in (tok.pad_token_id, tok.bos_token_id, tok.eos_token_id) if i is not None}
    msl, bs = mc["max_seq_len"], mc["batch_size"]

    # content-token count per sentence, from the tokenizer alone (same rule as the main cache)
    enc = [tok(s, truncation=True, max_length=msl)["input_ids"] for s in sents]
    n_tok = np.array([sum(i not in special for i in e) or len(e) for e in enc], dtype=np.int64)
    offsets = np.zeros(n + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(n_tok)
    T = int(offsets[-1])
    print(f"sentences {n}  content tokens {T}  ({T * 2304 * 4 / 1e9:.1f} GB per layer)", flush=True)

    mm = {L: np.lib.format.open_memmap(out / f"resid_L{L}.npy", mode="w+", dtype=np.float32,
                                       shape=(T, 2304)) for L in args.layers}
    model = AutoModelForCausalLM.from_pretrained(mc["name"], dtype=getattr(torch, mc["dtype"]),
                                                 attn_implementation=mc["attn_implementation"]).to(dev).eval()
    # Only layers <= max(layers) are needed. Keep one extra block: HF appends the FINAL-NORM
    # output as the last hidden_states entry, so hidden_states[L + 1] must not be the last one.
    model.model.layers = model.model.layers[:max(args.layers) + 2]
    enc_len = np.asarray([len(tok(s)["input_ids"]) for s in sents])
    order = np.argsort(enc_len, kind="stable")
    t0 = time.time()
    for start in tqdm(range(0, n, bs), desc="resid", mininterval=30):
        idx = order[start:start + bs]
        b = tok([sents[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=msl)
        ids, am = b["input_ids"].to(dev), b["attention_mask"].to(dev)
        hs_all = model(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states
        keep = am.bool()
        for sid in special:
            keep &= ids != sid
        empty = ~keep.any(dim=1)
        if empty.any():
            keep[empty] = am.bool()[empty]
        keep_np = keep.cpu().numpy()
        for L in args.layers:
            hs = hs_all[L + 1].float().cpu().numpy()
            for r, i in enumerate(idx):
                rows = hs[r][keep_np[r]]
                assert rows.shape[0] == n_tok[i], (i, rows.shape[0], n_tok[i])
                mm[L][offsets[i]:offsets[i + 1]] = rows
    for L in args.layers:
        mm[L].flush()
    np.savez(out / "resid_meta.npz", offsets=offsets, n_tokens=n_tok,
             labels=np.asarray(ds[dc["label_column"]], dtype=np.int8),
             sentence_ids=np.asarray(ds["idx"], dtype=np.int64), layers=np.asarray(args.layers))
    (out / "resid.done").write_text(json.dumps({"layers": args.layers, "n": n, "T": T, "device": dev,
                                                "minutes": (time.time() - t0) / 60}))
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
