"""Step 2b — sparse per-sentence latent cache for one (layer, width) SAE.

Encodes the content-token residual cache (Step 2a) in token chunks aligned to sentence
boundaries — never materialising (batch, T, d_sae) — and mean-aggregates per sentence exactly
as src/cache_activations.py does (sum over content tokens / count). JumpReLU as Gemma Scope
defines it: pre = x W_enc + b_enc, z = pre * (pre > threshold).

Writes  lat_{key}.npz   CSR (n x d_sae, float32) + firing_rate_all, labels, sentence_ids,
                        mean L0 on content tokens, explained variance, SAE path
        dec_{key}.npy   row-normalised W_dec, fp16 (Step 3)
Usage:  python concept/src/cache_sparse_latents.py --key L12_65k
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import scipy.sparse as sp
import torch

import _paths  # noqa: F401
from cseeds import ROOT, load_concept_config
from gpu import init_cuda


def load_params(path, dev):
    z = np.load(path)
    P = {k: torch.from_numpy(np.ascontiguousarray(z[k])).to(dev, torch.float32)
         for k in ("W_enc", "b_enc", "threshold", "b_dec")}
    return P, z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = load_concept_config()
    cc = cfg["concept"]
    sae = cc["saes"][args.key]
    cdir = ROOT / cc["cache_dir"]
    dev = init_cuda(args.device)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False          # fp32 encode: must reproduce the main cache

    meta = np.load(cdir / "resid_meta.npz")
    offsets = meta["offsets"]
    n = offsets.size - 1
    R = np.load(cdir / f"resid_L{sae['layer']}.npy", mmap_mode="r")
    t0 = time.time()
    P, raw = load_params(cdir / "sae" / sae["path"] / "params.npz", dev)
    d_sae = P["W_enc"].shape[1]
    assert d_sae == sae["width"], (d_sae, sae["width"])
    print(f"[{args.key}] d_sae={d_sae}  loaded in {time.time() - t0:.0f}s", flush=True)

    # chunks of whole sentences, about `budget` tokens each (smaller at 1M to bound memory)
    # <= 2 GB per fp32 activation buffer: the GPUs are shared (another user's jobs hold ~20 GB each)
    budget = int(min(cc["encode_chunk_tokens"], 5e8 / d_sae))
    bounds = [0]
    while bounds[-1] < n:
        s = bounds[-1]
        e = int(np.searchsorted(offsets, offsets[s] + budget, side="right")) - 1
        bounds.append(max(e, s + 1) if e < n else n)

    # decoder only feeds the explained-variance diagnostic: fp16 on the device (fp32 accumulate) so the
    # 1M SAE fits beside other users' jobs; fp32 for widths below 524k, as originally run
    dec_dtype = torch.float16 if d_sae >= 524288 else torch.float32
    W_dec_np = np.ascontiguousarray(raw["W_dec"])
    W_dec = torch.from_numpy(W_dec_np).to(dev, dec_dtype)
    rows, cols, vals = [], [], []
    fire_tok = 0.0; ntok = 0; sse = 0.0; ss_sum = torch.zeros(R.shape[1], dtype=torch.float64, device=dev)
    ss_sq = 0.0
    for bi in range(len(bounds) - 1):
        s, e = bounds[bi], bounds[bi + 1]
        a, b = int(offsets[s]), int(offsets[e])
        x = torch.from_numpy(np.asarray(R[a:b])).to(dev)
        pre = x @ P["W_enc"]
        pre += P["b_enc"]
        pre.mul_(pre > P["threshold"])                      # z, in place
        fire_tok += float(torch.count_nonzero(pre)); ntok += b - a
        rec = (pre.to(dec_dtype) @ W_dec).float() + P["b_dec"]
        sse += float(((x - rec) ** 2).sum()); del rec
        ss_sum += x.double().sum(0); ss_sq += float((x.double() ** 2).sum())
        # per-sentence mean over content tokens: segment-sum via a (sentences x tokens) indicator
        seg = torch.from_numpy(np.repeat(np.arange(e - s), np.diff(offsets[s:e + 1]))).to(dev)
        M = torch.zeros(e - s, b - a, device=dev)
        M[seg, torch.arange(b - a, device=dev)] = 1.0
        mean = (M @ pre) / torch.from_numpy(np.diff(offsets[s:e + 1]).astype(np.float32)).to(dev)[:, None]
        nz = mean.nonzero(as_tuple=True)
        rows.append((nz[0] + s).cpu().numpy().astype(np.int32)); cols.append(nz[1].cpu().numpy().astype(np.int32))
        vals.append(mean[nz].cpu().numpy())
        del pre, mean, M, x
        if bi % 50 == 0:
            print(f"  chunk {bi}/{len(bounds) - 1}  {time.time() - t0:.0f}s", flush=True)

    X = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, d_sae), dtype=np.float32)
    X.sort_indices()
    T = ntok
    mu = (ss_sum / T)
    ev = 1.0 - sse / (ss_sq - T * float((mu ** 2).sum()))
    fr = np.asarray((X > 0).mean(axis=0)).ravel().astype(np.float32)
    l0 = fire_tok / T
    out = cdir / f"lat_{args.key}.npz"
    sp.save_npz(out.with_suffix(".csr.npz"), X, compressed=False)
    np.savez(out, firing_rate_all=fr, labels=meta["labels"], sentence_ids=meta["sentence_ids"],
             n_tokens=meta["n_tokens"], mean_l0=np.asarray(l0), explained_variance=np.asarray(ev),
             sae_path=np.asarray(sae["path"]), layer=np.asarray(sae["layer"]), width=np.asarray(d_sae))
    del W_dec, P; torch.cuda.empty_cache() if dev.type == "cuda" else None
    dec = np.lib.format.open_memmap(cdir / f"dec_{args.key}.npy", mode="w+", dtype=np.float16, shape=W_dec_np.shape)
    for s0 in range(0, W_dec_np.shape[0], 65536):                     # row-normalise on the CPU, in chunks
        blk = W_dec_np[s0:s0 + 65536].astype(np.float32)
        dec[s0:s0 + 65536] = (blk / np.maximum(np.linalg.norm(blk, axis=1, keepdims=True), 1e-12)).astype(np.float16)
    dec.flush()
    info = {"key": args.key, "n": n, "d_sae": d_sae, "nnz": int(X.nnz), "nnz_per_sentence": X.nnz / n,
            "mean_l0": l0, "explained_variance": ev, "minutes": (time.time() - t0) / 60,
            "chunk_tokens": budget, "ev_decoder_dtype": str(dec_dtype).replace("torch.", ""),
            "n_firing_ge_0.1pct": int((fr >= 0.001).sum()), "n_firing_ge_1pct": int((fr >= 0.01).sum())}
    (cdir / f"lat_{args.key}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
