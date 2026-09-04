"""Empirically verify how the reference pipeline aggregates tokens.

The reference takes cache[hook][:, -1, :] after model.to_tokens(list_of_texts).
If TransformerLens right-pads, position -1 is a PAD token for every sentence
shorter than the batch maximum. This script checks that on their exact model.
"""
import torch
from transformer_lens import HookedTransformer

m = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device="cpu")
texts = ["a stirring film .", "this is a much longer sentence used to set the batch maximum length ."]
tok = m.to_tokens(texts)
print("padding_side           :", m.tokenizer.padding_side)
print("tokens shape           :", tuple(tok.shape))
for i, t in enumerate(texts):
    ids = tok[i].tolist()
    print(f"  row {i}: last id={ids[-1]} -> {m.tokenizer.decode([ids[-1]])!r}  full={m.to_str_tokens(tok[i])}")
pad_id = m.tokenizer.pad_token_id
n_pad_last = int((tok[:, -1] == pad_id).sum())
print(f"pad_token_id           : {pad_id}")
print(f"rows whose [-1] is PAD : {n_pad_last}/{len(texts)}")
