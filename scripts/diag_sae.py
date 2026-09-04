import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", dtype=torch.float32,
                                             attn_implementation="eager").to("cuda").eval()
loaded = SAE.from_pretrained(release="gemma-scope-2b-pt-res-canonical",
                             sae_id="layer_20/width_16k/canonical", device="cuda")
sae = (loaded[0] if isinstance(loaded,(tuple,list)) else loaded).to(torch.float32).eval()
print("type", type(sae).__name__)
for n,p in sae.named_parameters():
    print(f"  {n:12s} {tuple(p.shape)} mean={p.mean():+.4e} std={p.std():.4e} absmax={p.abs().max():.4e}")
for n,bf in sae.named_buffers():
    print(f"  buf {n:10s} {tuple(bf.shape)} mean={bf.mean():+.4e}")

b = tok(["a stirring , funny and finally transporting re-imagining of beauty and the beast ."],
        return_tensors="pt").to("cuda")
h = model(**b, output_hidden_states=True).hidden_states[21]
z = sae.encode(h); rec = sae.decode(z)
print(f"\nh   norm/token mean {h.norm(dim=-1).mean():.2f}")
print(f"rec norm/token mean {rec.norm(dim=-1).mean():.2f}")
print(f"err norm/token mean {(h-rec).norm(dim=-1).mean():.2f}")
print(f"L0 (nonzero latents/token) {(z>0).float().sum(-1).mean():.1f}")
print(f"z max {z.max():.3f}  z mean-nonzero {z[z>0].mean():.3f}")
print(f"b_dec norm {sae.b_dec.norm():.2f}")
# What if the SAE wants activations scaled?
for sc in (1.0,):
    pass
