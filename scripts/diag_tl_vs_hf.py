import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from sae_lens import SAE
torch.set_grad_enabled(False)
TXT = "a stirring , funny and finally transporting re-imagining of beauty and the beast ."

tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
hf = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", dtype=torch.float32,
                                          attn_implementation="eager").to("cuda").eval()
b = tok([TXT], return_tensors="pt").to("cuda")
hs = hf(**b, output_hidden_states=True).hidden_states[21]
print("HF ids   :", b["input_ids"][0].tolist()[:8])
print("HF  resid_post[20] norm/token:", [round(float(x),1) for x in hs.norm(dim=-1)[0][:8]])
del hf; torch.cuda.empty_cache()

tl = HookedTransformer.from_pretrained("google/gemma-2-2b", device="cuda", dtype=torch.float32)
toks = tl.to_tokens([TXT])
print("TL ids   :", toks[0].tolist()[:8])
_, cache = tl.run_with_cache(toks, names_filter=lambda n: n == "blocks.20.hook_resid_post")
tlh = cache["blocks.20.hook_resid_post"]
print("TL  resid_post[20] norm/token:", [round(float(x),1) for x in tlh.norm(dim=-1)[0][:8]])
n = min(hs.shape[1], tlh.shape[1])
print("max abs diff HF vs TL:", float((hs[:, :n] - tlh[:, :n]).abs().max()))
print("ratio of norms TL/HF :", float(tlh[:, :n].norm() / hs[:, :n].norm()))

loaded = SAE.from_pretrained(release="gemma-scope-2b-pt-res-canonical",
                             sae_id="layer_20/width_16k/canonical", device="cuda")
sae = (loaded[0] if isinstance(loaded,(tuple,list)) else loaded).to(torch.float32).eval()
for name, h in [("HF", hs[:, :n]), ("TL", tlh[:, :n])]:
    z = sae.encode(h); rec = sae.decode(h*0 + z@torch.zeros(0,0,device='cuda') if False else z)
    err = ((h-rec)**2).sum(); den = ((h-h.mean((0,1),keepdim=True))**2).sum()
    print(f"{name}: L0={(z>0).float().sum(-1).mean():.1f}  EV={1-err/den:.4f}")
