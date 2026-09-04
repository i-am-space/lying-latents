import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
torch.set_grad_enabled(False)
TXT = "a stirring , funny and finally transporting re-imagining of beauty and the beast ."
tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
hf = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", dtype=torch.float32,
                                          attn_implementation="eager").to("cuda").eval()
b = tok([TXT], return_tensors="pt").to("cuda")
h = hf(**b, output_hidden_states=True).hidden_states[21][0]      # (T, 2304)
loaded = SAE.from_pretrained(release="gemma-scope-2b-pt-res-canonical",
                             sae_id="layer_20/width_16k/canonical", device="cuda")
sae = (loaded[0] if isinstance(loaded,(tuple,list)) else loaded).to(torch.float32).eval()

z = sae.encode(h); rec = sae.decode(z)
# manual JumpReLU per official Gemma Scope code
pre = h @ sae.W_enc + sae.b_enc
zman = pre * (pre > sae.threshold)
recman = zman @ sae.W_dec + sae.b_dec
print("encode matches manual JumpReLU:", torch.allclose(z, zman, atol=1e-3),
      " maxdiff", float((z-zman).abs().max()))
print("decode matches manual        :", torch.allclose(rec, recman, atol=1e-2),
      " maxdiff", float((rec-recman).abs().max()))
strs = tok.convert_ids_to_tokens(b["input_ids"][0])
print(f"\n{'tok':>14} {'|h|':>8} {'L0':>6} {'cos(h,rec)':>10} {'|rec|':>8} {'relerr':>7}")
for t in range(h.shape[0]):
    cs = torch.nn.functional.cosine_similarity(h[t], rec[t], dim=0)
    rel = (h[t]-rec[t]).norm()/h[t].norm()
    print(f"{strs[t]:>14} {h[t].norm():8.1f} {int((z[t]>0).sum()):6d} {cs:10.4f} {rec[t].norm():8.1f} {rel:7.3f}")
