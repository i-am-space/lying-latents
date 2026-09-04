import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
torch.set_grad_enabled(False)

tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", dtype=torch.float32,
                                             attn_implementation="eager").to("cuda").eval()
loaded = SAE.from_pretrained(release="gemma-scope-2b-pt-res-canonical",
                             sae_id="layer_20/width_16k/canonical", device="cuda")
sae = (loaded[0] if isinstance(loaded,(tuple,list)) else loaded).to(torch.float32).eval()

cfgm = sae.cfg
print("SAE cfg fields:")
for k in ("normalize_activations","apply_b_dec_to_input","d_in","d_sae","dtype"):
    print("  ", k, getattr(cfgm, k, "<absent>"))
md = getattr(cfgm,"metadata",None)
for k in ("hook_name","hook_layer","normalize_activations","prepend_bos","model_name"):
    print("  meta", k, getattr(md,k,"<absent>"))

texts = ["a stirring , funny and finally transporting re-imagining of beauty and the beast .",
         "it 's so laddish and juvenile , only teenage boys could possibly find it funny ."]
b = tok(texts, return_tensors="pt", padding=True).to("cuda")
out = model(**b, output_hidden_states=True)

def ev(h, rec, mask):
    m = mask.unsqueeze(-1)
    num = (((h-rec)**2)*m).sum()
    mu  = (h*m).sum((0,1),keepdim=True)/m.sum()
    den = (((h-mu)**2)*m).sum()
    return (1-num/den).item()

am = b["attention_mask"].bool()
nobos = am.clone(); nobos[:,0] = False
for L in (19,20,21,22):
    h = out.hidden_states[L]
    rec = sae.decode(sae.encode(h))
    print(f"hidden_states[{L}]  EV(all)={ev(h,rec,am):8.4f}  EV(no BOS)={ev(h,rec,nobos):8.4f}  "
          f"|h| mean={h.norm(dim=-1).mean():.1f}  BOS|h|={h[0,0].norm():.1f}")
