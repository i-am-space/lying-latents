"""Step 2b sanity check: layer 20, 16k, restricted to the main cache's retained 2048 latents, against
the main cache's X_mean. The plan's criterion is max |diff| < 1e-4. Also reported: relative error,
and how many discrepancies are JumpReLU threshold flips (|diff| * n_tokens ~= threshold), so a
failure of the absolute criterion can be attributed rather than waved through."""
import json

import numpy as np
import scipy.sparse as sp

import _paths  # noqa: F401
from common import cache_path
from cseeds import ROOT, load_concept_config

cfg = load_concept_config()
cdir = ROOT / cfg["concept"]["cache_dir"]
main = np.load(cache_path(cfg))
ret = main["retained_idx"]
Xm = main["X_mean"].astype(np.float64)
Xn = sp.load_npz(cdir / "lat_L20_16k.csr.npz")[:, ret].toarray().astype(np.float64)
ntok = np.load(cdir / "resid_meta.npz")["n_tokens"]
thr = np.load(cdir / "sae" / cfg["concept"]["saes"]["L20_16k"]["path"] / "params.npz")["threshold"][ret]
d = np.abs(Xn - Xm)
over = d > 1e-4
r, c = np.nonzero(over)
ratio = d[r, c] * ntok[r] / thr[c]
flip = (ratio > 0.5)                                   # one or more token gates changed state
rel = d / np.maximum(np.maximum(np.abs(Xm), np.abs(Xn)), 1e-12)
nonflip_over = over.copy(); nonflip_over[r[flip], c[flip]] = False
res = {
    "criterion": "max |diff| < 1e-4 (plan)",
    "PASS_plan_criterion": bool(d.max() < 1e-4),
    "max_abs_diff": float(d.max()),
    "n_entries": int(d.size),
    "n_entries_over_1e-4": int(over.sum()),
    "n_threshold_flips": int(flip.sum()),
    "n_zero_pattern_mismatch": int(((Xn == 0) != (Xm == 0)).sum()),
    "token_counts_identical": bool(np.array_equal(ntok, main["n_tokens"])),
    "max_rel_diff_excluding_flips": float(rel[nonflip_over].max()) if nonflip_over.any() else 0.0,
    "median_rel_diff_over_1e-4_excluding_flips": float(np.median(rel[nonflip_over])) if nonflip_over.any() else 0.0,
    "max_abs_diff_excluding_flips": float(d[nonflip_over].max()) if nonflip_over.any() else 0.0,
    "frac_entries_rel_diff_le_1e-5": float((rel[(Xm != 0) | (Xn != 0)] <= 1e-5).mean()),
}
(ROOT / "concept" / "results" / "check_reproduction.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
