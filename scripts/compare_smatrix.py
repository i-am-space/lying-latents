"""Does a different S-matrix repair the near-copy problem? (pre-registered secondary)

Bears on power only, not validity: every second-order Gaussian knockoff is continuous,
so no choice of S can restore the zero atom. Run at p=512, the reference's dimension,
because MVR does not converge at p=2048 in reasonable time.
"""
import sys, time
import numpy as np
sys.path.insert(0, "src")
from knockoff_audit import standardise, estimate_cov
from knockpy import smatrix

d = np.load("data/cache/d33d210c5acb.npz")
X = d["X_mean"].astype(np.float64)
fr = d["firing_rate_all"][d["retained_idx"]]
sel = np.argsort(-fr)[:512]                       # top 512 by firing rate
Z, _, _ = standardise(X[:, sel])
Sig = estimate_cov(Z, "ledoit_wolf")
ev = np.linalg.eigvalsh(Sig)
print(f"p=512  cond={ev[-1]/ev[0]:.4g}  lam_min={ev[0]:.5f}", flush=True)
for m in ("equicorrelated", "mvr", "sdp"):
    t = time.time()
    try:
        s = np.diag(smatrix.compute_smatrix(Sig, method=m))
        print(f"{m:15s} s: mean={s.mean():.4f} median={np.median(s):.4f} "
              f"min={s.min():.4f} max={s.max():.4f} -> mean corr(X,Xk)={1-s.mean():.4f} "
              f"[{time.time()-t:.0f}s]", flush=True)
    except Exception as e:
        print(f"{m:15s} FAILED {type(e).__name__}: {str(e)[:100]} [{time.time()-t:.0f}s]", flush=True)
