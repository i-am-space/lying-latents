"""Shape of the positive part of cached SAE latents, per model and dataset (hurdle size-model fit)."""
import sys, json
import numpy as np
from scipy import stats
sys.path[:0] = ["src", "src/saebench"]
from common import load_config
from sbutil import cache_path
DS = ["Helsinki-NLP/europarl", "LabHC/bias_in_bios_class_set1", "canrager/amazon_reviews_mcauley_1and5"]
out = {}
for tag, cfgp in [("gemma2", "config/default.yaml"), ("gemma3", "config/models/gemma3_1b.yaml"), ("pythia", "config/models/pythia70m.yaml")]:
    cfg = load_config(cfgp)
    for d in DS:
        z = np.load(cache_path(cfg, d), allow_pickle=True)
        key = [k for k in z.files if k.startswith("X")][0]
        X = z[key].astype(np.float64)
        tiny, skew, ks, floor = [], [], [], []
        for j in range(X.shape[1]):
            x = X[:, j][X[:, j] > 0]
            if x.size < 50:
                continue
            lx = np.log(x)
            tiny.append((x < 0.01 * x.max()).mean())
            skew.append(stats.skew(lx))
            ks.append(stats.kstest((lx - lx.mean()) / lx.std(), "norm").statistic)
            floor.append(x.min() / np.median(x))
        r = {"key": key, "p": X.shape[1], "median_frac_below_1pct_of_max": float(np.median(tiny)),
             "median_skew_log_x": float(np.median(skew)), "median_KS_lognormal": float(np.median(ks)),
             "median_min_over_median": float(np.median(floor))}
        out[f"{tag} {d}"] = r
        print(f"{tag:7s} {d.split('/')[1][:24]:24s} below1%max {r['median_frac_below_1pct_of_max']:.3f}  "
              f"skew(log x) {r['median_skew_log_x']:+.2f}  KS vs lognormal {r['median_KS_lognormal']:.3f}  "
              f"min/median {r['median_min_over_median']:.4f}", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=2)
