"""Decision rule TAU (concept_amendment_1): from the full layer-12 census, the tau minimising over
widths the maximum |log(median sum-of-children firing / parent firing)|; ties to the larger tau."""
import json
import math

import _paths  # noqa: F401
from cseeds import ROOT

c = json.loads((ROOT / "concept/results/census_L12.json").read_text())
score = {}
for tau in c["taus"]:
    r = [c["per_width"][k]["tau"][str(tau)]["sumfire_over_parent_median"] for k in c["widths"] if not k.endswith("_16k")]
    score[tau] = max(abs(math.log(max(v, 1e-9))) for v in r)
best = min(score.values())
tau = max(t for t, s in score.items() if abs(s - best) < 1e-12)
out = {"rule": "TAU", "widths": c["widths"], "max_abs_log_ratio": score, "tau": tau}
(ROOT / "concept/results/tau.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
