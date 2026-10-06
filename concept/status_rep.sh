#!/bin/bash
# Replication progress at a glance:  bash concept/status_rep.sh
C=/home/vishalrao/lying-latents/data/cache/concept; LOG=$C/logs; cd /home/vishalrao/lying-latents
echo "== $(date '+%F %T') =="
echo "-- stages"; tail -n 6 $LOG/STATUS_REP 2>/dev/null | sed 's/^/   /'
echo "-- downloads"; for p in layer_19/width_1m layer_5/width_16k layer_5/width_65k layer_5/width_1m; do
  d=$(ls -d $C/sae/$p/* 2>/dev/null | head -1)
  if [ -z "$d" ]; then s=queued; elif [ -f $d/.verified ]; then s=verified; elif [ -d $d/parts ]; then s="$(ls $d/parts | grep -vc tmp) chunks"; else s=queued; fi
  printf "   %-18s %s\n" $p "$s"; done
echo "-- planted replicates (tag rep)"
for f in $(ls $C/ckpt_rep_*.npz 2>/dev/null); do printf "   %-10s %s/30\n" $(basename $f .npz | sed 's/ckpt_rep_//') $(python3 -c "import numpy as np;print(int(np.load('$f')['reps_done']))"); done
echo "-- real sweeps done: $(ls $C/real_rep_*.npz 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/real_rep_//;s/.npz//' | tr '\n' ' ')"
echo "-- outcome so far: $(python3 -c "import json;print(json.load(open('concept/results/replication.json')).get('outcome','(not yet)'))" 2>/dev/null || echo '(no report yet)')"
