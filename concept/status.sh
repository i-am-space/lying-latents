#!/bin/bash
# Progress at a glance:  bash concept/status.sh
C=/home/vishalrao/lying-latents/data/cache/concept; LOG=$C/logs; cd /home/vishalrao/lying-latents
echo "== $(date '+%F %T') =="
echo "-- pipeline stages (newest last)"; tail -n 6 $LOG/STATUS 2>/dev/null || echo "   (orchestrator not started yet)"
echo "-- SAE downloads (link ~5 MB/s)"
for p in $(ls -d $C/sae/layer_*/width_*/* 2>/dev/null); do
  k=$(echo $p | sed -E 's|.*layer_([0-9]+)/width_([0-9a-z]+)/.*|L\1_\2|')
  if [ -f $p/.verified ]; then s=verified
  elif [ -d $p/parts ]; then s="$(ls $p/parts | grep -vc tmp) chunks"; else s=queued; fi
  printf "   %-9s %s\n" $k "$s"
done
echo "-- latent caches: $(ls $C/lat_*.json 2>/dev/null | sed -E 's|.*lat_(.*)\.json|\1|' | tr '\n' ' ')"
echo "-- toy:  $(grep -cE '^(zinf|gauss)' $LOG/toy.log 2>/dev/null)/24 S solves; $(grep -c 'reps in' $LOG/toy.log 2>/dev/null)/36 fit blocks $(grep -q 'TOY DONE' $LOG/toy.log 2>/dev/null && echo '— DONE')"
for f in $(ls $C/ckpt_*.npz 2>/dev/null); do
  n=$(python3 -c "import numpy as np;print(int(np.load('$f')['reps_done']))" 2>/dev/null); echo "   ckpt $(basename $f .npz): $n replicates"; done | sort | sed 's/^/-- /' | head -30
echo "-- real sweeps done: $(ls $C/real_*.npz 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
echo "-- GPUs"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/   /'
echo "-- results: $(ls concept/results 2>/dev/null | tr '\n' ' ')"
