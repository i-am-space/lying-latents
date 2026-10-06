#!/bin/bash
# Replication orchestrator (concept_amendment_3). Detached, resumable (checkpoints per width and replicate).
cd "$(dirname "$0")/../.."
C=data/cache/concept; LOG=$C/logs; R=concept/results
stamp() { echo "$(date '+%F %T') $*" | tee -a $LOG/STATUS_REP; }
gpu_pair() {  # run "$1" with keys split across GPU 0 and 1; wait for both
  local script=$1; shift; local tag=$1; shift; local a=() b=() i=0
  for k in "$@"; do if (( i % 2 == 0 )); then a+=("$k"); else b+=("$k"); fi; i=$((i+1)); done
  for g in 0 1; do
    local ks; [ $g = 0 ] && ks="${a[*]}" || ks="${b[*]}"
    [ -z "$ks" ] && continue
    ( for t in $(seq 30); do
        CUDA_VISIBLE_DEVICES=$g python3 -u $script $ks >> $LOG/${tag}_gpu$g.log 2>&1 && break
        echo "retry $t" >> $LOG/${tag}_gpu$g.log; sleep 20; done ) &
  done; wait
}
ARMS="latent group group_sum group_lasso mkf_c1 mkf_c1.93"
PL="concept/src/planted_concept.py --tag rep --designs A B C D --arms $ARMS"
REAL="concept/src/width_sweep_real.py run --tag rep --arms latent group group_sum group_lasso"
keys() { python3 -c "import sys;sys.path.insert(0,'concept/src');from cseeds import load_concept_config,layer_keys;print(' '.join(layer_keys(load_concept_config()['concept'],$1)))"; }

pgrep -f "concept/scripts/cache_queue.py" >/dev/null || { setsid nohup python3 -u concept/scripts/cache_queue.py >> $LOG/queue.log 2>&1 < /dev/null & }
if [ ! -f $R/toy_glasso.json ]; then stamp "toy: group lasso validity (T1G)"
  for t in $(seq 10); do CUDA_VISIBLE_DEVICES=1 python3 -u concept/src/toy_sumstat.py --stat glasso > $LOG/toy_glasso.log 2>&1 && break; sleep 20; done; fi

stamp "layer 12 rerun (designs A-D)"
gpu_pair "$PL --layer 12 --keys" rep_L12 $(keys 12)

for layer in 19 5; do
  K=$(keys $layer)
  stamp "layer $layer: waiting for caches"
  for k in $K; do until [ -f $C/lat_$k.json ]; do sleep 30; done; done
  if [ ! -f $R/candidates_L$layer.json ]; then
    stamp "layer $layer: census + candidates (tau 0.3)"
    for t in $(seq 10); do CUDA_VISIBLE_DEVICES=0 python3 -u concept/src/family_census.py --layer $layer > $LOG/census_L$layer.log 2>&1 && break; sleep 20; done
    OMP_NUM_THREADS=2 python3 -u concept/src/build_candidates.py --layer $layer --tau 0.3 --no-cluster --workers 6 > $LOG/candidates_L$layer.log 2>&1 || { stamp "candidates L$layer failed"; exit 1; }
  fi
  stamp "layer $layer: planted benchmark"
  gpu_pair "$PL --layer $layer --keys" rep_L$layer $K
  stamp "layer $layer: real sweep"
  gpu_pair "$REAL --layer $layer --keys" realrep_L$layer $K
  python3 concept/src/report_replication.py > $LOG/report_rep.log 2>&1
  stamp "layer $layer done (interim report written)"
done
stamp "layer 12 real sweep"
gpu_pair "$REAL --layer 12 --keys" realrep_L12 $(keys 12)
python3 concept/src/report_replication.py > $LOG/report_rep.log 2>&1
stamp "ALL DONE"
