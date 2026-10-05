#!/bin/bash
# End-to-end orchestrator for Steps 3-8 (detached; survives hang-up). Resumable: every stage skips
# work already on disk (checkpoints per width and replicate).
cd "$(dirname "$0")/../.."
C=data/cache/concept; LOG=$C/logs; R=concept/results
stamp() { echo "$(date '+%F %T') $*" | tee -a $LOG/STATUS; }
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
L12="L12_16k L12_32k L12_65k L12_131k L12_262k L12_524k L12_1m"; L20="L20_16k L20_65k"

stamp "waiting for all layer-12 caches"
for k in $L12 $L20; do until [ -f $C/lat_$k.json ]; do sleep 30; done; done
stamp "caches complete"

if [ ! -f $R/tau.json ]; then
  stamp "census L12 + L20"
  CUDA_VISIBLE_DEVICES=0 python3 -u concept/src/family_census.py --layer 12 > $LOG/census_L12.log 2>&1
  CUDA_VISIBLE_DEVICES=0 python3 -u concept/src/family_census.py --layer 20 > $LOG/census_L20.log 2>&1
  grep -q '"verdict": "STOP' $R/census_L12.json && { stamp "GATE C0 FAILED: stop"; exit 1; }
  python3 concept/src/freeze_tau.py > $LOG/tau.log 2>&1
fi
TAU=$(python3 -c "import json;print(json.load(open('$R/tau.json'))['tau'])"); stamp "tau frozen at $TAU"

for layer in 20 12; do
  have=$(python3 -c "import json;print(json.load(open('$R/candidates_L$layer.json'))['tau'])" 2>/dev/null)
  if [ "$have" != "$TAU" ]; then
    stamp "candidates L$layer (tau $TAU)"; rm -f $C/S_*_L${layer}_*.npy $C/cand_L${layer}_*.npz
    OMP_NUM_THREADS=2 python3 -u concept/src/build_candidates.py --layer $layer --tau $TAU --workers 8 > $LOG/candidates_L$layer.log 2>&1 || { stamp "candidates failed"; exit 1; }
  fi
done
stamp "candidates and S solves done"

if [ ! -f $R/early_gate.json ]; then
  stamp "Step 6 gate"
  gpu_pair "concept/src/early_gate.py run --keys" gate $L12
  python3 concept/src/early_gate.py analyse > $LOG/gate_analyse.log 2>&1
fi
python3 -c "import json,sys;sys.exit(0 if json.load(open('$R/early_gate.json'))['G1']['PASS'] else 1)" || { stamp "GATE G1 FAILED: stop (thesis wrong at this layer)"; exit 2; }
stamp "gate G1 passed"

stamp "Step 7 planted benchmark"
gpu_pair "concept/src/planted_concept.py --layer 12 --tag full --keys" planted_L12 $L12
gpu_pair "concept/src/planted_concept.py --layer 20 --tag full --keys" planted_L20 $L20
python3 concept/src/report_concept.py --tag full > $LOG/report.log 2>&1
stamp "Step 8 real sweep"
gpu_pair "concept/src/width_sweep_real.py run --layer 12 --keys" real_L12 $L12
gpu_pair "concept/src/width_sweep_real.py run --layer 20 --keys" real_L20 $L20
python3 concept/src/width_sweep_real.py analyse > $LOG/real_analyse.log 2>&1
stamp "ALL DONE"
