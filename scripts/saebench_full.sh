#!/usr/bin/env bash
# SAEBench full runs (saebench_amendment_1). Waits for the prep queue to finish if its PID is given,
# then runs Stage 1 (GPU) and Stage 2 (CPU) side by side and aggregates both. Stage 2 starts only if
# its debug run (from scripts/saebench_prep.sh) produced a result. Both stages skip datasets without
# a cache and resume from their per-task / per-dataset files if restarted.
#
# Usage (from the repo root, after the pre-registration commit is pushed):
#   nohup bash scripts/saebench_full.sh [PREP_PID] > logs/saebench_full.log 2>&1 &
# Env: GPU (default 1)
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
GPU="${GPU:-1}"
CFG=config/default.yaml
stamp() { echo "[$(date '+%F %T')] $*"; }

if [ -n "${1:-}" ]; then
  stamp "waiting for prep job (PID $1) to finish"
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi
stamp "caches present: $(ls data/saebench/cache 2>/dev/null | wc -l) of 8"

stamp "=== Stage 1 (GPU $GPU) -> logs/saebench_stage1.log ==="
CUDA_VISIBLE_DEVICES=$GPU python src/saebench/stage1.py --config $CFG --device cuda > logs/saebench_stage1.log 2>&1 &
S1=$!

if ls results/saebench/stage2_debug/*.json >/dev/null 2>&1; then
  stamp "=== Stage 2 (CPU) -> logs/saebench_stage2.log ==="
  python src/saebench/stage2.py --config $CFG > logs/saebench_stage2.log 2>&1 &
  S2=$!
else
  stamp "Stage 2 NOT started: no debug result in results/saebench/stage2_debug/ (check logs/saebench_prep.log)"
  S2=""
fi

wait $S1 && stamp "Stage 1 finished" || stamp "Stage 1 EXITED WITH AN ERROR (see logs/saebench_stage1.log)"
[ -n "$S2" ] && { wait $S2 && stamp "Stage 2 finished" || stamp "Stage 2 EXITED WITH AN ERROR (see logs/saebench_stage2.log)"; }

stamp "=== aggregate ==="
python src/saebench/stage1.py --config $CFG --aggregate
[ -n "$S2" ] && python src/saebench/stage2.py --config $CFG --aggregate
stamp "=== done ==="
