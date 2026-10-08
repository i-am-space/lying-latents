#!/usr/bin/env bash
# SAEBench Stages 1-4 for another model / SAE (config/preregistration.yaml saebench_amendment_3).
#
#   scripts/models_pipeline.sh CONFIG PHASE
#
# CONFIG  config/models/pythia70m.yaml or config/models/gemma3_1b.yaml
# PHASE   cache    activation caches for the 8 datasets (GPU; needs data/saebench/tasks/*.json)
#         stage12  Stage 1 (GPU, one process) and Stage 2 (CPU, one process per dataset), then both aggregates
#         stage34  Stages 3-4: one process per (dataset, arm group) for the first N34 datasets of
#                  stage4_saebench.dataset_priority, spread over GPUS, then --stage analyse
#         all      cache, then stage12 and stage34 side by side
#         status   what is done so far
# Env: GPUS (default "0 1"; cache and Stage 1 use the first), N34 (default 3), S2_THREADS (default 4),
#      S34_THREADS (default 3), BATCH (cache batch size, default the config's)
# Logs go to logs/<config name>/. Every phase resumes: finished caches, tasks, datasets and records are
# skipped on a rerun. Run from anywhere, inside tmux, with the project environment active.
set -uo pipefail
cd "$(dirname "$0")/.."
[[ -f .venv/bin/activate && -z "${VIRTUAL_ENV:-}" ]] && source .venv/bin/activate
CFG=${1:?usage: $0 CONFIG cache|stage12|stage34|all|status}
PHASE=${2:?usage: $0 CONFIG cache|stage12|stage34|all|status}
read -ra GPUS <<< "${GPUS:-0 1}"
N34=${N34:-3}
S2_THREADS=${S2_THREADS:-4}
S34_THREADS=${S34_THREADS:-3}
TAG=$(basename "$CFG" .yaml)
LOG=logs/$TAG
mkdir -p "$LOG"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stamp() { echo "[$(date '+%F %T')] [$TAG] $*"; }
cfg() { python -c "import sys; sys.path.insert(0, 'src'); from common import load_config; c = load_config('$CFG'); $1"; }
threads() { OMP_NUM_THREADS=$1 MKL_NUM_THREADS=$1 OPENBLAS_NUM_THREADS=$1 "${@:2}"; }

mapfile -t DATASETS < <(cfg "print('\n'.join(c['saebench']['datasets']))")
mapfile -t PRIORITY < <(cfg "print('\n'.join(c['stage4_saebench'].get('dataset_priority', c['saebench']['datasets'])[:$N34]))")
RD=$(cfg "print(c['paths']['results_dir'])")/saebench
TASKS=$(cfg "print(c['saebench']['data_dir'])")/tasks

run_cache() {
    for d in "${DATASETS[@]}"; do
        [[ -f "$TASKS/${d//\//__}.json" ]] || { stamp "missing task texts $TASKS/${d//\//__}.json (copy data/saebench/tasks from a machine that has them)"; return 1; }
    done
    stamp "caches on GPU ${GPUS[0]} -> $LOG/cache.log"
    CUDA_VISIBLE_DEVICES=${GPUS[0]} python src/saebench/cache.py --config "$CFG" ${BATCH:+--batch-size $BATCH} > "$LOG/cache.log" 2>&1 \
        || { stamp "CACHE FAILED (see $LOG/cache.log)"; return 1; }
    stamp "caches done: $(grep -c 'wrote ' "$LOG/cache.log") written this run; $(grep -h 'BOS ' "$LOG/cache.log" | head -1)"
}

run_stage12() {
    stamp "Stage 1 on GPU ${GPUS[0]} -> $LOG/stage1.log; Stage 2: ${#DATASETS[@]} processes x $S2_THREADS threads -> $LOG/stage2_*.log"
    CUDA_VISIBLE_DEVICES=${GPUS[0]} python src/saebench/stage1.py --config "$CFG" --device cuda > "$LOG/stage1.log" 2>&1 &
    local s1=$! pids=()
    for d in "${DATASETS[@]}"; do
        threads "$S2_THREADS" python src/saebench/stage2.py --config "$CFG" --dataset "$d" > "$LOG/stage2_${d//\//__}.log" 2>&1 &
        pids+=($!)
    done
    wait $s1 && stamp "Stage 1 finished" || stamp "Stage 1 FAILED (see $LOG/stage1.log)"
    for p in "${pids[@]}"; do wait "$p" || stamp "a Stage 2 process FAILED (see $LOG/stage2_*.log)"; done
    python src/saebench/stage1.py --config "$CFG" --aggregate > "$LOG/stage1_aggregate.log" 2>&1 || stamp "Stage 1 aggregate FAILED"
    python src/saebench/stage2.py --config "$CFG" --aggregate > "$LOG/stage2_aggregate.log" 2>&1 || stamp "Stage 2 aggregate FAILED"
    stamp "Stages 1-2 done -> $RD/stage1_summary.json, $RD/stage2_summary.json"
}

run_stage34() {
    local pids=() i g gpu
    for i in "${!PRIORITY[@]}"; do
        d=${PRIORITY[$i]}
        for g in 0 1; do
            grp=$([[ $g == 0 ]] && echo hurdle || echo gauss)
            gpu=${GPUS[$(( (i + g) % ${#GPUS[@]} ))]}     # alternate so the slow hurdle jobs are split over the GPUs
            if [[ -f "$RD/stage34/${d//\//__}_${grp}_records.npz" ]]; then stamp "done already: $d $grp"; continue; fi
            CUDA_VISIBLE_DEVICES=$gpu threads "$S34_THREADS" python src/stage4_repairs.py --config "$CFG" --device cuda \
                --experiment saebench --dataset "$d" --arm-group "$grp" --stage full > "$LOG/stage34_${grp}_${d//\//__}.log" 2>&1 &
            pids+=($!)
            stamp "Stage 3-4 started: $d $grp on GPU $gpu"
        done
    done
    for p in "${pids[@]}"; do wait "$p" || stamp "a Stage 3-4 process FAILED (see $LOG/stage34_*.log)"; done
    python src/stage4_repairs.py --config "$CFG" --experiment saebench --stage analyse > "$LOG/stage34_analyse.log" 2>&1 \
        || stamp "Stage 3-4 analyse FAILED"
    stamp "Stages 3-4 done -> $(dirname "$RD")/saebench/stage34_summary.json"
}

status() {
    local n_c n_1 n_2 n_34
    n_c=$(python -c "
import sys; sys.path[:0] = ['src', 'src/saebench']
from common import load_config; from sbutil import cache_path
c = load_config('$CFG'); print(sum(cache_path(c, d).exists() for d in c['saebench']['datasets']))")
    n_1=$(ls "$RD"/stage1/*.json 2>/dev/null | wc -l)
    n_2=$(ls "$RD"/stage2/*.json 2>/dev/null | wc -l)
    n_34=$(ls "$RD"/stage34/*_records.npz 2>/dev/null | wc -l)
    echo "$TAG: caches $n_c/8 | Stage 1 files $n_1 (43 when done: 35 tasks + 8 gates) | Stage 2 $n_2/8 | Stage 3-4 records $n_34/$((2 * N34))"
    for f in stage1_summary.json stage2_summary.json stage34_summary.json; do
        [[ -f "$RD/$f" ]] && echo "  $RD/$f written"
    done
    grep -l "Traceback\|FAILED\|Error" "$LOG"/*.log 2>/dev/null | sed 's/^/  errors in: /'
    for f in "$LOG"/stage34_*_*.log; do [[ -f $f ]] && echo "  $(basename "$f" .log): $(grep -h 'rep ' "$f" | tail -1)"; done
}

case $PHASE in
    cache) run_cache ;;
    stage12) run_stage12 ;;
    stage34) run_stage34 ;;
    all) run_cache && { run_stage12 & run_stage34 & wait; } ;;
    status) status ;;
    *) echo "unknown phase $PHASE"; exit 2 ;;
esac
stamp "phase $PHASE finished"
