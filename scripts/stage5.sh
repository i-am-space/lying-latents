#!/usr/bin/env bash
# Stage 5: interventional FDR control (config/preregistration.yaml stage5_amendment_1), all model x task pairs on
# two GPUs, one pair at a time per GPU (Gemma-2-2B in float32 is about 12 GB). Each pair runs pilot -> screen ->
# baselines -> delta -> evaluate in one process; every stage resumes (finished stages are skipped, the delta stage
# checkpoints every 10 minutes). A pair whose pilot gate fails stops after the pilot and is reported as not run.
#
#   nohup scripts/stage5.sh > logs/stage5.log 2>&1 &
#   scripts/stage5.sh status
#
# Env: GPU_A (default 0), GPU_B (default 1), THREADS (default 8)
set -uo pipefail
cd "$(dirname "$0")/.."
[[ -f .venv/bin/activate && -z "${VIRTUAL_ENV:-}" ]] && source .venv/bin/activate
GPU_A=${GPU_A:-0}; GPU_B=${GPU_B:-1}; THREADS=${THREADS:-8}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=logs/stage5; mkdir -p "$LOG"
G2=config/default.yaml
G3=config/models/gemma3_1b.yaml
# balanced by expected cost (Gemma-2 replays 5 layers, Gemma-3 3 smaller ones; Bias in Bios / AG News prompts are longest)
WORKER_A=("$G2 sst2" "$G2 bias_in_bios" "$G3 sst2" "$G3 bias_in_bios")
WORKER_B=("$G2 ag_news" "$G2 europarl" "$G3 ag_news" "$G3 europarl")
stamp() { echo "[$(date '+%F %T')] $*"; }

status() {
    for cfg in $G2 $G3; do
        rd=$(python -c "import sys; sys.path.insert(0,'src'); from common import load_config; print(load_config('$cfg')['paths']['results_dir'])")
        for t in sst2 ag_news bias_in_bios europarl; do
            d=$rd/stage5/$t; s=""
            for f in pilot screen baselines delta evaluate; do [[ -f $d/$f.json ]] && s="$s $f"; done
            gate=$([[ -f $d/pilot.json ]] && python -c "import json; print('gate', json.load(open('$d/pilot.json'))['gate']['passed'])" || echo "")
            printf '%-36s %-14s done:%-40s %s\n' "$(basename "$cfg" .yaml)" "$t" "$s" "$gate"
        done
    done
    grep -l "Traceback" "$LOG"/*.log 2>/dev/null | sed 's/^/  errors in: /'
}
if [[ "${1:-}" == status ]]; then status; exit 0; fi

worker() {   # gpu jobs...
    local gpu=$1; shift
    for job in "$@"; do
        read -r cfg task <<< "$job"
        tag=$(basename "$cfg" .yaml)_$task
        stamp "GPU $gpu start: $tag"
        CUDA_VISIBLE_DEVICES=$gpu python src/stage5_interventional.py --config "$cfg" --task "$task" --stage all \
            > "$LOG/$tag.log" 2>&1 && stamp "GPU $gpu finished: $tag" || stamp "GPU $gpu FAILED: $tag (see $LOG/$tag.log)"
    done
}
stamp "stage 5: GPU $GPU_A and GPU $GPU_B"
worker "$GPU_A" "${WORKER_A[@]}" &
worker "$GPU_B" "${WORKER_B[@]}" &
wait
python src/stage5_interventional.py --config "$G2" --stage summary
python src/stage5_interventional.py --config "$G3" --stage summary
stamp "all done -> results/stage5/stage5_summary.json, results/models/gemma3_1b/stage5/stage5_summary.json"
