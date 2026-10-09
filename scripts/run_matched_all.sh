#!/usr/bin/env bash
# Everything of saebench_amendment_4 in one go, the two parts side by side (one GPU each by default):
#   A  Gemma-2 (default.yaml): calibrate -> run (gauss group) -> analyse           [steps 1 and 2]
#   B  Pythia-70M, JumpReLU-thresholded latents: calibrate -> run (hurdle + gauss) -> analyse   [step 5]
# Each part is scripts/saebench_matched.sh; this script starts both, waits for both and reports.
#
#   scripts/run_matched_all.sh
#
# Env (all optional):
#   GPU_A  CUDA device of part A                             (default 0)
#   GPU_B  CUDA device of part B                             (default 1)   (same value = share one GPU)
#   PAR_A  concurrent processes of part A                    (default 4: the 4 Gemma-2 datasets at once)
#   PAR_B  concurrent processes of part B                    (default 4: the 3 slow hurdle jobs + 1 gauss job)
#          Each process needs about 5-6 GB: 4 per 32 GB GPU. If both parts share a GPU, halve both.
#   THR    positive-threshold fraction for part B            (default 0.19, the JumpReLU floor of Gemma-2)
#   PILOT  1 = first a one-replicate timing pilot per part (prints the projected run time), then the real run
#          (default 0: go straight to the real run)
#   PILOT_ONLY  1 = calibrate and run the one-replicate pilot, print the projected times and the first coin
#          readings, then stop (check them, then start the real run without it)
#   DRY    1 = only print the commands
# Run from anywhere, inside tmux, on the GPU machine (needs data/saebench caches and the project environment).
# Every step resumes: rerun the same command after an interruption.
set -uo pipefail
cd "$(dirname "$0")/.."
[[ -f .venv/bin/activate && -z "${VIRTUAL_ENV:-}" ]] && source .venv/bin/activate
GPU_A=${GPU_A:-0}
GPU_B=${GPU_B:-1}
PAR_A=${PAR_A:-4}
PAR_B=${PAR_B:-4}
THR=${THR:-0.19}
PILOT=${PILOT:-0}
PILOT_ONLY=${PILOT_ONLY:-0}
[[ "$PILOT_ONLY" == 1 ]] && PILOT=1
DRY=${DRY:-0}
CFG_B=config/models/pythia70m.yaml
mkdir -p logs
stamp() { echo "[$(date '+%F %T')] $*"; }
x() { if [[ "$DRY" == 1 ]]; then echo "   $*"; else "$@"; fi; }

check_gpu() {   # GPU, processes: warn if less memory is free than the processes need
    command -v nvidia-smi > /dev/null && [[ "$DRY" != 1 ]] || return 0
    local free need=$(( $2 * 6000 ))
    free=$(nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits 2> /dev/null | head -1)
    stamp "GPU $1: ${free:-?} MiB free, about $need MiB needed for $2 processes"
    if [[ -n "${free:-}" ]] && (( free < need )); then stamp "WARNING: GPU $1 has less memory free than needed; lower PAR"; fi
}
if [[ "$GPU_A" == "$GPU_B" ]]; then check_gpu "$GPU_A" $(( PAR_A + PAR_B )); else check_gpu "$GPU_A" "$PAR_A"; check_gpu "$GPU_B" "$PAR_B"; fi

part_a() {
    stamp "[A] Gemma-2: calibrate, then run"
    x env GPU="$GPU_A" PAR="$PAR_A" scripts/saebench_matched.sh calibrate || return 1
    if [[ "$PILOT" == 1 ]]; then
        x env GPU="$GPU_A" PAR="$PAR_A" STAGE=diagnose REPS=1 scripts/saebench_matched.sh run
        stamp "[A] pilot done:"; x bash -c 'grep -h "projected\|coin" logs/matched_default/diagnose_*.log'
        [[ "$PILOT_ONLY" == 1 ]] && return 0
    fi
    x env GPU="$GPU_A" PAR="$PAR_A" scripts/saebench_matched.sh run || return 1
    x env scripts/saebench_matched.sh analyse
    stamp "[A] done -> results/saebench/stage34_matched_summary.json"
}

part_b() {
    stamp "[B] Pythia-70M thresholded at $THR: calibrate, then run (hurdle + gauss)"
    x env GPU="$GPU_B" PAR="$PAR_B" CFG="$CFG_B" THR="$THR" scripts/saebench_matched.sh calibrate || return 1
    if [[ "$PILOT" == 1 ]]; then
        x env GPU="$GPU_B" PAR="$PAR_B" CFG="$CFG_B" THR="$THR" ARMS="hurdle gauss" STAGE=diagnose REPS=1 scripts/saebench_matched.sh run
        stamp "[B] pilot done:"; x bash -c "grep -h 'projected\|coin' logs/matched_pythia70m_thr$THR/diagnose_*.log"
        [[ "$PILOT_ONLY" == 1 ]] && return 0
    fi
    x env GPU="$GPU_B" PAR="$PAR_B" CFG="$CFG_B" THR="$THR" ARMS="hurdle gauss" scripts/saebench_matched.sh run || return 1
    x env CFG="$CFG_B" THR="$THR" scripts/saebench_matched.sh analyse
    stamp "[B] done -> results/models/pythia70m/saebench/stage34_matched_thr${THR}_summary.json"
}

part_a > logs/matched_A.log 2>&1 &
pa=$!
part_b > logs/matched_B.log 2>&1 &
pb=$!
stamp "started A (pid $pa, log logs/matched_A.log) and B (pid $pb, log logs/matched_B.log) on GPUs $GPU_A (A) and $GPU_B (B)"
stamp "follow with:  tail -f logs/matched_A.log logs/matched_B.log   |   grep -h 'rep ' logs/matched_*/full_*.log | tail"
fail=0
wait "$pa" || { stamp "part A FAILED (see logs/matched_A.log and logs/matched_default/)"; fail=1; }
wait "$pb" || { stamp "part B FAILED (see logs/matched_B.log and logs/matched_pythia70m_thr$THR/)"; fail=1; }
(( fail )) && exit 1
stamp "both parts finished"
