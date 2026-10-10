#!/usr/bin/env bash
# SAEBench matched benchmark (config/preregistration.yaml saebench_amendment_4): per-dataset calibrated
# amplitudes (step 1), the fair-coin record of the null latents (step 2), and, for ReLU SAEs, the hurdle on
# JumpReLU-thresholded latents (step 5).
#
#   scripts/saebench_matched.sh calibrate|run|analyse|all [DATASET ...]
#
#   calibrate   parameters only (per-dataset amplitude matched to the real labels; minutes, no knockoffs)
#   run         the benchmark: one process per (dataset, arm group), PAR at a time on GPU; resumes from checkpoints
#   analyse     per-dataset JSON and the summary: breaches, FDR, power and the coin
#   exchange    the Stage 2 swap tests (and the zero-mass rule) on one knockoff draw of every arm of each group;
#               writes <dataset>_<group>_exchange.json next to the records
#   all         calibrate, run, analyse
#
# Env:  GPU   (default 2)            CUDA device
#       CFG   (default config/default.yaml; config/models/pythia70m.yaml for Pythia)
#       THR   (default empty)        positive threshold, e.g. THR=0.19: zero values below 0.19 x the latent's median
#                                    positive value first (JumpReLU-like; for the ReLU SAE of Pythia)
#       ARMS  (default "gauss")      arm groups to run, started in this order; "hurdle gauss" adds the (slow) hurdle,
#                                    first
#       PAR   (default 3)            concurrent processes on the GPU (each ~5-6 GB)
#       REPS  (default: config, 20)  replicates, e.g. REPS=1 for a timing pilot with STAGE=diagnose
#       STAGE (default full)         full | diagnose (one replicate, prints the projected time and the first coin)
# Run from the repository root inside tmux, with the project environment active.
#   GPU=2 scripts/saebench_matched.sh all
#   GPU=2 CFG=config/models/pythia70m.yaml THR=0.19 ARMS="hurdle gauss" scripts/saebench_matched.sh all
set -uo pipefail
cd "$(dirname "$0")/.."
MODE=${1:?usage: $0 calibrate|run|analyse|exchange|all [DATASET ...]}
shift
GPU=${GPU:-2}
CFG=${CFG:-config/default.yaml}
THR=${THR:-}
read -ra ARM_GROUPS <<< "${ARMS:-gauss}"   # (not GROUPS: that is a bash builtin)
PAR=${PAR:-3}
STAGE=${STAGE:-full}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
TAG=$(basename "$CFG" .yaml)${THR:+_thr$THR}
LOG=logs/matched_$TAG
mkdir -p "$LOG"
stamp() { echo "[$(date '+%F %T')] [$TAG] $*"; }
THRARG=(); [[ -n "$THR" ]] && THRARG=(--positive-threshold "$THR")
if [[ $# -gt 0 ]]; then DATASETS=("$@"); else
    mapfile -t DATASETS < <(python -c "import sys; sys.path.insert(0, 'src'); from common import load_config; print('\n'.join(load_config('$CFG')['stage4_saebench_matched']['datasets']))")
fi
RD=$(python -c "import sys; sys.path.insert(0, 'src'); from common import load_config; print(load_config('$CFG')['paths']['results_dir'])")/saebench
SUB=stage34_matched${THR:+_thr$THR}

do_calibrate() {
    if python -c "import sys; sys.path.insert(0, 'src'); from common import load_config; c = load_config('$CFG')['stage4_saebench_matched']; sys.exit(0 if (c.get('designs') or 'amplitudes' in c) else 1)"; then
        stamp "amplitude sweep design (config stage4_saebench_matched.amplitudes): no calibration needed"
        return 0
    fi
    for d in "${DATASETS[@]}"; do
        stamp "calibrate $d"
        CUDA_VISIBLE_DEVICES=$GPU python src/saebench/calibrate_amplitude.py --config "$CFG" --device cuda --dataset "$d" ${THRARG[@]+"${THRARG[@]}"} \
            > "$LOG/calibrate_${d//\//__}.log" 2>&1 || stamp "CALIBRATE FAILED: $d (see $LOG)"
    done
}

do_run() {
    local pids=() running=0
    for g in "${ARM_GROUPS[@]}"; do       # group-major: ARMS="hurdle gauss" starts every slow hurdle job first
        for d in "${DATASETS[@]}"; do
            if [[ "$STAGE" == full && -f "$RD/$SUB/${d//\//__}_${g}_records.npz" ]]; then stamp "done already: $d $g"; continue; fi
            while (( $(jobs -rp | wc -l) >= PAR )); do wait -n || true; done
            stamp "start: $d ($g, GPU $GPU, $STAGE)"
            CUDA_VISIBLE_DEVICES=$GPU python src/stage4_repairs.py --config "$CFG" --device cuda --experiment saebench --matched \
                --dataset "$d" --arm-group "$g" --stage "$STAGE" ${REPS:+--limit-reps $REPS} ${THRARG[@]+"${THRARG[@]}"} \
                > "$LOG/${STAGE}_${g}_${d//\//__}.log" 2>&1 || stamp "FAILED: $d $g (see $LOG)" &
        done
    done
    wait
    stamp "run finished"
}

do_exchange() {
    local g d
    for g in "${ARM_GROUPS[@]}"; do
        for d in "${DATASETS[@]}"; do
            if [[ -f "$RD/$SUB/${d//\//__}_${g}_exchange.json" ]]; then stamp "exchange done already: $d $g"; continue; fi
            while (( $(jobs -rp | wc -l) >= PAR )); do wait -n || true; done
            stamp "exchange: $d ($g, GPU $GPU)"
            CUDA_VISIBLE_DEVICES=$GPU python src/stage4_repairs.py --config "$CFG" --device cuda --experiment saebench --matched \
                --dataset "$d" --arm-group "$g" --stage exchange ${THRARG[@]+"${THRARG[@]}"} \
                > "$LOG/exchange_${g}_${d//\//__}.log" 2>&1 || stamp "EXCHANGE FAILED: $d $g (see $LOG)" &
        done
    done
    wait
    stamp "exchange finished"
}

do_analyse() {
    python src/stage4_repairs.py --config "$CFG" --experiment saebench --matched --stage analyse ${THRARG[@]+"${THRARG[@]}"} \
        2>&1 | tee "$LOG/analyse.log"
}

case $MODE in
    calibrate) do_calibrate ;;
    run) do_run ;;
    analyse) do_analyse ;;
    exchange) do_exchange ;;
    all) do_calibrate && do_run && do_analyse ;;
    *) echo "unknown mode $MODE"; exit 2 ;;
esac
