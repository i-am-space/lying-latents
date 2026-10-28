#!/usr/bin/env bash
# Stage 5: interventional FDR control (config/preregistration.yaml stage5_amendment_1), all model x task pairs on
# two GPUs, one pair at a time per GPU (Gemma-2-2B in float32 is about 12 GB). Each pair runs pilot -> screen ->
# baselines -> delta -> evaluate in one process; every stage resumes (finished stages are skipped, the delta stage
# checkpoints every 10 minutes). A pair whose pilot gate fails stops after the pilot and is reported as not run.
#
#   nohup scripts/stage5.sh > logs/stage5.log 2>&1 &
#   scripts/stage5.sh status
#
# stage5_amendment_2 (after the run above): baselines2 -> delta2 (GPU, only baselines2's new latents) -> evaluate2
# (CPU), same pairs and GPUs; pairs whose stage5_amendment_1 run is incomplete are skipped:
#   nohup scripts/stage5.sh v2 > logs/stage5_v2.log 2>&1 &
#
# stage5_amendment_3: baselines3 (Gaussian equicorrelated, block-MVR and hurdle knockoffs on the real labels, paired
# draws) -> delta3 (GPU, their new latents on split O) -> evaluate3 (CPU), then rule B3-1 over both models:
#   nohup scripts/stage5.sh v3 > logs/stage5_v3.log 2>&1 &
# v3 then also runs the exchangeability tests implemented but never run (the stage4_repairs --stage exchange suite:
# swap classifier at |S| = 0, 1, 10, 50, all, label-permutation band, 1{x = 0} rule): the hurdle on every SAEBench
# setting (Gemma-2 8 datasets, Gemma-3 3, Pythia-70M 3 unthresholded and 3 thresholded at 0.19), and the Gaussian arms
# where they are missing (Gemma-3, thresholded Pythia). Existing results are skipped. Only these tests:
#   nohup scripts/stage5.sh exchange > logs/exchange.log 2>&1 &
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
            for f in pilot screen baselines delta evaluate baselines2 delta2 evaluate2 baselines3 delta3 evaluate3; do [[ -f $d/$f.json ]] && s="$s $f"; done
            gate=$([[ -f $d/pilot.json ]] && python -c "import json; print('gate', json.load(open('$d/pilot.json'))['gate']['passed'])" || echo "")
            printf '%-36s %-14s done:%-66s %s\n' "$(basename "$cfg" .yaml)" "$t" "$s" "$gate"
        done
    done
    grep -l "Traceback" "$LOG"/*.log 2>/dev/null | sed 's/^/  errors in: /'
}
if [[ "${1:-}" == status ]]; then status; exit 0; fi
MODE=all; [[ "${1:-}" == v2 ]] && MODE=v2; [[ "${1:-}" == v3 ]] && MODE=v3; [[ "${1:-}" == exchange ]] && MODE=exchange
PY=config/models/pythia70m.yaml
SB=(LabHC/bias_in_bios_class_set1 LabHC/bias_in_bios_class_set2 LabHC/bias_in_bios_class_set3
    canrager/amazon_reviews_mcauley_1and5 canrager/amazon_reviews_mcauley_1and5_sentiment codeparrot/github-code
    fancyzhx/ag_news Helsinki-NLP/europarl)
SB3=(Helsinki-NLP/europarl LabHC/bias_in_bios_class_set1 canrager/amazon_reviews_mcauley_1and5)
# exchange jobs: "config|flags|dataset|group|result dir"
EXJ=()
for d in "${SB[@]}"; do EXJ+=("$G2||$d|hurdle|results/saebench/stage34"); done
for d in "${SB3[@]}"; do
    EXJ+=("$G3||$d|hurdle|results/models/gemma3_1b/saebench/stage34" "$G3||$d|gauss|results/models/gemma3_1b/saebench/stage34"
          "$PY||$d|hurdle|results/models/pythia70m/saebench/stage34"
          "$PY|--matched --positive-threshold 0.19|$d|hurdle|results/models/pythia70m/saebench/stage34_matched_thr0.19"
          "$PY|--matched --positive-threshold 0.19|$d|gauss|results/models/pythia70m/saebench/stage34_matched_thr0.19")
done

exchange_worker() {   # gpu jobs...
    local gpu=$1; shift
    for job in "$@"; do
        IFS='|' read -r cfg flags d g rd <<< "$job"
        out=$rd/${d//\//__}_${g}_exchange.json
        tag=exchange_$(basename "$cfg" .yaml)$([[ -n $flags ]] && echo _thr)_${d//\//__}_$g
        if [[ -f $out ]]; then stamp "GPU $gpu done already: $tag"; continue; fi
        stamp "GPU $gpu start: $tag"
        CUDA_VISIBLE_DEVICES=$gpu python src/stage4_repairs.py --config "$cfg" --device cuda --experiment saebench $flags \
            --dataset "$d" --arm-group "$g" --stage exchange > "$LOG/$tag.log" 2>&1 \
            && stamp "GPU $gpu finished: $tag" || stamp "GPU $gpu FAILED: $tag (see $LOG/$tag.log)"
    done
}
run_exchange() {
    local a=() b=() i=0
    for j in "${EXJ[@]}"; do if (( i++ % 2 == 0 )); then a+=("$j"); else b+=("$j"); fi; done
    stamp "exchange tests: ${#EXJ[@]} jobs on GPU $GPU_A and GPU $GPU_B"
    exchange_worker "$GPU_A" "${a[@]}" &
    exchange_worker "$GPU_B" "${b[@]}" &
    wait
    python - <<'PYEOF'
import json, glob
print("\nexchange tests (swap AUC at |S| = 1 / 50 / all; detected at 1):")
for f in sorted(glob.glob("results/**/*_exchange.json", recursive=True)):
    for arm, v in json.load(open(f))["arms"].items():
        a = v["summary"]["swap_auc"]; det = v["summary"]["detected_at"]
        print(f"  {f.split('results/')[1][:78]:78s} {arm:18s} {a.get('1', float('nan')):.3f} / {a.get('50', float('nan')):.3f} / "
              f"{a.get('all', float('nan')):.3f}  {det.get('1')}")
PYEOF
}
if [[ $MODE == exchange ]]; then run_exchange; stamp "exchange done"; exit 0; fi

worker() {   # gpu jobs...
    local gpu=$1; shift
    for job in "$@"; do
        read -r cfg task <<< "$job"
        tag=$(basename "$cfg" .yaml)_$task
        if [[ $MODE == v2 || $MODE == v3 ]]; then
            rd=$(python -c "import sys; sys.path.insert(0,'src'); from common import load_config; print(load_config('$cfg')['paths']['results_dir'])")
            [[ -f $rd/stage5/$task/evaluate.json ]] || { stamp "GPU $gpu skip: $tag (stage5_amendment_1 run incomplete)"; continue; }
            tag=${tag}_$MODE
        fi
        stamp "GPU $gpu start: $tag"
        CUDA_VISIBLE_DEVICES=$gpu python src/stage5_interventional.py --config "$cfg" --task "$task" --stage $MODE \
            > "$LOG/$tag.log" 2>&1 && stamp "GPU $gpu finished: $tag" || stamp "GPU $gpu FAILED: $tag (see $LOG/$tag.log)"
    done
}
stamp "stage 5 ($MODE): GPU $GPU_A and GPU $GPU_B"
worker "$GPU_A" "${WORKER_A[@]}" &
worker "$GPU_B" "${WORKER_B[@]}" &
wait
if [[ $MODE == v3 ]]; then
    python src/stage5_interventional.py --stage summary3
    stamp "v3 done -> results/stage5_baselines3_summary.json (rule B3-1), per task evaluate3.json"
    run_exchange
    stamp "all done (v3 and exchange tests)"
    exit 0
fi
python src/stage5_interventional.py --config "$G2" --stage summary
python src/stage5_interventional.py --config "$G3" --stage summary
stamp "all done -> results/stage5/stage5_summary.json, results/models/gemma3_1b/stage5/stage5_summary.json"
