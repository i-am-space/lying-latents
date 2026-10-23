#!/usr/bin/env bash
# Deep knockoffs on SAEBench (config/preregistration.yaml saebench_amendment_6), everything on one GPU:
#   Gemma-2  matched sweep (stage34_matched): Bias in Bios 1, GitHub code   (where block MVR breached)
#   Pythia   original cells (stage34):        EuroParl, Bias in Bios 1, Amazon category   (where the hurdle breached)
# Per dataset, one pipeline: swap tests of the deep arm (exchange) -> the benchmark (full) -> swap tests of the
# Gaussian arms (exchange; the Gemma-2 matched sweep has none yet). The five pipelines run side by side; then
# both analyses. Every step resumes (finished files are skipped; the trained generators are cached per dataset).
#
#   nohup scripts/deep_knockoffs.sh > logs/deep_knockoffs.log 2>&1 &
#   scripts/deep_knockoffs.sh status
#
# SET=gemma8 instead runs the deep arm on all 8 Gemma-2 datasets with saebench_amendment_2's 8 cells
# (stage34, paired with its gauss and hurdle records; saebench_amendment_6 revision_1):
#   SET=gemma8 GPU=1 nohup scripts/deep_knockoffs.sh > logs/deep_knockoffs_gemma8.log 2>&1 &
#
# GROUP=deep2 runs the gated deep knockoffs v2 (saebench_amendment_7: arms deep2 and its Gaussian-control
# diagnostic deep2_ctrl) on the same datasets and cells instead of v1:
#   GROUP=deep2 GPU=0 nohup scripts/deep_knockoffs.sh > logs/deep2_knockoffs.log 2>&1 &
#   GROUP=deep2 SET=gemma8 GPU=1 nohup scripts/deep_knockoffs.sh > logs/deep2_knockoffs_gemma8.log 2>&1 &
#
# Env: SET (main | gemma8, default main), GROUP (deep | deep2, default deep), GPU (default 0),
#      THREADS (CPU threads per process, default 6)
set -uo pipefail
cd "$(dirname "$0")/.."
[[ -f .venv/bin/activate && -z "${VIRTUAL_ENV:-}" ]] && source .venv/bin/activate
GPU=${GPU:-0}
THREADS=${THREADS:-6}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
SET=${SET:-main}
GROUP=${GROUP:-deep}
[[ $GROUP == deep || $GROUP == deep2 ]] || { echo "unknown GROUP=$GROUP"; exit 2; }
LOG=logs/${GROUP}_knockoffs$([[ $SET == main ]] || echo "_$SET")
mkdir -p "$LOG"
stamp() { echo "[$(date '+%F %T')] $*"; }
G2=config/default.yaml
PY=config/models/pythia70m.yaml
JOBS=("$G2|--matched|LabHC/bias_in_bios_class_set1|results/saebench/stage34_matched"
      "$G2|--matched|codeparrot/github-code|results/saebench/stage34_matched"
      "$PY||Helsinki-NLP/europarl|results/models/pythia70m/saebench/stage34"
      "$PY||LabHC/bias_in_bios_class_set1|results/models/pythia70m/saebench/stage34"
      "$PY||canrager/amazon_reviews_mcauley_1and5|results/models/pythia70m/saebench/stage34")
if [[ $SET == gemma8 ]]; then
    mapfile -t G8 < <(python -c "import sys; sys.path.insert(0, 'src'); from common import load_config; print('\n'.join(load_config('$G2')['saebench']['datasets']))")
    JOBS=()
    for d in "${G8[@]}"; do JOBS+=("$G2||$d|results/saebench/stage34"); done
elif [[ $SET != main ]]; then echo "unknown SET=$SET"; exit 2; fi

step() {   # cfg flag dataset rd group stage
    local cfg=$1 flag=$2 d=$3 rd=$4 g=$5 st=$6 out tag
    tag=$(basename "$cfg" .yaml)_${d//\//__}
    out=$rd/${d//\//__}_${g}_$([[ $st == full ]] && echo records.npz || echo exchange.json)
    if [[ -f $out ]]; then stamp "done already: $tag $g $st"; return 0; fi
    stamp "start: $tag $g $st"
    CUDA_VISIBLE_DEVICES=$GPU python src/stage4_repairs.py --config "$cfg" --device cuda --experiment saebench $flag \
        --dataset "$d" --arm-group "$g" --stage "$st" > "$LOG/${st}_${g}_${tag}.log" 2>&1 \
        && stamp "finished: $tag $g $st" || { stamp "FAILED: $tag $g $st (see $LOG/${st}_${g}_${tag}.log)"; return 1; }
}

pipeline() {
    local cfg flag d rd
    IFS='|' read -r cfg flag d rd <<< "$1"
    step "$cfg" "$flag" "$d" "$rd" "$GROUP" exchange
    step "$cfg" "$flag" "$d" "$rd" "$GROUP" full
    step "$cfg" "$flag" "$d" "$rd" gauss exchange
}

status() {
    local cfg flag d rd
    for j in "${JOBS[@]}"; do
        IFS='|' read -r cfg flag d rd <<< "$j"
        local s=${d//\//__} tag
        tag=$(basename "$cfg" .yaml)_$s
        printf '%-55s %s-exchange %-3s %s-records %-3s gauss-exchange %-3s | %s\n' "$tag" \
            "$GROUP" "$([[ -f $rd/${s}_${GROUP}_exchange.json ]] && echo yes || echo no)" \
            "$GROUP" "$([[ -f $rd/${s}_${GROUP}_records.npz ]] && echo yes || echo no)" \
            "$([[ -f $rd/${s}_gauss_exchange.json ]] && echo yes || echo no)" \
            "$(grep -h 'rep \|deep fold\|deep2 fold' "$LOG"/*_"$tag".log 2>/dev/null | tail -1)"
    done
    grep -l "Traceback\|Error" "$LOG"/*.log 2>/dev/null | sed 's/^/  errors in: /'
}

if [[ "${1:-}" == status ]]; then status; exit 0; fi
stamp "$GROUP knockoffs: ${#JOBS[@]} pipelines on GPU $GPU, $THREADS threads each; logs in $LOG/"
for j in "${JOBS[@]}"; do      # activation caches (data/saebench/cache, not in git): built here only if missing
    IFS='|' read -r cfg flag d rd <<< "$j"
    CUDA_VISIBLE_DEVICES=$GPU python src/saebench/cache.py --config "$cfg" --dataset "$d" >> "$LOG/cache.log" 2>&1 \
        || { stamp "CACHE FAILED: $cfg $d (see $LOG/cache.log)"; exit 1; }
done
grep -h "wrote " "$LOG/cache.log" | sed 's/^/  built cache: /'
pids=()
for j in "${JOBS[@]}"; do pipeline "$j" & pids+=($!); sleep 20; done   # staggered: the generator training is the memory peak
for p in "${pids[@]}"; do wait "$p"; done
stamp "analyse"
if [[ $SET == gemma8 ]]; then
    python src/stage4_repairs.py --config "$G2" --experiment saebench --stage analyse > "$LOG/analyse_gemma2.log" 2>&1 \
        || stamp "analyse FAILED (Gemma-2)"
    stamp "all done -> results/saebench/stage34_summary.json (DK keys)"
    exit 0
fi
python src/stage4_repairs.py --config "$G2" --experiment saebench --matched --stage analyse > "$LOG/analyse_gemma2_matched.log" 2>&1 \
    || stamp "analyse FAILED (Gemma-2)"
python src/stage4_repairs.py --config "$PY" --experiment saebench --stage analyse > "$LOG/analyse_pythia.log" 2>&1 \
    || stamp "analyse FAILED (Pythia)"
stamp "all done -> results/saebench/stage34_matched_summary.json, results/models/pythia70m/saebench/stage34_summary.json (DK keys)"
