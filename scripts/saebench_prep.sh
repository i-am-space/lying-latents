#!/usr/bin/env bash
# SAEBench preparation queue (no pre-registered results are produced here):
#   1. Stage 2 debug on AG News (cut-down; writes results/saebench/stage2_debug/)
#   2. task texts for all 8 datasets (SAEBench's code; needs the datasets-3.6 side folder)
#   3. activation caches for all 8 datasets (GPU)
# Each dataset is its own command, so one failure does not stop the others.
#
# Usage (from the repo root, with the project venv):
#   nohup bash scripts/saebench_prep.sh > logs/saebench_prep.log 2>&1 &
# Env: GPU (default 1), SB_DATA_PKGS (default ~/ANLP_sami/sb_data_pkgs), BATCH (default 32)
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
GPU="${GPU:-1}"
SB_DATA_PKGS="${SB_DATA_PKGS:-$HOME/ANLP_sami/sb_data_pkgs}"
BATCH="${BATCH:-32}"
CFG=config/default.yaml
DATASETS=$(python -c "import yaml; print(' '.join(yaml.safe_load(open('$CFG'))['saebench']['datasets']))")
stamp() { echo "[$(date '+%F %T')] $*"; }

stamp "=== 1. Stage 2 debug (AG News) ==="
python src/saebench/stage2.py --config $CFG --dataset fancyzhx/ag_news --debug || stamp "STAGE2 DEBUG FAILED"

stamp "=== 2. task texts ==="
for d in $DATASETS; do
  if [ -f "data/saebench/tasks/${d//\//__}.json" ]; then stamp "texts exist: $d"; continue; fi
  stamp "texts: $d"
  PYTHONPATH="$SB_DATA_PKGS" python src/saebench/build_tasks.py --config $CFG --dataset "$d" || stamp "TEXTS FAILED: $d"
done

stamp "=== 3. activation caches (GPU $GPU, batch $BATCH) ==="
for d in $DATASETS; do
  [ -f "data/saebench/tasks/${d//\//__}.json" ] || { stamp "SKIP cache (no texts): $d"; continue; }
  stamp "cache: $d"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python src/saebench/cache.py --config $CFG --dataset "$d" --batch-size $BATCH || stamp "CACHE FAILED: $d"
done

stamp "=== done ==="
ls -la data/saebench/tasks data/saebench/cache
cat results/saebench/cache_summary.json 2>/dev/null
