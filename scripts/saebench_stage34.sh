#!/usr/bin/env bash
# Stages 3-4 on SAEBench (saebench_amendment_2): run one arm group over a list of datasets, one after
# another, skipping datasets whose records already exist. Each dataset checkpoints per replicate, so
# rerunning the same command resumes.
#
#   CUDA_VISIBLE_DEVICES=N scripts/saebench_stage34.sh GROUP [DATASET ...]
#
# GROUP is hurdle or gauss. With no datasets, all eight in config saebench.datasets are run.
# Run from the repository root inside tmux, with the project environment active.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GROUP=${1:?usage: $0 hurdle|gauss [DATASET ...]}
shift
CFG=config/default.yaml
if [[ $# -gt 0 ]]; then DATASETS=("$@"); else
    mapfile -t DATASETS < <(python -c "import yaml; print('\n'.join(yaml.safe_load(open('$CFG'))['saebench']['datasets']))")
fi
stamp() { echo "[$(date '+%F %T')] $*"; }
for d in "${DATASETS[@]}"; do
    if [[ -f "results/saebench/stage34/${d//\//__}_${GROUP}_records.npz" ]]; then stamp "done already: $d"; continue; fi
    stamp "start: $d ($GROUP, GPU ${CUDA_VISIBLE_DEVICES:-default})"
    python src/stage4_repairs.py --config $CFG --device cuda --experiment saebench --dataset "$d" --arm-group "$GROUP" --stage full \
        || stamp "FAILED: $d"
done
stamp "queue finished ($GROUP)"
