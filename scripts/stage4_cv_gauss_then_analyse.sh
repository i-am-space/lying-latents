#!/usr/bin/env bash
# Stage 4 under the cross-validated penalty (stage4_amendment_4), Gaussian arm group:
# wait for a completely free GPU, run the group, then wait for the hurdle group's records and run the analysis.
# Run from the repository root inside tmux, with the project's environment active.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

if [[ ! -f results/stage4_cv_gauss_records.npz ]]; then
    scripts/run_when_gpu_free.sh -- \
        python src/stage4_repairs.py --config config/default.yaml --device cuda --experiment cv --arm-group gauss --stage full &
    wait $!
fi

echo "[$(date '+%F %T')] Gaussian group done; waiting for the hurdle group's records"
until [[ -f results/stage4_cv_hurdle_records.npz ]]; do sleep 60; done
python src/stage4_repairs.py --config config/default.yaml --experiment cv --stage analyse | tee results/stage4_cv_analyse.log
echo "[$(date '+%F %T')] all done: results/stage4_cv.json"
