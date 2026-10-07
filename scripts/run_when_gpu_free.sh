#!/usr/bin/env bash
# Wait for a completely free GPU (no compute process on it, almost no memory in use), then run a
# command on it.
#
#   scripts/run_when_gpu_free.sh -- command args...
#
# A GPU qualifies when nvidia-smi lists no compute process on it AND its used memory is at most
# MAX_USED_MIB (default 500; catches processes nvidia-smi cannot see, e.g. from other containers),
# for CHECKS consecutive polls (default 3) POLL seconds apart (default 30). The command runs with
# CUDA_VISIBLE_DEVICES set to that GPU. EXCLUDE: comma-separated GPU ids never to use.
set -euo pipefail

[[ "${1:-}" == "--" ]] && shift
[[ $# -gt 0 ]] || { echo "usage: $0 -- command args..."; exit 2; }
POLL=${POLL:-30}
CHECKS=${CHECKS:-3}
MAX_USED=${MAX_USED_MIB:-500}
EXCLUDE=",${EXCLUDE:-},"

declare -A streak
echo "[$(date '+%F %T')] waiting for a completely free GPU (no processes, <= ${MAX_USED} MiB used, ${CHECKS} polls every ${POLL}s)"
while true; do
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sort -u)
    while IFS=', ' read -r idx uuid used; do
        [[ "$EXCLUDE" == *",$idx,"* ]] && continue
        if ! grep -qx "$uuid" <<< "$busy" && (( used <= MAX_USED )); then
            streak[$idx]=$(( ${streak[$idx]:-0} + 1 ))
        else
            streak[$idx]=0
        fi
        if (( ${streak[$idx]} >= CHECKS )); then
            echo "[$(date '+%F %T')] GPU $idx is free (no processes, ${used} MiB used); starting: $*"
            CUDA_VISIBLE_DEVICES=$idx exec "$@"
        fi
    done < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)
    sleep "$POLL"
done
