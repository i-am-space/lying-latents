#!/bin/bash
# Label-free prep for the layer-20 bridge: census, candidates, S solves (provisional tau until frozen).
cd "$(dirname "$0")/../.."
TAU=${TAU:-0.5}
until [ -f data/cache/concept/lat_L20_65k.json ]; do sleep 10; done
CUDA_VISIBLE_DEVICES=${GPU:-1} python3 -u concept/src/family_census.py --layer 20
OMP_NUM_THREADS=2 python3 -u concept/src/build_candidates.py --layer 20 --tau $TAU --workers 4
echo "$(date +%T) L20 PREP DONE"
