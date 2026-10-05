#!/bin/bash
# Step 1 driver: 24 (flavour, design, m) S solves in parallel on CPU, then all fits on one GPU.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for f in zinf gauss; do for d in dilution redundancy; do for m in 1 2 4 8 16 32; do
  [ -f data/cache/concept/toy/${f}_${d}_m${m}.npz ] || echo "$f $d $m"; done; done; done |
  xargs -P 8 -L 1 bash -c 'python3 -u concept/src/toy_proposition.py solve --flavour $0 --design $1 --m $2'
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
echo "$(date +%T) solves done; fitting"
CUDA_VISIBLE_DEVICES=${TOY_GPU:-1} python3 -u concept/src/toy_proposition.py fit --R 50 --null-R 200
echo "$(date +%T) TOY DONE"
