#!/usr/bin/env bash
# Batch-size calibration sweep for the archive continuation on the target GPU (5090, 32 GB).
# One production-shape 512-frame clip per bs; reports peak CUDA alloc + wall time. Pick the
# largest bs with comfortable headroom (leave a few GiB — training also holds optimizer state
# for 41M+ params and the archive proxies grow with clip length).
# Usage: scripts/vast_run.sh --name memmaze-archive-calib -- \
#          bash experiments/hierarchical-archive-memory/calib_vast.sh [BS...]
set -uo pipefail
BSS=("$@"); [ ${#BSS[@]} -gt 0 ] || BSS=(1 2 4 8)

for bs in "${BSS[@]}"; do
  echo "########## calibrate bs=$bs (512 frames, dense_tbptt 64, hide 0.25) ##########"
  python -u experiments/hierarchical-archive-memory/calibrate.py \
    --resume checkpoints/memmaze/dynamics_mem2mem_noff9.pt \
    --batch-size "$bs" --frames 512 --dense-tbptt-frames 64 \
    --archive-interval 16 --archive-per-memory 1 \
    --fast-memory-hide-frac 0.25 \
    || echo "########## bs=$bs FAILED (likely OOM) ##########"
done
echo "########## CALIB DONE ##########"
