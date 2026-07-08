#!/usr/bin/env bash
# ColorField PIXEL steps-vs-quality curve — H100 extension (experiments/colorfield-pixcurve/).
#
# The local 4070 calcurve (autoresearch/runs/calcurve/, 1.32M, bs128, fixed n_ctx, sched 7000)
# was OOM-killed at step ~3750 by a GPU collision; snapshots 250-2000 + final survived and
# showed the APPEARANCE knee at ~3k steps with memory still absent. This run extends the same
# curve on an H100: same config + seed, 4h wall-clock enforced by train.py's BUDGET_STOP
# (per-step check; SLURM walltime is only the outer safety margin). Snapshot 3750 is the
# cross-backend anchor against the local kill point. Also yields the calibration's missing
# H100 s/step number (local anchor: 3.92 s/step at this config).
#
# RESIZED after first submission (416895, cancelled at ~2 min): measured H100 pace is
# 0.124 s/step (~31x local, NOT the guessed 4-6x) -> the original sched 14000 + default
# --epochs 50 (31k-step cap) would have ended the run in ~65 min. Now sched 110000
# (~3h48m; warmup stays min(200, 10%)=200, identical to the local run, so early snapshots
# remain LR-matched) + --epochs 200 so the 4h budget is the binding stop.
set -euo pipefail

# Cache guard: data + latent cache were built on ferranti by job 416225 (idempotent datagen,
# byte-identical proven). Rebuild only if missing; hash bd8f18857d71 = the FROZEN tokenizer.
CACHE=data/colorfield/latents-bd8f18857d71.npy
VCACHE=data/colorfield_val/latents-bd8f18857d71.npy
if [ ! -f "$CACHE" ] || [ ! -f "$VCACHE" ]; then
  echo "latent cache missing -> rebuilding via cache_job.sh"
  bash autoresearch/driver/cache_job.sh
fi

python -u autoresearch/editable/train.py \
  --data data/colorfield --val data/colorfield_val \
  --tokenizer checkpoints/colorfield/tokenizer.pt \
  --checkpoint runs/colorfield-pixcurve/dynamics.pt \
  --budget-s 14400 \
  --batch-size 128 --embedding-dim 128 --depth 6 --n-heads 8 \
  --fixed-n-ctx --seed 0 \
  --epochs 200 \
  --sched-steps 110000 \
  --snapshot-at 250,500,1000,2000,3750,6000,8000,10000,14000,20000,28000,40000,56000,80000,110000
