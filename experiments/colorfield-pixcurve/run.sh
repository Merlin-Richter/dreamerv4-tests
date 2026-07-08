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
  --sched-steps 14000 \
  --snapshot-at 250,500,1000,2000,3750,4000,6000,8000,10000,12000,14000
