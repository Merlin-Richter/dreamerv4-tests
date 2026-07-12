#!/usr/bin/env bash
# Memory Maze hierarchical archive continuation (W=32, N=16, M=8, R=1, clip=512).
# Usage: scripts/submit_job.sh --name memmaze-archive --hours H --cpus 8 -- \
#          bash experiments/hierarchical-archive-memory/train.sh EPOCHS BS [EXTRA_ARGS...]
set -euo pipefail
EPOCHS="${1:?usage: train.sh EPOCHS BS [EXTRA_ARGS...]}"
BS="${2:?usage: train.sh EPOCHS BS [EXTRA_ARGS...]}"
shift 2

python -u experiments/hierarchical-archive-memory/train_archive.py \
  --frames data/memmaze9x9.npy \
  --tokenizer checkpoints/memmaze/tokenizer.pt \
  --resume checkpoints/memmaze/dynamics_mem2mem_noff9.pt \
  --checkpoint checkpoints/memmaze/dynamics_archive.pt \
  --epochs "$EPOCHS" --batch-size "$BS" --lr 1e-4 \
  --clip-len 512 --dense-tbptt-frames 64 \
  --archive-interval 16 --archive-per-memory 1 \
  --compressor-depth 1 --compressor-mlp-ratio 2 \
  --seed 0 \
  --wandb --wandb-project transformer-archive-memory --wandb-name memmaze-archive-n16-r1 \
  --wandb-tags memmaze,dynamics,archive-memory,n16,r1 \
  "$@"

echo "########## HIERARCHICAL ARCHIVE TRAIN DONE ##########"
