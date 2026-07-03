#!/usr/bin/env bash
# Memory Maze VANILLA dynamics (baseline arm): 512/12/16, W=32, no memory. Latents from the disk cache.
# Usage: submit_job.sh --name memmaze-dyn-vanilla --hours H --cpus 8 -- \
#          bash experiments/memmaze-dynamics/train_vanilla.sh EPOCHS BS [EXTRA_ARGS...]
set -euo pipefail
EPOCHS="${1:?usage: train_vanilla.sh EPOCHS BS}"
BS="${2:?usage: train_vanilla.sh EPOCHS BS}"
shift 2

python -u src/training/train_dynamics.py \
  --frames data/memmaze9x9.npy --tokenizer checkpoints/memmaze/tokenizer.pt \
  --checkpoint checkpoints/memmaze/dynamics_vanilla.pt \
  --epochs "$EPOCHS" --batch-size "$BS" --context-length 32 \
  --embedding-dim 512 --depth 12 --n-heads 16 \
  --seed 0 \
  --wandb --wandb-project transformer-D-dynamics --wandb-name memmaze-dyn-vanilla \
  --wandb-tags memmaze,dynamics,vanilla \
  "$@"
echo "########## VANILLA TRAIN DONE ##########"
