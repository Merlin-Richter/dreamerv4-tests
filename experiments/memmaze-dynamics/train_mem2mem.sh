#!/usr/bin/env bash
# Memory Maze MEM2MEM ROLLOUT-ONLY dynamics (memory arm) [structure LOCKED by Merlin 2026-07-03]:
# the GridWorld 411133 winner config — mem2mem-frac 1.0, --no-bootstrap, FF9 on — scaled to memmaze:
# 512/12/16, W=32, clip 128, n_memory 8, ff9 3. Latents from the disk cache.
# Usage: submit_job.sh --name memmaze-dyn-mem2mem --hours H --cpus 8 -- \
#          bash experiments/memmaze-dynamics/train_mem2mem.sh EPOCHS BS [EXTRA_ARGS...]
set -euo pipefail
EPOCHS="${1:?usage: train_mem2mem.sh EPOCHS BS}"
BS="${2:?usage: train_mem2mem.sh EPOCHS BS}"
shift 2

python -u experiments/mem2mem/train_mem2mem.py \
  --frames data/memmaze9x9.npy --tokenizer checkpoints/memmaze/tokenizer.pt \
  --checkpoint checkpoints/memmaze/dynamics_mem2mem.pt \
  --epochs "$EPOCHS" --batch-size "$BS" \
  --clip-len 128 --context-length 32 \
  --embedding-dim 512 --depth 12 --n-heads 16 \
  --n-memory 8 --ff9 3 --mem2mem-frac 1.0 --no-bootstrap \
  --seed 0 \
  --wandb --wandb-project transformer-mem2mem --wandb-name memmaze-dyn-mem2mem \
  --wandb-tags memmaze,dynamics,mem2mem-rollout-only \
  "$@"
echo "########## MEM2MEM TRAIN DONE ##########"
