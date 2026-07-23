#!/usr/bin/env bash
# Continue the completed Memory Maze dense-memory/no-FF9 checkpoint while teaching the shortcut
# ladder needed for K=4 inference. The trainer owns a 12-hour ACTIVE optimization budget; request
# 13 SLURM hours so validation/checkpoint/setup overhead can finish cleanly.
set -euo pipefail

python -u experiments/mem2mem/train_mem2mem.py \
  --frames data/memmaze9x9.npy \
  --tokenizer checkpoints/memmaze/tokenizer.pt \
  --resume checkpoints/memmaze/dynamics_mem2mem_noff9.pt \
  --checkpoint checkpoints/memmaze/dynamics_mem2mem_noff9_k4.pt \
  --epochs 999 --batch-size 4 --lr 1e-4 \
  --clip-len 128 --context-length 32 \
  --embedding-dim 512 --depth 12 --n-heads 16 \
  --n-memory 8 --ff9 3 --mem2mem-frac 1.0 --no-ff9 \
  --wallclock-hours 12 \
  --curr-warmup-hours 1 \
  --curr-full-hours 6 \
  --curr-max-unlocked 6 \
  --checkpoint-every-hours 1 \
  --seed 0 \
  --wandb --wandb-project transformer-mem2mem \
  --wandb-name memmaze-dyn-mem2mem-noff9-k4-curriculum \
  --wandb-tags memmaze,dynamics,mem2mem,no-ff9,shortcut-k4,curriculum

echo "########## MEMMAZE K4 CURRICULUM DONE ##########"
