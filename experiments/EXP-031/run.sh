#!/usr/bin/env bash
# EXP-031 | FF9 rollout-training — DEEP (h=44, ~2.75x window). Tests P1's prediction that recall
# horizon == training rollout depth: should recall further than EXP-030 (h=24) if the relay tracks
# (and reveal whether GridWorld's DISCRETE bounded state extrapolates better than P1's continuous
# proxy). tbptt=16 (P1: needed to learn the relay to depth ~31). Budget-matched otherwise.
set -euo pipefail
TOK="runs/gridworld-tok-v3/tokenizer.pt"; [ -f "$TOK" ] || TOK="checkpoints/gridworld/tokenizer.pt"
echo "=== tokenizer: $TOK ==="; ls -la "$TOK"
[ -f gridworld.npy ] || python -u src/datagen/generate_gridworld.py --n_episodes 3000 --n_frames 200 --out gridworld.npy
OUT="runs/gridworld-ff9roll-d44-s0"; mkdir -p "$OUT" checkpoints/gridworld
python -u src/training/train_dynamics.py \
  --frames gridworld.npy --tokenizer "$TOK" \
  --checkpoint checkpoints/gridworld/dynamics_ff9roll_d44.pt \
  --epochs 80 --batch-size 64 --lr 3e-4 --seed 0 --fresh \
  --context-length 16 --rollout-clip-len 48 \
  --ff9 3 --n-memory 4 --lambda-ff9 1.0 \
  --ff9-rollout 44 --ff9-rollout-tbptt 16 --ff9-rollout-hide-mode tail \
  --lambda-ff9-rollout 1.0 --ff9-rollout-warmup 20 \
  --wandb --wandb-name gridworld-ff9roll-d44-s0
cp checkpoints/gridworld/dynamics_ff9roll_d44.pt "$OUT/"
echo "=== EXP-031 done ==="
