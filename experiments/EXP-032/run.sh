#!/usr/bin/env bash
# EXP-032 | VANILLA window-32 control. The "just grow the context window" baseline the memory relay
# must beat: a vanilla model with a 32-frame window trivially recalls to k~32 via attention, no
# memory machinery. Tells us whether rollout-training earns its keep vs brute-force context. Same
# bs64 lr3e-4 80ep seed0; only the window differs (32 vs the 16 of EXP-027). Frozen tokenizer.
set -euo pipefail
TOK="runs/gridworld-tok-v3/tokenizer.pt"; [ -f "$TOK" ] || TOK="checkpoints/gridworld/tokenizer.pt"
echo "=== tokenizer: $TOK ==="; ls -la "$TOK"
[ -f gridworld.npy ] || python -u src/datagen/generate_gridworld.py --n_episodes 3000 --n_frames 200 --out gridworld.npy
OUT="runs/gridworld-vanilla-w32-s0"; mkdir -p "$OUT" checkpoints/gridworld
python -u src/training/train_dynamics.py \
  --frames gridworld.npy --tokenizer "$TOK" \
  --checkpoint checkpoints/gridworld/dynamics_vanilla_w32.pt \
  --epochs 80 --batch-size 64 --lr 3e-4 --seed 0 --fresh \
  --context-length 32 \
  --wandb --wandb-name gridworld-vanilla-w32-s0
cp checkpoints/gridworld/dynamics_vanilla_w32.pt "$OUT/"
echo "=== EXP-032 done ==="
