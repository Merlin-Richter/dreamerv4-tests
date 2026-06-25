#!/usr/bin/env bash
# EXP-033 | FF9 rollout-training WIDER MEMORY (M=16 vs EXP-030's M=4). Tests capacity-vs-drift: if
# dynamic-position recall holds longer with more memory tokens, the EXP-030 decay was a CAPACITY
# limit; if it drifts the same, the limit is continuous-memory DRIFT (-> discrete/VQ, not capacity).
# Otherwise identical to EXP-030 (h24, window16, clip28, tbptt12, tail, +ff9 3, warmup20, 80ep).
set -euo pipefail
TOK="runs/gridworld-tok-v3/tokenizer.pt"; [ -f "$TOK" ] || TOK="checkpoints/gridworld/tokenizer.pt"
echo "=== tokenizer: $TOK ==="; ls -la "$TOK"
[ -f gridworld.npy ] || python -u src/datagen/generate_gridworld.py --n_episodes 3000 --n_frames 200 --out gridworld.npy
OUT="runs/gridworld-ff9roll-m24-M16-s0"; mkdir -p "$OUT" checkpoints/gridworld
python -u src/training/train_dynamics.py \
  --frames gridworld.npy --tokenizer "$TOK" \
  --checkpoint checkpoints/gridworld/dynamics_ff9roll_m24_M16.pt \
  --epochs 80 --batch-size 64 --lr 3e-4 --seed 0 --fresh \
  --context-length 16 --rollout-clip-len 28 \
  --ff9 3 --n-memory 16 --lambda-ff9 1.0 \
  --ff9-rollout 24 --ff9-rollout-tbptt 12 --ff9-rollout-hide-mode tail \
  --lambda-ff9-rollout 1.0 --ff9-rollout-warmup 20 \
  --wandb --wandb-name gridworld-ff9roll-m24-M16-s0
cp checkpoints/gridworld/dynamics_ff9roll_m24_M16.pt "$OUT/"
echo "=== EXP-033 done ==="
