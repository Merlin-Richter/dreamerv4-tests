#!/usr/bin/env bash
# EXP-028 | FF9 v2 (full-state memory token) GridWorld dynamics (D-047).
# Budget-matched to the EXP-027 vanilla baseline; only the FF9 objective differs. Frozen tokenizer.
set -euo pipefail
TOK="runs/gridworld-tok-v3/tokenizer.pt"; [ -f "$TOK" ] || TOK="checkpoints/gridworld/tokenizer.pt"
echo "=== tokenizer: $TOK ==="
ls -la "$TOK"
[ -f gridworld.npy ] || python -u src/datagen/generate_gridworld.py --n_episodes 3000 --n_frames 200 --out gridworld.npy
OUT="runs/gridworld-ff9-s0"
mkdir -p "$OUT" checkpoints/gridworld
python -u src/training/train_dynamics.py \
  --frames gridworld.npy \
  --tokenizer "$TOK" \
  --checkpoint checkpoints/gridworld/dynamics_ff9.pt \
  --epochs 80 --batch-size 64 --lr 3e-4 --seed 0 --fresh \
  --ff9 3 --n-memory 4 --lambda-ff9 1.0 \
  --wandb --wandb-name gridworld-ff9-s0
cp checkpoints/gridworld/dynamics_ff9.pt "$OUT/dynamics_ff9.pt"   # stage into run dir for pull_results
echo "=== EXP-028 FF9 training done ==="
