#!/usr/bin/env bash
# EXP-027 | vanilla GridWorld dynamics baseline (D-046)
# Cluster: ferranti (H100) | branch feat/motion-prediction | frozen tokenizer from EXP-025.
# Unmodified DreamerV4-style dynamics (no ff7/ff9/multistep) on the frozen GridWorld tokenizer.
set -euo pipefail

# Locate the frozen GridWorld tokenizer. EXP-025 saved it under runs/ on the cluster; the
# checkpoints/gridworld/ path is the local frozen copy. Fail fast (set -e + ls) if neither is
# present on the node (the tokenizer is a trained artifact, not regenerable here -> escalate).
TOK="runs/gridworld-tok-v3/tokenizer.pt"
[ -f "$TOK" ] || TOK="checkpoints/gridworld/tokenizer.pt"
echo "=== using tokenizer: $TOK ==="
ls -la "$TOK"

# Regenerate the gridworld dataset if absent. Deterministic (seed 42) -> byte-identical to the
# data EXP-025 trained the tokenizer on and EXP-026 evaluated.
if [ ! -f gridworld.npy ]; then
  echo "=== gridworld.npy absent -> regenerating (seed 42, == EXP-025 data) ==="
  python -u src/datagen/generate_gridworld.py --n_episodes 3000 --n_frames 200 --out gridworld.npy
fi

mkdir -p checkpoints/gridworld
python -u src/training/train_dynamics.py \
  --frames gridworld.npy \
  --tokenizer "$TOK" \
  --checkpoint checkpoints/gridworld/dynamics_vanilla.pt \
  --epochs 80 --batch-size 64 --lr 3e-4 --seed 0 --fresh \
  --wandb --wandb-name gridworld-vanilla-s0
echo "=== EXP-027 training done ==="
