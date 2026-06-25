#!/usr/bin/env bash
# EXP-030 | FF9 rollout-training (op-3 memory->memory relay, D-048) — MODERATE depth (the clean
# demonstration). Trains the relay to ~24 hops (1.5x the 16-frame window) so recall should extend
# clearly past the EXP-027/028 window cliff. Budget-matched to EXP-027/028 (bs64 lr3e-4 80ep seed0
# window16). FF9 sufficiency (--ff9 3) contains state; the rollout term (warmup 0->1 over 20 ep)
# trains the cross-window relay. hide=tail mirrors the eval's contiguous occlusion. Frozen tokenizer.
set -euo pipefail
TOK="runs/gridworld-tok-v3/tokenizer.pt"; [ -f "$TOK" ] || TOK="checkpoints/gridworld/tokenizer.pt"
echo "=== tokenizer: $TOK ==="; ls -la "$TOK"
[ -f gridworld.npy ] || python -u src/datagen/generate_gridworld.py --n_episodes 3000 --n_frames 200 --out gridworld.npy
OUT="runs/gridworld-ff9roll-m24-s0"; mkdir -p "$OUT" checkpoints/gridworld
python -u src/training/train_dynamics.py \
  --frames gridworld.npy --tokenizer "$TOK" \
  --checkpoint checkpoints/gridworld/dynamics_ff9roll_m24.pt \
  --epochs 80 --batch-size 64 --lr 3e-4 --seed 0 --fresh \
  --context-length 16 --rollout-clip-len 28 \
  --ff9 3 --n-memory 4 --lambda-ff9 1.0 \
  --ff9-rollout 24 --ff9-rollout-tbptt 12 --ff9-rollout-hide-mode tail \
  --lambda-ff9-rollout 1.0 --ff9-rollout-warmup 20 \
  --wandb --wandb-name gridworld-ff9roll-m24-s0
cp checkpoints/gridworld/dynamics_ff9roll_m24.pt "$OUT/"
echo "=== EXP-030 done ==="
