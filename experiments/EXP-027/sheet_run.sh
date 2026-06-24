#!/usr/bin/env bash
# EXP-027 qualitative rollout sheets (vanilla GridWorld dynamics). cv2-only (no matplotlib).
# Runs where dynamics_vanilla.pt lives; writes sheets into the run dir + stages the checkpoint back.
set -euo pipefail
TOK="runs/gridworld-tok-v3/tokenizer.pt"; [ -f "$TOK" ] || TOK="checkpoints/gridworld/tokenizer.pt"
DYN="checkpoints/gridworld/dynamics_vanilla.pt"
echo "=== tokenizer: $TOK | dynamics: $DYN ==="
ls -la "$TOK" "$DYN"
[ -f gridworld.npy ] || python -u src/datagen/generate_gridworld.py --n_episodes 3000 --n_frames 200 --out gridworld.npy
OUT="runs/gridworld-vanilla-s0-sheets"
mkdir -p "$OUT"
python -u experiments/EXP-027/make_sheets.py --tokenizer "$TOK" --dynamics "$DYN" --out-dir "$OUT" --n-samples 6
cp "$DYN" "$OUT/dynamics_vanilla.pt"   # stage authoritative checkpoint for pull_results (D-031)
echo "=== sheets done ==="
