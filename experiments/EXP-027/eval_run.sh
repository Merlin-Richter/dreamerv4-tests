#!/usr/bin/env bash
# EXP-027 recall eval (D-046) — vanilla GridWorld dynamics on the 150 held-out val episodes.
# Runs on the cluster where checkpoints/gridworld/dynamics_vanilla.pt lives. Writes results into the
# run dir (pullable) and stages the dynamics checkpoint back for archival.
set -euo pipefail
TOK="runs/gridworld-tok-v3/tokenizer.pt"; [ -f "$TOK" ] || TOK="checkpoints/gridworld/tokenizer.pt"
DYN="checkpoints/gridworld/dynamics_vanilla.pt"
echo "=== tokenizer: $TOK | dynamics: $DYN ==="
ls -la "$TOK" "$DYN"
[ -f gridworld.npy ] || python -u src/datagen/generate_gridworld.py --n_episodes 3000 --n_frames 200 --out gridworld.npy
OUT="runs/gridworld-vanilla-s0-eval"
mkdir -p "$OUT"
python -u experiments/EXP-027/eval.py --tokenizer "$TOK" --dynamics "$DYN" --out-dir "$OUT"
cp "$DYN" "$OUT/dynamics_vanilla.pt"   # stage authoritative checkpoint for pull_results (D-031)
echo "=== EXP-027 eval done ==="
