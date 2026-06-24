#!/usr/bin/env bash
# EXP-028 env-direct A/B recall (vanilla vs FF9) + FF9 sheets. No dataset needed (env-direct).
# Run AFTER FF9 training (needs checkpoints/gridworld/dynamics_ff9.pt). Writes into the run dir for pull.
set -euo pipefail
TOK="runs/gridworld-tok-v3/tokenizer.pt"; [ -f "$TOK" ] || TOK="checkpoints/gridworld/tokenizer.pt"
VAN="checkpoints/gridworld/dynamics_vanilla.pt"
FF9="checkpoints/gridworld/dynamics_ff9.pt"
echo "=== tokenizer: $TOK ==="; ls -la "$TOK" "$VAN" "$FF9"
OUT="runs/gridworld-recall-env"
mkdir -p "$OUT"
python -u experiments/EXP-028/recall_env.py --tokenizer "$TOK" --dynamics "$VAN" --tag vanilla --out-dir "$OUT"
python -u experiments/EXP-028/recall_env.py --tokenizer "$TOK" --dynamics "$FF9" --tag ff9     --out-dir "$OUT"
# FF9 qualitative sheets (occlusion belief + normal), env-based; needs gridworld.npy only for NORMAL.
[ -f gridworld.npy ] || python -u src/datagen/generate_gridworld.py --n_episodes 3000 --n_frames 200 --out gridworld.npy
python -u experiments/EXP-027/make_sheets.py --tokenizer "$TOK" --dynamics "$FF9" --out-dir "$OUT" --n-samples 6 || true
echo "=== EXP-028 recall A/B + FF9 sheets done ==="
