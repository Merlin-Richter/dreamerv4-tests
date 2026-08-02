#!/usr/bin/env bash
# Ferranti Phase 4: held-out qualitative and action-specific rollout evaluation.
set -euo pipefail

ROOT="$(pwd)"
EXP="$ROOT/experiments/dreamer4-community-baseline"
BASE="$ROOT/runs/dreamer4-community-baseline"
EVAL_OUT="$ROOT/data/d4_memmaze_community/eval-v2"
TOK_CKPT="$ROOT/runs/memmaze-d4-tokenizer-24h/tokenizer/final.pt"
DYN_CKPT="$ROOT/runs/memmaze-d4-dynamics-48h-v3/dynamics/final.pt"
RUN_NAME="${SLURM_JOB_NAME:-memmaze-d4-heldout-eval}"
RUN_DIR="$ROOT/runs/$RUN_NAME"
mkdir -p "$RUN_DIR"

bash "$EXP/setup_upstream.sh"
# shellcheck disable=SC1091
source "$BASE/current.env"
cp -a "$D4_PROVENANCE" "$RUN_DIR/provenance"

test -f "$EVAL_OUT/conversion_manifest.json"
test -f "$TOK_CKPT"
test -f "$DYN_CKPT"

SCRATCH_BASE="${SLURM_TMPDIR:-${TMPDIR:-}}"
test -n "$SCRATCH_BASE" || { echo "No node-local SLURM_TMPDIR/TMPDIR available" >&2; exit 1; }
EVAL_RUNTIME="$SCRATCH_BASE/d4_memmaze_community_eval"
mkdir -p "$EVAL_RUNTIME"
cp -a "$EVAL_OUT/shards" "$EVAL_RUNTIME/"
cp -a "$EVAL_OUT/demos" "$EVAL_RUNTIME/"

"$D4_PYTHON" -u "$EXP/evaluate_dynamics.py" \
  --dreamer4 "$D4_ROOT" \
  --dynamics-checkpoint "$DYN_CKPT" \
  --tokenizer-checkpoint "$TOK_CKPT" \
  --data-dir "$EVAL_RUNTIME/demos" \
  --frames-dir "$EVAL_RUNTIME/shards" \
  --out-dir "$RUN_DIR" \
  --n-sequences 4 --ctx 8 --horizon 16 --device cuda

sha256sum "$DYN_CKPT" "$TOK_CKPT" > "$RUN_DIR/checkpoint_sha256.txt"
echo "PHASE 4 HELD-OUT EVALUATION PASSED"
