#!/usr/bin/env bash
# Ferranti Phase 2: 24 active H100 hours of community-Dreamer4 tokenizer training.
set -euo pipefail

ROOT="$(pwd)"
EXP="$ROOT/experiments/dreamer4-community-baseline"
RUN_NAME="${SLURM_JOB_NAME:-memmaze-d4-tokenizer-24h}"
RUN_DIR="$ROOT/runs/$RUN_NAME"
BASE="$ROOT/runs/dreamer4-community-baseline"
RAW="$ROOT/data/memmaze9x9_raw"
TRAIN_OUT="$ROOT/data/d4_memmaze_community/train-part0-v2"
EVAL_OUT="$ROOT/data/d4_memmaze_community/eval-v2"
TOK_DIR="$RUN_DIR/tokenizer"
mkdir -p "$RUN_DIR"

bash "$EXP/setup_upstream.sh"
# shellcheck disable=SC1091
source "$BASE/current.env"
cp -a "$D4_PROVENANCE" "$RUN_DIR/provenance"

test -f "$TRAIN_OUT/conversion_manifest.json"
test -f "$EVAL_OUT/conversion_manifest.json"
"$D4_PYTHON" -u "$EXP/validate_converted.py" \
  --root "$TRAIN_OUT" --compare-other "$EVAL_OUT" | tee "$RUN_DIR/conversion_validation.log"

{
  echo "timestamp_utc=$(date -u +%FT%TZ)"
  echo "project_commit=$(git rev-parse HEAD)"
  echo "upstream_commit=$(git -C "$D4_ROOT" rev-parse HEAD)"
  echo "train_data=$TRAIN_OUT"
  echo "eval_data=$EVAL_OUT"
  echo "active_training_budget_hours=24"
  echo "batch_size=64"
  echo "checkpoint_every_steps=5000"
} > "$RUN_DIR/phase2_config.txt"

nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv -l 10 > "$RUN_DIR/gpu_samples.csv" &
MON_PID=$!
cleanup() { kill "$MON_PID" >/dev/null 2>&1 || true; }
trap cleanup EXIT

RESUME_ARGS=()
if [ -f "$TOK_DIR/latest.pt" ] && [ ! -f "$TOK_DIR/final.pt" ]; then
  RESUME_ARGS=(--resume "$TOK_DIR/latest.pt")
fi

if [ ! -f "$TOK_DIR/final.pt" ]; then
  "$D4_PYTHON" -u "$D4_ROOT/dreamer4/train_tokenizer.py" \
    --data_dirs "$TRAIN_OUT/shards" --H 64 --W 64 --patch 4 --seq_len 8 \
    --batch_size 64 --num_workers 8 --max_hours 24 --lpips_weight 0.2 \
    --log_every 50 --print_every 50 --viz_every 0 --save_every 5000 \
    --wandb_mode disabled --ckpt_dir "$TOK_DIR" "${RESUME_ARGS[@]}" \
    2>&1 | tee "$RUN_DIR/tokenizer_train.log"
fi

"$D4_PYTHON" "$EXP/summarize_checkpoint.py" "$TOK_DIR/final.pt" \
  --out "$RUN_DIR/tokenizer_summary.json"
"$D4_PYTHON" -u "$EXP/make_recon_sheet.py" --dreamer4 "$D4_ROOT" \
  --checkpoint "$TOK_DIR/final.pt" --raw-eval "$RAW/eval" \
  --out "$RUN_DIR/tokenizer_recon.png"

cleanup
trap - EXIT
echo "PHASE 2 TOKENIZER PASSED"
