#!/usr/bin/env bash
# Ferranti Phase 3: 48 active H100 hours of action-conditioned community-Dreamer4 dynamics training.
set -euo pipefail

ROOT="$(pwd)"
EXP="$ROOT/experiments/dreamer4-community-baseline"
RUN_NAME="${SLURM_JOB_NAME:-memmaze-d4-dynamics-48h}"
RUN_DIR="$ROOT/runs/$RUN_NAME"
BASE="$ROOT/runs/dreamer4-community-baseline"
TRAIN_OUT="$ROOT/data/d4_memmaze_community/train-part0-v2"
EVAL_OUT="$ROOT/data/d4_memmaze_community/eval-v2"
TOK_CKPT="$ROOT/runs/memmaze-d4-tokenizer-24h/tokenizer/final.pt"
TOK_EXPECTED_SHA256="347052fae0212ea2c6b943ae7c28a886298ce551d4155b882084d63a3ea48797"
DYN_DIR="$RUN_DIR/dynamics"
mkdir -p "$RUN_DIR"

bash "$EXP/setup_upstream.sh"
# shellcheck disable=SC1091
source "$BASE/current.env"
cp -a "$D4_PROVENANCE" "$RUN_DIR/provenance"

test -f "$TRAIN_OUT/conversion_manifest.json"
test -f "$EVAL_OUT/conversion_manifest.json"
test -f "$TOK_CKPT"
"$D4_PYTHON" -u "$EXP/validate_converted.py" \
  --root "$TRAIN_OUT" --compare-other "$EVAL_OUT" | tee "$RUN_DIR/conversion_validation.log"

TOK_ACTUAL_SHA256="$(sha256sum "$TOK_CKPT" | cut -d ' ' -f 1)"
if [ "$TOK_ACTUAL_SHA256" != "$TOK_EXPECTED_SHA256" ]; then
  echo "Approved tokenizer hash mismatch: expected=$TOK_EXPECTED_SHA256 actual=$TOK_ACTUAL_SHA256" >&2
  exit 1
fi
"$D4_PYTHON" "$EXP/summarize_checkpoint.py" "$TOK_CKPT" \
  --out "$RUN_DIR/tokenizer_input_summary.json"

{
  echo "timestamp_utc=$(date -u +%FT%TZ)"
  echo "project_commit=$(git rev-parse HEAD)"
  echo "upstream_commit=$(git -C "$D4_ROOT" rev-parse HEAD)"
  echo "train_data=$TRAIN_OUT"
  echo "heldout_data=$EVAL_OUT"
  echo "tokenizer_checkpoint=$TOK_CKPT"
  echo "tokenizer_sha256=$TOK_ACTUAL_SHA256"
  echo "active_training_budget_hours=48"
  echo "action_conditioning=true"
  echo "action_alignment=raw_action_t_produced_raw_image_t"
  echo "sequence_length=32"
  echo "batch_size=128"
  echo "bootstrap_start_step=5000"
  echo "log_every_steps=100"
  echo "train_batch_rollout_eval_every_steps=1000"
  echo "checkpoint_every_steps=5000"
} > "$RUN_DIR/phase3_config.txt"

GPU_LOG="$RUN_DIR/gpu_samples_$(date -u +%Y%m%dT%H%M%SZ).csv"
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv -l 10 > "$GPU_LOG" &
MON_PID=$!
cleanup() { kill "$MON_PID" >/dev/null 2>&1 || true; }
trap cleanup EXIT

RESUME_ARGS=()
if [ -f "$DYN_DIR/latest.pt" ] && [ ! -f "$DYN_DIR/final.pt" ]; then
  RESUME_ARGS=(--resume "$DYN_DIR/latest.pt")
fi

if [ ! -f "$DYN_DIR/final.pt" ]; then
  "$D4_PYTHON" -u "$D4_ROOT/dreamer4/train_dynamics.py" --use_actions \
    --frame_dirs "$TRAIN_OUT/shards" --data_dirs "$TRAIN_OUT/demos" \
    --tokenizer_ckpt "$TOK_CKPT" --img_size 64 --seq_len 32 \
    --batch_size 128 --num_workers 8 --max_hours 48 \
    --bootstrap_start 5000 --eval_every 1000 --eval_batch_size 4 \
    --eval_ctx 8 --eval_horizon 16 --log_every 100 --save_every 5000 \
    --wandb_mode disabled --wandb_run_name "$RUN_NAME" --tasks_json __none__ \
    --ckpt_dir "$DYN_DIR" "${RESUME_ARGS[@]}" \
    2>&1 | tee -a "$RUN_DIR/dynamics_train.log"
fi

"$D4_PYTHON" "$EXP/summarize_checkpoint.py" "$DYN_DIR/final.pt" \
  --out "$RUN_DIR/dynamics_summary.json"

cleanup
trap - EXIT
echo "PHASE 3 DYNAMICS PASSED"
