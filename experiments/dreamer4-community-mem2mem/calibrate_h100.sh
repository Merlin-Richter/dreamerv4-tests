#!/usr/bin/env bash
# Short H100 footprint/utilization + real checkpoint/resume calibration.
set -euo pipefail

ROOT="$(pwd)"
EXP="$ROOT/experiments/dreamer4-community-mem2mem"
BASE_EXP="$ROOT/experiments/dreamer4-community-baseline"
BASE="$ROOT/runs/dreamer4-community-mem2mem"
RUN_NAME="${SLURM_JOB_NAME:-memmaze-community-d4-mem2mem-calibrate}"
RUN_DIR="$ROOT/runs/$RUN_NAME"
TRAIN_OUT="$ROOT/data/d4_memmaze_community/train-part0-v2"
EVAL_OUT="$ROOT/data/d4_memmaze_community/eval-v2"
TOK_CKPT="$ROOT/runs/memmaze-d4-tokenizer-24h/tokenizer/final.pt"
VANILLA_CKPT="${D4_VANILLA_CKPT:-$ROOT/runs/memmaze-d4-dynamics-48h-v3/dynamics/final.pt}"
BATCH_SIZE="${D4_BATCH_SIZE:-4}"
NUM_WORKERS="${D4_NUM_WORKERS:-4}"
CACHE_MB="${D4_CACHE_MB:-128}"
FIRST_HOURS="${D4_FIRST_HOURS:-0.05}"
FINAL_HOURS="${D4_FINAL_HOURS:-0.10}"
mkdir -p "$RUN_DIR"

bash "$EXP/setup_upstream.sh"
# shellcheck disable=SC1091
source "$BASE/current.env"
cp -a "$D4_PROVENANCE" "$RUN_DIR/provenance"

test -f "$TOK_CKPT"
test -f "$VANILLA_CKPT"
"$D4_PYTHON" -u "$BASE_EXP/validate_converted.py" \
  --root "$TRAIN_OUT" --compare-other "$EVAL_OUT" | tee "$RUN_DIR/conversion-validation.log"
"$D4_PYTHON" -u "$EXP/validate_data.py" \
  --dreamer4 "$D4_ROOT" --train-root "$TRAIN_OUT" --eval-root "$EVAL_OUT" \
  --tokenizer "$TOK_CKPT" --report "$RUN_DIR/data-identity.json"
"$D4_PYTHON" -u "$EXP/validate_model.py" \
  --dreamer4 "$D4_ROOT" --vanilla-checkpoint "$VANILLA_CKPT" | tee "$RUN_DIR/model-gates.log"
"$D4_PYTHON" -u "$EXP/validate_resume.py" \
  --dreamer4 "$D4_ROOT" | tee "$RUN_DIR/resume-gate.log"

SCRATCH_BASE="${SLURM_TMPDIR:-${TMPDIR:-}}"
test -n "$SCRATCH_BASE" && test -d "$SCRATCH_BASE"
TRAIN_RUNTIME="$SCRATCH_BASE/d4_memmaze_community_mem2mem_train"
mkdir -p "$TRAIN_RUNTIME"
cp -a "$TRAIN_OUT/shards" "$TRAIN_RUNTIME/"
cp -a "$TRAIN_OUT/demos" "$TRAIN_RUNTIME/"

CKPT="$RUN_DIR/memory-latest.pt"
LEDGER="$RUN_DIR/active-clock.jsonl"
GPU_LOG="$RUN_DIR/gpu-samples.csv"
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv -l 10 > "$GPU_LOG" &
MON_PID=$!
cleanup() { kill "$MON_PID" >/dev/null 2>&1 || true; }
trap cleanup EXIT

COMMON=(
  --dreamer4 "$D4_ROOT"
  --frame-dirs "$TRAIN_RUNTIME/shards"
  --data-dirs "$TRAIN_RUNTIME/demos"
  --tokenizer "$TOK_CKPT"
  --checkpoint "$CKPT"
  --active-ledger "$LEDGER"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --cache-mb "$CACHE_MB"
  --save-every-hours 0.025
  --wandb-mode disabled
)

"$D4_PYTHON" -u "$EXP/train_mem2mem.py" "${COMMON[@]}" --max-hours "$FIRST_HOURS" \
  2>&1 | tee "$RUN_DIR/train-first.log"
"$D4_PYTHON" -u "$EXP/train_mem2mem.py" "${COMMON[@]}" --max-hours "$FINAL_HOURS" --resume "$CKPT" \
  2>&1 | tee "$RUN_DIR/train-resume.log"

cleanup
trap - EXIT
"$D4_PYTHON" -u "$EXP/summarize_telemetry.py" \
  --gpu-csv "$GPU_LOG" --active-ledger "$LEDGER" --out "$RUN_DIR/telemetry-summary.json"
sha256sum "$CKPT" "$TOK_CKPT" "$VANILLA_CKPT" > "$RUN_DIR/checksums.sha256"
echo "H100 CALIBRATION PASSED batch_size=$BATCH_SIZE active_hours=$FINAL_HOURS"
