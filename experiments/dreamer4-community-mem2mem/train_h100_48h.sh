#!/usr/bin/env bash
# Production: exactly 48 whole-training-loop H100 hours, matching community vanilla.
set -euo pipefail

ROOT="$(pwd)"
EXP="$ROOT/experiments/dreamer4-community-mem2mem"
BASE_EXP="$ROOT/experiments/dreamer4-community-baseline"
BASE="$ROOT/runs/dreamer4-community-mem2mem"
RUN_NAME="${SLURM_JOB_NAME:-memmaze-community-d4-mem2mem-48h}"
RUN_DIR="$ROOT/runs/$RUN_NAME"
TRAIN_OUT="$ROOT/data/d4_memmaze_community/train-part0-v2"
EVAL_OUT="$ROOT/data/d4_memmaze_community/eval-v2"
TOK_CKPT="$ROOT/runs/memmaze-d4-tokenizer-24h/tokenizer/final.pt"
CACHE_ROOT="$ROOT/data/d4_memmaze_community/train-part0-v2-community-window32-fp32"
CACHE_MANIFEST_SHA256="e7a4e57e63e357d1986154a1b6c3cea9f4220b1a716e4a553df8a345fb2f4fcf"
VANILLA_CKPT="${D4_VANILLA_CKPT:-$ROOT/runs/memmaze-d4-dynamics-48h-v3/dynamics/final.pt}"
BATCH_SIZE=24
NUM_WORKERS=4
CACHE_MB=128
TRAINING_SECONDS=172800
mkdir -p "$RUN_DIR"

bash "$EXP/setup_upstream.sh"
# shellcheck disable=SC1091
source "$BASE/current.env"
mkdir -p "$RUN_DIR/provenance"
cp -a "$D4_PROVENANCE"/. "$RUN_DIR/provenance"/
cp "$EXP/production-config.json" "$RUN_DIR/production-config.json"

test -f "$TOK_CKPT"
test -f "$VANILLA_CKPT"
test -f "$CACHE_ROOT/manifest.json"
test "$(sha256sum "$CACHE_ROOT/manifest.json" | awk '{print $1}')" = "$CACHE_MANIFEST_SHA256"
"$D4_PYTHON" -u "$BASE_EXP/validate_converted.py" \
  --root "$TRAIN_OUT" --compare-other "$EVAL_OUT" | tee "$RUN_DIR/conversion-validation.log"
"$D4_PYTHON" -u "$EXP/validate_data.py" \
  --dreamer4 "$D4_ROOT" --train-root "$TRAIN_OUT" --eval-root "$EVAL_OUT" \
  --tokenizer "$TOK_CKPT" --report "$RUN_DIR/data-identity.json"
"$D4_PYTHON" -u "$EXP/validate_model.py" \
  --dreamer4 "$D4_ROOT" --vanilla-checkpoint "$VANILLA_CKPT" | tee "$RUN_DIR/model-gates.log"
"$D4_PYTHON" -u "$EXP/validate_latent_cache.py" \
  --dreamer4 "$D4_ROOT" --data-dirs "$TRAIN_OUT/demos" --frame-dirs "$TRAIN_OUT/shards" \
  --tokenizer "$TOK_CKPT" --train-manifest "$TRAIN_OUT/conversion_manifest.json" \
  --cache "$CACHE_ROOT" --report "$RUN_DIR/cache-validation.json" --full-hash \
  --reference-batch-size 64 --comparison-batch-sizes 24 128 \
  --require-bit-exact-comparison-batches 24 --max-singleton-abs 0.002 \
  --max-replay-abs 0 --max-comparison-abs 0.002 --max-comparison-relative-l2 0.0005 \
  2>&1 | tee "$RUN_DIR/cache-validation.log"

LATEST="$RUN_DIR/memory-latest.pt"
FINAL="$RUN_DIR/memory-final.pt"
LEDGER="$RUN_DIR/training-clock.jsonl"
GPU_LOG="$RUN_DIR/gpu-samples-$(date -u +%Y%m%dT%H%M%SZ).csv"
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv -l 10 > "$GPU_LOG" &
MON_PID=$!
cleanup() { kill "$MON_PID" >/dev/null 2>&1 || true; }
trap cleanup EXIT

RESUME=()
if [ -f "$LATEST" ]; then
  RESUME=(--resume "$LATEST")
fi
if [ ! -f "$FINAL" ]; then
  "$D4_PYTHON" -u "$EXP/train_mem2mem.py" \
    --dreamer4 "$D4_ROOT" \
    --frame-dirs "$TRAIN_OUT/shards" --data-dirs "$TRAIN_OUT/demos" \
    --tokenizer "$TOK_CKPT" --latent-cache "$CACHE_ROOT" \
    --expected-latent-cache-manifest-sha256 "$CACHE_MANIFEST_SHA256" \
    --checkpoint "$LATEST" --training-ledger "$LEDGER" \
    --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" --cache-mb "$CACHE_MB" \
    --shard-size 2048 --window 32 --clip-length 128 --tbptt-frames 64 \
    --n-memory 8 --k-max 8 --bootstrap-start 5000 --self-fraction 0.25 \
    --d-model 512 --depth 8 --n-heads 4 --n-register 4 --n-agent 1 --time-every 1 \
    --packing-factor 2 --lr 1e-4 --weight-decay 1e-2 --grad-clip 1.0 --seed 0 \
    --max-hours 48 --lr-schedule-hours 48 --save-every-hours 1 --wandb-mode online \
    --wandb-run-name "$RUN_NAME" "${RESUME[@]}" \
    2>&1 | tee -a "$RUN_DIR/train.log"
  "$D4_PYTHON" -u "$EXP/validate_final_checkpoint.py" \
    --checkpoint "$LATEST" --expected-training-seconds "$TRAINING_SECONDS" \
    --expected-latent-cache-manifest-sha256 "$CACHE_MANIFEST_SHA256" \
    --out "$RUN_DIR/final-checkpoint-summary.json"
  cp "$LATEST" "$FINAL.tmp"
  mv "$FINAL.tmp" "$FINAL"
fi

cleanup
trap - EXIT
"$D4_PYTHON" -u "$EXP/summarize_telemetry.py" \
  --gpu-csv "$RUN_DIR"/gpu-samples-*.csv \
  --training-ledger "$LEDGER" --out "$RUN_DIR/telemetry-summary.json"
"$D4_PYTHON" -u "$EXP/validate_final_checkpoint.py" \
  --checkpoint "$FINAL" --expected-training-seconds "$TRAINING_SECONDS" \
  --expected-latent-cache-manifest-sha256 "$CACHE_MANIFEST_SHA256" \
  --out "$RUN_DIR/final-checkpoint-summary.json"
sha256sum "$FINAL" "$TOK_CKPT" "$VANILLA_CKPT" > "$RUN_DIR/checksums.sha256"
echo "MEM2MEM 48-TRAINING-WALL-HOUR TRAINING PASSED"
