#!/usr/bin/env bash
# Ferranti Phase 0/1: setup, real-data conversion, H100 batch calibration, tokenizer+dynamics smoke.
set -euo pipefail

ROOT="$(pwd)"
EXP="$ROOT/experiments/dreamer4-community-baseline"
RUN_NAME="${SLURM_JOB_NAME:-memmaze-d4-phase1-smoke}"
RUN_DIR="$ROOT/runs/$RUN_NAME"
BASE="$ROOT/runs/dreamer4-community-baseline"
RAW="$ROOT/data/memmaze9x9_raw"
TRAIN_OUT="$ROOT/data/d4_memmaze_community/train-part0-v1"
EVAL_OUT="$ROOT/data/d4_memmaze_community/eval-v1"
mkdir -p "$RUN_DIR"

bash "$EXP/setup_upstream.sh"
# shellcheck disable=SC1091
source "$BASE/current.env"
cp -a "$D4_PROVENANCE" "$RUN_DIR/provenance"

echo "timestamp_utc=$(date -u +%FT%TZ)" > "$RUN_DIR/phase1_config.txt"
echo "project_commit=$(git rev-parse HEAD)" >> "$RUN_DIR/phase1_config.txt"
echo "upstream_commit=$(git -C "$D4_ROOT" rev-parse HEAD)" >> "$RUN_DIR/phase1_config.txt"
echo "train_raw=$RAW/train-part0" >> "$RUN_DIR/phase1_config.txt"
echo "eval_raw=$RAW/eval" >> "$RUN_DIR/phase1_config.txt"

if ! find "$RAW/train-part0" -type f -name '*.npz' -print -quit 2>/dev/null | grep -q .; then
  "$D4_PYTHON" -u "$ROOT/experiments/memmaze-tokenizer/download_memmaze.py" \
    --parts train-part0 --out-dir "$RAW" --unzip
fi
if ! find "$RAW/eval" -type f -name '*.npz' -print -quit 2>/dev/null | grep -q .; then
  "$D4_PYTHON" -u "$ROOT/experiments/memmaze-tokenizer/download_memmaze.py" \
    --parts eval --out-dir "$RAW" --unzip
fi

if [ ! -f "$TRAIN_OUT/conversion_manifest.json" ]; then
  "$D4_PYTHON" -u "$EXP/memmaze_to_dreamer4.py" \
    --raw "$RAW/train-part0" --out-dir "$TRAIN_OUT" --task memmaze --shard-size 2048
fi
if [ ! -f "$EVAL_OUT/conversion_manifest.json" ]; then
  "$D4_PYTHON" -u "$EXP/memmaze_to_dreamer4.py" \
    --raw "$RAW/eval" --out-dir "$EVAL_OUT" --task memmaze --shard-size 2048
fi
"$D4_PYTHON" -u "$EXP/validate_converted.py" \
  --root "$TRAIN_OUT" --compare-other "$EVAL_OUT" | tee "$RUN_DIR/conversion_validation.log"

nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv -l 10 > "$RUN_DIR/gpu_samples.csv" &
MON_PID=$!
cleanup() { kill "$MON_PID" >/dev/null 2>&1 || true; }
trap cleanup EXIT

TOK_BS=""
for bs in 256 128 64 32 16; do
  dir="$RUN_DIR/tok-bs-$bs"
  echo "=== tokenizer batch calibration bs=$bs ==="
  if "$D4_PYTHON" -u "$D4_ROOT/dreamer4/train_tokenizer.py" \
      --data_dirs "$TRAIN_OUT/shards" --H 64 --W 64 --patch 4 --seq_len 8 \
      --batch_size "$bs" --num_workers 8 --max_steps 20 --lpips_weight 0.2 \
      --log_every 10 --print_every 10 --viz_every 0 --save_every 0 \
      --wandb_mode disabled --ckpt_dir "$dir" 2>&1 | tee "$RUN_DIR/tok-bs-$bs.log"; then
    TOK_BS="$bs"
    break
  fi
done
test -n "$TOK_BS" || { echo "No tokenizer batch size fit" >&2; exit 1; }
echo "tokenizer_batch_size=$TOK_BS" >> "$RUN_DIR/phase1_config.txt"

TOK_DIR="$RUN_DIR/tokenizer_smoke"
"$D4_PYTHON" -u "$D4_ROOT/dreamer4/train_tokenizer.py" \
  --data_dirs "$TRAIN_OUT/shards" --H 64 --W 64 --patch 4 --seq_len 8 \
  --batch_size "$TOK_BS" --num_workers 8 --max_hours 0.0833333 --lpips_weight 0.2 \
  --log_every 50 --print_every 50 --viz_every 0 --save_every 200 \
  --wandb_mode disabled --ckpt_dir "$TOK_DIR" 2>&1 | tee "$RUN_DIR/tokenizer_smoke.log"
"$D4_PYTHON" "$EXP/summarize_checkpoint.py" "$TOK_DIR/final.pt" \
  --out "$RUN_DIR/tokenizer_smoke_summary.json"
"$D4_PYTHON" -u "$EXP/make_recon_sheet.py" --dreamer4 "$D4_ROOT" \
  --checkpoint "$TOK_DIR/final.pt" --raw-eval "$RAW/eval" \
  --out "$RUN_DIR/tokenizer_smoke_recon.png"

DYN_BS=""
for bs in 128 64 32 16 8; do
  dir="$RUN_DIR/dyn-bs-$bs"
  echo "=== dynamics batch calibration bs=$bs ==="
  if "$D4_PYTHON" -u "$D4_ROOT/dreamer4/train_dynamics.py" --use_actions \
      --frame_dirs "$TRAIN_OUT/shards" --data_dirs "$TRAIN_OUT/demos" \
      --tokenizer_ckpt "$TOK_DIR/final.pt" --img_size 64 --seq_len 32 \
      --batch_size "$bs" --num_workers 8 --max_steps 10 --eval_every 0 \
      --log_every 5 --save_every 0 --wandb_mode disabled --tasks_json __none__ \
      --ckpt_dir "$dir" 2>&1 | tee "$RUN_DIR/dyn-bs-$bs.log"; then
    DYN_BS="$bs"
    break
  fi
done
test -n "$DYN_BS" || { echo "No dynamics batch size fit" >&2; exit 1; }
echo "dynamics_batch_size=$DYN_BS" >> "$RUN_DIR/phase1_config.txt"

DYN_DIR="$RUN_DIR/dynamics_smoke"
"$D4_PYTHON" -u "$D4_ROOT/dreamer4/train_dynamics.py" --use_actions \
  --frame_dirs "$TRAIN_OUT/shards" --data_dirs "$TRAIN_OUT/demos" \
  --tokenizer_ckpt "$TOK_DIR/final.pt" --img_size 64 --seq_len 32 \
  --batch_size "$DYN_BS" --num_workers 8 --max_hours 0.166667 \
  --eval_every 200 --eval_ctx 8 --eval_horizon 16 --log_every 25 --save_every 200 \
  --wandb_mode disabled --tasks_json __none__ --ckpt_dir "$DYN_DIR" \
  2>&1 | tee "$RUN_DIR/dynamics_smoke.log"
"$D4_PYTHON" "$EXP/summarize_checkpoint.py" "$DYN_DIR/final.pt" \
  --out "$RUN_DIR/dynamics_smoke_summary.json"

cleanup
trap - EXIT
echo "PHASE 1 SMOKE PASSED"
