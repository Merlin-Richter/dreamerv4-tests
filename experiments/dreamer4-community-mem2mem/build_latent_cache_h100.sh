#!/usr/bin/env bash
# One-time exact community-tokenizer window cache. This is outside the 48h dynamics budget.
set -euo pipefail

ROOT="$(pwd)"
EXP="$ROOT/experiments/dreamer4-community-mem2mem"
BASE_EXP="$ROOT/experiments/dreamer4-community-baseline"
BASE="$ROOT/runs/dreamer4-community-mem2mem"
RUN_NAME="${SLURM_JOB_NAME:-memmaze-community-d4-window-cache}"
RUN_DIR="$ROOT/runs/$RUN_NAME"
TRAIN_OUT="$ROOT/data/d4_memmaze_community/train-part0-v2"
EVAL_OUT="$ROOT/data/d4_memmaze_community/eval-v2"
CACHE_ROOT="$ROOT/data/d4_memmaze_community/train-part0-v2-community-window32-fp32"
TOK_CKPT="$ROOT/runs/memmaze-d4-tokenizer-24h/tokenizer/final.pt"
mkdir -p "$RUN_DIR"

bash "$EXP/setup_upstream.sh"
# shellcheck disable=SC1091
source "$BASE/current.env"
mkdir -p "$RUN_DIR/provenance"
cp -a "$D4_PROVENANCE"/. "$RUN_DIR/provenance"/

"$D4_PYTHON" -u "$BASE_EXP/validate_converted.py" \
  --root "$TRAIN_OUT" --compare-other "$EVAL_OUT" | tee "$RUN_DIR/conversion-validation.log"
"$D4_PYTHON" -u "$EXP/validate_data.py" \
  --dreamer4 "$D4_ROOT" --train-root "$TRAIN_OUT" --eval-root "$EVAL_OUT" \
  --tokenizer "$TOK_CKPT" --report "$RUN_DIR/data-identity.json"

"$D4_PYTHON" -u "$EXP/build_latent_cache.py" \
  --dreamer4 "$D4_ROOT" --data-dirs "$TRAIN_OUT/demos" --frame-dirs "$TRAIN_OUT/shards" \
  --tokenizer "$TOK_CKPT" --train-manifest "$TRAIN_OUT/conversion_manifest.json" \
  --out "$CACHE_ROOT" --window 32 --packing-factor 2 --batch-size 64 \
  --num-workers 4 --cache-mb 128 --shard-size 2048 \
  2>&1 | tee -a "$RUN_DIR/cache-build.log"

"$D4_PYTHON" -u "$EXP/validate_latent_cache.py" \
  --dreamer4 "$D4_ROOT" --data-dirs "$TRAIN_OUT/demos" --frame-dirs "$TRAIN_OUT/shards" \
  --tokenizer "$TOK_CKPT" --train-manifest "$TRAIN_OUT/conversion_manifest.json" \
  --cache "$CACHE_ROOT" --report "$RUN_DIR/cache-validation.json" \
  --reference-batch-size 64 --comparison-batch-sizes 24 128 \
  --require-bit-exact-comparison-batches 24 --max-singleton-abs 0.002 \
  --max-replay-abs 0 --max-comparison-abs 0.002 --max-comparison-relative-l2 0.0005 \
  2>&1 | tee "$RUN_DIR/cache-validation.log"
cp "$CACHE_ROOT/manifest.json" "$RUN_DIR/cache-manifest.json"
sha256sum "$CACHE_ROOT/manifest.json" "$CACHE_ROOT/row-by-start.npy" \
  "$TOK_CKPT" > "$RUN_DIR/checksums.sha256"
echo "COMMUNITY WINDOW LATENT CACHE PASSED: $CACHE_ROOT"
