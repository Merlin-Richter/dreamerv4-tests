#!/usr/bin/env bash
# Quantify batch-shape numerical differences without rebuilding the durable cache.
set -euo pipefail

ROOT="$(pwd)"
EXP="$ROOT/experiments/dreamer4-community-mem2mem"
BASE="$ROOT/runs/dreamer4-community-mem2mem"
RUN_NAME="${SLURM_JOB_NAME:-memmaze-community-d4-window-cache-diagnostic}"
RUN_DIR="$ROOT/runs/$RUN_NAME"
TRAIN_OUT="$ROOT/data/d4_memmaze_community/train-part0-v2"
CACHE_ROOT="$ROOT/data/d4_memmaze_community/train-part0-v2-community-window32-fp32"
TOK_CKPT="$ROOT/runs/memmaze-d4-tokenizer-24h/tokenizer/final.pt"
mkdir -p "$RUN_DIR"

bash "$EXP/setup_upstream.sh"
# shellcheck disable=SC1091
source "$BASE/current.env"
test -f "$CACHE_ROOT/manifest.json"

"$D4_PYTHON" -u "$EXP/validate_latent_cache.py" \
  --dreamer4 "$D4_ROOT" --data-dirs "$TRAIN_OUT/demos" --frame-dirs "$TRAIN_OUT/shards" \
  --tokenizer "$TOK_CKPT" --train-manifest "$TRAIN_OUT/conversion_manifest.json" \
  --cache "$CACHE_ROOT" --report "$RUN_DIR/cache-numerical-diagnostic.json" \
  --reference-batch-size 64 --comparison-batch-sizes 24 128 \
  --max-singleton-abs inf --max-replay-abs inf \
  2>&1 | tee "$RUN_DIR/cache-numerical-diagnostic.log"
cp "$CACHE_ROOT/manifest.json" "$RUN_DIR/cache-manifest.json"
sha256sum "$CACHE_ROOT/manifest.json" "$CACHE_ROOT/row-by-start.npy" "$TOK_CKPT" \
  > "$RUN_DIR/checksums.sha256"
echo "CACHE NUMERICAL DIAGNOSTIC PASSED"
