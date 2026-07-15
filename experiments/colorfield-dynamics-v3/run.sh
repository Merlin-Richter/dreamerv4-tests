#!/usr/bin/env bash
# Exact ColorField pixel-v3 dynamics arms. All memory rollouts are fixed at W=16.
set -euo pipefail

ARM="${1:?usage: run.sh vanilla-or-memory-base-or-memory-control-or-archive}"
MAX_STEPS="${MAX_STEPS:-20000}"
BUDGET_S="${BUDGET_S:-14400}"
BATCH_SIZE="${BATCH_SIZE:-32}"
COMMON=(
  --data data/colorfield --val data/colorfield_val
  --tokenizer checkpoints/colorfield/tokenizer.pt
  --budget-s "$BUDGET_S" --max-steps "$MAX_STEPS"
  --embedding-dim 128 --depth 6 --n-heads 8
  --fixed-n-ctx --seed 0 --epochs 1000
)

case "$ARM" in
  vanilla)
    CHECKPOINT="${CHECKPOINT:-checkpoints/colorfield/dynamics_vanilla_tau0.pt}"
    python -u autoresearch/editable/train.py "${COMMON[@]}" \
      --checkpoint "$CHECKPOINT" \
      --batch-size "${VANILLA_BATCH_SIZE:-128}" --clip-len 64 \
      --n-memory 0 --mem2mem-frac 0 --tau0-anchor 0.5 \
      --sched-steps "${VANILLA_SCHED_STEPS:-$MAX_STEPS}"
    ;;
  memory-base)
    CHECKPOINT="${CHECKPOINT:-checkpoints/colorfield/dynamics_memory_shared90m.pt}"
    python -u autoresearch/editable/train.py "${COMMON[@]}" \
      --checkpoint "$CHECKPOINT" \
      --batch-size "$BATCH_SIZE" --clip-len 256 --max-frames 256 \
      --tbptt-frames 32 --blockwise-rollout-backward \
      --n-memory 4 --mem2mem-frac 1 --ff9 0 --tau0-anchor 0 \
      --sched-steps "${BASE_SCHED_STEPS:-100000}"
    ;;
  memory-control)
    RESUME="${RESUME:-checkpoints/colorfield/dynamics_memory_shared90m.pt}"
    CHECKPOINT="${CHECKPOINT:-checkpoints/colorfield/dynamics_memory_rollout_noff9.pt}"
    python -u autoresearch/editable/train.py "${COMMON[@]}" \
      --resume "$RESUME" \
      --checkpoint "$CHECKPOINT" \
      --batch-size "$BATCH_SIZE" --clip-len 256 --max-frames 256 \
      --tbptt-frames 32 --blockwise-rollout-backward \
      --n-memory 4 --mem2mem-frac 1 --ff9 0 --tau0-anchor 0 \
      --sched-steps "${CONTROL_SCHED_STEPS:-$MAX_STEPS}"
    ;;
  archive)
    RESUME="${RESUME:-checkpoints/colorfield/dynamics_memory_shared90m.pt}"
    CHECKPOINT="${CHECKPOINT:-checkpoints/colorfield/dynamics_memory_archive_noff9.pt}"
    python -u experiments/colorfield-dynamics-v3/train_archive.py \
      --data data/colorfield --val data/colorfield_val \
      --tokenizer checkpoints/colorfield/tokenizer.pt \
      --resume "$RESUME" --checkpoint "$CHECKPOINT" \
      --budget-s "$BUDGET_S" --max-steps "$MAX_STEPS" \
      --sched-steps "${ARCHIVE_SCHED_STEPS:-$MAX_STEPS}" \
      --batch-size "${ARCHIVE_BATCH_SIZE:-32}" --clip-len 256 \
      --archive-interval 16 --archive-per-memory 1 \
      --dense-tbptt-frames 32 --fast-memory-hide-frac 0.25 \
      --hide-latents-frac 0.5 --seed 0
    ;;
  *)
    echo "unknown arm: $ARM" >&2
    exit 2
    ;;
esac
