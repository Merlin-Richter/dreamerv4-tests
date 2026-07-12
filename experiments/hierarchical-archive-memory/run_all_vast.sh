#!/usr/bin/env bash
# Fully autonomous vast pipeline: data prep -> bs auto-calibration -> archive training.
# One detached job, no orchestrator round-trips — load-bearing on this box, whose network
# blacks out for ~35 min at a time (a blackout pauses/retries stages instead of requiring a
# human/agent to notice and relaunch). Every stage is idempotent/skippable on re-run.
# Usage: scripts/vast_run.sh --name memmaze-archive -- \
#          bash experiments/hierarchical-archive-memory/run_all_vast.sh [EPOCHS] [PART]
set -euo pipefail
EPOCHS="${1:-50}"
PART="${2:-train-part8}"

echo "########## STAGE A: data prep (skips if latent cache already present) ##########"
if ls data/memmaze9x9.latents-*.npy >/dev/null 2>&1 && [ -f data/memmaze9x9_actions.npy ]; then
  echo "latent cache + actions already present — skipping prep"
else
  bash experiments/hierarchical-archive-memory/prep_vast.sh "$PART"
fi

echo "########## STAGE B: batch-size auto-calibration (512-frame clip, hide 0.25) ##########"
BS_PICKED=""
for bs in 8 6 4 2 1; do
  echo "=== calibrate bs=$bs ==="
  out="$(python -u experiments/hierarchical-archive-memory/calibrate.py \
    --resume checkpoints/memmaze/dynamics_mem2mem_noff9.pt \
    --batch-size "$bs" --frames 512 --dense-tbptt-frames 64 \
    --archive-interval 16 --archive-per-memory 1 \
    --fast-memory-hide-frac 0.25 2>&1)" || { echo "$out" | tail -3; echo "=== bs=$bs failed (OOM?) ==="; continue; }
  echo "$out" | tail -2
  peak="$(echo "$out" | grep -oP 'cuda_peak_alloc=\K[0-9.]+' || echo 999)"
  # keep ~6 GiB headroom on the 32 GiB card: optimizer state, allocator slack, fragmentation
  if python3 -c "import sys; sys.exit(0 if float('$peak') < 26.0 else 1)"; then
    BS_PICKED="$bs"; echo "=== picked bs=$bs (peak ${peak} GiB) ==="; break
  fi
  echo "=== bs=$bs peak ${peak} GiB too close to 32 GiB — trying smaller ==="
done
[ -n "$BS_PICKED" ] || { echo "no batch size fits — abort"; exit 1; }

echo "########## STAGE C: archive training ($EPOCHS epochs, bs=$BS_PICKED) ##########"
bash experiments/hierarchical-archive-memory/train.sh "$EPOCHS" "$BS_PICKED" \
  --fast-memory-hide-frac 0.25 --hide-latents-frac 0.5
