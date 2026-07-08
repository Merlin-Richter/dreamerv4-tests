#!/usr/bin/env bash
# Autoresearch loop — remote H100 job payload (NOT agent-editable).
# Usage: bash autoresearch/loop/job_payload.sh <run-name>
# Stages: data sha-gate -> 60s pace micro-run (outside budget) -> sched-sized 600s
# budgeted train (train_sym.py, hyperparams live IN that file) -> in-window probe +
# window-pin config check + reduced frozen eval -> grep-able summary block.
set -euo pipefail
RUN="${1:?usage: job_payload.sh <run-name>}"
OUT="runs/$RUN"
mkdir -p "$OUT"

# ---- data (idempotent, byte-identity gated) ----
[ -d data/colorfield_sym ] || \
  python -u -m autoresearch.frozen_sym.datagen --out data/colorfield_sym --n-episodes 5000 --T 1024 --seed 0
[ -d data/colorfield_sym_val ] || \
  python -u -m autoresearch.frozen_sym.datagen --out data/colorfield_sym_val --n-episodes 250 --T 1024 --seed 777
echo "d8cb99f24671ddcc5925733c9d3be0947aa84125125b6d67144645bfad79ffb0  data/colorfield_sym/actions.npy" | sha256sum -c -
echo "68c22b9616ba16cc451dc1e6308a777b28493163649a541d8f7117eff5110dff  data/colorfield_sym_val/actions.npy" | sha256sum -c -

# ---- GPU sampler (peak VRAM + mean util over the training phase) ----
( while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits; sleep 5; done \
  > "$OUT/gpu_samples.csv" 2>/dev/null ) & GPUMON=$!
trap 'kill $GPUMON 2>/dev/null || true' EXIT

# ---- pace micro-run (OUTSIDE the budget; sizes the LR schedule so speed => steps) ----
python -u autoresearch/editable/train_sym.py --num-workers 8 \
  --budget-s 60 --sched-steps 500 --checkpoint "$OUT/pace.pt" 2>&1 | tee "$OUT/pace.log"
RATE=$(grep -oE 'BUDGET_STOP step=[0-9]+' "$OUT/pace.log" | grep -oE '[0-9]+')
SCHED=$(( RATE * 10 * 95 / 100 )); [ "$SCHED" -ge 100 ] || SCHED=100
echo "pace_steps_per_min: $RATE"
echo "sched_steps:      $SCHED"

# ---- the budgeted run (hyperparams = train_sym.py defaults; the agent edits the file) ----
python -u autoresearch/editable/train_sym.py --num-workers 8 \
  --budget-s 600 --epochs 500 --sched-steps "$SCHED" \
  --checkpoint "$OUT/dynamics_sym.pt" 2>&1 | tee "$OUT/train.log"
kill $GPUMON 2>/dev/null || true

STEPS=$(grep -oE '(BUDGET_STOP|EPOCHS_DONE) step=[0-9]+' "$OUT/train.log" | grep -oE '[0-9]+' | tail -1)
ELAPSED=$(grep -oE 'elapsed=[0-9.]+' "$OUT/train.log" | tail -1 | cut -d= -f2)
FLOW=$(grep -oE 'flow [0-9.]+' "$OUT/train.log" | tail -1 | awk '{print $2}')
PEAK_MB=$(cut -d, -f1 "$OUT/gpu_samples.csv" | sort -n | tail -1)
UTIL=$(cut -d, -f2 "$OUT/gpu_samples.csv" | awk '{s+=$1; n++} END {if (n>0) printf "%.0f", s/n; else print 0}')

# ---- carried-state byte budget (replaces the old hard window pin) ----
python -u autoresearch/loop/state_probe.py --checkpoint "$OUT/dynamics_sym.pt"
WFRAMES=$(python -c "
import torch
c = torch.load('$OUT/dynamics_sym.pt', map_location='cpu', weights_only=False)['config']
print(c.get('max_temporal_length', '?'))")

# ---- probes + reduced frozen eval ----
python -u autoresearch/loop/probe_inwindow.py --checkpoint "$OUT/dynamics_sym.pt"
python -u autoresearch/loop/eval_reduced.py --checkpoint "$OUT/dynamics_sym.pt"

# ---- summary block (agent greps below the --- marker) ----
echo "---"
echo "steps:            ${STEPS:-0}"
echo "train_seconds:    ${ELAPSED:-0}"
echo "sec_per_step:     $(python -c "print(f'{${ELAPSED:-0}/max(1,${STEPS:-1}):.3f}')")"
echo "flow_final:       ${FLOW:-nan}"
echo "peak_vram_mb:     ${PEAK_MB:-0}"
echo "gpu_util_pct:     ${UTIL:-0}"
echo "window_frames:    $WFRAMES"
