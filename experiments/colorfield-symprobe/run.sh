#!/usr/bin/env bash
# ColorField-SYM 10-minute H100 budget probe (experiments/colorfield-symprobe/).
#
# Q: is the SYM tier (one-hot viewport, no tokenizer) better suited than the pixel tier
# for the autoresearch loop at a ~10-min budget? Pixel H100 numbers exist (0.124 s/step,
# knee ~10k steps => 10 min ~ 4800 steps, "barely in reach"). Sym local pace was 5.08
# s/step at bs128 on the 4070 — DATALOADER-BOUND (per-clip on-the-fly grid rendering),
# so the H100 pace is unknown and workers may matter more than the GPU.
#
# Stages: (1) idempotent datagen + sha256 determinism gate (must match the local
# sidecars byte-for-byte or every downstream claim breaks); (2) two 60s pace probes,
# --num-workers 0 vs 8; (3) the real --budget-s 600 run with --sched-steps sized from
# the measured pace (the driver's intended mechanism), snapshots incl step 237 = the
# local sym20 probe's final step, as a same-step cross-backend quality anchor.
set -euo pipefail

[ -d data/colorfield_sym ] || \
  python -u -m autoresearch.frozen_sym.datagen --out data/colorfield_sym --n-episodes 5000 --T 1024 --seed 0
[ -d data/colorfield_sym_val ] || \
  python -u -m autoresearch.frozen_sym.datagen --out data/colorfield_sym_val --n-episodes 250 --T 1024 --seed 777
echo "d8cb99f24671ddcc5925733c9d3be0947aa84125125b6d67144645bfad79ffb0  data/colorfield_sym/actions.npy" | sha256sum -c -
echo "68c22b9616ba16cc451dc1e6308a777b28493163649a541d8f7117eff5110dff  data/colorfield_sym_val/actions.npy" | sha256sum -c -

OUT=runs/colorfield-symprobe
mkdir -p "$OUT"
COMMON="--batch-size 128 --embedding-dim 128 --depth 6 --n-heads 8 --fixed-n-ctx --seed 0"

for W in 0 8; do
  python -u autoresearch/editable/train_sym.py $COMMON --num-workers $W \
    --budget-s 60 --sched-steps 500 --checkpoint "$OUT/pace_w$W.pt" 2>&1 | tee "$OUT/pace_w$W.log"
done
S0=$(grep -oE 'BUDGET_STOP step=[0-9]+' "$OUT/pace_w0.log" | grep -oE '[0-9]+')
S8=$(grep -oE 'BUDGET_STOP step=[0-9]+' "$OUT/pace_w8.log" | grep -oE '[0-9]+')
echo "PACE_PROBE: workers0=$S0 steps/min, workers8=$S8 steps/min"
if [ "$S8" -ge "$S0" ]; then W=8; S=$S8; else W=0; S=$S0; fi
SCHED=$(( S * 10 * 95 / 100 ))
[ "$SCHED" -ge 100 ] || SCHED=100
echo "REAL_RUN: workers=$W sched_steps=$SCHED"

python -u autoresearch/editable/train_sym.py $COMMON --num-workers $W \
  --budget-s 600 --epochs 200 --sched-steps $SCHED \
  --snapshot-at 100,237,500,1000,2000,4000,8000,16000 \
  --checkpoint "$OUT/dynamics_sym.pt"

python -u -m autoresearch.driver.sheets_sym \
  --checkpoint "$OUT/dynamics_sym.pt" --out "$OUT/sheets" || echo "sheets_sym failed (non-fatal)"
