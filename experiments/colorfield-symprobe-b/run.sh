#!/usr/bin/env bash
# ColorField-SYM 10-min H100 probe, arm B: + tau0-anchor 0.5 (experiments/colorfield-symprobe-b/).
#
# Single-varying-factor rerun of colorfield-symprobe (job 417029): identical config, data,
# seed, budget and sched, PLUS --tau0-anchor 0.5 — the GridWorld Arm-D fix seeded into the
# mem2mem clean mode (measured: without it only ~1.6% of clean-mode frames train
# visible-context next-frame prediction; the 10-min model's teacher-forced shift-copy acc
# was 0.42 vs the 1.0 floor). Pre-registered: shift-acc jumps toward ~1.0; memory columns
# (window/past) become the live signal. Pace known from 417029 (329 steps/min, workers 8)
# -> sched 3125, no pace probes.
set -euo pipefail

[ -d data/colorfield_sym ] || \
  python -u -m autoresearch.frozen_sym.datagen --out data/colorfield_sym --n-episodes 5000 --T 1024 --seed 0
[ -d data/colorfield_sym_val ] || \
  python -u -m autoresearch.frozen_sym.datagen --out data/colorfield_sym_val --n-episodes 250 --T 1024 --seed 777
echo "d8cb99f24671ddcc5925733c9d3be0947aa84125125b6d67144645bfad79ffb0  data/colorfield_sym/actions.npy" | sha256sum -c -
echo "68c22b9616ba16cc451dc1e6308a777b28493163649a541d8f7117eff5110dff  data/colorfield_sym_val/actions.npy" | sha256sum -c -

OUT=runs/colorfield-symprobe-b
mkdir -p "$OUT"

python -u autoresearch/editable/train_sym.py \
  --batch-size 128 --embedding-dim 128 --depth 6 --n-heads 8 --fixed-n-ctx --seed 0 \
  --num-workers 8 --tau0-anchor 0.5 \
  --budget-s 600 --epochs 200 --sched-steps 3125 \
  --snapshot-at 100,237,500,1000,2000 \
  --checkpoint "$OUT/dynamics_sym.pt"

python -u -m autoresearch.driver.sheets_sym \
  --checkpoint "$OUT/dynamics_sym.pt" --out "$OUT/sheets" || echo "sheets_sym failed (non-fatal)"
