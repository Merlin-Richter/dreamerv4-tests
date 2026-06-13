#!/usr/bin/env bash
# EXP-017 — FF9 v2 memory-token baseline (D-024). Overnight run.
# Trains the distinct-MEMORY-token model with the FF9 v2 memory-only-sufficiency loss (ops 1&2:
# write-mem<-latents, read-mem->latents) at the EXACT EXP-010/012 budget. This is the architectural
# baseline (V-T013: expected ~FF7 on the frozen probe; the cross-window relay / op-3 is built on top
# of this NEXT). Registers stay pure scratch; FF9 uses distinct memory tokens.
#
# TRAIN ONLY tonight — the frozen-probe eval is deferred: it needs generate_full_state_memory (the
# memory-carry inference path), not yet built. Without it, generate() would NOT carry memory, so a
# probe tonight would mis-measure the model as ~vanilla. Eval tomorrow once that path exists.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root
PY="${PY:-venv/Scripts/python.exe}"   # CUDA torch build; bare python is CPU-only

"$PY" -u src/D_dynamics_model/train_dynamics_model.py \
  --frames occluded.npy --actions occluded_actions.npy --tokenizer trained_autoencoder.pt \
  --epochs 100 --batch-size 32 --lr 3e-4 --seed 0 \
  --ff9 3 --n-memory 4 --lambda-ff9 1.0 --fresh \
  --checkpoint experiments/EXP-017/ff9v2_s0.pt \
  > experiments/EXP-017/train.log 2>&1

echo "EXP-017 train done: experiments/EXP-017/ff9v2_s0.pt"
