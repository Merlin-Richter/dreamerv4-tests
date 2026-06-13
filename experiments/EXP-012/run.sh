#!/usr/bin/env bash
# EXP-012 — budget-matched vanilla baseline (D-016). Reproducible launch. Run from repo root.
# Trains the vanilla dynamics model with the EXACT EXP-010 FF7 budget minus the FF7 loss,
# then evaluates on the frozen probe. Chain aborts on first failure (&&).
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

# Use the project venv python (has the CUDA torch build); bare `python` here is CPU-only.
PY="${PY:-venv/Scripts/python.exe}"

"$PY" src/D_dynamics_model/train_dynamics_model.py \
  --frames occluded.npy --actions occluded_actions.npy --tokenizer trained_autoencoder.pt \
  --epochs 100 --batch-size 32 --lr 3e-4 --seed 0 --ff7 0 --fresh \
  --checkpoint experiments/EXP-012/vanilla_s0.pt \
  --wandb --wandb-project transformer-D-dynamics --wandb-name exp012-vanilla-s0 \
  > experiments/EXP-012/train.log 2>&1 \
&& "$PY" -m src.probe.revisit_probe \
  --dynamics experiments/EXP-012/vanilla_s0.pt --tokenizer trained_autoencoder.pt \
  --out experiments/EXP-012/results.json \
  > experiments/EXP-012/probe.log 2>&1

echo "EXP-012 done: experiments/EXP-012/{vanilla_s0.pt,results.json}"
