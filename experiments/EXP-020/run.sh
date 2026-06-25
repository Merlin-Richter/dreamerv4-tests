#!/usr/bin/env bash
# EXP-019 (vanilla CONTROL) + EXP-020 (C1 multistep TREATMENT) — D-027/D-028.
# Budget-matched A/B for the C1 time-axis multi-step DAgger loss on curtain-up motion.
# Run from repo root with the CUDA venv python. Provenance: code @ a07fdee (C1 impl, T-018).
#
# Both share: occluded.npy (+ occluded_actions.npy, n_actions=2), frozen tokenizer
# trained_autoencoder.pt, 250-episode TRAIN subset (val unchanged), 40 epochs, bs32, lr3e-4, seed0,
# trained fresh. ONLY the three --multistep* flags differ -> clean attribution of the C1 loss.
set -euo pipefail
PY=./venv/Scripts/python.exe
TRAIN=src/D_dynamics_model/train_dynamics_model.py
COMMON=(--frames occluded.npy --actions occluded_actions.npy --tokenizer trained_autoencoder.pt \
        --fresh --seed 0 --epochs 40 --batch-size 32 --lr 3e-4 --max-episodes 250)

# --- EXP-019 control (vanilla, no extra loss) ---  [already run: vanilla_ctrl_s0.pt]
# "$PY" -u "$TRAIN" "${COMMON[@]}" --checkpoint experiments/EXP-019/vanilla_ctrl_s0.pt

# --- EXP-020 treatment (C1: multistep_h=4, lambda=1.0, warmup=10 epochs) ---
"$PY" -u "$TRAIN" "${COMMON[@]}" \
    --multistep 4 --lambda-multistep 1.0 --multistep-warmup 10 \
    --checkpoint experiments/EXP-020/c1_h4_s0.pt
