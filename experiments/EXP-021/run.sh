#!/usr/bin/env bash
# EXP-021 (D-029) — C1 on the FULL 1000-episode occluded data (confound-free compounding test).
# Same C1 hyperparameters as EXP-020 (multistep_h=4, lambda=1.0, warmup=10); the ONLY change is
# dropping --max-episodes so all 1000 episodes are used (vs EXP-020's 250-ep subset). Compared in
# eval against the EXISTING competent reference set (vanilla_s0 / ff7_k3 / ff9v2, all full-data),
# so NO new control run is needed. Not epoch-matched to the references' 100ep (infeasible ~24min/
# epoch); periodic per-epoch checkpoints + reported TF curves control for per-step accuracy.
# Run from repo root with the CUDA venv python. Provenance: code @ a07fdee.
set -euo pipefail
PY=./venv/Scripts/python.exe
"$PY" -u src/D_dynamics_model/train_dynamics_model.py \
    --frames occluded.npy --actions occluded_actions.npy --tokenizer trained_autoencoder.pt \
    --fresh --seed 0 --epochs 40 --batch-size 32 --lr 3e-4 \
    --multistep 4 --lambda-multistep 1.0 --multistep-warmup 10 \
    --checkpoint experiments/EXP-021/c1_full_s0.pt
