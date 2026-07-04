#!/usr/bin/env bash
# GridWorldV2 campaign prep: dataset (5000 eps) + latent cache (frozen v1 tokenizer).
# Usage: submit_job.sh --name gwv2-datagen --hours 2 -- bash experiments/gridworldv2-arms/prep.sh
set -euo pipefail
python -u src/datagen/generate_gridworldv2.py --n_episodes 5000
python -u src/training/train_dynamics.py --build-latent-cache-only \
  --frames data/gridworldv2.npy --tokenizer checkpoints/gridworld/tokenizer.pt \
  --checkpoint checkpoints/gridworldv2/cachebuild.pt --cache-batch 16
echo "########## GWV2 PREP DONE ##########"
