#!/usr/bin/env bash
# EXP-025 | GridWorld tokenizer v3 (foreground-weighted recon + D-043 training-stability fixes)
# Cluster: ferranti (H100) | branch feat/motion-prediction @ 725a988 | job 408737
# Submitted via scripts/submit_job.sh (renders job_template.sbatch). The training invocation:
set -e
python -u src/datagen/generate_gridworld.py --n_episodes 3000 --n_frames 200 --out gridworld.npy
python -u src/training/train_tokenizer.py --frames gridworld.npy \
  --epochs 30 --batch-size 64 --lr 3e-4 --lpips --lpips-net vgg --fresh \
  --fg-weight 10 --adam-beta2 0.95 --grad-spike-mult 5.0 --log-every 50 \
  --checkpoint runs/gridworld-tok-v3/tokenizer.pt \
  --wandb --wandb-project transformer-C-tokenizer --wandb-name gridworld-tok-v3
python -u src/training/train_tokenizer.py --frames gridworld.npy \
  --checkpoint runs/gridworld-tok-v3/tokenizer.pt \
  --save-recon runs/gridworld-tok-v3/recon.png --n-samples 6
