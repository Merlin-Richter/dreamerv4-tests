#!/usr/bin/env bash
# Full Memory-Maze 9x9 tokenizer training (LOCKED config) + a recon sheet, in one H100 allocation.
# LOCKED: embedding_dim=512 depth=12 n_heads=16 n_latents=32 bottleneck_dim=16, L=64, LPIPS on, fg-weight off.
# bs=6 (bs-search max was 8 @75GB; throughput plateaus past bs4, so 6 keeps headroom over a long run).
# Usage (via submit_job.sh -- bash experiments/memmaze-tokenizer/train_and_recon.sh [EPOCHS]):
set -euo pipefail
EPOCHS="${1:-15}"
CKPT=checkpoints/memmaze/tokenizer.pt
RECON=experiments/memmaze-tokenizer/_recon_memmaze.png

echo "########## train tokenizer (${EPOCHS} epochs, bs6, LOCKED 512/12/16/32/16, LPIPS) ##########"
python -u src/training/train_tokenizer.py \
  --frames data/memmaze9x9.npy --checkpoint "$CKPT" \
  --epochs "$EPOCHS" --batch-size 6 --context-length 64 \
  --embedding-dim 512 --depth 12 --n-heads 16 --n-latents 32 --bottleneck-dim 16 \
  --lpips --grad-spike-mult 5.0 \
  --wandb --wandb-project transformer-C-tokenizer --wandb-name memmaze-tok-full --wandb-tags memmaze,tokenizer

echo "########## recon sheet -> ${RECON} ##########"
python -u src/training/train_tokenizer.py \
  --frames data/memmaze9x9.npy --checkpoint "$CKPT" \
  --save-recon "$RECON" --n-samples 6 --seed 1

echo "########## TRAIN+RECON DONE ##########"
