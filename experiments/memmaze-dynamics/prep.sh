#!/usr/bin/env bash
# Memory Maze dynamics PREP (one GPU job): actions/labels extraction + latent cache + invariance probe.
# Usage: submit_job.sh --name memmaze-dyn-prep --hours 2 --cpus 8 -- bash experiments/memmaze-dynamics/prep.sh
set -euo pipefail

RAW=data/memmaze9x9_raw
FRAMES=data/memmaze9x9.npy
TOK=checkpoints/memmaze/tokenizer.pt

echo "########## sanity: inputs ##########"
ls -la "$FRAMES" "$TOK"
if [ ! -d "$RAW" ]; then
  echo "ERROR: $RAW missing (raw npz dir was cleaned?) — re-download train-part0 via" \
       "experiments/memmaze-tokenizer/cluster_prep.sh before re-running." >&2
  exit 1
fi

echo "########## 1/3 extract actions + labels ##########"
python -u experiments/memmaze-dynamics/extract_actions_labels.py --raw "$RAW" --frames "$FRAMES"

echo "########## 2/3 latent cache build ##########"
python -u src/training/train_dynamics.py \
  --frames "$FRAMES" --tokenizer "$TOK" \
  --checkpoint checkpoints/memmaze/_prep_dummy.pt \
  --build-latent-cache-only --cache-batch 16

echo "########## 3/3 window-invariance probe (offset 32) ##########"
python -u experiments/memmaze-dynamics/probe_window_invariance.py \
  --frames "$FRAMES" --tokenizer "$TOK" --n-episodes 8 --offset 32

echo "########## PREP DONE ##########"
