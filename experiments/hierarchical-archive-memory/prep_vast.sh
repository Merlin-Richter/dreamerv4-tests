#!/usr/bin/env bash
# One-shot Memory Maze data prep for the DISK-CONSTRAINED vast box (32 GB container disk):
# download ONE public train shard from Drive, stream its npz straight into the fp16 latent
# cache + actions sidecar + sparse frames placeholder (prep_vast.py), then delete the raw.
# Peak disk ~27 GB (venv ~7 + zip 9.6 + extracted ~9.7); steady state ~11 GB.
#
# Usage: scripts/vast_run.sh --name memmaze-archive-prep -- \
#          bash experiments/hierarchical-archive-memory/prep_vast.sh [PART]
# Requires checkpoints/memmaze/tokenizer.pt on the box (push_file.sh) BEFORE launching.
set -euo pipefail
PART="${1:-train-part8}"   # Merlin's chosen public shard (2026-07-12), ~10% of the train set

echo "########## [0/3] disk before ##########"
df -h / | tail -1
pip install --quiet gdown
pip cache purge >/dev/null 2>&1 || true   # the venv build leaves a ~3 GB torch wheel cached

echo "########## [1/3] download + unzip $PART ##########"
python -u experiments/memmaze-tokenizer/download_memmaze.py \
  --parts "$PART" --out-dir data/memmaze9x9_raw --unzip
rm -f "data/memmaze9x9_raw/${PART}.zip"   # zip + extracted must not coexist longer than needed
df -h / | tail -1

echo "########## [2/3] stream npz -> latent cache + actions + placeholder ##########"
python -u experiments/hierarchical-archive-memory/prep_vast.py \
  --raw data/memmaze9x9_raw --frames data/memmaze9x9.npy \
  --tokenizer checkpoints/memmaze/tokenizer.pt

echo "########## [3/3] cleanup raw npz ##########"
rm -rf data/memmaze9x9_raw
df -h / | tail -1
ls -la data/
echo "########## VAST PREP DONE ##########"
