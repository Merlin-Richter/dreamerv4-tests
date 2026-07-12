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
# This host's network blacks out for ~35 min at a time (3x on 2026-07-12 alone). gdown can't
# resume, and download_memmaze.py treats any size>0 zip as complete — so a blackout mid-download
# leaves a corrupt zip that would sail into unzip. Retry with integrity checks, tolerating one
# full blackout window; skip straight through if the extracted dir already has the npz.
ZIP="data/memmaze9x9_raw/${PART}.zip"
EXDIR="data/memmaze9x9_raw/${PART}"
# NB: guard the dir — under `set -euo pipefail` a failing find inside $() kills the script.
NPZ_COUNT=0
if [ -d "$EXDIR" ]; then NPZ_COUNT="$(find "$EXDIR" -name '*.npz' | wc -l)"; fi
if [ "$NPZ_COUNT" -gt 100 ]; then
  echo "extracted dir already has $NPZ_COUNT npz — skipping download"
else
  ok=0
  for attempt in $(seq 12); do
    if [ -f "$ZIP" ] && ! python -c "import sys,zipfile; sys.exit(0 if zipfile.is_zipfile('$ZIP') and zipfile.ZipFile('$ZIP').testzip() is None else 1)" 2>/dev/null; then
      echo "=== attempt $attempt: existing zip is partial/corrupt — deleting ==="
      rm -f "$ZIP"
    fi
    if python -u experiments/memmaze-tokenizer/download_memmaze.py \
         --parts "$PART" --out-dir data/memmaze9x9_raw --unzip \
       && [ "$(find "$EXDIR" -name '*.npz' 2>/dev/null | wc -l)" -gt 100 ]; then
      ok=1; break
    fi
    echo "=== download/unzip attempt $attempt/12 failed — sleeping 240s (network blackout?) ==="
    sleep 240
  done
  [ "$ok" = 1 ] || { echo "download failed after 12 attempts"; exit 1; }
fi
rm -f "$ZIP"   # zip + extracted must not coexist longer than needed
df -h / | tail -1

echo "########## [2/3] stream npz -> latent cache + actions + placeholder ##########"
# The tokenizer may still be mid-upload (push_file.sh from local) — wait up to 30 min for it.
for i in $(seq 180); do
  [ -f checkpoints/memmaze/tokenizer.pt ] && break
  [ "$i" = 1 ] && echo "waiting for checkpoints/memmaze/tokenizer.pt (push_file.sh in flight)..."
  sleep 10
done
[ -f checkpoints/memmaze/tokenizer.pt ] || { echo "tokenizer.pt never arrived"; exit 1; }
python -u experiments/hierarchical-archive-memory/prep_vast.py \
  --raw data/memmaze9x9_raw --frames data/memmaze9x9.npy \
  --tokenizer checkpoints/memmaze/tokenizer.pt

echo "########## [3/3] cleanup raw npz ##########"
rm -rf data/memmaze9x9_raw
df -h / | tail -1
ls -la data/
echo "########## VAST PREP DONE ##########"
