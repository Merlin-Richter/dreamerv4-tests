#!/usr/bin/env bash
# ColorField latent-cache job (cluster): idempotent datagen (byte-identical, proven)
# then encode both datasets through the frozen tokenizer (window-invariance probe
# runs first and aborts loudly on failure). Pull results back with pull_file.sh —
# data flows cluster->local directly, never through GitHub.
set -euo pipefail

python -u -m autoresearch.frozen.datagen --out data/colorfield --n-episodes 5000 --T 1024 --seed 0
python -u -m autoresearch.frozen.datagen --out data/colorfield_val --n-episodes 250 --T 1024 --seed 777
python -u -m autoresearch.driver.latent_cache
ls -la data/colorfield/latents-*.npy data/colorfield_val/latents-*.npy
sha256sum data/colorfield/latents-*.npy data/colorfield_val/latents-*.npy
