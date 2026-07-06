#!/usr/bin/env bash
# ColorField tokenizer prep job (cluster): datagen (deterministic, hash-echoed for
# cross-check vs local) -> train -> readout-exactness verify. Submitted via
# scripts/submit_job.sh -- bash autoresearch/frozen/tok_job.sh
# (a committed script because submit_job re-joins args without re-quoting,
# so inline compound commands cannot be passed reliably).
set -euo pipefail

python -u -m autoresearch.frozen.datagen --out data/colorfield --n-episodes 5000 --T 1024 --seed 0
python -u -m autoresearch.frozen.datagen --out data/colorfield_val --n-episodes 250 --T 1024 --seed 777
sha256sum data/colorfield/*.npy data/colorfield_val/*.npy
python -u -m autoresearch.frozen.train_tokenizer \
  --checkpoint checkpoints/colorfield/tokenizer.pt --epochs 20 --batch-size 32
