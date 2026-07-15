#!/usr/bin/env bash
# ColorField tokenizer prep job (cluster): datagen (deterministic, hash-echoed for
# cross-check vs local) -> train -> readout-exactness verify. Submitted via
# scripts/submit_job.sh -- bash autoresearch/frozen/tok_job.sh
# (a committed script because submit_job re-joins args without re-quoting,
# so inline compound commands cannot be passed reliably).
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-checkpoints/colorfield/tokenizer.pt}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BUDGET_ARGS=()
if [[ -n "${BUDGET_S:-}" ]]; then
  BUDGET_ARGS=(--budget-s "$BUDGET_S")
fi

python -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}"); assert torch.cuda.is_available(); print((torch.ones(1, device="cuda") + 1).item())'
python -u -m autoresearch.frozen.datagen --out data/colorfield --n-episodes 5000 --T 1024 --seed 0
python -u -m autoresearch.frozen.datagen --out data/colorfield_val --n-episodes 250 --T 1024 --seed 777
sha256sum data/colorfield/*.npy data/colorfield_val/*.npy
python -u -m autoresearch.frozen.train_tokenizer \
  --checkpoint "$CHECKPOINT" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
  "${BUDGET_ARGS[@]}"
