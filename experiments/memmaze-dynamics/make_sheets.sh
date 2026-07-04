#!/usr/bin/env bash
# Qualitative rollout sheets for a memmaze dynamics checkpoint, into a pullable runs/ dir — plus a
# small held-out-episode frames slice for local iteration (the full 35.7GB frames npy is cluster-only).
# Usage: submit_job.sh --name memmaze-sheets-<arm> --hours 1 -- \
#          bash experiments/memmaze-dynamics/make_sheets.sh CHECKPOINT OUTDIR
set -euo pipefail
CKPT="${1:?usage: make_sheets.sh CHECKPOINT OUTDIR}"
OUT="${2:?usage: make_sheets.sh CHECKPOINT OUTDIR}"
mkdir -p "$OUT"

# in-window rollout (n_ctx 8 | n_gen 24 = the native W=32) and past-window (n_gen 56: window slides)
python -u src/evals/memmaze/sheets.py --checkpoint "$CKPT" \
  --tokenizer checkpoints/memmaze/tokenizer.pt --frames data/memmaze9x9.npy \
  --n-samples 4 --n-ctx 8 --out-dir "$OUT/in_window"
python -u src/evals/memmaze/sheets.py --checkpoint "$CKPT" \
  --tokenizer checkpoints/memmaze/tokenizer.pt --frames data/memmaze9x9.npy \
  --n-samples 4 --n-ctx 8 --n-gen 56 --out-dir "$OUT/past_window"

# 12 held-out episodes (frames+actions+ids) -> ~150MB, pullable for local sheet/eval iteration
python -u - "$OUT" <<'PY'
import sys
import numpy as np
sys.path.insert(0, 'src')
from evals.memmaze.sheets import val_episodes
out = sys.argv[1]
frames = np.load('data/memmaze9x9.npy', mmap_mode='r')
acts = np.load('data/memmaze9x9_actions.npy', mmap_mode='r')
ids = val_episodes(len(frames))[:12]
np.save(f'{out}/memmaze9x9_val12.npy', np.asarray(frames[ids]))
np.save(f'{out}/memmaze9x9_val12_actions.npy', np.asarray(acts[ids]))
np.save(f'{out}/memmaze9x9_val12_ids.npy', ids)
print('val slice episodes:', ids.tolist())
PY
echo "########## SHEETS DONE ##########"
