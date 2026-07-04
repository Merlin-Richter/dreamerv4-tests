#!/usr/bin/env bash
# Full probe battery for the dip investigation. Run from repo root.
set -e
PY=venv/Scripts/python.exe
RUN=experiments/gridworldv2-arms/dip-investigation/run.py

# 1) H4: 256-rollout fine-grid re-measure, both arms, both windows
$PY -u $RUN --arm D --n-rollouts 256 --max-k 16 --window 8  --out d_w8.json
$PY -u $RUN --arm D --n-rollouts 256 --max-k 16 --window 16 --out d_w16.json
$PY -u $RUN --arm A --n-rollouts 256 --max-k 16 --window 8  --out a_w8.json
$PY -u $RUN --arm A --n-rollouts 256 --max-k 16 --window 16 --out a_w16.json

# 2) H3: fully-revealed driver validation (teacher-forced = pure alignment check;
#    free-running = model dynamics under the same driver)
$PY -u $RUN --arm D --n-rollouts 128 --max-k 12 --window 8  --no-hide --teacher-forced --out d_nohide_tf_w8.json
$PY -u $RUN --arm D --n-rollouts 128 --max-k 12 --window 16 --no-hide --teacher-forced --out d_nohide_tf_w16.json
$PY -u $RUN --arm A --n-rollouts 128 --max-k 12 --window 8  --no-hide --teacher-forced --out a_nohide_tf_w8.json
$PY -u $RUN --arm D --n-rollouts 128 --max-k 12 --window 8  --no-hide --out d_nohide_free_w8.json

# 3) H1: branch-only attention ablations (arm D)
$PY -u $RUN --arm D --n-rollouts 256 --max-k 16 --window 8  --mask no_mem_read --out d_w8_nomem.json
$PY -u $RUN --arm D --n-rollouts 256 --max-k 16 --window 16 --mask no_mem_read --out d_w16_nomem.json
$PY -u $RUN --arm D --n-rollouts 256 --max-k 16 --window 8  --mask mem_only --out d_w8_memonly.json
$PY -u $RUN --arm D --n-rollouts 256 --max-k 16 --window 16 --mask mem_only --out d_w16_memonly.json

# 4) Phase-shift probe: n_ctx=8 moves the hide tick onto a write slot; first occluded write at 16.
#    Phase-locked story predicts the dip moves to k=9-11; hide-locked story keeps it at k=4-6.
$PY -u $RUN --arm D --n-rollouts 256 --n-ctx 8 --max-k 20 --window 8  --out d_nctx8_w8.json
$PY -u $RUN --arm D --n-rollouts 256 --n-ctx 8 --max-k 20 --window 16 --out d_nctx8_w16.json

echo ALL_PROBES_DONE
