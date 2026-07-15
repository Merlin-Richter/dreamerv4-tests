#!/usr/bin/env bash
# Cost-calibrated RTX 5090 sequence. Stops immediately if any stage fails.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MAX_STEPS=40000 BUDGET_S=1200 VANILLA_BATCH_SIZE=128 \
  bash "$HERE/run.sh" vanilla

MAX_STEPS=5000 BUDGET_S=2700 BATCH_SIZE=128 BASE_SCHED_STEPS=100000 \
  bash "$HERE/run.sh" memory-base

MAX_STEPS=5000 BUDGET_S=2700 BATCH_SIZE=128 \
  bash "$HERE/run.sh" memory-control

MAX_STEPS=5000 BUDGET_S=3600 ARCHIVE_BATCH_SIZE=128 \
  bash "$HERE/run.sh" archive
