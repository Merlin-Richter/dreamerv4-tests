#!/usr/bin/env bash
# Cost-calibrated RTX 5090 sequence. Stops immediately if any stage fails.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MAX_STEPS=1000000 BUDGET_S=10800 VANILLA_BATCH_SIZE=128 VANILLA_SCHED_STEPS=490000 \
  bash "$HERE/run.sh" vanilla

MAX_STEPS=100000 BUDGET_S=5400 BATCH_SIZE=128 BASE_SCHED_STEPS=100000 \
  bash "$HERE/run.sh" memory-base

MAX_STEPS=100000 BUDGET_S=5400 BATCH_SIZE=128 CONTROL_SCHED_STEPS=13000 \
  bash "$HERE/run.sh" memory-control

MAX_STEPS=100000 BUDGET_S=5400 ARCHIVE_BATCH_SIZE=128 ARCHIVE_SCHED_STEPS=10200 \
  bash "$HERE/run.sh" archive
