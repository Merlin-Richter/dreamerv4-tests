#!/usr/bin/env bash
# Sheets sweep over pulled pixcurve snapshots -> sheets_step{N}/ + one parseable log.
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=experiments/colorfield-pixcurve/sheets_sweep.log
: > "$LOG"
for f in $(ls experiments/colorfield-pixcurve/dynamics_step*.pt | sed 's/.*_step//; s/\.pt//' | sort -n); do
  ck=experiments/colorfield-pixcurve/dynamics_step${f}.pt
  echo "=== STEP $f ===" | tee -a "$LOG"
  venv/Scripts/python.exe -u -m autoresearch.driver.sheets \
    --checkpoint "$ck" --out experiments/colorfield-pixcurve/sheets_step${f} 2>&1 | tee -a "$LOG"
done
echo "SWEEP_DONE" | tee -a "$LOG"
