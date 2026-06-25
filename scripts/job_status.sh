#!/usr/bin/env bash
# scripts/job_status.sh --cluster {ferranti|galvani} [jobid ...]
#
# No ids  -> squeue for your jobs (live queue).
# With ids -> sacct: state, exit code, elapsed, MaxRSS (post-mortem accounting).
# Run on cold-start to reconcile believed vs actual cluster state (protocol §1).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_common.sh"
init_verb "$@"
require_master

if [ "${#WRAP_ARGS[@]}" -eq 0 ]; then
  echo "== squeue (${USERNAME}@${CLUSTER}) =="
  ssh_cluster "squeue -u '${USERNAME}' -o '%.12i %.24j %.9T %.10M %.6D %.20R' 2>&1" \
    | { out="$(cat)"; scan_quota "$out"; echo "$out"; }
else
  ids="$(IFS=,; echo "${WRAP_ARGS[*]}")"
  echo "== sacct (${CLUSTER}) jobs: $ids =="
  ssh_cluster "sacct -j '$ids' --format=JobID%14,JobName%24,State%12,ExitCode%8,Elapsed%12,MaxRSS%10,AllocTRES%30 2>&1" \
    | { out="$(cat)"; scan_quota "$out"; echo "$out"; }
fi
