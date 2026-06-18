#!/usr/bin/env bash
# scripts/wait_for_jobs.sh --cluster {ferranti|galvani} <jobid> [jobid ...] [--poll SECONDS]
#
# Blocks until all jobs reach a terminal state. Returns EARLY (non-zero) on the first
# FAILED/TIMEOUT/CANCELLED/OOM job OR a "Traceback" appearing in a running job's log,
# so a crashing run is caught fast. This blocking wait is the intended WAIT step (§3).
# Exit: 0 all COMPLETED; 7 a job failed / crashed (inspect logs, diagnose, fix or escalate).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_common.sh"

POLL=30
CLEAN_ARGS=()
i=0; A=("$@")
while [ $i -lt ${#A[@]} ]; do
  case "${A[$i]}" in
    --poll) POLL="${A[$((i+1))]:-30}"; i=$((i+2));;
    --poll=*) POLL="${A[$i]#*=}"; i=$((i+1));;
    *) CLEAN_ARGS+=("${A[$i]}"); i=$((i+1));;
  esac
done
set -- "${CLEAN_ARGS[@]}"
init_verb "$@"
require_master
[ "${#WRAP_ARGS[@]}" -gt 0 ] || die_config "usage: wait_for_jobs.sh --cluster X <jobid> [...] [--poll S]"
IDS=("${WRAP_ARGS[@]}")
idcsv="$(IFS=,; echo "${IDS[*]}")"

log "waiting on $CLUSTER jobs ${idcsv} (poll ${POLL}s)"
while true; do
  # state per job from sacct (-X = allocation row only)
  states="$(ssh_cluster "sacct -j '$idcsv' -X -n --format=JobID%20,State%20 2>&1")" || true
  scan_quota "$states"
  all_done=1
  for id in "${IDS[@]}"; do
    st="$(echo "$states" | awk -v j="$id" '$1==j || $1 ~ ("^"j"\\.") {print $2; exit}')"
    st="${st:-PENDING}"
    case "$st" in
      COMPLETED) ;;
      FAILED|TIMEOUT|CANCELLED*|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE)
        log "job $id terminal-FAILED: $st"
        die "job $id on $CLUSTER ended $st — fetch_logs.sh --cluster $CLUSTER $id" 7 ;;
      *) all_done=0 ;;  # PENDING/RUNNING/CONFIGURING/etc.
    esac
  done
  [ "$all_done" -eq 1 ] && { log "all jobs COMPLETED: $idcsv"; exit 0; }

  # early crash detection: Traceback in any running job's log
  for id in "${IDS[@]}"; do
    lp="$(ssh_cluster "scontrol show job '$id' -o 2>/dev/null | grep -oE 'StdOut=[^ ]+' | cut -d= -f2" || true)"
    [ -z "$lp" ] && continue
    if ssh_cluster "test -f '$lp' && tail -n 200 '$lp' 2>/dev/null | grep -qE 'Traceback \(most recent call last\)|CUDA out of memory'"; then
      log "job $id: crash signature in log $lp"
      die "job $id on $CLUSTER shows a Traceback/OOM in its log — fetch_logs.sh --cluster $CLUSTER $id" 7
    fi
  done
  sleep "$POLL"
done
