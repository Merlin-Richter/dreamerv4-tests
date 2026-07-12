#!/usr/bin/env bash
# scripts/vast_cancel.sh [--cluster vast] <RUN>
#
# The Vast.ai equivalent of cancel_job.sh: kill -TERM the run's self-registered PID
# (no scancel/scheduler). Same ownership guard as cancel_job.sh — REFUSES a run name
# not recorded in EXPERIMENTS.md, so it can't kill something you forgot to log.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_common.sh"

CLEAN_ARGS=()
i=0; A=("$@")
while [ $i -lt ${#A[@]} ]; do
  case "${A[$i]}" in
    --cluster) [ "${A[$((i+1))]:-}" = vast ] || die_config "vast_cancel.sh only targets vast"; i=$((i+2));;
    --cluster=*) [ "${A[$i]#*=}" = vast ] || die_config "vast_cancel.sh only targets vast"; i=$((i+1));;
    *) CLEAN_ARGS+=("${A[$i]}"); i=$((i+1));;
  esac
done
RUN="${CLEAN_ARGS[0]:-}"
[ -n "$RUN" ] || die_config "usage: vast_cancel.sh [--cluster vast] <RUN>"
echo "$RUN" | grep -qE '^[A-Za-z0-9._-]+$' || die_config "run name must be [A-Za-z0-9._-]"

IDX="$REPO_DIR/agent/EXPERIMENTS.md"
[ -f "$IDX" ] || IDX="$REPO_DIR/EXPERIMENTS.md"
[ -f "$IDX" ] || die "EXPERIMENTS.md not found (agent/ or root) — cannot verify job ownership"
if ! grep -qF "$RUN" "$IDX"; then
  die "refusing to cancel '$RUN' — not found in EXPERIMENTS.md (record it there first if it's ours)"
fi

CLUSTER=vast
load_config
require_master
RUN_DIR="$RUNS_DIR/$RUN"
ssh_cluster "test -d '$RUN_DIR'" || die_badref "no such run on vast: $RUN"

PID="$(ssh_cluster "cat '$RUN_DIR/run.pid' 2>/dev/null" || true)"
[ -n "$PID" ] || die "run $RUN has no live pidfile — already finished?"
# Kill the PROCESS GROUP, not just job.sh: setsid gives the job its own PGID (= the pidfile
# PID), and TERMing only the shell runs its EXIT trap and exits while the python child keeps
# training as an orphan (live failure 2026-07-12: an orphaned trainer held 22 GiB and OOM'd
# the next launch). Group-kill takes the shell AND every child down together; fall back to
# the single PID if the group is already gone.
out="$(ssh_cluster "kill -TERM -- -'$PID' 2>&1 || kill -TERM '$PID' 2>&1" || true)"
log "sent SIGTERM to vast run $RUN (process group $PID)"
echo "cancelled: $RUN (pgid $PID)"
[ -z "$out" ] || echo "$out"
