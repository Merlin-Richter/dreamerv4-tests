#!/usr/bin/env bash
# scripts/vast_status.sh [--cluster vast] [RUN] [--tail N]
#
# The Vast.ai equivalent of job_status.sh + fetch_logs.sh combined (no squeue/sacct
# to call — status here just means "is the pidfile's PID alive", and the log is
# whatever run.sh wrote to $RUN_DIR/run.log).
#
# No RUN given -> lists every run dir under $RUNS_DIR with RUNNING/DONE + last line
#   (the squeue-equivalent "what's on this box" view).
# RUN given -> that run's status + log (--tail N last lines, default whole file).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_common.sh"

TAIL=""
CLEAN_ARGS=()
i=0; A=("$@")
while [ $i -lt ${#A[@]} ]; do
  case "${A[$i]}" in
    --cluster) [ "${A[$((i+1))]:-}" = vast ] || die_config "vast_status.sh only targets vast"; i=$((i+2));;
    --cluster=*) [ "${A[$i]#*=}" = vast ] || die_config "vast_status.sh only targets vast"; i=$((i+1));;
    --tail) TAIL="${A[$((i+1))]:-}"; i=$((i+2));;
    --tail=*) TAIL="${A[$i]#*=}"; i=$((i+1));;
    *) CLEAN_ARGS+=("${A[$i]}"); i=$((i+1));;
  esac
done

CLUSTER=vast
load_config
require_master
export SSH_CALL_TIMEOUT=45   # see _common.sh's ssh_cluster() — bounds a hung vast mux channel

RUN="${CLEAN_ARGS[0]:-}"

if [ -z "$RUN" ]; then
  echo "== runs on vast ($RUNS_DIR) =="
  ssh_cluster "for d in '$RUNS_DIR'/*/; do
    [ -f \"\$d/run.log\" ] || continue
    name=\$(basename \"\$d\")
    state=DONE
    if [ -f \"\$d/run.pid\" ] && kill -0 \"\$(cat \"\$d/run.pid\" 2>/dev/null)\" 2>/dev/null; then state=RUNNING; fi
    last=\$(tail -n1 \"\$d/run.log\" 2>/dev/null)
    printf '%-8s %-30s %s\n' \"\$state\" \"\$name\" \"\$last\"
  done" 2>&1
  exit 0
fi

echo "$RUN" | grep -qE '^[A-Za-z0-9._-]+$' || die_config "run name must be [A-Za-z0-9._-]"
RUN_DIR="$RUNS_DIR/$RUN"
# Retried: a mux channel reset fails this call even when the dir exists (live 2026-07-12).
EXISTS=""
for _try in 1 2 3; do
  ssh_cluster "test -d '$RUN_DIR'" && { EXISTS=1; break; }
  sleep 10
done
[ -n "$EXISTS" ] || die_badref "no such run on vast: $RUN"

STATE="DONE"
PID="$(ssh_cluster "cat '$RUN_DIR/run.pid' 2>/dev/null" || true)"
if [ -n "$PID" ] && ssh_cluster "kill -0 '$PID' 2>/dev/null"; then STATE="RUNNING"; fi
echo "== $RUN: $STATE $([ -n "$PID" ] && [ "$STATE" = RUNNING ] && echo "(pid $PID)") =="

if [ -n "$TAIL" ]; then
  out="$(ssh_cluster "tail -n '$TAIL' '$RUN_DIR/run.log' 2>&1")" || true
else
  out="$(ssh_cluster "cat '$RUN_DIR/run.log' 2>&1")" || true
fi
scan_quota "$out"
echo "$out"
