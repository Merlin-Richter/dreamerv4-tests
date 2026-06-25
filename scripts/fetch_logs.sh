#!/usr/bin/env bash
# scripts/fetch_logs.sh --cluster {ferranti|galvani} <jobid> [--tail N]
#
# Prints the job's slurm log (works while running or after completion). --tail N limits
# to the last N lines. Resolves the log path via `scontrol show job` (falls back to the
# templated path $RUNS_DIR/<name>/slurm-<id>.out via sacct job name).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_common.sh"

TAIL=""
CLEAN_ARGS=()
i=0; A=("$@")
while [ $i -lt ${#A[@]} ]; do
  case "${A[$i]}" in
    --tail) TAIL="${A[$((i+1))]:-}"; i=$((i+2));;
    --tail=*) TAIL="${A[$i]#*=}"; i=$((i+1));;
    *) CLEAN_ARGS+=("${A[$i]}"); i=$((i+1));;
  esac
done
set -- "${CLEAN_ARGS[@]}"
init_verb "$@"
require_master
JOBID="${WRAP_ARGS[0]:-}"
[ -n "$JOBID" ] || die_config "usage: fetch_logs.sh --cluster X <jobid> [--tail N]"

# find the log path
LOGPATH="$(ssh_cluster "scontrol show job '$JOBID' -o 2>/dev/null | grep -oE 'StdOut=[^ ]+' | cut -d= -f2" || true)"
if [ -z "$LOGPATH" ]; then
  NAME="$(ssh_cluster "sacct -j '$JOBID' --format=JobName%50 -n -X 2>/dev/null | head -1 | tr -d ' '" || true)"
  [ -n "$NAME" ] && LOGPATH="$RUNS_DIR/$NAME/slurm-$JOBID.out"
fi
[ -n "$LOGPATH" ] || die "could not locate log for job $JOBID on $CLUSTER"

if [ -n "$TAIL" ]; then
  out="$(ssh_cluster "tail -n '$TAIL' '$LOGPATH' 2>&1")" || true
else
  out="$(ssh_cluster "cat '$LOGPATH' 2>&1")" || true
fi
scan_quota "$out"
echo "== $LOGPATH ($CLUSTER) =="
echo "$out"
