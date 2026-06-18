#!/usr/bin/env bash
# scripts/clean_run.sh --cluster {feranti|galvani} <run>
#
# Deletes $RUNS_DIR/<run> on the cluster. Restricted to the runs/ subtree (protocol §6):
# refuses absolute paths, '..', slashes, or anything that would escape $RUNS_DIR.
# Use to reclaim quota from superseded runs you own.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_common.sh"
init_verb "$@"
# local guards first (fail fast, no socket needed) ---
RUN="${WRAP_ARGS[0]:-}"
[ -n "$RUN" ] || die_config "usage: clean_run.sh --cluster X <run>"
# strict whitelist: single path component, no traversal
echo "$RUN" | grep -qE '^[A-Za-z0-9._-]+$' || die_config "refusing run name '$RUN' (must be a single [A-Za-z0-9._-] component — no slashes/.. /absolute)"
case "$RUN" in .|..) die_config "refusing '$RUN'";; esac

require_master
TARGET="$RUNS_DIR/$RUN"
# defense-in-depth: confirm the resolved path is inside RUNS_DIR before rm
out="$(ssh_cluster "
  set -e
  rd=\$(cd '$RUNS_DIR' 2>/dev/null && pwd) || { echo 'NORUNS'; exit 0; }
  tgt='$TARGET'
  case \"\$tgt\" in \"\$rd\"/*) : ;; *) echo 'ESCAPE'; exit 0;; esac
  [ -e \"\$tgt\" ] || { echo 'MISSING'; exit 0; }
  rm -rf \"\$tgt\" && echo 'OK'
" 2>&1)" || { scan_quota "$out"; die "clean_run failed: $out"; }

case "$(echo "$out" | tail -1)" in
  OK) log "removed $CLUSTER:$TARGET"; echo "cleaned: $TARGET" ;;
  MISSING) echo "nothing to clean (not present): $TARGET" ;;
  ESCAPE) die "refused — resolved path escapes $RUNS_DIR" ;;
  NORUNS) die "runs dir $RUNS_DIR does not exist on $CLUSTER" ;;
  *) die "unexpected clean_run result: $out" ;;
esac
