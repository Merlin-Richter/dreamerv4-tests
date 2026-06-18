#!/usr/bin/env bash
# scripts/cancel_job.sh --cluster {feranti|galvani} <jobid>
#
# scancel — but REFUSES any job id not recorded in EXPERIMENTS.md (protocol §6: only
# jobs present in the index). This guards against cancelling someone else's job or a
# mistyped id. Record submitted job ids in EXPERIMENTS.md immediately so cancel works.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_common.sh"
init_verb "$@"
# local guards first (fail fast, no socket needed) ---
JOBID="${WRAP_ARGS[0]:-}"
[ -n "$JOBID" ] || die_config "usage: cancel_job.sh --cluster X <jobid>"
echo "$JOBID" | grep -qE '^[0-9]+$' || die_config "job id must be numeric"
IDX="$REPO_DIR/EXPERIMENTS.md"
[ -f "$IDX" ] || die "EXPERIMENTS.md not found — cannot verify job ownership"
if ! grep -qE "(^|[^0-9])${JOBID}([^0-9]|$)" "$IDX"; then
  die "refusing to cancel job $JOBID — not found in EXPERIMENTS.md (record it there first if it's ours)"
fi

require_master
out="$(ssh_cluster "scancel '$JOBID' 2>&1")" || { scan_quota "$out"; die "scancel failed: $out"; }
log "cancelled job $JOBID on $CLUSTER"
echo "cancelled: $JOBID"
