#!/usr/bin/env bash
# scripts/pull_results.sh --cluster {ferranti|galvani} <run> [--what all|logs|metrics|checkpoints] [--dest DIR]
#
# rsyncs $RUNS_DIR/<run>/ from the cluster to a local dir (default experiments/<run>/).
# Checkpoints (*.pt) are EXCLUDED unless --what explicitly includes 'checkpoints' or 'all'
# (they are large; pull only on demand — protocol §6).
#   logs        -> *.out / *.log / *.txt
#   metrics     -> *.json / *.csv / *.yaml / *.md   (default)
#   checkpoints -> include *.pt
#   all         -> everything
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_common.sh"

WHAT="metrics"; DEST=""
CLEAN_ARGS=()
i=0; A=("$@")
while [ $i -lt ${#A[@]} ]; do
  case "${A[$i]}" in
    --what) WHAT="${A[$((i+1))]:-}"; i=$((i+2));;
    --what=*) WHAT="${A[$i]#*=}"; i=$((i+1));;
    --dest) DEST="${A[$((i+1))]:-}"; i=$((i+2));;
    --dest=*) DEST="${A[$i]#*=}"; i=$((i+1));;
    *) CLEAN_ARGS+=("${A[$i]}"); i=$((i+1));;
  esac
done
set -- "${CLEAN_ARGS[@]}"
init_verb "$@"
require_master
RUN="${WRAP_ARGS[0]:-}"
[ -n "$RUN" ] || die_config "usage: pull_results.sh --cluster X <run> [--what ...] [--dest DIR]"
echo "$RUN" | grep -qE '^[A-Za-z0-9._-]+$' || die_config "run name must be [A-Za-z0-9._-]"
DEST="${DEST:-$REPO_DIR/experiments/$RUN}"
mkdir -p "$DEST"

# rsync filter by --what
RSYNC_FILTERS=()
case "$WHAT" in
  all) ;;  # no filter
  checkpoints) ;;  # everything incl *.pt
  logs)    RSYNC_FILTERS=(--include='*/' --include='*.out' --include='*.log' --include='*.txt' --exclude='*');;
  metrics) RSYNC_FILTERS=(--include='*/' --include='*.json' --include='*.csv' --include='*.yaml' --include='*.md' --include='*.png' --exclude='*');;
  *) die_config "unknown --what '$WHAT' (all|logs|metrics|checkpoints)";;
esac
# exclude checkpoints unless asked
case "$WHAT" in all|checkpoints) :;; *) RSYNC_FILTERS=(--exclude='*.pt' "${RSYNC_FILTERS[@]}");; esac

SRC="${USERNAME}@${HOST}:$RUNS_DIR/$RUN/"
log "pulling $CLUSTER:$RUNS_DIR/$RUN -> $DEST (what=$WHAT)"
out="$(rsync_cluster -az --partial "${RSYNC_FILTERS[@]}" "$SRC" "$DEST/" 2>&1)" || { scan_quota "$out"; die "rsync failed: $(echo "$out" | tail -3)"; }
echo "pulled to: $DEST"
ls -la "$DEST" | tail -n +2
