#!/usr/bin/env bash
# scripts/sync_code.sh --cluster {ferranti|galvani} <branch> [sha]
#
# On the remote clone: git fetch origin, checkout <sha> (or <branch> tip if no sha),
# and echo the RESOLVED commit SHA (record it in EXPERIMENTS.md for provenance).
# Pulls from GitHub origin (Merlin's chosen sync model). BAD_REF if the ref is unknown.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_common.sh"
init_verb "$@"
require_master

BRANCH="${WRAP_ARGS[0]:-}"
SHA="${WRAP_ARGS[1]:-}"
[ -n "$BRANCH" ] || die_config "usage: sync_code.sh --cluster X <branch> [sha]"
TARGET="${SHA:-origin/$BRANCH}"

out="$(ssh_repo "git fetch --prune origin '$BRANCH' 2>&1 && git checkout -q '$TARGET' 2>&1 && git rev-parse HEAD 2>&1" 2>&1)" || {
  scan_quota "$out"
  if echo "$out" | grep -qiE 'did not match|unknown revision|pathspec|couldn.t find remote ref|fatal: .*not'; then
    die_badref "ref '$TARGET' not found on $CLUSTER remote — $(echo "$out" | tail -1)"
  fi
  die "sync_code failed on $CLUSTER: $(echo "$out" | tail -3)"
}
RESOLVED="$(echo "$out" | tail -1)"
echo "$RESOLVED" | grep -qE '^[0-9a-f]{40}$' || die "could not resolve SHA: $(echo "$out" | tail -3)"
log "synced $CLUSTER to $BRANCH @ $RESOLVED"
echo "SHA: $RESOLVED"
