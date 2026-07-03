#!/usr/bin/env bash
# scripts/clean_untracked.sh --cluster {ferranti|galvani} <repo-relative-path> [...]
#
# Remove UNTRACKED file(s)/dir(s) from the remote repo clone. Use case: a cluster job wrote an
# artifact into a tracked directory (e.g. experiments/**), the artifact was pulled + committed
# locally, and the next sync_code checkout now refuses to overwrite the remote's untracked copy.
#
# Safe by construction: delegates to `git clean -f -- <paths>`, which NEVER touches tracked files
# (git refuses) nor gitignored files (no -x flag) — only plain untracked paths are removed.
# Paths must be repo-relative (no absolute paths, no '..').
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_common.sh"
init_verb "$@"
require_master

[ ${#WRAP_ARGS[@]} -ge 1 ] || die_config "usage: clean_untracked.sh --cluster X <repo-relative-path>..."
QUOTED=""
for p in "${WRAP_ARGS[@]}"; do
  case "$p" in
    /*) die_config "absolute path not allowed: $p" ;;
    *..*) die_config "'..' not allowed in path: $p" ;;
  esac
  QUOTED+=" '$p'"
done

out="$(ssh_repo "git clean -f --$QUOTED" 2>&1)" || { scan_quota "$out"; die "clean_untracked failed on $CLUSTER: $(echo "$out" | tail -3)"; }
[ -n "$out" ] && echo "$out"
log "clean_untracked done on $CLUSTER:$QUOTED"
