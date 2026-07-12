#!/usr/bin/env bash
# scripts/push_file.sh --cluster {ferranti|galvani|vast} <local-path> [--dest REMOTE]
#
# Copy a single LOCAL file up to the cluster — the inverse of pull_file.sh, for binary
# INPUTS that cannot go up via git (checkpoints to warm-start from, tokenizers). Code
# still goes up ONLY via sync_code.sh (protocol §6's asymmetric transport is unchanged:
# this verb is for data, never source).
#
# <local-path> is taken relative to the local repo root unless absolute. Default remote
# destination MIRRORS the repo-relative path under REMOTE_PATH, so
#   push_file.sh --cluster vast checkpoints/memmaze/tokenizer.pt
# lands at REMOTE_PATH/checkpoints/memmaze/tokenizer.pt. Override with --dest:
#   --dest ending in '/'  -> file dropped inside that remote dir (keeps basename)
#   --dest as a full path -> exact remote file path (relative => under REMOTE_PATH)
#
# Sanctioned transport: rsync over the authenticated ControlMaster socket (protocol §6).
# Same error contract as the other verbs (AUTH_DEAD / QUOTA / BAD_REF / ...).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_common.sh"

DEST=""
CLEAN_ARGS=()
i=0; A=("$@")
while [ $i -lt ${#A[@]} ]; do
  case "${A[$i]}" in
    --dest) DEST="${A[$((i+1))]:-}"; i=$((i+2));;
    --dest=*) DEST="${A[$i]#*=}"; i=$((i+1));;
    *) CLEAN_ARGS+=("${A[$i]}"); i=$((i+1));;
  esac
done
set -- "${CLEAN_ARGS[@]}"
init_verb "$@"
require_master

LOCAL="${WRAP_ARGS[0]:-}"
[ -n "$LOCAL" ] || die_config "usage: push_file.sh --cluster X <local-path> [--dest REMOTE]"
[ "${#WRAP_ARGS[@]}" -le 1 ] || die_config "expected exactly one <local-path> (got ${#WRAP_ARGS[@]}); the remote target goes in --dest"

# Resolve the local source (relative => under the local repo root).
case "$LOCAL" in
  *..*) die_config "<local-path> must not contain '..'" ;;
  /*)   SRC_FILE="$LOCAL"; REL="" ;;
  *)    SRC_FILE="$REPO_DIR/$LOCAL"; REL="$LOCAL" ;;
esac
[ -e "$SRC_FILE" ] || die_badref "local path not found: $SRC_FILE"
[ -f "$SRC_FILE" ] || die_config "local path is not a regular file (this verb pushes one file): $SRC_FILE"

# Resolve the remote destination.
base="$(basename "$SRC_FILE")"
if [ -n "$DEST" ]; then
  case "$DEST" in
    *..*) die_config "--dest must not contain '..'" ;;
    */)   DEST_PATH="$DEST$base" ;;
    *)    DEST_PATH="$DEST" ;;
  esac
  case "$DEST_PATH" in
    /*) : ;;
    *)  DEST_PATH="$REMOTE_PATH/$DEST_PATH" ;;
  esac
elif [ -n "$REL" ]; then
  DEST_PATH="$REMOTE_PATH/$REL"   # mirror the repo-relative path
else
  DEST_PATH="$REMOTE_PATH/$base"  # absolute local -> drop basename in the remote repo root
fi

ssh_cluster "mkdir -p '$(dirname "$DEST_PATH")'"
log "pushing $SRC_FILE -> $CLUSTER:$DEST_PATH"
out="$(rsync_cluster -az --partial "$SRC_FILE" "${USERNAME}@${HOST}:$DEST_PATH" 2>&1)" || { scan_quota "$out"; die "rsync failed: $(echo "$out" | tail -3)"; }
echo "pushed to: $CLUSTER:$DEST_PATH"
ssh_cluster "ls -la '$DEST_PATH'"
