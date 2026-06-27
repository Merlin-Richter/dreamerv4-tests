#!/usr/bin/env bash
# scripts/pull_file.sh --cluster {ferranti|galvani} <remote-path> [--dest LOCAL]
#
# Copy a single file back from the cluster to local — for things that live OUTSIDE
# runs/<run>/ and so are unreachable by pull_results.sh. The common case: a checkpoint a
# training job wrote to checkpoints/<env>/ (the job cd's to the repo root, so --checkpoint
# checkpoints/gridworld/x.pt lands at REMOTE_PATH/checkpoints/..., a sibling of runs/).
#
# <remote-path> is taken relative to the cluster repo root (REMOTE_PATH) unless it starts
# with '/' (absolute). Default destination MIRRORS the repo-relative path locally, so
#   pull_file.sh --cluster ferranti checkpoints/gridworld/x.pt
# writes the local repo's checkpoints/gridworld/x.pt. Override with --dest:
#   --dest ending in '/' or an existing directory  -> file dropped inside it (keeps basename)
#   --dest as a full path                          -> exact local file path
#
# Sanctioned transport: rsync over Merlin's authenticated ControlMaster socket (protocol §6).
# Never raw scp/ssh. Same error contract as the other verbs (AUTH_DEAD / QUOTA / BAD_REF / ...).
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

REMOTE="${WRAP_ARGS[0]:-}"
[ -n "$REMOTE" ] || die_config "usage: pull_file.sh --cluster X <remote-path> [--dest LOCAL]"
[ "${#WRAP_ARGS[@]}" -le 1 ] || die_config "expected exactly one <remote-path> (got ${#WRAP_ARGS[@]}); the local target goes in --dest"

# Resolve the remote source path (relative => under the repo root REMOTE_PATH).
case "$REMOTE" in
  *..*) die_config "<remote-path> must not contain '..'" ;;
  /*)   SRC_PATH="$REMOTE"; REL="" ;;
  *)    SRC_PATH="$REMOTE_PATH/$REMOTE"; REL="$REMOTE" ;;
esac

# Confirm it exists remotely (clean error instead of a cryptic rsync failure) and is a file.
ssh_cluster "test -e '$SRC_PATH'" || die_badref "remote path not found on $CLUSTER: $SRC_PATH"
ssh_cluster "test -f '$SRC_PATH'" || die_config "remote path is not a regular file (this verb pulls one file; use pull_results.sh for run dirs): $SRC_PATH"

# Resolve the local destination.
base="$(basename "$REMOTE")"
if [ -n "$DEST" ]; then
  case "$DEST" in
    */) DEST_FILE="${DEST}${base}" ;;
    *)  if [ -d "$DEST" ]; then DEST_FILE="${DEST%/}/$base"; else DEST_FILE="$DEST"; fi ;;
  esac
elif [ -n "$REL" ]; then
  DEST_FILE="$REPO_DIR/$REL"      # mirror the repo-relative path
else
  DEST_FILE="$REPO_DIR/$base"     # absolute remote -> drop basename in the repo root
fi

mkdir -p "$(dirname "$DEST_FILE")"
SRC="${USERNAME}@${HOST}:$SRC_PATH"
log "pulling $CLUSTER:$SRC_PATH -> $DEST_FILE"
out="$(rsync_cluster -az --partial "$SRC" "$DEST_FILE" 2>&1)" || { scan_quota "$out"; die "rsync failed: $(echo "$out" | tail -3)"; }
echo "pulled to: $DEST_FILE"
ls -la "$DEST_FILE"
