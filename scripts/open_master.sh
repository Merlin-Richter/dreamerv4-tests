#!/usr/bin/env bash
# scripts/open_master.sh --cluster {ferranti|galvani}
#
# INTERACTIVE — run this yourself (Merlin) in a normal terminal. It opens the shared
# ControlMaster socket the wrappers reuse, completing 2FA here once. The orchestrator
# CANNOT run this (it needs your interactive auth). Re-run when a socket has expired
# (a wrapper reporting ERROR: AUTH_DEAD means this needs re-running).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_common.sh"

init_verb "$@"

opts=(-M -S "$CONTROL_PATH" -o ControlPersist=8h -o ServerAliveInterval=30 -fN)
[ -n "$PROXY_JUMP" ] && opts+=(-o "ProxyJump=$PROXY_JUMP")

# already alive?
if ssh -S "$CONTROL_PATH" -O check "${USERNAME}@${HOST}" >/dev/null 2>&1; then
  echo "Master socket for $CLUSTER already alive ($CONTROL_PATH)."
  exit 0
fi

# Fell through: no live master. A stale socket file may still be on disk (e.g. the box
# rebooted or the connection dropped without a clean exit). ssh -M refuses to open while
# the file exists ("already exists, disabling multiplexing"), so tear it down first:
# ask any lingering master to exit, then force-remove whatever's left.
#
# $CONTROL_PATH carries ssh ControlPath tokens (%r@%h:%p). ssh expands them itself for
# -S/-O, but a bare [ -e ]/rm would test the literal token string and never match the
# real file — so resolve the tokens here for the shell-level ops.
SOCKET_FILE="$CONTROL_PATH"
SOCKET_FILE="${SOCKET_FILE//%r/$USERNAME}"
SOCKET_FILE="${SOCKET_FILE//%h/$HOST}"
SOCKET_FILE="${SOCKET_FILE//%n/$HOST}"
SOCKET_FILE="${SOCKET_FILE//%p/22}"
SOCKET_FILE="${SOCKET_FILE//%%/%}"
if [ -e "$SOCKET_FILE" ]; then
  echo "Removing dead socket for $CLUSTER ($SOCKET_FILE)..."
  ssh -S "$CONTROL_PATH" -O exit "${USERNAME}@${HOST}" >/dev/null 2>&1 || true
  rm -f "$SOCKET_FILE"
fi

echo "Opening ControlMaster to ${USERNAME}@${HOST} ($CLUSTER) — complete 2FA when prompted..."
ssh "${opts[@]}" "${USERNAME}@${HOST}"
ssh -S "$CONTROL_PATH" -O check "${USERNAME}@${HOST}" \
  && echo "Master socket up. Wrappers can now reach $CLUSTER for ~8h." \
  || { echo "Failed to establish master socket." >&2; exit 1; }
