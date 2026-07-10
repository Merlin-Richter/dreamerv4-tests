#!/usr/bin/env bash
# scripts/open_master.sh --cluster {ferranti|galvani|vast}
#
# INTERACTIVE — run this yourself (Merlin) in a normal terminal for ferranti/galvani.
# It opens the shared ControlMaster socket the wrappers reuse, completing 2FA here
# once. The orchestrator CANNOT run this for those two (needs your interactive auth).
# Re-run when a socket has expired (a wrapper reporting ERROR: AUTH_DEAD means this
# needs re-running).
#
# EXCEPTION — vast: plain SSH-key auth, no 2FA. The orchestrator MAY run this itself
# for --cluster vast (nothing interactive to complete) and self-heal an AUTH_DEAD
# without escalating. Same socket mechanics either way.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_common.sh"

init_verb "$@"

opts=(-M -S "$CONTROL_PATH" -o ControlPersist=8h -o ServerAliveInterval=30 -fN)
[ -n "$PROXY_JUMP" ] && opts+=(-o "ProxyJump=$PROXY_JUMP")
[ -n "$PORT" ] && opts+=(-p "$PORT")
[ -n "$IDENTITY" ] && opts+=(-i "$IDENTITY" -o IdentitiesOnly=yes)

# -O check/exit don't reconnect, but ssh still expands %p in ControlPath from the port
# it WOULD use — omitting -p here defaults that to 22, silently mismatching a real
# non-default port (vast) and misreporting a live master as dead. Same opts minus the
# master-only flags (-M/-fN/ControlPersist/ServerAliveInterval).
check_opts=(-S "$CONTROL_PATH")
[ -n "$PROXY_JUMP" ] && check_opts+=(-o "ProxyJump=$PROXY_JUMP")
[ -n "$PORT" ] && check_opts+=(-p "$PORT")

# already alive?
if ssh "${check_opts[@]}" -O check "${USERNAME}@${HOST}" >/dev/null 2>&1; then
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
SOCKET_FILE="${SOCKET_FILE//%p/${PORT:-22}}"
SOCKET_FILE="${SOCKET_FILE//%%/%}"
if [ -e "$SOCKET_FILE" ]; then
  echo "Removing dead socket for $CLUSTER ($SOCKET_FILE)..."
  ssh "${check_opts[@]}" -O exit "${USERNAME}@${HOST}" >/dev/null 2>&1 || true
  rm -f "$SOCKET_FILE"
fi

echo "Opening ControlMaster to ${USERNAME}@${HOST} ($CLUSTER)$([ "$CLUSTER" = vast ] || echo ' — complete 2FA when prompted')..."
ssh "${opts[@]}" "${USERNAME}@${HOST}"
ssh "${check_opts[@]}" -O check "${USERNAME}@${HOST}" \
  && echo "Master socket up. Wrappers can now reach $CLUSTER for ~8h." \
  || { echo "Failed to establish master socket." >&2; exit 1; }
