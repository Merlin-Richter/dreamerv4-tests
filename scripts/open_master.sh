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

echo "Opening ControlMaster to ${USERNAME}@${HOST} ($CLUSTER) — complete 2FA when prompted..."
ssh "${opts[@]}" "${USERNAME}@${HOST}"
ssh -S "$CONTROL_PATH" -O check "${USERNAME}@${HOST}" \
  && echo "Master socket up. Wrappers can now reach $CLUSTER for ~8h." \
  || { echo "Failed to establish master socket." >&2; exit 1; }
