#!/usr/bin/env bash
# scripts/_common.sh — shared core for the cluster wrappers.
#
# Sourced by every verb. Provides: config loading + --cluster selection, the single
# ssh_cluster() entry point (reuses Merlin's authenticated ControlMaster socket — NEVER
# re-auths), and the machine-parseable error contract (first stderr line = ERROR: <CODE>).
#
# Design rules (protocol §6):
#  - The wrappers are the ONLY sanctioned cluster interface; all ssh goes through ssh_cluster().
#  - We NEVER attempt to (re)authenticate. If the master socket is down → ERROR: AUTH_DEAD, exit.
#  - Site specifics live ONLY in cluster.env. Missing required keys → ERROR: BAD_CONFIG.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- error contract: first stderr line is "ERROR: <CODE>" -------------------
# Exit codes: 2 BAD_CONFIG, 3 AUTH_DEAD, 4 QUOTA, 5 BAD_REF, 6 generic/unrecognized.
die_config()  { echo "ERROR: BAD_CONFIG${1:+ — $1}" >&2; exit 2; }
die_auth()    { echo "ERROR: AUTH_DEAD${1:+ — $1}" >&2; exit 3; }
die_quota()   { echo "ERROR: QUOTA${1:+ — $1}" >&2; exit 4; }
die_badref()  { echo "ERROR: BAD_REF${1:+ — $1}" >&2; exit 5; }
die()         { echo "ERROR: ${1:-unspecified}" >&2; exit "${2:-6}"; }
log()         { echo "[$(date +%H:%M:%S)] $*" >&2; }

# ---- --cluster parsing ------------------------------------------------------
# Pulls "--cluster X" (or "--cluster=X") out of "$@", sets CLUSTER, and re-exports
# the REMAINING args into the global array WRAP_ARGS for the caller to consume.
CLUSTER=""
WRAP_ARGS=()
parse_cluster() {
  WRAP_ARGS=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --cluster) CLUSTER="${2:-}"; shift 2 || die_config "--cluster needs a value" ;;
      --cluster=*) CLUSTER="${1#*=}"; shift ;;
      *) WRAP_ARGS+=("$1"); shift ;;
    esac
  done
  case "$CLUSTER" in
    ferranti|galvani) : ;;
    "") die_config "--cluster {ferranti|galvani} is required (no default — pick per live fairshare/queue)" ;;
    *) die_config "unknown cluster '$CLUSTER' (expected ferranti|galvani)" ;;
  esac
}

# ---- config loading ---------------------------------------------------------
# Loads cluster.env and maps the chosen cluster's PREFIX_* vars onto generic names.
load_config() {
  local envf="$SCRIPT_DIR/cluster.env"
  [ -f "$envf" ] || die_config "scripts/cluster.env not found — copy cluster.env.example and fill it in"
  # shellcheck disable=SC1090
  set -a; . "$envf"; set +a
  local P; P="$(echo "$CLUSTER" | tr '[:lower:]' '[:upper:]')"
  # required keys
  HOST="$(_cfg "${P}_SSH_HOST")"
  USERNAME="$(_cfg "${P}_SSH_USER")"
  CONTROL_PATH="$(_cfg "${P}_CONTROL_PATH")"
  REMOTE_PATH="$(_cfg "${P}_REMOTE_PATH")"
  VENV_ROOT="$(_cfg "${P}_VENV_ROOT")"
  RUNS_SUBDIR="$(_cfg "${P}_RUNS_SUBDIR")"
  # optional keys (may be empty)
  PROXY_JUMP="$(_cfg_opt "${P}_PROXY_JUMP")"
  PARTITION="$(_cfg_opt "${P}_PARTITION")"
  ACCOUNT="$(_cfg_opt "${P}_ACCOUNT")"
  GRES="$(_cfg_opt "${P}_GRES")"
  CONSTRAINT="$(_cfg_opt "${P}_CONSTRAINT")"
  DEFAULT_HOURS="$(_cfg_opt "${P}_DEFAULT_HOURS")"
  MODULES="$(_cfg_opt "${P}_MODULES")"
  RUNS_DIR="$REMOTE_PATH/$RUNS_SUBDIR"
  # tilde-expand the control path (ssh does not expand ~ inside -S in all builds)
  CONTROL_PATH="${CONTROL_PATH/#\~/$HOME}"
}
# required: error if blank
_cfg() { local v="${!1:-}"; [ -n "$v" ] || die_config "missing required key $1 in cluster.env"; printf '%s' "$v"; }
# optional: empty allowed
_cfg_opt() { printf '%s' "${!1:-}"; }

# ---- ssh ControlMaster plumbing --------------------------------------------
# Common ssh opts: reuse the master socket, never become master, never prompt.
_ssh_opts() {
  local opts=(-S "$CONTROL_PATH" -o ControlMaster=no -o BatchMode=yes
              -o ConnectTimeout=10 -o ServerAliveInterval=15)
  [ -n "$PROXY_JUMP" ] && opts+=(-o "ProxyJump=$PROXY_JUMP")
  printf '%s\n' "${opts[@]}"
}

# Verify the master socket is alive; AUTH_DEAD if not (Merlin must re-open it).
require_master() {
  local opts; mapfile -t opts < <(_ssh_opts)
  if ! ssh "${opts[@]}" -O check "${USERNAME}@${HOST}" >/dev/null 2>&1; then
    die_auth "no live ControlMaster socket for $CLUSTER at $CONTROL_PATH — run: scripts/open_master.sh --cluster $CLUSTER (interactive, completes 2FA)"
  fi
}

# Run a command on the cluster over the master socket. Echoes remote stdout.
# Usage: ssh_cluster '<remote shell command>'
ssh_cluster() {
  local opts; mapfile -t opts < <(_ssh_opts)
  ssh "${opts[@]}" "${USERNAME}@${HOST}" "$@"
}

# Run a remote command IN the repo dir (most verbs want this).
ssh_repo() {
  ssh_cluster "cd '$REMOTE_PATH' && $*"
}

# rsync over the master socket (for pull_results). Args after the function are rsync args.
rsync_cluster() {
  local opts; mapfile -t opts < <(_ssh_opts)
  local sshcmd="ssh"; local o
  for o in "${opts[@]}"; do sshcmd+=" $o"; done
  rsync -e "$sshcmd" "$@"
}

# Detect quota/disk-full signatures in a remote command's combined output → QUOTA.
# Usage: scan_quota "<captured output>"
scan_quota() {
  if echo "$1" | grep -qiE 'disk quota exceeded|no space left on device|quota exceeded'; then
    die_quota "remote reported quota/disk-full — run cluster_health.sh --cluster $CLUSTER and clean_run.sh superseded runs"
  fi
}

# Standard init for a verb: parse --cluster from "$@", load config, set WRAP_ARGS.
# Caller then reads positional args from "${WRAP_ARGS[@]}".
init_verb() {
  parse_cluster "$@"
  load_config
}
