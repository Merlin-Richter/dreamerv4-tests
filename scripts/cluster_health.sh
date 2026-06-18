#!/usr/bin/env bash
# scripts/cluster_health.sh [--cluster {feranti|galvani|both}]
#
# Reports fairshare, queue depth, your running/pending jobs, and disk/quota for each
# cluster — run BEFORE every submit, and to PICK which cluster (no default; choice
# depends on live fairshare + queue, per Merlin). Defaults to reporting BOTH.
# Best-effort: does NOT die on a single failed probe; does NOT raise QUOTA (it reports it).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_common.sh"

# parse --cluster (allow "both" / default both)
WANT="both"
while [ $# -gt 0 ]; do
  case "$1" in
    --cluster) WANT="${2:-}"; shift 2 ;;
    --cluster=*) WANT="${1#*=}"; shift ;;
    *) shift ;;
  esac
done
case "$WANT" in feranti|galvani|both) : ;; *) die_config "unknown cluster '$WANT' (feranti|galvani|both)";; esac
[ "$WANT" = both ] && CLUSTERS=(feranti galvani) || CLUSTERS=("$WANT")

report_one() {
  CLUSTER="$1"
  if ! load_config 2>/tmp/.health_cfg_err; then cat /tmp/.health_cfg_err >&2; return; fi
  echo "================ $CLUSTER (${USERNAME}@${HOST}) ================"
  local sopts; mapfile -t sopts < <(_ssh_opts)
  if ! ssh "${sopts[@]}" -O check "${USERNAME}@${HOST}" >/dev/null 2>&1; then
    echo "  socket DOWN — run: scripts/open_master.sh --cluster $CLUSTER"; echo; return
  fi
  echo "-- fairshare (sshare) --"
  ssh_cluster "sshare -U -u '${USERNAME}' 2>/dev/null || sshare -u '${USERNAME}' 2>/dev/null || echo '(sshare unavailable)'"
  echo "-- your jobs (squeue) --"
  ssh_cluster "squeue -u '${USERNAME}' -o '%.10i %.20j %.8T %.10M %.6D %R' 2>/dev/null || echo '(squeue failed)'"
  echo "-- queue depth ${PARTITION:+partition $PARTITION} --"
  ssh_cluster "squeue ${PARTITION:+-p '$PARTITION'} -h -t pending -o '%i' 2>/dev/null | wc -l | sed 's/^/  pending jobs: /'; squeue ${PARTITION:+-p '$PARTITION'} -h -t running -o '%i' 2>/dev/null | wc -l | sed 's/^/  running jobs: /'"
  echo "-- disk / quota ($REMOTE_PATH) --"
  ssh_cluster "df -h '$REMOTE_PATH' 2>/dev/null | tail -n +1; lfs quota -u '${USERNAME}' '$REMOTE_PATH' 2>/dev/null || true"
  echo
}

for c in "${CLUSTERS[@]}"; do report_one "$c"; done
