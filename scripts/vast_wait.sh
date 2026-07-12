#!/usr/bin/env bash
# scripts/vast_wait.sh [--cluster vast] <RUN> [--poll SECONDS]
#
# The Vast.ai equivalent of wait_for_jobs.sh. No sacct/squeue to poll — "terminal"
# means the job's self-registered pidfile is gone (or its PID is no longer alive).
# Distinguishes clean completion from a crash by grepping the trailer job.sh always
# writes on a clean exit ("=== done <RUN> rc=N ..."); PID-dead with no trailer means
# it was killed/crashed before reaching it. Also does the same early Traceback/OOM
# log-scan wait_for_jobs.sh does, so a crashing run is caught fast rather than only
# at the end. Exit: 0 clean rc=0; 7 failed/crashed (inspect $RUN_DIR/run.log).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_common.sh"

POLL=30
CLEAN_ARGS=()
i=0; A=("$@")
while [ $i -lt ${#A[@]} ]; do
  case "${A[$i]}" in
    --cluster) [ "${A[$((i+1))]:-}" = vast ] || die_config "vast_wait.sh only targets vast"; i=$((i+2));;
    --cluster=*) [ "${A[$i]#*=}" = vast ] || die_config "vast_wait.sh only targets vast"; i=$((i+1));;
    --poll) POLL="${A[$((i+1))]:-30}"; i=$((i+2));;
    --poll=*) POLL="${A[$i]#*=}"; i=$((i+1));;
    *) CLEAN_ARGS+=("${A[$i]}"); i=$((i+1));;
  esac
done
RUN="${CLEAN_ARGS[0]:-}"
[ -n "$RUN" ] || die_config "usage: vast_wait.sh [--cluster vast] <RUN> [--poll S]"
echo "$RUN" | grep -qE '^[A-Za-z0-9._-]+$' || die_config "run name must be [A-Za-z0-9._-]"

CLUSTER=vast
load_config
require_master
RUN_DIR="$RUNS_DIR/$RUN"

# Hard-bound every polling ssh call. This box's ssh mux layer has shown BOTH an
# instant reset+fallback (~1-3s tax, harmless — see vast_run.sh) AND a rarer TRUE
# HANG (observed live 2026-07-10: a poll call blocked 3+ minutes with no error at
# all, long after the remote job had already finished). ConnectTimeout only bounds
# the initial TCP handshake, not a channel that hangs after connecting — this turns
# a hang into a detected failure that feeds the FAILS/MAX_FAILS retry below, instead
# of blocking this script — and a real autoresearch loop iteration — forever.
export SSH_CALL_TIMEOUT=45

# Retry the existence gate: a mux channel reset makes ssh_cluster fail even though the
# dir exists (observed live 2026-07-12, while a concurrent rsync push saturated the mux
# connection) — one flaky call must not BAD_REF a wait on a real run.
EXISTS=""
for _try in 1 2 3; do
  ssh_cluster "test -d '$RUN_DIR'" && { EXISTS=1; break; }
  sleep 10
done
[ -n "$EXISTS" ] || die_badref "no such run on vast: $RUN"

log "waiting on vast run $RUN (poll ${POLL}s)"
FAILS=0; MAX_FAILS=3
while true; do
  # NOTE: capture with 2>&1 so a real ssh-level failure is visible in the diagnostic
  # (raw) below — but every fallback connection ALSO prints the login banner + a
  # harmless "Connection reset by peer" mux warning on its OWN success path (see
  # vast_run.sh's notes), so `raw` is NOT the bare ALIVE/DEAD answer even on success.
  # Bug caught live 2026-07-10: comparing the whole blob against "DEAD" never
  # matched, so this loop polled forever past job completion. Fix: the real answer
  # is always the LAST line; take that, keep `raw` only for failure logging.
  if ! raw="$(ssh_cluster "p=\$(cat '$RUN_DIR/run.pid' 2>/dev/null) || { echo DEAD; exit 0; }; kill -0 \"\$p\" 2>/dev/null && echo ALIVE || echo DEAD" 2>&1)"; then
    FAILS=$((FAILS + 1))
    log "poll failed (${FAILS}/${MAX_FAILS}): ${raw:-<no output>}"
    scan_quota "$raw"
    if [ "$FAILS" -ge "$MAX_FAILS" ]; then
      require_master  # exits AUTH_DEAD if the socket is gone (the expected cause)
      die "polling vast failed ${FAILS}x but the master socket is alive — investigate" 7
    fi
    sleep "$POLL"; continue
  fi
  FAILS=0
  alive="$(printf '%s\n' "$raw" | tail -1)"

  if [ "$alive" = DEAD ]; then
    log "run $RUN: pidfile gone, checking trailer"
    tail3="$(ssh_cluster "tail -n 5 '$RUN_DIR/run.log' 2>/dev/null" || true)"
    if echo "$tail3" | grep -qE '^=== done '"$RUN"' rc=0 '; then
      log "run $RUN COMPLETED"
      exit 0
    fi
    log "run $RUN: no clean rc=0 trailer — treating as FAILED"
    die "vast run $RUN ended without a clean rc=0 trailer — vast_status.sh $RUN (last lines: $(echo "$tail3" | tail -3))" 7
  fi

  # early crash detection, same signatures as wait_for_jobs.sh
  if ssh_cluster "tail -n 200 '$RUN_DIR/run.log' 2>/dev/null | grep -qE 'Traceback \(most recent call last\)|CUDA out of memory'"; then
    log "run $RUN: crash signature in log"
    die "vast run $RUN shows a Traceback/OOM in its log — vast_status.sh $RUN" 7
  fi
  sleep "$POLL"
done
