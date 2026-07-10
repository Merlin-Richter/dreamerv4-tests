#!/usr/bin/env bash
# scripts/vast_run.sh --name RUN -- <command...>
#
# The Vast.ai equivalent of submit_job.sh. There is NO SLURM on a Vast rental — it's
# a single box you already own outright, not a shared login node with a scheduler.
# "Submit" here means: render vast_job_template.sh, write it to the box, and launch
# it DETACHED (setsid+nohup) so it outlives this ssh session. The job self-registers
# its own PID as run.pid — that PID is this backend's job id (echoed as "JOB_ID: <RUN>"
# to slot into the same EXPERIMENTS.md-recording habit as the SLURM verbs).
#
# One job at a time: refuses to launch if any run under $RUNS_DIR still has a live
# PID (no queue to hide behind — a single GPU can only run one thing anyway).
# Always targets vast (implied — there's only one non-SLURM backend; an explicit
# --cluster must equal vast if given, just so a copy-pasted SLURM invocation errors
# loudly instead of silently doing the wrong thing).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_common.sh"

RUN_NAME=""; CMD_ARGS=()
ARGS=("$@"); i=0
while [ $i -lt ${#ARGS[@]} ]; do
  case "${ARGS[$i]}" in
    --cluster) [ "${ARGS[$((i+1))]:-}" = vast ] || die_config "vast_run.sh only targets vast"; i=$((i+2));;
    --cluster=*) [ "${ARGS[$i]#*=}" = vast ] || die_config "vast_run.sh only targets vast"; i=$((i+1));;
    --name) RUN_NAME="${ARGS[$((i+1))]:-}"; i=$((i+2));;
    --) i=$((i+1)); while [ $i -lt ${#ARGS[@]} ]; do CMD_ARGS+=("${ARGS[$i]}"); i=$((i+1)); done;;
    *) die_config "unexpected arg '${ARGS[$i]}' (command must come after --)";;
  esac
done
[ -n "$RUN_NAME" ] || die_config "--name RUN is required"
[ "${#CMD_ARGS[@]}" -gt 0 ] || die_config "no command given (put it after --)"
echo "$RUN_NAME" | grep -qE '^[A-Za-z0-9._-]+$' || die_config "run name must be [A-Za-z0-9._-]"

CLUSTER=vast
load_config
require_master
export SSH_CALL_TIMEOUT=60   # see _common.sh's ssh_cluster() — bounds a hung vast mux channel; a bit longer here since the launch call also does the remote pid-registration poll

CMD="${CMD_ARGS[*]}"
RUN_DIR="$RUNS_DIR/$RUN_NAME"

# W&B export (same convention as job_template.sbatch)
WB_LINE="# (WandB auth via remote ~/.netrc if present)"
[ -n "${WANDB_API_KEY:-}" ] && WB_LINE="export WANDB_API_KEY='${WANDB_API_KEY}'"
[ -n "${WANDB_ENTITY:-}" ] && WB_LINE="$WB_LINE
export WANDB_ENTITY='${WANDB_ENTITY}'"

# Same escaping discipline as submit_job.sh's render() — see its comment for why.
_esc() { local s="$1"; s="${s//\\/\\\\}"; s="${s//&/\\&}"; printf '%s' "$s"; }
render() {
  local t; t="$(cat "$SCRIPT_DIR/vast_job_template.sh")"
  t="${t//@RUN_NAME@/$(_esc "$RUN_NAME")}"
  t="${t//@RUN_DIR@/$(_esc "$RUN_DIR")}"
  t="${t//@REMOTE_PATH@/$(_esc "$REMOTE_PATH")}"
  t="${t//@VENV_ROOT@/$(_esc "$VENV_ROOT")}"
  t="${t//@WANDB_EXPORT@/$(_esc "$WB_LINE")}"
  t="${t//@CMD@/$(_esc "$CMD")}"
  printf '%s\n' "$t"
}

# Serialize "check busy + write job.sh + launch" against another concurrent
# vast_run.sh with an atomic remote `mkdir` lock — plain sequential ssh calls left a
# real TOCTOU race (verified live 2026-07-10: two vast_run.sh invocations fired ~1s
# apart both passed the busy-check before either had registered its PID, and both
# launched concurrently on the one GPU). `mkdir` on the remote fs is atomic even
# though our own ssh transport falls back through reconnects — the OS still
# serializes which caller's mkdir wins. Held only for this critical section, not
# the job's lifetime (that's tracked by the per-run PID file as before).
LOCKDIR="$RUNS_DIR/.launch.lock"
ssh_cluster "mkdir -p '$RUNS_DIR'"
ssh_cluster "mkdir '$LOCKDIR'" >/dev/null 2>&1 || die "another vast_run.sh launch is racing this one (lock held: $LOCKDIR) — retry in a few seconds"
_release_lock() { ssh_cluster "rmdir '$LOCKDIR'" >/dev/null 2>&1 || true; }
trap _release_lock EXIT

# refuse if the box is already busy (any run dir with a live PID)
BUSY="$(ssh_cluster "for f in '$RUNS_DIR'/*/run.pid; do [ -f \"\$f\" ] || continue; p=\$(cat \"\$f\" 2>/dev/null) || continue; kill -0 \"\$p\" 2>/dev/null && echo \"\$f (pid \$p)\"; done" 2>/dev/null || true)"
[ -z "$BUSY" ] || die "vast box already busy: $BUSY — wait (vast_wait.sh) or vast_cancel.sh it first"

ssh_cluster "mkdir -p '$RUN_DIR'"
SCRIPT_TXT="$(render)"
out="$(printf '%s\n' "$SCRIPT_TXT" | ssh_cluster "cat > '$RUN_DIR/job.sh' && chmod +x '$RUN_DIR/job.sh'" 2>&1)" || { scan_quota "$out"; die "failed to write job.sh: $out"; }

# Launch detached AND wait for the pidfile IN THE SAME ssh call. setsid+nohup fully
# decouples the job from this ssh channel (verified live 2026-07-10: it survives the
# channel closing; stdout/stderr/stdin all redirected so nothing ties it to the
# session). The job writes its OWN pidfile as its first action — more robust than
# trusting `$!` across an ssh round-trip. The poll loop MUST run remotely, not as
# repeated separate ssh_cluster calls: this box's ssh mux layer reliably fails to
# open a session channel over the master (every call eats a "Connection reset by
# peer" before falling back to a fresh direct connection, ~1-3s tax EACH TIME) —
# polling via N separate calls cost 15-30s and raced against short jobs finishing
# (and removing their own pidfile) before the last retry ever landed. One call that
# loops server-side pays the reconnect tax exactly once.
PID="$(ssh_cluster "cd '$REMOTE_PATH' && setsid nohup '$RUN_DIR/job.sh' > '$RUN_DIR/run.log' 2>&1 < /dev/null & disown
for i in 1 2 3 4 5 6 7 8 9 10; do [ -f '$RUN_DIR/run.pid' ] && break; sleep 1; done
cat '$RUN_DIR/run.pid' 2>/dev/null")"
[ -n "$PID" ] || die "job.sh did not register a pid within 10s — check $RUN_DIR/run.log"
log "launched $RUN_NAME on vast -> pid $PID (log: $RUN_DIR/run.log)"
echo "JOB_ID: $RUN_NAME"
