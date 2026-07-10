#!/usr/bin/env bash
# Autoresearch loop — local runner (run inside WSL; NOT agent-editable).
# The loop agent's single "run the experiment" command. Assumes the agent has already
# COMMITTED its editable/ change on the current autoresearch/<tag> branch.
# Does: push -> sync_code -> submit -> wait -> fetch log (which ends in the summary
# block + score lines). Everything goes to stdout; the agent redirects to run.log.
#
# Backend: --cluster {ferranti|vast} (default ferranti). ferranti = SLURM verbs
# (submit_job/wait_for_jobs/fetch_logs); vast = the schedulerless verbs
# (vast_run/vast_wait/vast_status) — same payload, same summary block. The 600s
# budget is wall-clock on whatever GPU runs it, so scores are comparable only
# WITHIN a backend (pace-sized sched keeps the LR horizon right on both).
set -euo pipefail
cd "$(dirname "$0")/../.."

CLUSTER=ferranti
while [ $# -gt 0 ]; do
  case "$1" in
    --cluster) CLUSTER="${2:?--cluster needs a value}"; shift 2;;
    --cluster=*) CLUSTER="${1#*=}"; shift;;
    *) echo "ERROR: BAD_CONFIG — unexpected arg '$1' (only --cluster {ferranti|vast})"; exit 2;;
  esac
done
case "$CLUSTER" in ferranti|vast) :;; *) echo "ERROR: BAD_CONFIG — unsupported cluster '$CLUSTER'"; exit 2;; esac

BRANCH=$(git rev-parse --abbrev-ref HEAD)
SHA=$(git rev-parse --short=7 HEAD)
case "$BRANCH" in autoresearch/*|exp/*) :;; *) echo "WARNING: unusual branch '$BRANCH' for a loop run";; esac
[ -z "$(git status --porcelain)" ] || { echo "ERROR: DIRTY_TREE — commit your change first"; exit 2; }

# Integrity preflight: frozen_sym/ + loop/ must match MANIFEST-sym (code files
# only here — the job payload re-checks INCLUDING the dataset sidecars). A
# mismatch means the tree moved the goalposts: the run is refused, not scored.
PY=$(command -v python3 || command -v python)
"$PY" -m autoresearch.driver.manifest --check --tier sym --no-artifacts \
  || { echo "ERROR: TAMPERED — frozen_sym/ or loop/ differs from MANIFEST-sym; only editable/ may change"; exit 8; }

RUN="loop-$SHA"
echo "branch: $BRANCH  sha: $SHA  run: $RUN  cluster: $CLUSTER"
# NO git push here: WSL git has no credential helper and hangs silently on private
# remotes — the AGENT pushes from its own (Windows) shell after committing. If the
# commit was not pushed, sync_code fails loudly with ERROR: BAD_REF below.
bash scripts/sync_code.sh --cluster "$CLUSTER" "$BRANCH" "$(git rev-parse HEAD)"

if [ "$CLUSTER" = vast ]; then
  JOB=$(bash scripts/vast_run.sh --name "$RUN" \
          -- bash autoresearch/loop/job_payload.sh "$RUN" | sed -n 's/^JOB_ID: //p')
  [ -n "$JOB" ] || { echo "ERROR: submit failed (no JOB_ID)"; exit 6; }
  echo "job: $JOB"
  bash scripts/vast_wait.sh "$JOB" --poll 60 || echo "wait rc=$? (job failed or early-exit — log follows)"
  bash scripts/vast_status.sh "$JOB" --tail 500
else
  JOB=$(bash scripts/submit_job.sh --cluster "$CLUSTER" --name "$RUN" --hours 1 --cpus 16 \
          -- bash autoresearch/loop/job_payload.sh "$RUN" | sed -n 's/^JOB_ID: //p')
  [ -n "$JOB" ] || { echo "ERROR: submit failed (no JOB_ID)"; exit 6; }
  echo "job: $JOB"
  bash scripts/wait_for_jobs.sh --cluster "$CLUSTER" "$JOB" --poll 60 || echo "wait rc=$? (job failed or early-exit — log follows)"
  bash scripts/fetch_logs.sh --cluster "$CLUSTER" "$JOB" --tail 500
fi
