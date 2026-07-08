#!/usr/bin/env bash
# Autoresearch loop — local runner (run inside WSL; NOT agent-editable).
# The loop agent's single "run the experiment" command. Assumes the agent has already
# COMMITTED its editable/ change on the current autoresearch/<tag> branch.
# Does: push -> sync_code -> submit -> wait -> fetch log (which ends in the summary
# block + score lines). Everything goes to stdout; the agent redirects to run.log.
set -euo pipefail
cd "$(dirname "$0")/../.."

BRANCH=$(git rev-parse --abbrev-ref HEAD)
SHA=$(git rev-parse --short=7 HEAD)
case "$BRANCH" in autoresearch/*|exp/*) :;; *) echo "WARNING: unusual branch '$BRANCH' for a loop run";; esac
[ -z "$(git status --porcelain)" ] || { echo "ERROR: DIRTY_TREE — commit your change first"; exit 2; }

RUN="loop-$SHA"
echo "branch: $BRANCH  sha: $SHA  run: $RUN"
git push -q -u origin "$BRANCH"
bash scripts/sync_code.sh --cluster ferranti "$BRANCH" "$(git rev-parse HEAD)"
JOB=$(bash scripts/submit_job.sh --cluster ferranti --name "$RUN" --hours 1 --cpus 16 \
        -- bash autoresearch/loop/job_payload.sh "$RUN" | sed -n 's/^JOB_ID: //p')
[ -n "$JOB" ] || { echo "ERROR: submit failed (no JOB_ID)"; exit 6; }
echo "job: $JOB"
bash scripts/wait_for_jobs.sh --cluster ferranti "$JOB" --poll 60 || echo "wait rc=$? (job failed or early-exit — log follows)"
bash scripts/fetch_logs.sh --cluster ferranti "$JOB" --tail 500
