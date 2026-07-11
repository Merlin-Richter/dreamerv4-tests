#!/usr/bin/env bash
# Rendered by vast_run.sh from scripts/vast_job_template.sh. Placeholder tokens are
# substituted at launch time. Written to $RUN_DIR/job.sh on the vast box and started
# detached (setsid+nohup) — there is no scheduler here, this file IS the "job".
#
# Mirrors job_template.sbatch's venv-by-requirements-hash convention exactly, so the
# vast box gets byte-identical dependency versions to ferranti/galvani. Differences
# from the sbatch template: no #SBATCH directives (nothing to submit to); self-
# registers its own PID (the process-tracking analogue of a SLURM job id); the run
# command is NOT wrapped in `set -e` so a failure still reaches the done/rc trailer
# and cleans up the pidfile (vast_wait.sh / vast_status.sh key off both).
set -euo pipefail

echo $$ > "@RUN_DIR@/run.pid"
trap 'rm -f "@RUN_DIR@/run.pid"' EXIT   # a set-e death before the trailer (e.g. venv build) must still free the box
echo "=== job @RUN_NAME@ on $(hostname) | pid $$ | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "@REMOTE_PATH@"
echo "=== commit: $(git rev-parse HEAD) ==="

# --- venv keyed on sha256(requirements.txt): build once, reuse thereafter (same as ferranti/galvani) ---
# Reuse keys on the .ready marker, NOT on bin/python: `python3 -m venv` succeeding says nothing
# about `pip install -r` having finished, and an interrupted install used to poison the cache
# forever (live failure 2026-07-11: torch wheel download broke mid-transfer, next run "reused"
# a venv with zero packages). pip re-install into a half-built venv is idempotent, so a
# markerless dir is simply resumed, and flaky-network installs get 3 attempts.
REQ_HASH="$(sha256sum requirements.txt | cut -c1-16)"
VENV="@VENV_ROOT@/venv-${REQ_HASH}"
if [ ! -f "${VENV}/.ready" ]; then
  echo "=== building venv ${VENV} (requirements hash ${REQ_HASH}) ==="
  [ -x "${VENV}/bin/python" ] || python3 -m venv "${VENV}"
  "${VENV}/bin/pip" install --upgrade pip
  ok=0
  for attempt in 1 2 3; do
    if "${VENV}/bin/pip" install -r requirements.txt; then ok=1; break; fi
    echo "=== pip install failed (attempt ${attempt}/3) — retrying in 15s ==="
    sleep 15
  done
  [ "$ok" = 1 ] || { echo "=== venv build FAILED after 3 attempts ==="; exit 1; }
  touch "${VENV}/.ready"
else
  echo "=== reusing cached venv ${VENV} ==="
fi
export PATH="${VENV}/bin:${PATH}"

# --- W&B auth (from cluster.env; falls back to remote ~/.netrc if blank) ---
@WANDB_EXPORT@

echo "=== running: @CMD@ ==="
set +e
@CMD@
RC=$?
set -e
echo "=== done @RUN_NAME@ rc=$RC $(date) ==="
rm -f "@RUN_DIR@/run.pid"
exit "$RC"
