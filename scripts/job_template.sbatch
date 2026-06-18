#!/usr/bin/env bash
#SBATCH --job-name=@RUN_NAME@
#SBATCH --output=@RUNS_DIR@/@RUN_NAME@/slurm-%j.out
#SBATCH --error=@RUNS_DIR@/@RUN_NAME@/slurm-%j.out
#SBATCH --gres=@GRES@
#SBATCH --time=@TIME@
@SBATCH_PARTITION@
@SBATCH_ACCOUNT@
@SBATCH_CONSTRAINT@
#
# Rendered by submit_job.sh from scripts/job_template.sbatch. Placeholder tokens are
# substituted at submit time; lines that resolve empty (partition/account/constraint)
# are omitted. Piped to `sbatch` on the login node over the ControlMaster socket.
set -euo pipefail

echo "=== job @RUN_NAME@ on $(hostname) | jobid ${SLURM_JOB_ID:-?} | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "@REMOTE_PATH@"
echo "=== commit: $(git rev-parse HEAD) ==="

# --- optional module loads (blank = pure venv) ---
@MODULE_LOADS@

# --- venv keyed on sha256(requirements.txt): build once, reuse thereafter ---
REQ_HASH="$(sha256sum requirements.txt | cut -c1-16)"
VENV="@VENV_ROOT@/venv-${REQ_HASH}"
if [ ! -x "${VENV}/bin/python" ]; then
  echo "=== building venv ${VENV} (requirements hash ${REQ_HASH}) ==="
  python3 -m venv "${VENV}"
  "${VENV}/bin/pip" install --upgrade pip
  "${VENV}/bin/pip" install -r requirements.txt
else
  echo "=== reusing cached venv ${VENV} ==="
fi
export PATH="${VENV}/bin:${PATH}"

# --- W&B auth (from cluster.env; falls back to remote ~/.netrc if blank) ---
@WANDB_EXPORT@

echo "=== running: @CMD@ ==="
@CMD@
echo "=== done @RUN_NAME@ rc=$? $(date) ==="
