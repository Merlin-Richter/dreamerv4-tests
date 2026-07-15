#!/usr/bin/env bash
set -o pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
VAST_VENV="${VAST_VENV:-/workspace/venvs/venv-b05c6eb3f672f99e}"
source "$VAST_VENV/bin/activate"

cd /workspace/dreamerv4-tests
echo "COLORFIELD_DYNAMICS_V3_START commit=$(git rev-parse HEAD) utc=$(date -u +%FT%TZ)"
bash experiments/colorfield-dynamics-v3/run_sequence.sh
rc=$?
echo "COLORFIELD_DYNAMICS_V3_DONE rc=$rc utc=$(date -u +%FT%TZ)"
exit "$rc"
