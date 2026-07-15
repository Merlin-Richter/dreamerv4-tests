#!/usr/bin/env bash
set -uo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
source /venv/main/bin/activate

cd /workspace/dreamerv4-tests
echo "COLORFIELD_DYNAMICS_V3_START commit=$(git rev-parse HEAD) utc=$(date -u +%FT%TZ)"
bash experiments/colorfield-dynamics-v3/run_sequence.sh
rc=$?
echo "COLORFIELD_DYNAMICS_V3_DONE rc=$rc utc=$(date -u +%FT%TZ)"
exit "$rc"
