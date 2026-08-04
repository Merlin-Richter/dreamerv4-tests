#!/usr/bin/env bash
# Build the pinned community checkout with the accepted baseline patch plus memory.
set -euo pipefail

ROOT="$(pwd)"
EXP="$ROOT/experiments/dreamer4-community-mem2mem"
BASE_EXP="$ROOT/experiments/dreamer4-community-baseline"
BASE="$ROOT/runs/dreamer4-community-mem2mem"
UPSTREAM_SHA="b8abafbf4da72c59b6aa09f8499ccde0d6a37fd6"
BASE_PATCH="$BASE_EXP/upstream-memmaze.patch"
MEMORY_PATCH="$EXP/upstream-memory.patch"
REQ="$BASE_EXP/requirements-community.txt"
PATCH_ID="$(sha256sum "$BASE_PATCH" "$MEMORY_PATCH" | sha256sum | cut -c1-12)"
REQ_ID="$(sha256sum "$REQ" | cut -c1-12)"
D4_ROOT="$BASE/upstream-${UPSTREAM_SHA:0:8}-${PATCH_ID}"
D4_VENV="$ROOT/runs/dreamer4-community-baseline/envs/venv-${REQ_ID}"
PROV="$BASE/provenance"
mkdir -p "$BASE" "$PROV"

if [ ! -d "$D4_ROOT/.git" ]; then
  test ! -e "$D4_ROOT" || { echo "Refusing non-git path $D4_ROOT" >&2; exit 2; }
  git clone https://github.com/nicklashansen/dreamer4.git "$D4_ROOT"
  git -C "$D4_ROOT" checkout --detach "$UPSTREAM_SHA"
  git -C "$D4_ROOT" apply --unidiff-zero --check "$BASE_PATCH"
  git -C "$D4_ROOT" apply --unidiff-zero "$BASE_PATCH"
  git -C "$D4_ROOT" apply --unidiff-zero --check "$MEMORY_PATCH"
  git -C "$D4_ROOT" apply --unidiff-zero "$MEMORY_PATCH"
fi

test "$(git -C "$D4_ROOT" rev-parse HEAD)" = "$UPSTREAM_SHA"
git -C "$D4_ROOT" diff --check
git -C "$D4_ROOT" diff > "$PROV/upstream-baseline-plus-memory.patch"
sha256sum "$BASE_PATCH" "$MEMORY_PATCH" "$REQ" > "$PROV/integration-input-sha256.txt"
git -C "$D4_ROOT" rev-parse HEAD > "$PROV/upstream-head.txt"

if [ ! -x "$D4_VENV/bin/python" ]; then
  echo "Accepted baseline environment is missing: $D4_VENV" >&2
  echo "Run $BASE_EXP/setup_upstream.sh once to build it." >&2
  exit 2
fi
D4_PYTHON="$D4_VENV/bin/python"

"$D4_PYTHON" -u "$BASE_EXP/validate_integration.py" --dreamer4 "$D4_ROOT"
"$D4_PYTHON" -u "$EXP/validate_model.py" --dreamer4 "$D4_ROOT"

{
  printf 'D4_ROOT=%q\n' "$D4_ROOT"
  printf 'D4_PYTHON=%q\n' "$D4_PYTHON"
  printf 'D4_PROVENANCE=%q\n' "$PROV"
} > "$BASE/current.env"

echo "MEM2MEM SETUP PASSED"
echo "D4_ROOT=$D4_ROOT"
echo "D4_PYTHON=$D4_PYTHON"
