#!/usr/bin/env bash
# Build a fresh, pinned, patched community Dreamer 4 checkout and its isolated environment.
set -euo pipefail

ROOT="$(pwd)"
EXP="$ROOT/experiments/dreamer4-community-baseline"
BASE="$ROOT/runs/dreamer4-community-baseline"
UPSTREAM_SHA="b8abafbf4da72c59b6aa09f8499ccde0d6a37fd6"
PATCH="$EXP/upstream-memmaze.patch"
REQ="$EXP/requirements-community.txt"
PATCH_ID="$(sha256sum "$PATCH" | cut -c1-12)"
REQ_ID="$(sha256sum "$REQ" | cut -c1-12)"
D4_ROOT="$BASE/upstream-${UPSTREAM_SHA:0:8}-${PATCH_ID}"
D4_VENV="$BASE/envs/venv-${REQ_ID}"
PROV="$BASE/provenance"

mkdir -p "$BASE" "$BASE/envs" "$PROV"

if [ ! -d "$D4_ROOT/.git" ]; then
  test ! -e "$D4_ROOT" || { echo "Refusing non-git path $D4_ROOT" >&2; exit 2; }
  git clone https://github.com/nicklashansen/dreamer4.git "$D4_ROOT"
  git -C "$D4_ROOT" checkout --detach "$UPSTREAM_SHA"
  git -C "$D4_ROOT" apply --unidiff-zero --check "$PATCH"
  git -C "$D4_ROOT" apply --unidiff-zero "$PATCH"
fi

test "$(git -C "$D4_ROOT" rev-parse HEAD)" = "$UPSTREAM_SHA"
git -C "$D4_ROOT" diff --check
git -C "$D4_ROOT" diff > "$PROV/upstream-adaptations.patch"
git -C "$D4_ROOT" apply --unidiff-zero --reverse --check "$PATCH"

if [ ! -f "$D4_VENV/.ready" ]; then
  BOOTSTRAP_PY=""
  for candidate in python3.10 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      minor="$($candidate -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || true)"
      if [ "$minor" = 10 ] || [ "$minor" = 11 ]; then
        BOOTSTRAP_PY="$candidate"
        break
      fi
    fi
  done
  test -n "$BOOTSTRAP_PY" || { echo "Need Python 3.10 or 3.11 for pinned community dependencies" >&2; exit 2; }
  "$BOOTSTRAP_PY" -m venv "$D4_VENV"
  "$D4_VENV/bin/pip" install --upgrade pip
  for attempt in 1 2 3; do
    "$D4_VENV/bin/pip" install -r "$REQ" && break
    test "$attempt" -lt 3 || exit 1
    sleep 5
  done
  touch "$D4_VENV/.ready"
fi

D4_PYTHON="$D4_VENV/bin/python"
"$D4_PYTHON" -u "$EXP/validate_integration.py" --dreamer4 "$D4_ROOT"

git -C "$D4_ROOT" rev-parse HEAD > "$PROV/upstream-head.txt"
sha256sum "$PATCH" "$REQ" > "$PROV/integration-input-sha256.txt"
{
  echo "project_commit=$(git -C "$ROOT" rev-parse HEAD)"
  echo "upstream_commit=$UPSTREAM_SHA"
  echo "patch_id=$PATCH_ID"
  echo "requirements_id=$REQ_ID"
  echo "python=$($D4_PYTHON --version 2>&1)"
  "$D4_PYTHON" -c 'import torch, torchvision; print(f"torch={torch.__version__} torchvision={torchvision.__version__} cuda={torch.version.cuda}")'
} > "$PROV/environment.txt"

{
  printf 'D4_ROOT=%q\n' "$D4_ROOT"
  printf 'D4_PYTHON=%q\n' "$D4_PYTHON"
  printf 'D4_PROVENANCE=%q\n' "$PROV"
} > "$BASE/current.env"

echo "SETUP PASSED"
echo "D4_ROOT=$D4_ROOT"
echo "D4_PYTHON=$D4_PYTHON"
