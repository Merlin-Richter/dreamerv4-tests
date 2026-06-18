#!/usr/bin/env bash
# scripts/submit_job.sh --cluster {ferranti|galvani} --name RUN [--gpus N] [--hours H] -- <command...>
#
# Renders job_template.sbatch with the cluster's SLURM settings + your command, pipes it
# to `sbatch` over the master socket, and echoes the new job id as "JOB_ID: <n>".
# The command is the EXACT training invocation, e.g.:
#   submit_job.sh --cluster ferranti --name gw-tok-s0 --hours 6 -- \
#       python -u src/training/train_tokenizer.py --wandb --epochs 10 --batch-size 16
# Record JOB_ID + the resolved SHA (from sync_code.sh) in EXPERIMENTS.md immediately.
# Provenance: commit a run spec per experiment (experiments/EXP-NNN/run.sh) holding this line.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_common.sh"

# custom parse: pull --cluster/--name/--gpus/--hours, then everything after `--` is CMD
RUN_NAME=""; GPUS=""; HOURS=""; DRYRUN=0; CMD_ARGS=()
ARGS=("$@")
i=0
while [ $i -lt ${#ARGS[@]} ]; do
  case "${ARGS[$i]}" in
    --cluster) CLUSTER="${ARGS[$((i+1))]:-}"; i=$((i+2));;
    --cluster=*) CLUSTER="${ARGS[$i]#*=}"; i=$((i+1));;
    --name) RUN_NAME="${ARGS[$((i+1))]:-}"; i=$((i+2));;
    --gpus) GPUS="${ARGS[$((i+1))]:-}"; i=$((i+2));;
    --hours) HOURS="${ARGS[$((i+1))]:-}"; i=$((i+2));;
    --dry-run) DRYRUN=1; i=$((i+1));;
    --) i=$((i+1)); while [ $i -lt ${#ARGS[@]} ]; do CMD_ARGS+=("${ARGS[$i]}"); i=$((i+1)); done;;
    *) die_config "unexpected arg '${ARGS[$i]}' (command must come after --)";;
  esac
done
case "${CLUSTER:-}" in ferranti|galvani) :;; "") die_config "--cluster required";; *) die_config "unknown cluster '$CLUSTER'";; esac
[ -n "$RUN_NAME" ] || die_config "--name RUN is required"
[ "${#CMD_ARGS[@]}" -gt 0 ] || die_config "no command given (put it after --)"
echo "$RUN_NAME" | grep -qE '^[A-Za-z0-9._-]+$' || die_config "run name must be [A-Za-z0-9._-]"

load_config

CMD="${CMD_ARGS[*]}"
HOURS="${HOURS:-$DEFAULT_HOURS}"; HOURS="${HOURS:-8}"
TIME="$(printf '%02d:00:00' "$HOURS")"
# gres count override
if [ -n "$GPUS" ]; then GRES="$(echo "$GRES" | sed -E "s/:[0-9]+$//; s/$/:$GPUS/")"; fi

# optional SLURM directives only if configured
PART_LINE=""; [ -n "$PARTITION" ] && PART_LINE="#SBATCH --partition=$PARTITION"
ACCT_LINE=""; [ -n "$ACCOUNT" ] && ACCT_LINE="#SBATCH --account=$ACCOUNT"
CONS_LINE=""; [ -n "$CONSTRAINT" ] && CONS_LINE="#SBATCH --constraint=$CONSTRAINT"
# module loads
MOD_LINE="# (no modules configured)"
[ -n "$MODULES" ] && MOD_LINE="$(for m in $MODULES; do echo "module load $m"; done)"
# wandb export
WB_LINE="# (WandB auth via remote ~/.netrc if present)"
[ -n "${WANDB_API_KEY:-}" ] && WB_LINE="export WANDB_API_KEY='${WANDB_API_KEY}'"
[ -n "${WANDB_ENTITY:-}" ] && WB_LINE="$WB_LINE
export WANDB_ENTITY='${WANDB_ENTITY}'"

# Render via bash pattern substitution. NOTE: bash 5.2 (like sed/awk) treats `&` in the
# replacement as the matched text, so every value is run through _esc (escapes \ then &)
# to keep replacements literal — commands/paths with &, |, \, / are then safe.
_esc() { local s="$1"; s="${s//\\/\\\\}"; s="${s//&/\\&}"; printf '%s' "$s"; }
render() {
  local t; t="$(cat "$SCRIPT_DIR/job_template.sbatch")"
  t="${t//@RUN_NAME@/$(_esc "$RUN_NAME")}"
  t="${t//@RUNS_DIR@/$(_esc "$RUNS_DIR")}"
  t="${t//@REMOTE_PATH@/$(_esc "$REMOTE_PATH")}"
  t="${t//@VENV_ROOT@/$(_esc "$VENV_ROOT")}"
  t="${t//@GRES@/$(_esc "$GRES")}"
  t="${t//@TIME@/$(_esc "$TIME")}"
  t="${t//@SBATCH_PARTITION@/$(_esc "$PART_LINE")}"
  t="${t//@SBATCH_ACCOUNT@/$(_esc "$ACCT_LINE")}"
  t="${t//@SBATCH_CONSTRAINT@/$(_esc "$CONS_LINE")}"
  t="${t//@MODULE_LOADS@/$(_esc "$MOD_LINE")}"
  t="${t//@WANDB_EXPORT@/$(_esc "$WB_LINE")}"
  t="${t//@CMD@/$(_esc "$CMD")}"
  printf '%s\n' "$t"
}

SCRIPT_TXT="$(render)"
if [ "$DRYRUN" -eq 1 ]; then
  echo "=== DRY RUN: rendered sbatch for $RUN_NAME on $CLUSTER (not submitted) ===" >&2
  printf '%s\n' "$SCRIPT_TXT"
  exit 0
fi
require_master
# ensure the run dir exists for slurm logs, then submit (script via stdin)
ssh_cluster "mkdir -p '$RUNS_DIR/$RUN_NAME'" >/dev/null
out="$(printf '%s\n' "$SCRIPT_TXT" | ssh_cluster "sbatch --parsable" 2>&1)" || { scan_quota "$out"; die "sbatch failed: $out"; }
scan_quota "$out"
JOB_ID="$(echo "$out" | grep -oE '^[0-9]+' | head -1)"
[ -n "$JOB_ID" ] || die "sbatch did not return a job id: $out"
log "submitted $RUN_NAME to $CLUSTER -> job $JOB_ID (logs: $RUNS_DIR/$RUN_NAME/slurm-$JOB_ID.out)"
echo "JOB_ID: $JOB_ID"
