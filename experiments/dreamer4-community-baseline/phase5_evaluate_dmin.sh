#!/usr/bin/env bash
# Ferranti Phase 5 scoring: the d_min-only arm vs the vanilla control, on the same held-out split.
#
# Three checkpoints x three sampling schedules, all through one instrument with one seed:
#
#   dmin_final          runs/memmaze-d4-dynamics-dmin-24h/dynamics/final.pt   (24 active H100 h)
#   control_final       runs/memmaze-d4-dynamics-48h-v3/dynamics/final.pt     (48 active H100 h)
#   control_stepmatched the control periodic checkpoint nearest the arm final step
#
#   K=1  (--schedule shortcut -d 1.0) the setting a d_min-only x-prediction arm should be READ at.
#                                     With K=1 the sampler degenerates to z = x1_hat: one forward,
#                                     no re-noising.  Every K>1 step re-noises the model's OWN
#                                     prediction to an intermediate tau, an input never produced in
#                                     training, and that is the DOMINANT penalty -- ColorField
#                                     measured a d_min-only model at K=1 0.978 / K=2 0.680 /
#                                     K=4 0.615 / K=8 0.532.  In-distribution for the control too:
#                                     its self rows sample step_idx uniformly over {0,1,2}.
#   K=8  (--schedule finest)          the step size the arm actually trained (d_min = 1/k_max), and
#                                     in-distribution for the control, which trains d_min on 75% of
#                                     its rows.  Fair, but pays the re-noising cost above.
#   K=4  (--schedule shortcut -d .25) the historic protocol, kept for continuity.  Worst case for
#                                     the arm: re-noised AND reading step_embed row 2, which
#                                     nn.Embedding(log2(k_max)+1, d_model) leaves at random init
#                                     because d_min-only training never gives it a gradient.
#
# Report all three.  A d_min-only arm that wins at K=1 and loses at K=4 has not "broken shortcut
# sampling" -- it declined to learn it, which is what dropping the bootstrap term buys and costs.
#
# control_stepmatched exists because the arm gets 24 h against the control 48 h.  Compute-matched
# and step-matched disagreeing is itself the finding; scoring both means the 2x budget gap cannot
# silently decide the result.
set -euo pipefail

ROOT="$(pwd)"
EXP="$ROOT/experiments/dreamer4-community-baseline"
BASE="$ROOT/runs/dreamer4-community-baseline"
EVAL_OUT="$ROOT/data/d4_memmaze_community/eval-v2"
TOK_CKPT="$ROOT/runs/memmaze-d4-tokenizer-24h/tokenizer/final.pt"
TOK_EXPECTED_SHA256="347052fae0212ea2c6b943ae7c28a886298ce551d4155b882084d63a3ea48797"
DMIN_DIR="$ROOT/runs/${D4_DMIN_RUN:-memmaze-d4-dynamics-dmin-24h}/dynamics"
CTRL_DIR="$ROOT/runs/${D4_CTRL_RUN:-memmaze-d4-dynamics-48h-v3}/dynamics"
CTRL_EXPECTED_SHA256="7b077938fec776c74e62201ab79194a7a06e10e54856c69d47b65dda6367d674"
N_SEQ="${D4_EVAL_SEQS:-64}"
CTX="${D4_EVAL_CTX:-8}"
HORIZON="${D4_EVAL_HORIZON:-16}"
SEED="${D4_EVAL_SEED:-20260729}"
RUN_NAME="${SLURM_JOB_NAME:-memmaze-d4-dmin-eval}"
RUN_DIR="$ROOT/runs/$RUN_NAME"
mkdir -p "$RUN_DIR"

bash "$EXP/setup_upstream.sh"
# shellcheck disable=SC1091
source "$BASE/current.env"
cp -a "$D4_PROVENANCE" "$RUN_DIR/provenance"

test -f "$EVAL_OUT/conversion_manifest.json"
test -f "$TOK_CKPT"
test -f "$DMIN_DIR/final.pt"
test -f "$CTRL_DIR/final.pt"

TOK_ACTUAL_SHA256="$(sha256sum "$TOK_CKPT" | cut -d ' ' -f 1)"
test "$TOK_ACTUAL_SHA256" = "$TOK_EXPECTED_SHA256" \
  || { echo "tokenizer hash mismatch: $TOK_ACTUAL_SHA256" >&2; exit 1; }
CTRL_ACTUAL_SHA256="$(sha256sum "$CTRL_DIR/final.pt" | cut -d ' ' -f 1)"
test "$CTRL_ACTUAL_SHA256" = "$CTRL_EXPECTED_SHA256" \
  || { echo "control checkpoint hash mismatch: $CTRL_ACTUAL_SHA256 (baseline must be immutable)" >&2; exit 1; }

# Arm identity: the checkpoint itself must record self_fraction 0, and the control must not.
"$D4_PYTHON" - "$DMIN_DIR/final.pt" "$CTRL_DIR/final.pt" <<'PY' | tee "$RUN_DIR/arm_identity.txt"
import sys, torch
dmin = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
ctrl = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
d_args, c_args = dict(dmin["args"]), dict(ctrl["args"])
d_sf = float(d_args.get("self_fraction", 0.25))
c_sf = float(c_args.get("self_fraction", 0.25))
print(f"dmin_step={int(dmin['step'])} dmin_self_fraction={d_sf}")
print(f"ctrl_step={int(ctrl['step'])} ctrl_self_fraction={c_sf}")
assert d_sf == 0.0, f"d_min arm checkpoint has self_fraction={d_sf}, not 0.0"
assert c_sf == 0.25, f"control checkpoint has self_fraction={c_sf}, not 0.25"
for k in ("d_model_dyn", "dyn_depth", "n_heads", "k_max", "seq_len", "packing_factor",
          "n_register", "n_agent", "time_every", "space_mode", "lr", "weight_decay",
          "grad_clip", "batch_size", "seed", "use_actions"):
    dv, cv = d_args.get(k), c_args.get(k)
    print(f"{'OK ' if dv == cv else 'DIFF'} {k}: dmin={dv} ctrl={cv}")
    assert dv == cv, f"{k} differs between arms: {dv} vs {cv}; no longer a 1-variable A/B"
print("ARM IDENTITY OK: the only training difference is self_fraction")
PY

# The control checkpoint whose step count is closest to the arm final step.
CTRL_MATCHED="$("$D4_PYTHON" - "$DMIN_DIR/final.pt" "$CTRL_DIR" <<'PY'
import sys, re, pathlib, torch
target = int(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["step"])
cands = []
for p in pathlib.Path(sys.argv[2]).glob("step_*.pt"):
    m = re.fullmatch(r"step_(\d+)\.pt", p.name)
    if m:
        cands.append((abs(int(m.group(1)) - target), int(m.group(1)), str(p)))
print(sorted(cands)[0][2] if cands else "", end="")
PY
)"
if [ -n "$CTRL_MATCHED" ]; then
  echo "step-matched control checkpoint: $CTRL_MATCHED"
else
  echo "WARNING: no periodic control checkpoints under $CTRL_DIR; step-matched read-off skipped" >&2
fi

SCRATCH_BASE="${SLURM_TMPDIR:-${TMPDIR:-}}"
test -n "$SCRATCH_BASE" || { echo "No node-local SLURM_TMPDIR/TMPDIR available" >&2; exit 1; }
EVAL_RUNTIME="$SCRATCH_BASE/d4_memmaze_community_eval"
mkdir -p "$EVAL_RUNTIME"
cp -a "$EVAL_OUT/shards" "$EVAL_RUNTIME/"
cp -a "$EVAL_OUT/demos" "$EVAL_RUNTIME/"

# score <tag> <checkpoint> <n-sequences> <schedule-args...>
score() {
  local tag="$1"; shift
  local ckpt="$1"; shift
  local nseq="$1"; shift
  local out="$RUN_DIR/$tag"
  mkdir -p "$out"
  echo "=== scoring $tag ($ckpt) nseq=$nseq $* ==="
  "$D4_PYTHON" -u "$EXP/evaluate_dynamics.py" \
    --dreamer4 "$D4_ROOT" \
    --dynamics-checkpoint "$ckpt" \
    --tokenizer-checkpoint "$TOK_CKPT" \
    --data-dir "$EVAL_RUNTIME/demos" \
    --frames-dir "$EVAL_RUNTIME/shards" \
    --out-dir "$out" \
    --n-sequences "$nseq" --ctx "$CTX" --horizon "$HORIZON" --seed "$SEED" \
    --device cuda "$@" 2>&1 | tee "$out/eval.log"
}

score dmin_final_K1    "$DMIN_DIR/final.pt" "$N_SEQ" --schedule shortcut --eval-d 1.0
score control_final_K1 "$CTRL_DIR/final.pt" "$N_SEQ" --schedule shortcut --eval-d 1.0
score dmin_final_K8    "$DMIN_DIR/final.pt" "$N_SEQ" --schedule finest
score control_final_K8 "$CTRL_DIR/final.pt" "$N_SEQ" --schedule finest
score dmin_final_K4    "$DMIN_DIR/final.pt" "$N_SEQ" --schedule shortcut --eval-d 0.25
score control_final_K4 "$CTRL_DIR/final.pt" "$N_SEQ" --schedule shortcut --eval-d 0.25
if [ -n "$CTRL_MATCHED" ]; then
  score control_stepmatched_K1 "$CTRL_MATCHED" "$N_SEQ" --schedule shortcut --eval-d 1.0
  score control_stepmatched_K8 "$CTRL_MATCHED" "$N_SEQ" --schedule finest
  score control_stepmatched_K4 "$CTRL_MATCHED" "$N_SEQ" --schedule shortcut --eval-d 0.25
fi

# Reproduction check against the recorded 2026-08-02 result (4 sequences, K=4, same seed):
# mse_correct_actions 0.007988 / PSNR 20.98 dB.  A mismatch means the instrument drifted and
# none of the numbers above are comparable to the historic record.
score control_historic_4seq_K4 "$CTRL_DIR/final.pt" 4 --schedule shortcut --eval-d 0.25 \
  || echo "WARNING: historic reproduction eval failed" >&2

"$D4_PYTHON" - "$RUN_DIR" <<'PY' | tee "$RUN_DIR/summary.txt"
import json, pathlib, sys
run = pathlib.Path(sys.argv[1])
rows = []
for d in sorted(run.iterdir()):
    f = d / "heldout_rollout_metrics.json"
    if d.is_dir() and f.is_file():
        rows.append((d.name, json.loads(f.read_text())))
hdr = (f"{'arm':28s} {'step':>8s} {'K':>3s} {'seqs':>5s} {'mse':>10s} "
       f"{'psnr_dB':>8s} {'/wrong':>7s} {'/copy':>7s}")
print(hdr); print("-" * len(hdr))
for name, m in rows:
    print(f"{name:28s} {m['checkpoint_step']:8d} {m['shortcut_steps']:3d} "
          f"{m.get('n_sequences', len(m['dataset_indices'])):5d} "
          f"{m['mse_correct_actions']:10.6f} {m['psnr_correct_actions_db']:8.2f} "
          f"{m['correct_over_wrong_mse']:7.3f} {m['correct_over_copy_mse']:7.3f}")
PY

sha256sum "$TOK_CKPT" "$CTRL_DIR/final.pt" "$DMIN_DIR/final.pt" > "$RUN_DIR/checkpoint_sha256.txt"
echo "PHASE 5 DMIN EVALUATION PASSED"
