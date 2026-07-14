# Train an independent Dreamer 4 baseline on Memory Maze

## Goal
Establish a credible **vanilla** Dreamer 4 baseline for Memory Maze using the independent community
implementation at [nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4).

Our in-repository vanilla model failed to learn reliable in-context or action-specific behavior (even
with the tau0 addition), so it cannot serve as a fair baseline for judging mem2mem / archive models.
This task tests whether a separate, normal Dreamer 4 implementation *can* learn Memory Maze given
substantial compute. **Do not** claim "our vanilla is buggy" vs "mem2mem is just more compute-efficient"
from training loss alone — this baseline can establish an independent reference or falsify "our vanilla
run was representative", but it does not by itself distinguish an implementation bug from a
compute-efficiency advantage without matched evidence.

---

## STATUS (2026-07-14): cluster-free preparation COMPLETE. Blocked on cluster availability only.
All non-cluster work is done and validated. A fresh agent can start at **Phase 0 → Phase 2** below once a
cluster is available. What was done:
- Studied the community repo end-to-end; pinned the upstream commit; catalogued every memmaze-specific
  adaptation needed (below).
- Wrote + **validated** the data converter and a regression test (`experiments/dreamer4-community-baseline/`)
  against the REAL community dataset classes and a 64×64 model forward. No cluster / real data needed.
- Confirmed the Memory-Maze ↔ Dreamer4 action alignment from first principles + the memory-maze README.

### What is NOT done (needs the cluster)
Download the ~100 GB dataset onto the cluster, convert it, run the 24 h tokenizer + 48 h dynamics H100
runs, review gates, evaluate, make it playable. Steps below.

---

## Upstream repo + provenance (RECORD THIS)
- Repo: `https://github.com/nicklashansen/dreamer4`
- **Pinned commit: `b8abafbf4da72c59b6aa09f8499ccde0d6a37fd6`** ("update README", 2026-07-09). Clone fresh
  on the cluster and `git checkout b8abafbf` BEFORE any edits; record the commit + your local diff in the
  provenance notes. Do NOT port the model into our `src/` — keep it a separate checkout (task requires an
  *independent* baseline). This is experimental integration code: edit data plumbing / training controls
  freely, but preserve the substantive Dreamer 4 architecture + shortcut-forcing objective.
- It is an *unofficial* PyTorch Dreamer 4 for **multi-task DMControl (continuous actions)**, loosely based
  on edwhu's JAX impl. Note the authors' newer `nicklashansen/mmbench2` ("strictly better implementation")
  — we deliberately use `dreamer4` per this task; do not swap without sign-off.
- Package layout (`dreamer4/dreamer4/`): `model.py` (Encoder/Decoder/Tokenizer/Dynamics + shortcut loss),
  `train_tokenizer.py`, `train_dynamics.py`, `sharded_frame_dataset.py` (frames), `wm_dataset.py`
  (frames+actions), `preprocess_dataset.py` (their DMControl PNG→shards; we replace it), `task_set.py`
  (`TASK_SET`, 30 DMControl tasks), `interactive.py` (web UI), `tasks.json` (per-task metadata; optional).

## Environment (Phase 1 install)
`environment.yaml`: conda env `dreamer4`, **python 3.10, torch 2.8.0**, torchvision 0.23, torchrl 0.10,
tensordict 0.10, transformers 4.56, lpips 0.1.4, wandb 0.22, plus glew/glib/glfw (rendering, for the web
UI). `conda env create -f environment.yaml`. GPU: ≥1 CUDA GPU (their recommended full config is 8× GPU
≥24 GB + >256 GB RAM; we run **1× H100** — see Compute).

---

## DATA — how to feed Memory Maze in (the crux)

### Source
Public Google Drive folder `memory-maze-9x9` (id `1RcnkTZVwEHnAQeEuw7X8Y1RPSmrFLDFB`): 11 single-file
zip shards = `eval.zip` + `train-part0..9.zip`. 30k trajectories total (29k train / 1k eval), all **1000
steps (1001 frames)**, 64×64. Full set ≈100 GB zipped. Per-trajectory `.npz` after unzip.
- Download with our existing script (resumable gdown, blackout-tolerant — file IDs are baked in):
  `python -u experiments/memmaze-tokenizer/download_memmaze.py --parts train-part0 --out-dir data/memmaze9x9_raw --unzip`
  (`--parts all-train` / `all` / `eval`). Each `train-partN` ≈10 GB zipped, ≈2900 trajectories.

### Raw `.npz` format (from the memory-maze README, verified)
Keys per trajectory: `image (1001,64,64,3) uint8 **RGB**`, `action (1001,6) binary one-hot` = **"last
action"**, `reward (1001,) float` (sparse: ~1–2 rewards/traj). `Discrete(6)` = (noop, forward, left,
right, forward_left, forward_right). **Alignment**: `image[0]` is *before* the first action; `action[t]`
= the action that **produced** `image[t]` (`action[0]` is a dummy no-op). Other keys (agent_pos,
maze_layout, target_*) are privileged labels, unused by this baseline.

### Conversion (ready + validated): `experiments/dreamer4-community-baseline/memmaze_to_dreamer4.py`
Produces the two paired trees the community trainers consume, per task `<task>` (default `memmaze`):
- **frame shards** `<out>/shards/<task>/<task>_shard0000.pt = {"frames": (2048,3,64,64) uint8}` (CHW, RGB),
  read by `ShardedFrameDataset` (tokenizer) AND as the dynamics frame source.
- **demo** `<out>/demos/<task>.pt = {"episode":(N,) int64, "action":(N,16) f32, "reward":(N,) f32}`,
  read by `WMDataset` (dynamics `--use_actions`). `action` = raw one-hot placed in the **first 6 of 16**
  columns (the repo hardcodes `ActionEncoder(action_dim=16)`); cols 6..15 stay 0. **No time-shift** —
  raw MM `action[t]` already equals what `WMDataset` wants (`demo_action[g]` pairs with `frame[g]`).
```
# on the cluster, after download+unzip:
python -u experiments/dreamer4-community-baseline/memmaze_to_dreamer4.py \
    --raw data/memmaze9x9_raw/train-part0 --out-dir data/d4_memmaze/train --task memmaze --shard-size 2048
python -u experiments/dreamer4-community-baseline/memmaze_to_dreamer4.py \
    --raw data/memmaze9x9_raw/eval        --out-dir data/d4_memmaze/eval  --task memmaze --shard-size 2048
```
Disk: ~24 MB/shard → ~34 GB per `train-partN` of shards (2900 traj). Their full preprocessed DMControl set
is ~350 GB for scale reference. **Recommendation:** start with `train-part0` (≈2900 traj, matches the data
scale our own memmaze models trained on → fairer comparison); scale to `part0..2` if disk+time allow.
Keep **eval** shards OUT of the training `--frame_dirs`/`--data_dirs` (no eval leakage). Our own held-out
`data/memmaze9x9_val12*` (12 eval trajectories + `_ids.npy`) is a subset of `eval` — use those same 12 if
you want a head-to-head with our models.

### ⚠ Integration constraints (found during validation — do not skip)
1. **`--shard-size` MUST equal `WMDataset`'s `shard_size`** (default **2048**, and `train_dynamics.py`
   does NOT override it). Mismatch → `WMDataset` frame-indexing RuntimeError. Keep 2048 everywhere.
   (`ShardedFrameDataset`/tokenizer reads real shard lengths, so it is not affected.)
2. **`TASK_SET` filters both datasets.** The task name must be in `dreamer4/task_set.py::TASK_SET`. Edit it
   to `TASK_SET = ['memmaze']` (single-task run) or add `'memmaze'`. Otherwise 0 tasks load.
3. **`tasks.json`** (per-task lang embedding + action_dim) has no `memmaze` entry → `WMDataset` warns and
   falls back to zero language embedding + `action_dim=16` mask. That is fine (no change needed). If you
   want the mask to be exactly the 6 real dims, add a `{"memmaze": {"action_dim": 6}}` entry and pass
   `--tasks_json`; not required.

---

## CODE ADAPTATIONS (deltas from the stock repo)
Minimal — the architecture needs **no** change for discrete actions or 64×64.
- **Resolution 64×64:** train the tokenizer with `--H 64 --W 64 --patch 4` (→ 256 patches). The dynamics
  model reads H/W/C/patch from the tokenizer checkpoint's saved `args`, so it inherits 64 automatically.
  (Alternative: `--target-size 128` in the converter to use the repo's exact 128 default — wasteful upscale
  of native 64; not recommended.)
- **Discrete actions:** handled entirely in the converter (one-hot into 16 dims). `ActionEncoder` takes
  continuous `[-1,1]`; one-hot {0,1} is in range. No model edit.
- **Single GPU:** run `torchrun --nproc_per_node=1 …` (DDP auto-off when world_size==1) or plain
  `python train_*.py …`.
- **Time-budgeted stopping + LR schedule (REQUIRED by this task; the stock scripts have neither — constant
  LR, stop only at `--max_steps`).** Add a `--max_hours` arg and, in the training loop of BOTH
  `train_tokenizer.py` and `train_dynamics.py` (each already defines `t0 = time.time()` before the loop):
  ```python
  # argparse:  p.add_argument("--max_hours", type=float, default=0.0)   # 0 = disabled
  # first line inside the `for batch in loader:` / `for x in loader:` body:
  if step >= args.max_steps or (args.max_hours > 0 and (time.time()-t0)/3600.0 >= args.max_hours):
      step = args.max_steps            # make the outer `while step < args.max_steps` exit
      break
  # after each optimizer step, cosine-decay LR over the time budget:
  if args.max_hours > 0:
      frac = min(1.0, (time.time()-t0)/(args.max_hours*3600.0))
      for g in opt.param_groups: g["lr"] = args.lr * 0.5 * (1.0 + math.cos(math.pi*frac))
  ```
  Then set `--max_hours 24` (tokenizer) / `--max_hours 48` (dynamics) and leave `--max_steps` at its huge
  default. This satisfies "stopping + LR scheduling driven by elapsed training time". (`math` is already
  imported in `train_dynamics.py`; add `import math` to `train_tokenizer.py`.)
- **wandb** defaults to `mode="online"` in both trainers → set `WANDB_API_KEY` on the cluster, or change
  to `mode="offline"` / `os.environ["WANDB_MODE"]="offline"`, or (cleanest) add `--wandb_mode` plumbing.
  Use our W&B entity/project conventions if logging online.

---

## Compute & scheduling
- **1× H100 on ferranti** (SLURM). Access ferranti ONLY through the `scripts/` wrappers (never raw
  ssh/scp/sbatch — protocol §6); code goes up via GitHub (`sync_code.sh` = remote fetch+checkout, so
  commit+push first), results come back via the pull wrappers. See `scripts/README.md`, `HOWTO/cluster.md`.
  BUT the community repo is a *separate checkout* on the cluster, not part of our synced tree — clone it
  directly on ferranti (record the commit) and keep its checkpoints under a job dir you can pull back.
- Budgets by **active training wall-clock**, not epochs: tokenizer **24 h**, dynamics **48 h** (after
  tokenizer approval). The repo's reference is 8×3090 for 24 h/48 h; on 1× H100 you will do fewer steps —
  fine, the budget is time. Measure throughput on a short smoke to pick batch size (raise `--batch_size`
  and/or `--grad_accum` to saturate the 80 GB H100; their defaults are per-GPU bs 8 tokenizer / 24 dynamics
  on a 3090). Retain periodic checkpoints (`--save_every`) so a late failure doesn't lose the run.

---

## Phase 0 — prepare the checkout (cluster)
Clone `dreamer4` on ferranti, `git checkout b8abafbf`, create the conda env, apply the adaptations above
(`task_set.py` → memmaze; `--max_hours` + LR snippets; wandb mode). Sanity: run
`experiments/dreamer4-community-baseline/validate_integration.py --dreamer4 <checkout>` (cluster-free, ~10 s)
to confirm the converter + datasets + 64×64 forward still pass on that checkout.

## Phase 1 — adapt & smoke-validate the stack (cluster)
Download `train-part0` (+`eval`), convert both (commands above). Point the trainers at the converted trees
and run a SHORT smoke of each (a few hundred steps) to confirm: data loads, loss is finite, checkpoints
write, and — for dynamics — `stats/action_shuffle_loss_ratio` climbs clearly **above 1** (the decisive
check that the model is genuinely action-conditioned; if it sits at ~1, re-examine the action alignment —
try the converter's `--action-shift`, though 0 is expected-correct). Measure H100 throughput → pick
batch size + estimate data exposure over the budgets.

## Phase 2 — train the tokenizer (24 h H100), then HARD REVIEW GATE
```
python -u dreamer4/train_tokenizer.py \
    --data_dirs data/d4_memmaze/train/shards \
    --H 64 --W 64 --patch 4 --seq_len 8 --batch_size <tuned> --max_hours 24 \
    --lpips_weight 0.2 --ckpt_dir ./logs/memmaze_tok_ckpts
```
Produce reconstruction sheets on held-out **eval** episodes (geometry, walls, objects, motion, fine
detail — not just aggregate loss/PSNR). Pull the checkpoint + sheets + metrics + provenance local.
**HARD GATE: stop and present the tokenizer reconstruction sheets to Merlin. Do NOT start dynamics until
Merlin explicitly approves.** If rejected, record why and wait for direction.

## Phase 3 — train the action-conditioned dynamics (48 h H100, after approval)
```
python -u dreamer4/train_dynamics.py --use_actions \
    --frame_dirs data/d4_memmaze/train/shards \
    --data_dirs  data/d4_memmaze/train/demos \
    --tokenizer_ckpt ./logs/memmaze_tok_ckpts/latest.pt \
    --seq_len 32 --batch_size <tuned> --max_hours 48 --ckpt_dir ./logs/memmaze_dyn_ckpts
```
Watch during the run (do NOT judge success by aggregate loss alone):
- `stats/action_shuffle_loss_ratio` — **the core question**: does a *normal* Dreamer 4 learn
  action-conditioning on Memory Maze? Should rise meaningfully above 1.
- `eval/mse_ratio_pred_over_floor` (<1 = beats copy-last), `eval/psnr_gain_over_floor_db` (>0), and the
  `eval/viz` GT-vs-pred filmstrips (in-context tracking, does the maze stay coherent past the window).
Retain checkpoints + full provenance (upstream commit, local diff, data split, config, wall-clock, seed).

## Phase 4 — evaluate & make it playable
- **Quantitative:** the repo's built-in `run_dynamics_eval` already logs per-horizon pixel MSE vs a
  copy-last floor — the community analogue of our `src/evals/memmaze/rollout_error.py`. For a head-to-head
  with our models, compare in **pixel space** (the tokenizers/latent spaces differ, so latent metrics are
  NOT comparable across stacks). Optional nice-to-have: write an adapter that drives the community model
  through our rollout-error protocol (128-frame streamed prefill → 32-frame scored rollout, pixel MSE) on
  the SAME held-out `val12` episodes, so it drops onto our comparison plot.
- **Playable:** `python dreamer4/interactive.py` (web UI at :7860; forward via SSH). Verify paths/ckpts,
  drive it with the 6 discrete actions (its action input is the 16-dim continuous vector — feed one-hots).
  Make controls/reset consistent with our real + learned Memory Maze players where practical. Interaction
  must be genuine action-conditioned autoregressive rollout, not frame replay.

---

## Notes verified locally (cluster-free, upstream `b8abafbf`, 2026-07-14)
- Converter output is accepted end-to-end by the REAL `ShardedFrameDataset` (→ `(T,3,64,64)` float in
  [0,1]) and `WMDataset` (→ `obs (B,T+1,3,64,64)`, `act (B,T,16)` clean one-hot in first 6 dims).
- `Encoder/Decoder/Tokenizer/Dynamics` run a 64×64 forward with correct shapes (`z (B,T,16,32)` →
  packed `(B,T,8,64)`). **The untrained dynamics output is exactly 0** because `flow_x_head` is
  **zero-initialized** (a flow-matching init trick), NOT because actions are disconnected — de-zeroing the
  head shows the action changes the prediction in both space modes. Don't be alarmed by ~0 predictions at
  step 0; trust `action_shuffle_loss_ratio` once training has moved the head off zero.
- Action modality: `[ACTION, SHORTCUT_SIGNAL, SHORTCUT_STEP, SPATIAL, REGISTER, AGENT]`. `ACTION` is a
  distinct NON-agent modality; the prediction head reads `SPATIAL`, which attends to `ACTION` in both
  `wm_agent_isolated` (default) and `wm_agent`. Agent tokens are the isolated/inert branch in pretrain.
- Re-run the regression test anytime: `python experiments/dreamer4-community-baseline/validate_integration.py --dreamer4 <checkout>`.

## Done means
Independent community Dreamer 4 stack (its OWN tokenizer + action-conditioned dynamics, NOT our frozen
tokenizer), upstream revision + local adaptations recorded; tokenizer did 24 h time-budgeted H100 training,
produced held-out reconstruction sheets, got Merlin's explicit approval before dynamics; dynamics did 48 h
time-budgeted training with elapsed-time stopping + LR schedule; the model is playable via genuine
action-conditioned rollout; artifacts/metrics/qual evidence/provenance are local, including whether this
independent vanilla learned in-context + action-specific behavior. Framed correctly (see Goal caveat).

## Progress
- **2026-07-14** — Cluster-free prep COMPLETE (this document). Pinned upstream `b8abafbf`; wrote + validated
  the data converter and regression test under `experiments/dreamer4-community-baseline/`; catalogued all
  adaptations (TASK_SET, time-budget stop + LR, wandb mode, shard_size=2048, 64×64 via tokenizer args) and
  the action alignment (no shift). Blocked on cluster availability. NEXT: Phase 0 → Phase 2 on ferranti.
