# dreamer4-community-baseline

Integration glue for training the independent community Dreamer 4
([nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4)) on Memory Maze as an independent
vanilla baseline. The task is `tasks/backlog/memmaze-community-dreamer4-baseline.md`. This directory is
experimental integration code and does not follow `src/`'s one-spec-per-file discipline. The community
repo is not vendored here: `setup_upstream.sh` clones the pinned commit into a gitignored run directory,
applies the committed patch, and records the exact upstream diff and environment.

- `memmaze_to_dreamer4.py` converts raw Memory Maze `.npz` trajectories into the community repo's paired
  frame-shard and demo trees. The native six-way one-hot action occupies the first six of the upstream
  stack's 16 action dimensions. No time shift is applied: raw `action[t]` produced raw `image[t]`.
- `validate_integration.py` synthesizes Memory Maze data, runs the converter, loads it through the real
  upstream `ShardedFrameDataset` and `WMDataset`, then exercises 64x64 tokenizer and dynamics forwards.
- `upstream-memmaze.patch` is the narrow adaptation against upstream `b8abafbf`: it selects `memmaze`,
  accepts native 64px dynamics shards, adds resumable elapsed-time stop/cosine LR, selectable W&B mode,
  guaranteed final checkpoints, and a matched-noise action-shuffle diagnostic. It does not change the
  model architecture or training loss.
- `setup_upstream.sh` and `requirements-community.txt` reproducibly create the external checkout and
  isolated pinned environment, then rerun the integration regression.
- `validate_converted.py` checks all converted shard/demo tensors and uses trajectory content fingerprints
  to prove train/eval disjointness.
- `phase1_smoke.sh` is the ferranti Phase 0/1 driver: setup, resumable public-data download, full train
  part 0 plus eval conversion, H100 batch calibration, tokenizer smoke, held-out reconstruction sheet,
  action-conditioned dynamics smoke, checkpoints, and GPU telemetry.
- `phase2_tokenizer.sh` runs the accepted tokenizer configuration for 24 active H100 hours, with periodic
  resumable checkpoints, GPU telemetry, a final checkpoint summary, and held-out reconstruction sheet.
- `phase3_dynamics.sh` verifies the approved tokenizer by SHA-256, then trains the action-conditioned
  dynamics model for 48 active H100 hours at the calibrated H100 batch size, with elapsed-time cosine
  scheduling, resumable checkpoints, action-shuffle diagnostics, rollout evaluation, and GPU telemetry.
  It stages the 2.9M-frame training conversion from Weka to per-job node-local scratch before the active
  timer starts; direct cold random reads across 1,418 Weka shards can block the first batch for minutes.
  The production loader uses four workers with 128 MB cache each, and the job requests 16 CPUs for a
  wider host-memory allocation; this stayed below 13 GiB RSS for the completed 48-hour run.
- `phase4_evaluate.sh` stages the content-disjoint held-out split and runs `evaluate_dynamics.py` to
  produce a four-sequence rollout sheet and JSON metrics. Each sequence compares exact ground truth,
  an autoregressive rollout with correct actions, a matched-noise rollout with cyclically wrong future
  actions, and copy-last. This tests visible rollout quality and action specificity independently of the
  training loss.
- `play_dynamics.py` is the local pygame player for the completed community baseline. Reset encodes and
  replays eight real held-out context frames (green border), then every frame is an action-conditioned
  autoregressive model sample. It uses the same native six-action keymap as `src/interactive/play_memmaze.py`
  and keeps both a rolling dynamics history and a trailing temporal-decoder history.
- `make_recon_sheet.py` and `summarize_checkpoint.py` provide held-out visual acceptance and stable
  checkpoint throughput/provenance summaries.

- `gate_dmin_only.py` is the correctness gate for the d_min-only arm, run against the pinned
  upstream checkout itself. It proves `--self_fraction 0` reduces upstream's
  `dynamics_pretrain_loss` exactly to its finest-step flow term, and that the `step_embed` rows
  for `K < k_max` then receive exactly zero gradient -- which is why that arm must be scored at
  `K=8` (`--schedule finest`) and not at the default `K=4`, whose embedding row never trains.
- `phase5_dynamics_dmin.sh` trains the d_min-only arm (24 active H100 hours; `--self_fraction 0.0`
  and `--eval_schedule finest` are its ONLY differences from `phase3_dynamics.sh`).
- `phase5_evaluate_dmin.sh` scores the arm against the immutable vanilla control at both `K=8` and
  `K=4`, plus the control's own step-matched periodic checkpoint, through one instrument and seed.
- `evaluate_dynamics.py` gained `--schedule`/`--eval-d`/`--sheet-sequences`; the defaults reproduce
  the original 2026-08-02 protocol exactly.

The cluster-free integration test passed locally against upstream `b8abafbf` on 2026-07-20.

Submit the smoke only through the academic-cluster wrapper:

```bash
bash scripts/submit_job.sh --cluster ferranti --name memmaze-d4-phase1-smoke --hours 8 -- \
  bash experiments/dreamer4-community-baseline/phase1_smoke.sh
```

After explicit approval of the Phase 2 tokenizer sheet:

```bash
bash scripts/submit_job.sh --cluster ferranti --name memmaze-d4-dynamics-48h --hours 54 --cpus 8 -- \
  bash experiments/dreamer4-community-baseline/phase3_dynamics.sh
```

## Local imagined-world player

The player imports `model.py` from a separate checkout of the pinned upstream revision. If the local
integration checkout does not already exist below `runs/dreamer4-community-baseline/upstream-*`, make a
source-only checkout once:

```powershell
git clone https://github.com/nicklashansen/dreamer4.git runs/dreamer4-community-baseline/upstream-player-b8abafbf
git -C runs/dreamer4-community-baseline/upstream-player-b8abafbf checkout --detach b8abafbf4da72c59b6aa09f8499ccde0d6a37fd6
```

This machine's durable installation lives in the sibling community checkout at
`C:/Users/richt/OneDrive/Desktop/Code/dreamer4`. It contains the exact approved checkpoints, a CUDA
environment with pygame, and a self-contained launcher. From this repository run:

```powershell
powershell.exe -ExecutionPolicy Bypass -File ../dreamer4/run_memmaze_community.ps1
```

The installed artifacts are `dreamer4/checkpoints/memmaze-community-d4/dynamics-final.pt` and
`tokenizer-final.pt`; `CHECKSUMS.sha256` pins them to the evaluated final models. The general
`play_dynamics.py` remains the versioned source of truth on this branch.

Controls: up = forward, left/right = turn, up+left/up+right = moving turns, space = pause,
backspace = reset to a new held-out context, tab = remove frame pacing, escape = quit. The green-border
frames after reset are the only recorded frames; once the border disappears, play is entirely inside
the model. A deterministic headless gate is available with `--episode 0 --start 0 --selftest 33`.
