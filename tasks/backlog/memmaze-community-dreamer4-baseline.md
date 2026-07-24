# Train an independent Dreamer 4 baseline on Memory Maze

## Goal
Establish a credible vanilla Dreamer 4 baseline for Memory Maze using the independent community
implementation at [nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4).

Our in-repository vanilla model failed to learn even reliable in-context or action-specific behavior,
including with the tau0 addition. It therefore cannot serve as a fair baseline for judging mem2mem or
archive models. This task tests whether a separate, normal Dreamer 4 implementation can learn Memory Maze
when given substantial compute.

Possible explanations for the current gap remain open: our vanilla path may be broken, or mem2mem training
may simply be more compute-efficient. Do not claim either explanation from training loss alone.

## Scope and ownership
- Use a separate copy of the community repository rather than porting its model into this repository's
  spec-backed `src/` architecture.
- Adapt the complete external stack to the Memory Maze training and held-out evaluation data, including
  its own tokenizer and action-conditioned dynamics model. Do not substitute our frozen tokenizer.
- Preserve the external implementation's substantive Dreamer 4 architecture and learning objectives so
  it remains an independent baseline. Data plumbing, training controls, checkpointing, visualization, and
  interaction code may be changed freely to fit Memory Maze and the cluster.
- The copied external code is experimental integration code: it may be modified aggressively and does not
  need the cleanliness, one-to-one specs, or long-term maintainability required under this repository's
  `src/`. Correctness, usable throughput, and provenance still matter.
- Record the exact upstream commit before modifications so the origin of the baseline is reproducible.

## Compute and scheduling
- Run on one H100 on ferranti.
- Training budgets are measured by active training wall-clock time, not by a fixed epoch count.
- Drive stopping and learning-rate scheduling from elapsed training time. Epoch counts may be estimated
  from a short throughput measurement for planning and reporting, but must not determine when either main
  run stops.
- Tokenizer training budget: 24 hours.
- Dynamics training budget: 48 hours, after tokenizer approval.
- Use the ferranti/galvani wrapper-only access policy and retain normal job, commit, configuration, and
  checkpoint provenance.

## Phase 1: adapt and validate the external stack
- Make the community repository consume the Memory Maze training data, actions, and held-out split without
  changing their meaning or leaking evaluation episodes into training.
- Establish that the tokenizer and action-conditioned dynamics paths can train and checkpoint on a small
  smoke run before spending the main budgets.
- Measure H100 throughput and resource use sufficiently to select a practical configuration and estimate
  how much data exposure the wall-clock budgets will provide.
- Keep the model configuration faithful to the external implementation unless a Memory Maze compatibility
  change is necessary; record and justify material deviations.

## Phase 2: train and review the tokenizer
- Train the external repository's tokenizer on Memory Maze for 24 hours of active H100 training time.
- Track training health and retain useful checkpoints so a late failure does not lose the full run.
- Produce reconstruction sheets on held-out Memory Maze episodes that expose geometry, objects, motion,
  and visually important fine detail rather than relying on aggregate reconstruction loss alone.
- Pull the checkpoint, sheets, metrics, and provenance locally and present the sheets to Merlin for
  personal review.

**Hard review gate:** stop after presenting the tokenizer reconstruction sheets. Do not begin the dynamics
run until Merlin explicitly approves the tokenizer. If it is rejected, record why and wait for direction
on retraining or adaptation rather than silently continuing.

## Phase 3: train the action-conditioned dynamics model
- After tokenizer approval, freeze the accepted external tokenizer and train the external repository's
  action-conditioned dynamics model for 48 hours of active H100 training time.
- Use the real Memory Maze action stream and preserve correct observation/action temporal alignment.
- Monitor whether the model learns action-specific and in-context behavior during the run. A low aggregate
  loss by itself is not evidence that the failed-baseline problem has been solved.
- Retain checkpoints and all provenance needed to identify the exact external revision, local adaptations,
  data split, configuration, wall-clock exposure, and resulting model.

## Phase 4: make the trained world model playable
- Provide a playable Memory Maze experience driven by the trained external world model, with controls and
  reset behavior consistent with the existing real and learned Memory Maze players where practical.
- Interaction must use action-conditioned autoregressive model rollouts rather than replaying recorded
  frames.
- Make the qualitative result easy to compare with our vanilla, mem2mem, and archive models.
- Where compatible, make the checkpoint evaluable by the quantitative Memory Maze rollout-error task so
  baseline conclusions do not rest on playability or training loss alone.

## Done means
- The adapted baseline is recognizably the independent community Dreamer 4 stack, with its own tokenizer
  and action-conditioned dynamics model, and its upstream revision and adaptations are recorded.
- The tokenizer completed 24 hours of time-budgeted H100 training, produced held-out reconstruction
  sheets, and received Merlin's explicit approval before dynamics training began.
- The dynamics model completed 48 hours of time-budgeted H100 training with elapsed-time stopping and
  learning-rate scheduling.
- The final model can be played interactively on Memory Maze through genuine action-conditioned rollouts.
- Artifacts, metrics, qualitative evidence, and provenance are available locally, including evidence about
  whether this independent vanilla Dreamer learned in-context and action-specific behavior.
- The result is framed correctly: it can establish a usable independent baseline or falsify the idea that
  our failed vanilla run was representative, but it does not by itself distinguish an implementation bug
  from a compute-efficiency advantage without matched evidence.

## Progress
Maintain this section throughout the task. After every meaningful transition, append a dated entry stating
the current phase, what completed, active or completed ferranti job identifiers, commit/revision, artifact
locations, measured training time, and the next gate or blocker. This section must always make it possible
to tell where the multi-step task currently stands without reconstructing state from logs.

- **2026-07-20 — Phase 0 local integration complete; ferranti ready.** Ferranti health checked through
  the wrappers (H100 partition healthy, no active user jobs, 81 TB free) and project prep SHA
  `b6b07774b82794a458e406938dcbd235cca5d16c` synced. Upstream remains pinned to
  `b8abafbf4da72c59b6aa09f8499ccde0d6a37fd6`. Integration commit `beef6a6` adds a reproducible
  external-checkout patch/bootstrap, isolated pinned environment, real conversion manifests plus
  train/eval content-disjointness validation, H100 batch calibration, tokenizer/dynamics smoke driver,
  held-out reconstruction sheets, GPU telemetry, and checkpoint summaries. The upstream architecture
  and objectives are unchanged; compatibility/control fixes are native 64px dynamics loading,
  resumable elapsed-time stop/cosine LR, W&B mode, guaranteed final checkpoints, and a matched-noise
  action-shuffle diagnostic. Local gates passed: converter + real upstream datasets + 64px model forward;
  clean patch apply/reverse; Python/Bash syntax; elapsed stop/final save; resume consumes no extra steps;
  tiny action-independent dynamics control reports shuffle ratio exactly 1.0. NEXT: push/sync `beef6a6`,
  submit `memmaze-d4-phase1-smoke`, record the ferranti job id, then inspect throughput/checkpoints and
  smoke reconstruction before starting the 24-hour tokenizer.
- **2026-07-20 — Phase 1 submitted.** Branch `codex/memmaze-community-d4-baseline` pushed and ferranti
  synced to exact project commit `945170b8dbf4d0b8cdd198bccc183fb0149f823e`; upstream remains pinned
  at `b8abafbf4da72c59b6aa09f8499ccde0d6a37fd6`. Ferranti job **418360**
  (`memmaze-d4-phase1-smoke`, 1x H100, 16 CPUs, 8-hour allocation) submitted at 17:09 Europe/Berlin.
  Active training time: 0 h so far (queued/submitted). Remote artifacts will land under
  `runs/memmaze-d4-phase1-smoke/`; stable converted data under `data/d4_memmaze_community/`.
  NEXT: monitor 418360 through the wrappers; on success pull logs/metrics/checkpoints, inspect the smoke
  reconstruction and action-shuffle/throughput evidence, then configure the 24-hour tokenizer run.
- **2026-07-22 — Phase 1 conversion OOM diagnosed and fixed.** Ferranti job **418360** ended
  `OUT_OF_MEMORY` after 27m30s with MaxRSS 32,656,728 KiB while converting trajectory 2,400/2,900 of
  `train-part0`; no H100 training began and eval conversion had not begun. The old converter repeatedly
  allocated full-trajectory/concatenation temporaries and its RSS grew approximately with raw frames
  processed. The replacement uses one fixed frame-shard buffer, preallocated aligned demo tensors, and
  buffer-protocol hashing (no full-frame `bytes` copy), and reports peak RSS every 200 trajectories.
  Synthetic validation passed for 2,701 frames / 22 shards with exact frame-demo alignment. The rerun
  writes fresh `train-part0-v2` / `eval-v2` outputs so job 418360's partial v1 output cannot be mixed in.
  NEXT: push/sync the bounded-memory fix, resubmit Phase 1, and verify RSS remains bounded before training.
- **2026-07-22 — Phase 1 resubmitted.** Bounded-memory conversion commit
  `84ebb28fad34574fe637a74fbef35c216266c227` was pushed and synced exactly to ferranti. Replacement job
  **419859** (`memmaze-d4-phase1-smoke-v2`, 1x H100, 16 CPUs, 8-hour allocation) was submitted and is
  pending for priority. Active training time remains 0 h. Artifacts will land under
  `runs/memmaze-d4-phase1-smoke-v2/`; converted data under `data/d4_memmaze_community/*-v2`.
  NEXT: inspect the first conversion progress reports for bounded peak RSS, then follow smoke training.
- **2026-07-23 — Attempt 2 timed out; final-attempt gates hardened.** Job **419859** timed out after
  8h00m16s. Conversion completed correctly and stayed bounded (train: 2,900 trajectories, peak RSS
  0.75 GiB; eval: 1,000 trajectories, peak RSS 0.62 GiB), and split validation passed. Tokenizer batch
  calibration established bs=64 as the largest tested fit (128/256 OOM), but upstream's nested training
  loop failed to propagate the normal `max_steps` break and repeated `step=20` without forward passes;
  GPU telemetry consequently showed 0% utilization while retaining 44.9 GiB. Both tokenizer and dynamics
  now use one propagated stop flag for max-step and elapsed-time exits. Actual local GPU gates passed for
  both trainers: max-step exits at exactly step 2 with final checkpoints, sub-second time budgets exit at
  step 1, tokenizer resume advances exactly one step, and the patched upstream stack still passes its
  data/model integration test. A subsequently exposed OpenCV stride bug in the held-out reconstruction
  sheet was also fixed and the sheet was rendered and visually checked locally. The final Phase 1 attempt
  will reuse the validated v2 conversion, reconfirm known-good tokenizer bs=64, and retain dynamics batch
  calibration. Active useful training time remains limited to the 20 calibration steps from attempt 2.
  NEXT: commit/push/sync these final-attempt fixes, submit attempt 3, and monitor through sustained
  tokenizer and dynamics training rather than accepting scheduler RUNNING as evidence.
- **2026-07-23 — Final Phase 1 attempt submitted.** Commit
  `829a58ff3f722232ca13bb2810b967eded254c81` was pushed and synced exactly to ferranti. Job **420912**
  (`memmaze-d4-phase1-smoke-v3`, 1x H100, 16 CPUs, 8-hour allocation) started immediately on `mlcbm012`.
  Its clean patched-upstream setup and synthetic end-to-end integration gate passed. Artifacts land under
  `runs/memmaze-d4-phase1-smoke-v3/`; the completed bounded v2 conversion is reused.
  NEXT: require advancing tokenizer and dynamics steps plus sustained nonzero GPU telemetry, then pull and
  inspect all smoke artifacts before declaring Phase 1 successful.
- **2026-07-23 — Phase 1 tokenizer path passed; move to tokenizer-only production.** Job **420912**
  completed the useful Phase 1 gates: tokenizer calibration reconfirmed bs=64; the timed tokenizer smoke
  trained 962 steps in 300.11 active seconds (3.205 steps/s), with sustained 98-100% H100 utilization,
  wrote a valid final checkpoint, and rendered a held-out sheet (mean MSE 0.03618 / PSNR 14.41 dB).
  Action-conditioned dynamics calibration trained and checkpointed 10 steps at bs=128. The subsequent
  redundant timed dynamics smoke completed step 0 but then spent 22 minutes idle in its step-0 periodic
  save, so job 420912 was deliberately cancelled at 28m59s rather than wasting the H100; no production
  dynamics budget was started. Periodic checkpointing now explicitly skips step 0. A resumable
  `phase2_tokenizer.sh` production driver was added for exactly 24 active H100 training hours, bs=64,
  periodic checkpoints, telemetry, final summary, and held-out reconstruction. NEXT: sync and submit the
  tokenizer-only production run; monitor advancing steps and GPU telemetry, then pull the final sheet for
  Merlin's hard review gate. Do not begin the 48-hour dynamics run without explicit approval.
- **2026-07-23 — Phase 2 tokenizer production started.** Production driver commit
  `041716e43acbe9f9f24c727c65c83f88f0dea395` was pushed and synced exactly. Ferranti job **420962**
  (`memmaze-d4-tokenizer-24h`, 1x H100, 16 CPUs) started immediately on `mlcbm007`. The scheduler
  allocation is 30 hours, while the trainer's authoritative active budget is exactly 24 hours; the margin
  covers setup, validation, and final checkpoint/sheet I/O. Clean patched-upstream integration passed and
  the real 2.9M-frame conversion validation is in progress/passing. Artifacts land under
  `runs/memmaze-d4-tokenizer-24h/`. NEXT: verify advancing optimizer steps plus sustained utilization,
  monitor checkpoints/health, then pull the final checkpoint, metrics, and held-out reconstruction for
  the mandatory review gate.
- **2026-07-24 — Phase 2 complete; stopped at Merlin's tokenizer review gate.** Ferranti job **420962**
  completed successfully (exit 0) after 24:01:11 scheduler elapsed. The trainer stopped at its exact
  24-hour active-time budget (`86400.27 s`) at step 281,559 / epoch 6, averaging 3.259 optimizer steps/s.
  GPU telemetry recorded 8,640 active samples at 98.1% mean utilization and 44,915 MiB mean HBM use.
  Held-out reconstruction across four content-disjoint sequences measured mean MSE **0.00138073** /
  PSNR **28.60 dB**. Manual inspection found faithful maze geometry, colors, objects, and motion with
  modest smoothing of fine wall texture and the largest error on close-up geometry; there is no
  mean-image collapse, gross color drift, or missing-object failure. The pulled final checkpoint loads
  successfully and has SHA-256
  `347052fae0212ea2c6b943ae7c28a886298ce551d4155b882084d63a3ea48797`. Checkpoint, sheet, metrics,
  logs, telemetry, and provenance are local under `runs/memmaze-d4-tokenizer-24h/`. **NEXT / BLOCKER:**
  waiting for Merlin to approve or reject `tokenizer_recon.png`. No dynamics production job has been
  submitted; Phase 3 remains forbidden until explicit approval.
