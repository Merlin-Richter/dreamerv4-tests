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
