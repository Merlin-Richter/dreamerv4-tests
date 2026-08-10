# Community Dreamer 4 with memory tokens and mem2mem rollout training

## Goal

Extend the independent community Dreamer 4 Memory Maze baseline with the smallest faithful version of
our memory-token idea, train it with long contiguous mem2mem rollout training and truncated
backpropagation through time (TBPTT), and let Merlin compare it interactively against the completed
community vanilla model.

This is the externally grounded comparison we actually want:

- **Vanilla:** the completed community Dreamer 4 baseline.
- **Memory:** the same community implementation, tokenizer, data, actions, transformer dimensions,
  shortcut objective, optimizer family, seed, and 48-hour cumulative training-loop wall budget, with only:
  1. per-frame memory tokens and their read-old/write-new inference semantics; and
  2. long sequential rollout training that teaches memory-to-memory construction.

The memory arm must remain recognizably the community implementation. Do not port our full
`src/models/dynamics_model.py`, tokenizer, FF9 machinery, archive hierarchy, or evaluation stack into
it.

## Primary question

When a normal external Dreamer 4 implementation is given a bounded recurrent memory channel and the
training flow needed to learn repeated memory read/write, does its interactively played Memory Maze
world remain coherent and controllable longer than the same implementation's vanilla sliding-window
rollout?

The answer for this task is Merlin's paired interactive assessment. Training loss, memory-token
variance, and held-out pixel metrics are health diagnostics only and must not be presented as the
result.

## Locked reference baseline

The comparison target is the completed community baseline, not our in-repository vanilla model:

- Upstream architecture origin: `nicklashansen/dreamer4` at
  `b8abafbf4da72c59b6aa09f8499ccde0d6a37fd6`.
- Integration branch: `codex/memmaze-community-d4-baseline` (durable-install ledger at `1ae9cdc`).
- Approved community tokenizer SHA-256:
  `347052fae0212ea2c6b943ae7c28a886298ce551d4155b882084d63a3ea48797`.
- Completed vanilla dynamics SHA-256:
  `7b077938fec776c74e62201ab79194a7a06e10e54856c69d47b65dda6367d674`.
- Vanilla production job: ferranti **423141**, step 298,164, seed 0. Its `elapsed_train_s` was
  **172,800.10 seconds** from a timer started immediately before DataLoader iteration, so it counted
  batch waits, online tokenizer encoding, dynamics optimization, logging, and periodic checkpoint time.
- Data: exact `train-part0-v2` training conversion and content-disjoint `eval-v2` held-out conversion;
  preserve the manifest, episode membership, RGB channel order, and action convention
  `raw action[t] produced raw image[t]`.
- Frozen tokenizer: native 64px RGB, 4x4 patches, 16 latent tokens x 32 dimensions, packed 2:1 into
  8 x 64-dimensional dynamics scene tokens.
- Vanilla dynamics configuration: sequence length 32, `d_model=512`, depth 8, 4 heads,
  `time_every=1`, packing factor 2, 4 registers, 1 isolated agent token, `k_max=8`, bootstrap start
  step 5,000, self fraction 0.25, action conditioning on, LR 1e-4, weight decay 0.01, gradient clip
  1.0, K=4 interactive sampling.

Do not warm-start the primary memory arm from the final vanilla checkpoint. That would give it the
vanilla model's full 48-hour training plus additional memory training and invalidate the comparison.
Construct both architectures at seed 0 and prove that every shared parameter has the same initial value;
initialize only the new memory-token parameter separately, after shared initialization, without advancing
the RNG stream used by shared weights.

## What “same tokenizer and same tokenized dataset” means

Use the exact approved community tokenizer and exact converted community train/eval data above. No frame
resizing, color conversion, altered action shift, alternate tokenizer, or in-repository latent cache is
allowed.

The completed vanilla trainer encoded frames online; it did not train from a permanent canonical latent
cache. Therefore the primary memory arm must preserve the vanilla encoder's window semantics. Long-rollout
training may:

- encode each W-frame rollout window through the same frozen community encoder; or
- cache those exact `(episode, start, W)` encoder outputs if a gate proves they are numerically equivalent
  to the online path and the cache is only a transport optimization.

It may not encode whole 128/1001-frame sequences under a different temporal-position/context convention
and call that “the same tokenized data.” The selected transport is an exact FP32 cache of every independent
`(episode, start, W=32)` encoder call, flattened by valid window start. It is deliberately not a single
`(episode,time)` latent tensor: this temporal causal tokenizer gives a frame different latents when its
position/history inside the 32-frame window changes. Cache construction happens before the 48-hour
training clock, and its manifest pins the tokenizer hash, source conversion hash, window, dtype, and full
array hashes.

## Memory-token architecture

Add memory as an optional feature of the community `Dynamics` model:

- New configuration/CLI field `n_memory`; default **0** means exact vanilla behavior and checkpoint
  compatibility. The production arm uses **8 memory tokens per frame**, matching the existing Memory Maze
  mem2mem reference at the same 512-dimensional transformer width.
- Add a MEMORY modality to the per-frame layout. Memory participates in ordinary within-frame spatial
  mixing and has its own causal same-slot temporal channels, like the current community transformer's
  other per-frame slots. The isolated agent-token behavior must remain unchanged.
- `forward(..., memory_in=None, return_memory=False)` injects provided memory activations or learned blank
  memory tokens, and optionally returns the final-layer written memory states. Memory has no supervised
  target and no projection into the clean-latent prediction head.
- A new memory state is **written from** prior memory, current scene latents, and actions. Do not copy a
  frozen activation unchanged from frame to frame and do not treat registers as the memory carrier.
- `n_memory=0` must preserve the existing token layout, outputs, state dict, and vanilla checkpoint load
  path. Add no unused parameters in this mode.

### Inference semantics

Extend the community autoregressive sampler and pygame player, which currently recompute a sliding latent
history without a KV cache:

1. Keep the recent packed scene latents, aligned actions, and written memory activations together.
2. During a target frame's K denoising passes, inject written memories for past frames and blank memory
   for the target frame.
3. After the target latent is generated, run a dedicated near-clean **commit/write pass** with the same
   action and prior memory history. Store its final target memory activation alongside the generated
   latent.
4. When old scene frames leave the 32-frame window, their information can survive only insofar as newer
   written memory activations learned to absorb it.

Training and inference must agree about where blank versus written memory is injected and which forward
produces the committed memory. A shortcut-denoising intermediate is not a committed memory state.

## Rollout training (“mem2mem”)

The primary memory arm is **rollout-only**. It replaces independent randomly sampled 32-frame dynamics
snippets with long contiguous trajectory clips and sequential overlapping windows. Do not mix ordinary
snippet batches back in merely to improve headline loss unless Merlin explicitly approves a separate
ablation.

Use the established references as the mechanism guide:

- `experiments/mem2mem/rollout.py`
- `experiments/mem2mem/train_mem2mem.py`
- the historical tasks `memmaze-dynamics-mem2mem` and `memmaze-dynamics-mem2mem-noff9`
- memory read/write and commit semantics in `specs/models/dynamics_model.md`

Locked initial rollout geometry:

- model window `W=32`;
- long contiguous training clip `L=128` (4W), sampled without crossing episode boundaries;
- slide by `W/2=16` frames;
- first window uses learned blank memory to construct the initial written states;
- each later window injects graph-attached written memory into its old half and learned blank memory into
  its new half;
- compute shortcut/x-prediction loss on the **new half only**, so every absolute target frame is scored
  once rather than overweighting overlapping frames;
- preserve exact action-to-produced-frame alignment through every slide.

### Prediction modes without FF9

Use the current rollout-only/no-FF9 reference's per-sequence 50/50 modes:

- **Latent-present mode:** old-half latents are held near-clean; new-half latents use the normal community
  shortcut corruption/sampling. This preserves ordinary visible dynamics and rollout quality.
- **Memory-load-bearing mode:** after a normally grounded initialization window, rollout-path latents are
  pure noise at tau=0 while written memory continues to relay. The new-half clean-latent prediction loss
  is unchanged, but memory is the only scene carrier.

The second mode is not FF9 and introduces no auxiliary loss: it is the same rollout prediction objective
under a context ablation. Without it, the model can minimize the new-half loss while ignoring memory.

### Preserve the community shortcut objective

Port the community implementation's own finest-step flow and self-bootstrap objective to the new-half
loss; do not silently replace it with our model's loss or a d-min-only objective. Preserve `k_max=8`,
`bootstrap_start=5000`, self fraction 0.25, ramp/weighting behavior, and K=4 inference. Old-half context
must remain fixed and consistent across the main and bootstrap sub-forwards; bootstrap targets are
stop-gradient exactly as in vanilla.

If an exact new-half formulation is impossible, stop and document the mathematical mismatch before
training. A different loss would make this a multi-variable comparison.

## Real TBPTT, not just a detach label

Backpropagate through the written-memory relay for **64 frames (2W)**, then detach at a declared boundary.
Implement footprint-bounded TBPTT:

- accumulate and backward each TBPTT segment with correct total-loss normalization;
- release the completed segment's graph before continuing;
- carry only detached boundary memory into the next segment;
- do not sum every slide loss for the full 128-frame clip and call one final backward—the old reference's
  detach limited gradient reach but retained all slide graphs, so memory still scaled with total rollout
  length;
- optimizer stepping may occur per complete long clip or per normalized TBPTT segment, but the choice must
  be pre-registered and kept fixed for the production run.

TBPTT truncation is a resource boundary, not a change to forward memory state. The carried activation
continues across the full 128-frame clip even where its gradient is detached.

## Explicit non-goals

- **No FF9** sufficiency loss, lookahead reconstruction, FF9 scaler, or FF9 injection path.
- **No archive** tokens, compression, sparse-write hierarchy, retrieval, or archival loss.
- No changes to the accepted community tokenizer.
- No our-repo tokenizer or dynamics-model substitution.
- No policy/value learning, rewards, or environment-state labels.
- No extra training data and no held-out episode leakage.
- No architecture-width/depth increase other than the eight memory tokens and their unavoidable learned
  initialization parameter.
- No quantitative metric is allowed to overrule Merlin's interactive evaluation for this task.

## Correctness gates before production

All gates must pass on local CUDA or a short ferranti probe before submitting the 48-hour run:

1. **Vanilla parity:** `n_memory=0` loads the completed vanilla checkpoint strictly and produces the same
   outputs for identical latents/actions/tau/d/noise seeds. Report max absolute difference; target 0.
2. **Shared initialization parity:** seed-0 vanilla and memory models have bit-identical shared parameters;
   only memory-token parameters are new.
3. **Shape and causality:** memory-enabled forward/loss/sample shapes are correct; perturbing future
   latents/actions/memory cannot change earlier outputs.
4. **Relay-gradient proof:** in memory-load-bearing mode, a loss-bearing frame beyond the first window has
   nonzero gradient to the mechanism that constructed initial/earlier written memory. With a deliberate
   detach before that dependency, the gradient becomes exactly zero.
5. **Read/write proof:** replacing or permuting old written memories changes new written memory and future
   latent predictions under matched noise; blanking memory after eviction has a measurable effect.
6. **No shortcut:** zero/blank registers, memory, and actions separately under matched noise to ensure the
   implementation is measuring distinct channels and not accidentally routing privileged state.
7. **Loss accounting:** every new-half absolute frame is scored once; overlap does not duplicate targets;
   short-clip segmented TBPTT matches a monolithic backward within numerical tolerance when no detach
   boundary is crossed.
8. **Data identity:** tokenizer hash, train/eval manifests, episode disjointness, RGB order, action
   alignment, and window-keyed encoding equivalence pass.
9. **Resume determinism:** checkpoint/resume preserves model, optimizer, cumulative training-loop clock, shortcut
   bootstrap state, data position/seed, and TBPTT configuration without repeating or skipping effective
   training budget.
10. **Player gate:** vanilla and memory checkpoints both load through the same pygame player; scripted
    action self-tests run beyond 32 generated frames, cross window eviction, stay finite, and never read
    recorded future frames.

## Training and compute contract

- Cluster: ferranti, **one H100**.
- Primary production budget: exactly **48 hours = 172,800 seconds on the same cumulative training-loop
  wall clock used by vanilla**.
- The clock starts immediately before DataLoader iteration and counts first-batch startup/waits, cached
  latent reads, host-to-device transfers, dynamics forward/backward/optimizer work, logging, and periodic
  checkpoint serialization. Upstream setup, cache construction/validation, W&B setup, final checkpoint
  serialization/evaluation, queueing, and preemption downtime remain outside it, matching vanilla's timer
  boundaries.
- GPU telemetry is mandatory at fine cadence. Target mean utilization is >=95%; the hard health gate is
  >=90% with no unexplained idle interval over five minutes. Utilization is a health/efficiency diagnostic,
  not a second narrower clock and never grants extra training time.
- If utilization is below the gate, profile/fix cache I/O, rollout batching, or TBPTT scheduling before
  production. Do not compensate by extending the 48-hour budget.
- Use resumable periodic checkpoints in one **54-hour** scheduler allocation, leaving setup/finalization
  margin. Completion is determined by audited cumulative training-loop time, not Slurm elapsed time,
  optimizer-only time, or epoch count.
- Match vanilla optimizer hyperparameters and 48-hour elapsed-time LR schedule. Memory rollout batch size
  may be smaller for HBM, but record batch size, slides/step, target frames/step, optimizer steps,
  unique/source latent windows read, total scored frames, and achieved utilization.
- The primary comparison is time-matched at 48 training-loop wall hours. Do not extend the memory run to
  match vanilla optimizer steps or frames after seeing the interactive result; that would be a separate,
  pre-declared secondary experiment.

## Implementation phases

### Phase 1 — independent community memory option

- Add optional `n_memory` model/config/checkpoint support to the versioned community integration.
- Keep `n_memory=0` byte-compatible and prove the parity gate.
- Add written-memory input/output and commit semantics to the community sampler.
- Record the exact upstream revision and all new diffs; keep the change experimental and outside
  spec-backed `src/`.

### Phase 2 — rollout trainer and TBPTT

- Add long contiguous data loading, 32/16 sliding rollout, 50/50 prediction modes, community-equivalent
  new-half shortcut loss, and real 64-frame segmented TBPTT.
- Add the correctness probes above, including the relay-gradient and segmented-backward tests.
- Establish a small train-data-only smoke where loss is finite, optimizer steps advance, memory receives
  nonzero gradients, memory activations do not collapse to a constant, and checkpoint/resume works.

### Phase 3 — H100 calibration

- Build and validate the exact window-keyed FP32 latent cache, then calibrate long-clip batch size, mmap
  loader workers/prefetch, and TBPTT segment scheduling on an H100.
- Demonstrate sustained utilization and bounded host/HBM memory through multiple TBPTT boundaries and at
  least one checkpoint save/resume.
- Freeze the production configuration before the main run. Do not tune against held-out interactive
  behavior.

### Phase 4 — 48-training-loop-hour memory training

- Train from seed-0 scratch for exactly 172,800 counted cumulative training-loop seconds.
- Monitor finite losses/gradients, written-memory statistics, action-shuffle sensitivity, relay influence,
  throughput, HBM/host RAM, utilization, and periodic checkpoints.
- Retain the exact final checkpoint regardless of whether intermediate health diagnostics look better.

### Phase 5 — paired interactive comparison

- Extend the durable community pygame installation at
  `C:/Users/richt/OneDrive/Desktop/Code/dreamer4` to select **vanilla** or **memory** without changing
  tokenizer, held-out reset data, context length, K, action mapping, rendering, or noise schedule.
- Install the final memory checkpoint beside
  `checkpoints/memmaze-community-d4/dynamics-final.pt`, with an unambiguous name and SHA-256.
- Show the active model prominently in the UI.
- Support fixed `--episode`, `--start`, and noise `--seed`. Add action-trace record/replay (or an equivalent
  paired mode) so Merlin can drive once and replay the identical actions/noise initialization on both
  models, while still allowing ordinary free interactive play.
- The useful comparison must continue beyond the 32-frame latent window; short in-window behavior alone
  cannot demonstrate memory.

## Interactive review protocol

Merlin is the evaluator. Present two one-line launch commands or a single explicit model selector. For a
paired review:

1. Use the same held-out episode, start, eight real context frames, K=4, and seed.
2. Drive or replay the same action trace on vanilla and memory.
3. Continue well past 32 generated frames and include revisits/turns where off-screen geometry matters.
4. Judge controllability, action response, room/turn continuity, object persistence, revisit consistency,
   texture collapse, and catastrophic drift.
5. Record Merlin's verdict verbatim. Do not translate a subjective rejection into a success because
   training diagnostics were healthy.

## Artifacts and provenance

Retain and pull locally:

- final and periodic memory checkpoints plus SHA-256;
- full resolved config and code/upstream commits;
- tokenizer/data manifest hashes and disjointness report;
- training-loop clock ledger, scheduler accounting, optimizer/scored-frame exposure, resume events;
- GPU telemetry, HBM/host RSS, training logs, and health diagnostics;
- correctness-gate outputs and H100 calibration results;
- durable player command, action traces used for review, and Merlin's final assessment.

Keep the baseline branch/checkpoint immutable. New artifacts must use a distinct run/checkpoint name and
must never overwrite the completed vanilla model.

## Done means

- The community model supports optional memory tokens with exact `n_memory=0` vanilla parity.
- A no-FF9, no-archive, rollout-only mem2mem trainer performs sequential 32/16 read-old/write-new memory
  training with real 64-frame TBPTT over 128-frame contiguous clips.
- The primary memory arm used the exact approved tokenizer, exact train split/actions, seed-0 shared
  initialization, community shortcut objective, and **48 hours on the vanilla-matched cumulative
  training-loop wall clock**, with utilization reported separately.
- The final memory checkpoint is pulled, hash-verified, durably installed beside vanilla, and both models
  run through the same interactive player beyond window eviction.
- Merlin completed the paired interactive comparison and the outcome—positive, negative, or mixed—is
  recorded without overclaiming what subjective play establishes.

## Progress

Maintain this section throughout the task. After each meaningful transition, append a dated entry with
the exact branch/commit, ferranti job IDs, cumulative training-loop seconds, utilization, artifacts, completed
gates, and next blocker. Do not reconstruct status later from chat or scheduler history.

**Supersession note:** progress entries through the 2026-08-06 startup snapshot preserve what was run and
observed, but their optimizer-only “active” clock, online re-encoding, and two-allocation conclusions are
rejected. The 2026-08-07 correction below is the current contract.

- **2026-08-04 — Task specified.** Locked an external community vanilla-vs-memory comparison using the
  same approved tokenizer/data/actions and a seed-0 scratch memory arm. Memory uses eight optional tokens,
  rollout-only 32/16 mem2mem over 128-frame clips, 64-frame real TBPTT, the community shortcut objective,
  and explicitly no FF9 or archives. Production compute is exactly 48 effective high-utilization H100
  hours. Final evaluation is Merlin's paired interactive play beyond the 32-frame window. NEXT: audit the
  community source and write the n_memory=0 parity design before implementation.
- **2026-08-05 — Phases 1/2 implemented and locally gated; no server connection used.** Created branch
  `codex/memmaze-community-d4-mem2mem` from the immutable accepted-baseline ledger commit `1ae9cdc` and
  reconstructed upstream `b8abafbf` plus the accepted baseline patch in a disposable local checkout.
  Added an optional MEMORY modality whose only new `n_memory>0` state key is `memory_tokens`; its private
  RNG initialization occurs after all shared parameters. Added dedicated near-clean commit/write sampling,
  exact online `(episode,start,W)` tokenizer-window transport, rollout-only W=32/stride=16/L=128 training,
  per-sequence 50/50 latent-present vs memory-load-bearing modes, the community flow/bootstrap objective,
  and footprint-bounded TBPTT: four slides backwarded/released, boundary memory detached, final two slides
  backwarded, all scaled by 1/6 before one complete-clip optimizer step. Checkpointing preserves optimizer,
  scaler, every RNG, counter-keyed data position, active clock/ledger, W&B id, and periodic snapshots.
  Local gates passed: approved vanilla checkpoint strict load and output max-abs **0** (SHA-256
  `7b077938...d674`); seed-0 shared initialization exact; future-causality max-abs **0**; relay activation
  grad **5.209043e-06** vs deliberate-detach **0**; memory-prediction/write effects **0.01096/0.81896**;
  unique target accounting and segmented-vs-monolithic grad max-abs **0**; blank-after-eviction effect
  **0.004109**; interrupted-vs-uninterrupted two-step model/optimizer/RNG state exact. The unified pygame
  player loaded the approved production vanilla checkpoint and generated **33** finite frames across
  eviction at ~12.25 fps; a production-sized temporary `n_memory=8` smoke checkpoint exercised the same
  player's memory read/write/commit path for **33** finite frames at ~10.42 fps, and a recorded five-action
  trace replayed with the pinned episode/start/seed. Prepared H100 calibration/production drivers,
  data-identity gate, active-clock
  ledger, GPU-utilization gate, and paired action-trace record/replay. NEXT: finish delta/syntax review and
  commit the local implementation; then ferranti connection is required for the full converted-data gate
  and H100 calibration. No cluster job has been submitted; active production seconds remain **0**.
- **2026-08-05 — Local implementation committed.** Branch
  `codex/memmaze-community-d4-mem2mem`, implementation commit **`bf2c117`**. The accepted vanilla branch
  and durable player/checkpoints remain unmodified. NEXT / BLOCKER: push this branch, then Merlin must
  open the ferranti master connection so the exact converted-data identity gate and H100 calibration can
  run through the repository wrappers. Ferranti jobs: none; active production seconds: **0**.
- **2026-08-05 — Branch pushed; ferranti is now the blocker.** Implementation `bf2c117` plus provenance
  commit `098c255` are on `origin/codex/memmaze-community-d4-mem2mem`. All work possible without the
  academic server connection is complete. NEXT / BLOCKER: Merlin opens the ferranti master socket; then
  sync this exact branch through `scripts/sync_code.sh`, submit the short H100 calibration through
  `scripts/submit_job.sh`, and record its job ID. Ferranti jobs: none; active production seconds: **0**.
- **2026-08-06 — Ferranti connected; H100 calibration submitted.** The approved wrappers synced branch
  `codex/memmaze-community-d4-mem2mem` at exact commit
  **`9eec994d930460279309e1dcc1f49b23b8685f24`**. Ferranti job **429420** is the short calibration run
  `memmaze-community-d4-mem2mem-calib-b4`: one H100, batch size 4, 16 CPUs, four-hour outer allocation.
  It runs the full converted-data identity, vanilla parity/model, deterministic resume, checkpoint/resume,
  and active-interval GPU-utilization gates before any production submission. Active production seconds:
  **0**. NEXT: inspect calibration artifacts; freeze the batch/loader/TBPTT configuration only if all
  correctness and >=90% utilization health gates pass.
- **2026-08-06 — Calibration 429420 stopped at the data-identity schema gate; no training ran.** The
  baseline validator completed its full 2,902,900-train-frame / 1,001,000-eval-frame shard scan and proved
  content-disjoint splits, and setup/model gates passed. The stricter checker then failed because it
  expected manifest `target_size=64`; both approved manifests correctly store `target_size=null`, meaning
  no resize from native 64x64, while the baseline gate independently verifies every shard frame is 64x64.
  Pulled the manifests through `pull_file.sh`, pinned their exact SHA-256 values
  (`834c9b29e4436614694635826d570d3695542058e020487500691e8954ab673c` train;
  `3739484c11a87dca14c714b3b491e24e923f9a0a0c48c11cc4bf2e6950e62d20` eval), corrected the assertion,
  and validated the checker against both exact files locally. Ferranti job **429420**: FAILED before
  training; active production seconds: **0**. NEXT: commit/push/sync this gate fix and rerun calibration.
- **2026-08-06 — Corrected H100 calibration retry submitted.** Committed the exact-manifest schema fix as
  **`6d279a91500fa6c8d41409fff475796da53829db`**, pushed it, and synced that exact SHA through the approved
  wrapper. Ferranti job **429421**, run `memmaze-community-d4-mem2mem-calib-b4-r2`, retries with one H100,
  batch size 4, 16 CPUs, and a four-hour outer allocation. Active production seconds: **0**. NEXT: inspect
  all correctness, resume, HBM, throughput, and active-interval utilization results before freezing the
  production configuration.
- **2026-08-06 — Calibration 429421 exposed a CUDA-only resume bug after a healthy first segment.** All
  data/model gates passed, then batch-size-4 training completed **881** optimizer steps and
  **180.028626** active calibration seconds with finite losses/gradients, noncollapsed written-memory
  standard deviation, and a checkpoint at 90 seconds plus the segment-final checkpoint. Resume failed
  before another optimizer step because loading the checkpoint with `map_location=cuda` moved the saved
  CPU RNG `ByteTensor` to CUDA, which `torch.set_rng_state` rejects. Fixed `restore_rng` to normalize CPU,
  CUDA-global, and rollout-generator byte states to CPU before the generator APIs consume them. Extended
  `validate_resume.py` with the exact CUDA-map-location path; local RTX CUDA gate passes while preserving
  the prior exact two-step loss/model/optimizer/RNG equality. Ferranti jobs: **429420**, **429421** failed
  as documented; active production seconds: **0**. NEXT: commit/push/sync the resume fix and run a fresh
  calibration through the forced resume and telemetry health gate.
- **2026-08-06 — Clean post-fix calibration submitted.** CUDA-verified resume fix commit
  **`d10e17d28d341feee3a9e0301a08576ad6f3057a`** was pushed and synced exactly. Ferranti job **429431**,
  run `memmaze-community-d4-mem2mem-calib-b4-r3`, uses one H100, batch size 4, 16 CPUs, and a four-hour
  outer allocation. Active production seconds: **0**. NEXT: require the first 180-second segment, forced
  resume to 360 cumulative seconds, checkpoint consistency, and telemetry health gate all to pass.
- **2026-08-06 — Calibration 429431 passed correctness/resume but rejected batch size 4 utilization.**
  The run completed **1,780** optimizer steps and **360.041798** cumulative active calibration seconds.
  Forced restart was exact: first process stopped at step 897 / 180.194211 seconds; the second restored
  exactly that counter/clock and continued. Losses and gradients remained finite; written-memory standard
  deviation stayed noncollapsed. Telemetry hard-failed batch size 4: mean H100 utilization **76.4054%**,
  p05 **27%**, longest below 90% **230.022s**, mean/max HBM **10,989/11,337 MiB** of 81,559 MiB. Full logs,
  manifests, config, GPU samples, and telemetry summary were pulled to
  `experiments/memmaze-community-d4-mem2mem-calib-b4-r3/`. Active production seconds: **0**. NEXT:
  calibrate batch size **24** (sixfold batch with substantial measured HBM margin); freeze only if it fits,
  resumes, and clears the >=90% mean-utilization hard gate.
- **2026-08-06 — Batch-size-24 utilization calibration submitted.** Retained and committed the batch-4
  artifacts as `735ee38926b2d9c3dc738506e18bb4f7aea9b87b`, synced that exact SHA, and submitted ferranti job
  **429432**, run `memmaze-community-d4-mem2mem-calib-b24`: one H100, batch size 24, 16 CPUs, four-hour
  outer allocation. Active production seconds: **0**. NEXT: inspect OOM/HBM, forced-resume exactness,
  throughput, and active-interval utilization; freeze bs24 only if every gate passes.
- **2026-08-06 — H100 calibration passed and production configuration frozen.** Ferranti job **429432**
  completed exit 0 after 13m56s scheduler elapsed: **695** steps / **360.055772** active seconds, exact
  resume from step 350 / 180.337252 seconds, mean utilization **98.75%**, p05 **97%**, no sample below
  90%, mean/max HBM **54,625.78/54,843 MiB** of 81,559 MiB, and finite/noncollapsed training throughout.
  Passing logs, GPU samples, exact active-clock ledger (SHA-256
  `75ca1451c83da9441143e6f1b736576e6f42f443347fd82ebeee15c30426305a`), config, checksums, and gates are
  retained under `experiments/memmaze-community-d4-mem2mem-calib-b24/`. Frozen production is bs24,
  workers4, cache128 MiB/worker, seed0, W32/stride16/L128, TBPTT64, M8, k_max8, bootstrap step5000,
  self-fraction0.25, AdamW lr1e-4/wd1e-2, grad clip1, hourly active-time checkpoints. Calibration's
  active dynamics work occupied **47.8410%** of its training wall span, projecting **100.3323 wall hours**
  for 48 active hours; the outer scheduler request is therefore frozen at **120h**. Added a hard final
  checkpoint validator and separate `memory-latest.pt`/`memory-final.pt` semantics, so an early scheduler
  stop cannot be mislabeled complete; synthetic complete/partial acceptance/rejection and bash syntax
  gates pass. Active production seconds: **0**. NEXT: commit/push/sync the frozen production driver and
  submit the 48-active-hour run.
- **2026-08-06 — 48-active-hour production run submitted.** Frozen production commit
  **`0fb8759a41dd237d1daf0a3ef92fb847a4aeb909`** was pushed and synced exactly. Ferranti job **429437**,
  run `memmaze-community-d4-mem2mem-48h`, has one H100, 16 CPUs, and a **120-hour** outer allocation; the
  calibrated trainer owns the exact **172,800 active-second** stop. It starts seed0 from scratch, writes
  `memory-latest.pt` plus hourly active-time snapshots, and can create `memory-final.pt` only after the
  hard active-clock/config/exposure gate passes. Active production seconds at submission: **0**. NEXT:
  require full startup identity gates, advancing finite optimizer steps, >=90% active-interval utilization,
  bounded HBM/host RSS, and healthy transition through shortcut bootstrap step 5,000.
- **2026-08-06 — Initial production request 429437 cancelled before start: partition time limit.**
  Ferranti left the 120-hour request pending with reason `PartitionTimeLimit`; it consumed **0** production
  active seconds and was cancelled through the ownership-guarded wrapper. The calibrated projection is
  100.3323 wall hours, while a 54-hour request is known accepted from the completed vanilla run. Frozen
  scheduling is therefore corrected to **two resumable 54-hour allocations** with the same run name and
  hourly `memory-latest.pt`; their calibrated combined capacity is about 51.67 active hours, leaving
  roughly 3.67 active hours of margin around the 48-hour target. The hard final gate prevents allocation
  1 from being called complete. NEXT: commit/push/sync the corrected scheduler provenance and submit
  allocation 1. Production active seconds: **0**.
- **2026-08-06 — Production allocation 1 submitted.** Corrected scheduling commit
  **`0963308c07e0b081f902140b2ab2a392e924f9cb`** was pushed and synced exactly. Ferranti job **429438**,
  run `memmaze-community-d4-mem2mem-48h`, requests one H100, 16 CPUs, and the accepted 54-hour envelope.
  It uses the frozen bs24 configuration and the shared resumable `memory-latest.pt`/active ledger that
  allocation 2 will continue. Active production seconds at submission: **0**. NEXT: require all startup
  identity/model gates, advancing finite training, calibrated utilization/HBM, bounded host RSS, first
  hourly checkpoint, and healthy bootstrap activation at optimizer step 5,000.
- **2026-08-06 — Production startup healthy through step 246.** Allocation 1 job **429438** is RUNNING
  on ferranti node `mlcbm010` from exact production code
  **`0963308c07e0b081f902140b2ab2a392e924f9cb`** (submission ledger
  **`783aff68e4a17dd7743a43db71de722d853f35df`**). The full setup, 3,903,900-frame conversion scan,
  train/eval disjointness, exact tokenizer/manifests, vanilla parity, shared initialization, causality,
  relay-gradient, distinct-channel, loss-accounting, and beyond-eviction gates all passed again on the
  production node. Training started from seed 0 with 42,646,592 parameters, 2,531,700 valid sequences,
  and W&B run `gh3n2zlg`. Loss remained finite and fell from **0.061322** at step 1 to **0.012850** at
  step 220; written-memory standard deviation remained noncollapsed (**1.95** at step 220), and gradients
  remained finite. Bootstrap loss is correctly zero before its frozen step-5,000 activation. The exact
  active-clock snapshot is step **246 / 127.120510 seconds**. Production-start telemetry over 12 active
  samples passed both gates: mean utilization **98.5833%**, p05 **96%**, longest sustained below 90%
  **0s**, mean/max HBM **54,383/54,383 MiB** of 81,559 MiB. Live `sacct` does not yet report batch-step
  MaxRSS; retain final scheduler MaxRSS when the allocation closes. NEXT: verify the first hourly
  active-time checkpoint/resume artifact and the healthy transition through shortcut bootstrap step
  **5,000**; allocation 2 remains required after this 54-hour envelope unless the audited 172,800-second
  target has already been reached.
- **2026-08-07 — Online run cancelled; exact latent-cache and fair-clock correction committed.** Inspection
  of the accepted vanilla trainer proved that its timer starts immediately before DataLoader iteration and
  counts loader waits, frozen-tokenizer encoding, optimization, logging, and periodic checkpoint time. The
  mem2mem trainer's optimizer-only clock therefore was not compute-comparable, and repeated encoding of
  the same frozen-tokenizer windows was avoidable. Ferranti job **429438** was cancelled through the
  ownership-guarded wrapper after **04:03:56** scheduler elapsed (Slurm state `CANCELLED`, batch exit
  `0:15`, MaxRSS **9,718,628 KiB**); its partial weights will not be used. Correction commit
  **`d4cca6800f5fd1eaa8f58ea1706e09cedc5edd7e`** adds a resumable exact FP32 cache for all **2,810,100**
  independent W=32 windows (`(2810100,32,8,64)`, **171.515 GiB**), a pixel-free cached training path,
  cache identity/equality gates, and the vanilla-matched cumulative 172,800-second training-loop clock.
  A local two-episode cache passed interrupt/resume, full hash, sampled online equality
  (`max_abs=0`), seven-window clip lookup, cached training, exact checkpoint resume, and finalization-only
  recovery after a simulated interrupted hash. NEXT: push/sync the correction, build the full durable
  cache on ferranti, pin its manifest SHA-256, run cached-input H100 calibration, then start a fresh seed-0
  48-hour run in one 54-hour allocation.
- **2026-08-07 — Full exact latent-cache build submitted.** Correction/provenance SHA
  **`70b278f8f8efd0a6742ec04f00837b8e3d04ac03`** was pushed and synced exactly through the approved
  wrapper. Ferranti job **429610**, run `memmaze-community-d4-window-cache-v1`, requests one H100,
  16 CPUs, and 12 hours. It validates the approved conversion/tokenizer, writes the durable resumable
  171.515 GiB FP32 window cache outside the dynamics budget, computes its complete SHA-256, and runs
  sampled bit-exact online equality plus long-clip lookup gates. Initial scheduler state is `PENDING`;
  cumulative production training-loop seconds remain **0**. NEXT: monitor 429610 to completion, retain
  its manifest/validation artifacts, pin the manifest SHA-256 in the production driver, and run cached-I/O
  H100 calibration.
- **2026-08-07 — Cache startup path fixed; corrected build submitted.** Job **429610** failed before
  encoding because the new drivers requested nonexistent `manifest.json`; the approved converter writes
  `conversion_manifest.json`. All setup, conversion, model, and data-identity gates had passed, and no
  cache rows or H100 training budget were consumed. Fix **`bd4e40dff3e3660452aeb247ee2e4c5ac299a2f6`**
  updates build, calibration, and production consistently and was pushed/synced exactly. Corrected
  ferranti job **429612**, run `memmaze-community-d4-window-cache-v2`, requests one H100, 16 CPUs, and
  12 hours. Cumulative production training-loop seconds remain **0**. NEXT: monitor 429612 through cache
  construction/hash/equality gates and pin the completed manifest.
- **2026-08-08 — Full cache encoded/hashed; H100 batch-numerics diagnostic submitted.** Ferranti job
  **429612** encoded all **2,810,100 / 2,810,100** windows at **307.2 windows/s** and finalized the
  `(2810100,32,8,64)` FP32 array. Data SHA-256 is
  **`48d9a9f4bd504f36e5ea81af2f35089e92dff4f1260d6ca1b1dd811a8ea53cb8`**; row-map SHA-256 is
  **`1ec521e31facf6cb8fd2404e66122d28bf6b6ff3e5732ce218332cabf26fc861`**; current manifest SHA-256 is
  **`e7a4e57e63e357d1986154a1b6c3cea9f4220b1a716e4a553df8a345fb2f4fcf`**. The job then failed only
  because its validator demanded bit equality between batch-64 cache construction and batch-1 online
  H100 encoding. Rather than weakening that assertion blindly, diagnostic commit
  **`a8fb01c5805456cb2241304c2df1571875e347cd`** adds per-row absolute/RMSE/relative/cosine measurements
  plus same-batch-64 replay, which should distinguish harmless GEMM batch-shape rounding from a bad cache
  write. It passed against the local cache (same-builder-batch replay max-abs 0). Ferranti diagnostic job
  **430087**, run `memmaze-community-d4-window-cache-numerics`, requests one H100/16 CPUs/1h. Cumulative
  production training-loop seconds remain **0**. NEXT: freeze an evidence-based numerical gate from
  430087; accept the completed cache only if same-batch replay and singleton error support it.
- **2026-08-08 — Exact cache accepted and pinned.** H100 diagnostics **430087** and **430088** proved
  builder batch-64 replay is bit-exact (`max_abs=0`) and, more importantly, batch-24 re-encoding—the
  actual mem2mem online reference—is also bit-exact across sampled groups (`max_abs=0`). Batch-1 max
  absolute error was **0.00061658** with max relative L2 **0.00019420**; accepted-vanilla batch-128 was
  exact for full groups and its final partial group had max absolute **0.00085971** / relative L2
  **0.00018268**. These are batch-shape GEMM rounding, not cache corruption. Commit
  **`ff089c5e83574fcfd0eb5d75bf247ab9d37dd835`** pins manifest
  `e7a4e57e63e357d1986154a1b6c3cea9f4220b1a716e4a553df8a345fb2f4fcf`, requires exact batch-64 and
  batch-24 replay, bounds alternate batch shapes at max-abs 0.002 / relative-L2 0.0005, and retains the
  small build/diagnostic artifacts locally. The 171.515 GiB array remains durable on Weka and was not
  pulled. Cumulative production training-loop seconds remain **0**. NEXT: sync the pinned-cache commit
  and run the forced-resume cached-I/O H100 calibration before freezing batch size/utilization.
- **2026-08-08 — Production-shaped cached calibration submitted.** Accepted-cache ledger SHA
  **`2a8f10c05e143b4e1d648f6998c47f9bc4eb12ee`** was pushed and synced exactly. Ferranti job
  **430089**, run `memmaze-community-d4-mem2mem-cached-calib-b24`, requests one H100, 16 CPUs, and two
  hours. It uses batch 24/four mmap workers, the pinned durable cache, 180 seconds before a forced process
  restart, and 360 cumulative vanilla-matched training-loop seconds. The job re-runs model/resume/cache
  gates and must pass utilization, HBM, finite-loss/gradient, noncollapsed-memory, clock-ledger, and exact
  checkpoint-resume checks before production is frozen. Cumulative production training-loop seconds remain
  **0**. NEXT: inspect 430089 and freeze/launch only if every cached-input gate passes.
- **2026-08-08 — Cached H100 calibration passed; production configuration frozen.** Ferranti job
  **430089** completed exit 0 after 6m56s scheduler elapsed. It performed **685** optimizer steps in
  **360.181498** cumulative whole-loop seconds (**1.90182 steps/s**). Forced restart was exact: process 1
  stopped at step **343 / 180.120748s**, and process 2 restored exactly that step/clock before continuing.
  Losses and gradients stayed finite; written-memory standard deviation remained noncollapsed. GPU health
  passed: mean utilization **96.9167%**, p05 **91%**, longest sub-90% interval **0s**, mean/max HBM
  **52,950/52,991 MiB** of 81,559 MiB; scheduler MaxRSS **9,124,100 KiB**. Clock-ledger SHA-256 is
  **`89d4bc67abda36047f7784ce63539b5f23363f410424e65ac94810f1699a8d98`** and calibration-checkpoint
  SHA-256 is **`b7dc41112d7c9fc4b5abed93aabff5472bbc57c38e4ef6b6998bf5aad3f7a80a`**. Freeze commit
  **`cc5327432a3ef908d66e902a11a2df96c2c380d6`** retains all small artifacts and locks batch24/workers4,
  exact cache manifest, the 48-hour LR schedule, one 54-hour allocation, and the vanilla-matched
  172,800-second whole-loop stop. Cumulative production training-loop seconds remain **0**. NEXT: sync
  the frozen commit and submit the fresh seed-0 production run under a new cached run name.
- **2026-08-08 — Fresh cached 48-hour production run launched.** Passing-calibration ledger SHA
  **`1e49c2ddb3632d4b63839cf0d12f0d9b17257d97`** was pushed and synced exactly. Ferranti job
  **430090**, run `memmaze-community-d4-mem2mem-cached-48h`, is `RUNNING` with one H100, 16 CPUs, and
  one accepted **54-hour** scheduler allocation. It starts seed 0 from scratch under the new run name,
  reads only the pinned exact latent cache during training, uses the frozen batch24/workers4 configuration,
  and stops after exactly **172,800 cumulative vanilla-matched training-loop seconds**; cache/setup/full
  hash and final serialization remain outside that clock. `memory-final.pt` can be created only after the
  hard config/cache/exposure/time gate passes. Production training-loop seconds at submission: **0**.
  NEXT: verify startup identity/full-cache-hash gates, first advancing optimizer steps and utilization,
  first hourly checkpoint, then healthy shortcut-bootstrap activation at step 5,000.
- **2026-08-08 — Fresh production startup and full-cache integrity passed.** Job **430090** completed the
  full 171.515 GiB SHA-256 scan plus exact manifest, batch64 replay, batch24 online-reference equality,
  bounded alternate-batch numerics, data identity, vanilla parity, shared initialization, causality,
  relay-gradient, distinct-channel, loss-accounting, and beyond-eviction gates. W&B run is **`ai5rh826`**.
  Fresh seed-0 training then advanced through step **280 / ~146.45 cumulative whole-loop seconds**;
  loss/gradients remained finite and written-memory standard deviation noncollapsed. Bootstrap loss is
  correctly zero before frozen step 5,000. NEXT: retain live utilization/HBM health, verify the first
  hourly checkpoint, and inspect the transition through bootstrap step 5,000.
- **2026-08-09 — Fresh production reached 74.6% of the exact training clock.** Job **430090** remains
  `RUNNING` after **1-11:51:24** scheduler elapsed and reached step **226,140** at
  **35.790674 / 48.000000 cumulative whole-loop hours**. Recent loss, flow loss, bootstrap loss, gradients,
  and written-memory standard deviation are finite and healthy (`loss=0.001089`, `flow=0.006135`,
  `boot=0.000675`, `grad=0.0083`, `mem_std=3.9342`). The frozen bootstrap boundary behaved exactly as
  intended: bootstrap loss was zero at steps 4,980 and 5,000 and became nonzero at step 5,020. Hourly
  checkpoints are advancing normally; the latest verified one is step **221,140** at **126,000.44s**.
  About **12.21 training hours** remain, with roughly **5.9 hours** of scheduler headroom beyond the
  projected finish. NEXT: monitor through the exact 172,800-second stop, final checkpoint gates, artifact
  pull, and player/evaluation work.
- **2026-08-09 — Immutable 35-hour checkpoint pulled for optional mid-run play.** Without pausing or
  mutating job **430090**, the immutable hourly snapshot at step **221,140 / 126,000.438123s** was pulled
  through the approved wrapper. It is **514,488,117 bytes** with local SHA-256
  **`ca6e7fb3ba17a16b94ddd55fe1734863abbbb3b450c91eddbad057537fc453bd`** and retains `n_memory=8` plus
  the pinned cache-manifest SHA. A Windows portability defect exposed by the real cluster checkpoint
  (`pathlib.PosixPath` in saved arguments) was fixed in the player loader. The actual snapshot then passed
  a **33-generated-frame** CUDA self-test across eviction at **14.90 fps** using a separate pinned/patched
  upstream checkout; the accepted vanilla installation remained untouched. This checkpoint is for
  provisional interactive inspection only and does not replace the exact 48-hour final artifact. A
  post-transfer check found the production run still healthy and advancing at step **247,160 / 39.115272h**;
  the pull caused no observable interruption.
- **2026-08-10 — Exact 48-hour production checkpoint completed, pulled, and locally gated.** Ferranti job
  **430090** completed exit `0:0` after **2-00:04:13** scheduler elapsed. The trainer stopped at step
  **303,248** after **172,800.299007 cumulative whole-loop seconds** and passed the final production
  config/cache/exposure/time gate. GPU health also passed: mean utilization **95.8517%**, p05 **90%**,
  longest sub-90% interval **30.003s**, and mean/max HBM **52,910.97/52,911 MiB** of 81,559 MiB. The
  immutable `memory-final.pt` is **514,477,149 bytes** with cluster and local SHA-256
  **`aff682549e03616e30d46d83e19bfe79b69aa60fb1bfe714ae225fb98872ab93`**. It was installed beside the
  accepted vanilla checkpoint as `dynamics-mem2mem-final.pt` without modifying the vanilla artifact.
  The actual final model passed a 33-generated-frame CUDA player self-test across eviction at **12.18
  fps**. Compact final summaries are retained under `experiments/memmaze-community-d4-mem2mem-48h/`;
  the 64.3 MB exact clock ledger remains local/ignored and the 171.515 GiB cache was not pulled. NEXT:
  Merlin's paired interactive vanilla-vs-memory assessment and recording the qualitative outcome.
