# Community Dreamer 4 with memory tokens and mem2mem rollout training

## Goal

Extend the independent community Dreamer 4 Memory Maze baseline with the smallest faithful version of
our memory-token idea, train it with long contiguous mem2mem rollout training and truncated
backpropagation through time (TBPTT), and let Merlin compare it interactively against the completed
community vanilla model.

This is the externally grounded comparison we actually want:

- **Vanilla:** the completed community Dreamer 4 baseline.
- **Memory:** the same community implementation, tokenizer, data, actions, transformer dimensions,
  shortcut objective, optimizer family, seed, and effective H100 budget, with only:
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
- Vanilla production job: ferranti **423141**, step 298,164, exactly 172,800.10 seconds of active
  training, seed 0.
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
and call that “the same tokenized data.” Any cache construction happens before the 48-hour dynamics clock
and its manifest must pin the tokenizer hash, source conversion hash, window, dtype, and action alignment.

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
9. **Resume determinism:** checkpoint/resume preserves model, optimizer, active-training clock, shortcut
   bootstrap state, data position/seed, and TBPTT configuration without repeating or skipping effective
   training budget.
10. **Player gate:** vanilla and memory checkpoints both load through the same pygame player; scripted
    action self-tests run beyond 32 generated frames, cross window eviction, stay finite, and never read
    recorded future frames.

## Training and compute contract

- Cluster: ferranti, **one H100**.
- Primary production budget: exactly **48 hours = 172,800 seconds of effective H100 training**.
- The active clock advances only during forward/backward/optimizer rollout training. Repository setup,
  data validation/staging, cache construction, compilation/warmup, checkpoint serialization, telemetry,
  evaluation, queueing, and preemption downtime do not consume the 48-hour budget.
- GPU telemetry is mandatory at fine cadence. During counted active segments require sustained high H100
  use: target mean utilization >=95%, hard health gate >=90%, with no unexplained idle interval over five
  minutes. A nominal 48-hour wall clock with low utilization does **not** satisfy the task.
- If utilization is below the gate, profile/fix loader, encoding, rollout batching, or TBPTT scheduling
  before production. Do not compensate by quietly counting idle hours.
- Use resumable periodic checkpoints and an outer scheduler allocation with enough setup/finalization
  margin. Completion is determined by the trainer's audited active clock, not Slurm elapsed time or epoch
  count.
- Match vanilla optimizer hyperparameters and elapsed-time LR schedule. Memory rollout batch size may be
  smaller for HBM, but record batch size, slides/step, target frames/step, optimizer steps, unique/source
  frames read, total scored frames, and achieved utilization so the compute-matched comparison is honest.
- The primary comparison is compute-matched at 48 effective H100 hours. Do not extend the memory run to
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

- Calibrate long-clip batch size, loader workers/cache, online/window encoding strategy, and TBPTT segment
  scheduling on an H100.
- Demonstrate sustained utilization and bounded host/HBM memory through multiple TBPTT boundaries and at
  least one checkpoint save/resume.
- Freeze the production configuration before the main run. Do not tune against held-out interactive
  behavior.

### Phase 4 — 48-effective-hour memory training

- Train from seed-0 scratch for exactly 172,800 counted active seconds.
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
- active-clock ledger, scheduler accounting, optimizer/scored-frame exposure, resume events;
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
  initialization, community shortcut objective, and **48 hours of verified high-utilization effective H100
  training**.
- The final memory checkpoint is pulled, hash-verified, durably installed beside vanilla, and both models
  run through the same interactive player beyond window eviction.
- Merlin completed the paired interactive comparison and the outcome—positive, negative, or mixed—is
  recorded without overclaiming what subjective play establishes.

## Progress

Maintain this section throughout the task. After each meaningful transition, append a dated entry with
the exact branch/commit, ferranti job IDs, active training seconds, utilization, artifacts, completed
gates, and next blocker. Do not reconstruct status later from chat or scheduler history.

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
