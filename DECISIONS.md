# DECISIONS.md — append-only decision log

> Entries D-001 through D-008 were **backfilled on 2026-06-11** from Merlin's
> journal, git history, file timestamps, and W&B run metadata. Dates marked `~` are
> approximate. Backfilled entries reconstruct intent honestly but were not written
> before the work, unlike everything from D-009 onward. Never edit or delete a past
> entry; corrections are new entries referencing the old ID.

## D-001 | ~2026-05-02 (backfilled)
Context: Project start. Goal is a DreamerV4-style world-model pipeline; need
working transformer infrastructure first.
Decision: Implement a minimal char-level transformer LM in PyTorch (`src/A_LM`),
no extras.
Alternatives rejected: Starting directly with video models — too many moving parts
to debug at once.
Expected outcome: Coherent character-level text generation on Shakespeare.
Would change my mind: n/a (infrastructure stepping stone).
Spawns: EXP-001

## D-002 | ~2026-05-10 (backfilled)
Context: Read the DreamerV4 paper. It attributes training stability to specific
architecture components.
Decision: Extend the transformer blocks with RoPE (temporal axis), RMSNorm, SwiGLU,
QK-norm, and attention logit soft-capping, per the paper. These blocks become the
shared basis for tokenizer and dynamics model.
Alternatives rejected: Vanilla blocks until problems appear — rejected because the
point is to replicate the paper's recipe.
Expected outcome: Stable training at small scale; reusable components.
Would change my mind: Divergence or instability traceable to one of these
components.
Spawns: code changes in `src/A_LM/model.py` (later reused in B, C, D)

## D-003 | ~2026-05-17 (backfilled)
Context: Architecture components in place. Need a video testbed and the tokenizer
half of the pipeline.
Decision: (a) Generate BouncingBall dataset (`bouncing.npy`, 2026-05-17): DVD-style
ball on black background. (b) Implement single-image autoencoder (B) as baseline:
image → patch tokens → attention → 4-latent bottleneck → decoder. (c) Extend to
temporal autoencoder (C): alternating spatial/temporal layers (every 4th layer
temporal, causal), learned position embeddings spatially, RoPE temporally, MAE
patch dropout, restricted latent↔patch attention to force the bottleneck.
Alternatives rejected: CNN encoder (not the paper's design); real video data (too
slow to iterate).
Expected outcome: Faithful reconstruction through the 4×64 bottleneck.
Would change my mind: Bottleneck information provably insufficient for recon.
Spawns: EXP-002, EXP-003

## D-004 | 2026-05-31 (backfilled)
Context: EXP-003 failed — latent collapse. Black background + ~5%-area ball is a
pathological MSE optimum: predicting all-black scores well, latents collapse
(pairwise cosine similarity near 1). Commit 60a4b67 "still has MAE latent collapse".
Decision: Build a new environment, OccludedBouncing / "CurtainsEnv"
(`src/data_generators/occluded_bouncing.py`): dense per-episode gradient
backgrounds (removes the all-black optimum), bright ball, and two **absolute**
curtain actions (0 = reveal, 1 = occlude whole frame). The curtain makes it a
*memory* environment — ball physics continue behind the curtain — so the same env
later serves H2/H3. Assorted training tweaks were applied alongside (not
individually recorded — provenance gap, acknowledged).
Alternatives rejected: Tuning MAE rate / bottleneck size on the black env — treats
the symptom; the data distribution is the cause.
Expected outcome: No collapse on dense backgrounds; latent_cos low.
Would change my mind: Collapse persisting despite dense backgrounds.
Spawns: `occluded.npy` (2026-05-31), EXP-004

## D-005 | ~2026-06-01 (backfilled)
Context: Tokenizer (C) reconstructing acceptably. Second half of pipeline needed.
Decision: Implement dynamics model (D) with shortcut forcing per the paper:
per-frame (τ, d) sampling, x-prediction, bootstrap distillation for coarse steps,
ramp weight w(τ); token layout [action | latents | registers | shortcut]. Train
unconditionally on `bouncing.npy` over the frozen C tokenizer.
Alternatives rejected: Plain next-latent regression (not the paper; no few-step
sampling).
Expected outcome: Plausible short rollouts on the simple bouncing data.
Would change my mind: Rollouts no better than copying the last context frame.
Spawns: EXP-005 (`dynamics_bouncing.pt`, 2026-06-01)

## D-006 | 2026-06-09 (backfilled)
Context: Training plateaued (commit 7cb30c1 "fixed plateu issue"); no experiment
tracking existed — results lived in terminal scrollback.
Decision: Integrate W&B via `src/wlog.py` (no-op unless --wandb); instrument
latent_cos (collapse detector), pred_std, train/val MSE, perf counters; fix the
LR/plateau issue.
Alternatives rejected: Continuing with print-log archaeology.
Expected outcome: Comparable runs; plateau broken.
Would change my mind: n/a (observability).
Spawns: iteration runs of 2026-06-09/10 (see EXP-004 notes), prerequisite for
EXP-006/007

## D-007 | 2026-06-10 (backfilled)
Context: Local RTX 4070 too slow for 100-epoch tokenizer runs; reconstructions not
sharp enough to trust latents as a dynamics substrate.
Decision: (a) More patches — smaller patch size (commit 0ac0daa). (b) Add VGG LPIPS
perceptual loss, **normalized relative to the MSE loss** (commit 3205e8e); VGG
chosen over AlexNet for more learnable gradients. (c) Move heavy training to the
university H100 clusters (manual SSH by Merlin for now; wrapper scripts pending,
T-003). A/B the LPIPS decision with two long cluster runs.
Alternatives rejected: AlexNet LPIPS (weaker gradients in practice); unnormalized
LPIPS (scale mismatch with MSE).
Expected outcome: LPIPS run beats non-LPIPS on val/mse and visual sharpness.
Would change my mind: LPIPS run worse or equal on both.
Spawns: EXP-006

## D-008 | 2026-06-10 (backfilled)
Context: EXP-006 done; LPIPS tokenizer adopted and frozen
(`trained_autoencoder.pt` = product of W&B run rc01geau).
Decision: Train the dynamics model action-conditioned (n_actions=2, one action
token per frame) on `occluded.npy` over the frozen LPIPS tokenizer; 100 epochs on
the cluster.
Alternatives rejected: Staying unconditional on bouncing — cannot test memory.
Expected outcome: Rollouts preserve ball color and background always, and ball
position when context contains ≥2 visible frames.
Would change my mind: Rollouts failing to preserve per-episode constants (ball
color, background) even with fully visible context.
Spawns: EXP-007 (`my_dynamics.pt`, W&B sm0kr1cf)

## D-009 | 2026-06-11
Context: EXP-007 reconciliation: val/loss healthy (1.93e-3) but decoded rollouts
randomize ball color and position from the first generated frame; background is
preserved; decoding random bottleneck tokens yields *no* ball, so the model has
learned "predict latents containing some ball" but not which or where. The D-008
tripwire is triggered. Candidate causes, mutually compatible: (a) shortcut-forcing
objective bug or self-chasing bootstrap; (b) tokenizer latent geometry — temporally
adjacent frames may map to distant latents, making dynamics ill-conditioned;
(c) undertraining / capacity. Escalated as ESC-001; Merlin's verdict: diagnose the
dynamics failure first, probe suite second. Simultaneously: the orchestrator
protocol (`.claude/agents/research-orchestrator.md`) is adopted and prior history
backfilled into these state files.
Decision: Next work is diagnosis task T-001, designed to discriminate (a)/(b)/(c):
measure latent-space continuity of the frozen tokenizer (distance between latents
of temporally adjacent frames vs. random pairs), verify the shortcut-forcing
implementation against the paper's Eq. 7, and run context/teacher-forcing ablations
on the trained model. Probe suite (T-002) is built after the diagnosis verdict.
Alternatives rejected: Building the probe suite first (protocol §8 default) —
rejected by Merlin because a broken dynamics model blocks everything the probe
suite would measure. Jumping straight to H3 method work — rejected: H1 baseline
must function first, and baselines are sacred.
Expected outcome: One of (a)/(b)/(c) clearly implicated. Prediction: if (b), the
fix lives in the tokenizer objective — which is exactly the H3 research direction
arriving early.
Would change my mind: Evidence that val/loss is miscomputed or leaks context — that
would reframe the entire reconciliation of EXP-007.
Spawns: T-001 (diagnosis), T-002 (probe suite, queued), T-003 (cluster wrappers),
T-004 (pre-register H2/H3 criteria)

## D-010 | 2026-06-11
Context: Cold-start code read of `src/D_dynamics_model/dynamics_model.py` while
scoping T-001's candidate (a) (shortcut-forcing/inference bug). Found a specific,
mechanism-level candidate for the EXP-007 rollout failure in the rollout
context-noising. The codebase convention is **tau = signal level** (loss:
`z_tilde = (1-tau)*noise + tau*z1`; tau=1 clean, tau=0 noise — verified internally
consistent with `_denoise_next`'s sampling loop). But `_denoise_next` builds the
context as `ctx_noised = (1 - tau_ctx)*randn + tau_ctx*context` with
`config.context_noise = tau_ctx = 0.1`, i.e. **0.9*noise + 0.1*context = 90% noise
on the context frames**. The comment intends "lightly corrupt the clean context";
the implementation nearly destroys it. The model is told (tau_col=round(0.1*128)=13)
the context is at signal 0.1, so it is self-consistent, but there is almost no
ball color/position information left to condition on → it falls back to its prior
and emits a plausible ball at a random color/position. This matches ALL EXP-007
symptoms: healthy val/loss (training never exercises the rollout context-noising
path), ball randomized from the first generated frame even with fully-visible
context, background roughly preserved (large low-frequency signal survives), and
the random-latent control (decoder emits no ball from random latents — so the model
genuinely learned the ball manifold, just isn't being given the conditioning info).
If correct, this is an **inference-only bug: a one-line fix, no retraining**.
Decision: First diagnostic (smallest, most decisive — protocol §3/§5) is an
inference-only sweep of `context_noise` on the *existing* `my_dynamics.pt` over the
frozen `trained_autoencoder.pt` tokenizer, decoding rollouts to images and pixel-MSE
vs ground truth. Sweep tau_ctx ∈ {0.1 (current/broken), 0.5, 0.9, 0.99}. Implemented
by worker T-001 (builds + smoke-tests the headless script); I run the full sweep as
EXP-008 and reconcile. This is candidate (a) sharpened — within the D-009 plan, not
a deviation. Latent-geometry (b) and undertraining (c) remain queued follow-ups,
run only if the context-noise fix does NOT restore ball identity.
Alternatives rejected: (1) Running the broader (a)/(b)/(c) battery first — rejected:
this single test is cheaper and, per the symptom-match, far more likely; do the
decisive cheap test before the expensive broad one. (2) Just patching the line and
re-running without a controlled sweep — rejected: I need the broken-vs-fixed
comparison as evidence, not a silent fix (a clean A/B is the artifact).
Expected outcome: At tau_ctx≈0.9–0.99 the decoded rollout preserves ball color and
approximate position for the first several generated frames and pixel-MSE-vs-GT
drops sharply relative to tau_ctx=0.1; at 0.1 the failure reproduces. Concretely:
visible ball at correct color in ≥1 of the first 2 generated frames at high tau_ctx,
clearly absent/random at 0.1.
Would change my mind: If high tau_ctx does NOT restore ball identity (ball still
random with near-clean context), the context-noising is not the (sole) cause →
escalate and open the (b)/(c) threads. Also: if even tau_ctx=0.99 rollouts diverge
within 1-2 frames, suspect latent geometry / error accumulation, not conditioning.
Spawns: T-001 (rescoped to the context-noise diagnostic), EXP-008 (the sweep run)

## D-011 | 2026-06-12 (milestone)
Context: ESC-002 resolved — Merlin agrees the context-noise fix closes the EXP-007
dynamics failure and **completes H1**. He called a milestone to plan Phase 2 and
explicitly said: don't mechanically continue the backlog, reevaluate. During that
conversation he corrected two architecture misunderstandings of mine, which reshape
H2/H3:
  1. **M<N inference needs no retrain.** RoPE is relative; running the model at a
     shorter context window than it was trained on is free. (I had wrongly planned a
     "retrain to shorten the window" task — dropped.)
  2. **A sliding-window transformer has no persistent state.** I had framed H2 as
     "never trained to carry across the window boundary" — wrong. There is no
     boundary and nothing to carry: each step sees N−1 frames, predicts next, drops
     the oldest; RoPE-relative so step 1000 == step N. Info older than N−1 frames is
     simply absent from the model (not in input, not stored). Confirmed the register
     tokens are per-frame scratch, not a cross-window carrier. So the dynamics
     `generate()` (line ~391, `window = seq[:, -max_ctx:]`) *already* slides the
     window — the beyond-window regime is reachable today with NO new architecture
     and NO retrain, just by rolling out longer than the window.
Also established: KV caching is real, missing work but is **efficiency, not a
prerequisite** for measuring memory. And a careful flag (Merlin): once KV cache
exists, the current fixed `cos/sin` RoPE table (size `max_temporal_length`) is
cache-incompatible — cached K/V can't be re-rotated when the window slides, so the
cache requires a **continuously-advancing absolute position / on-the-fly rotation**,
never reset. Captured in `HOWTO/rope_kv_cache_caveat.md`.
H3 reframed by Merlin: it is an **open-ended end-goal**, not one pre-registered
hypothesis — "force the encoder and/or dynamics model to include hidden info in the
latent space; how is up in the air; we'll try many things and keep what sticks."
Working intuition: the autoregressive latent chain is the carrier (see GOAL H3).
Decision: Proceed to Phase 2 (H2) on the **cheap-signal-first** path I proposed and
Merlin endorsed ("you have enough; continue as you see fit"):
  (a) Apply the agreed cleanup: rename `context_noise`→`context_signal`, fix the
      misleading comment, set default 0.9 (keep tau=signal-level convention,
      consistent with the loss). Inference-only; no retrain.
  (b) Build & freeze the revisit-consistency probe suite (T-002) on the EXISTING
      frozen tokenizer + `my_dynamics.pt`, using a chosen inference window N and
      rolling out occlusion length k spanning below→above N. Primary metric:
      latent-token MSE (predicted reveal latent vs frozen-tokenizer GT latent),
      validated against a pixel-space color/position decomposition. Include controls:
      chance floor (no-context/random latent), ceiling (fully visible), no-occlusion
      drift control to difference out ordinary autoregressive drift.
  (c) Pre-register H2 criteria (T-004) against the calibration controls, before
      reading the H2 result.
  (d) Measure the H2 baseline → present-then-stop (§5).
T-003 (cluster wrappers) and KV cache are deferred to when H3 method work needs heavy
training / long horizons. Shorten the window at **inference** only — frozen tokenizer
untouched, H1 baseline preserved.
Alternatives rejected: (1) Architecture-first (KV cache + short-window retrain before
probing) — rejected: the sliding window already exists and M<N is free, so retrain
buys nothing now; do the decisive cheap measurement first (cf. D-010). (2) Shortening
the tokenizer's temporal length — rejected: un-freezes H1's tokenizer for no benefit;
memory is tested in the dynamics chain, where the tokenizer just encodes a blank
curtain frame during occlusion.
Expected outcome: latent-MSE recall of color/position decays toward the chance floor
as occlusion length k exceeds the window N, cleanly above the no-occlusion drift
control. If latent-MSE tracks the color/position decomposition, it becomes the H2
headline metric.
Would change my mind: (1) If recall does NOT collapse beyond the window — that would
mean some unaccounted persistent state exists (re-examine architecture; big surprise,
escalate). (2) If latent-MSE and the color/position decomposition disagree, the metric
choice is wrong — that divergence is itself a finding to escalate before pre-reg.
Spawns: T-002 (probe suite, rescoped), T-004 (pre-reg, rescoped), T-007 (context_signal
rename cleanup). Closes T-001b (dropped). ESC-003 records the milestone.

## D-012 | 2026-06-12
Context: EXP-009 (frozen probe @ f1cf860) showed the H2 cliff cleanly — color recall at
ceiling for n_occ<=6, chance for n_occ>=7, matching N=8/P=3 geometry; drift-controlled;
latent-MSE↔color r=0.952; detector gate pass. Presented (ESC-004); Merlin: "Yes I agree.
This proofs H2."
Decision: Declare **H2 supported**. Lock the T-004 pre-registration (Merlin-approved):
headline = color ΔRGB occluded-vs-matched-drift; latent-MSE validating secondary; position
reported-but-confounded (not a success metric); H3 bar = color ΔRGB < ~63 (halfway
ceiling→chance) at n_occ ∈ {12,16,24} on the identical frozen probe. Write tasks/T-004.md;
set GOAL H2 → supported with criteria + evidence EXP-009.
Alternatives rejected: re-running with more episodes/seeds (cliff is unambiguous, ceiling/
chance well-separated — no power problem); making position a co-headline (drift-confounded,
EXP-009 confirmed occluded≈drift position).
Expected outcome: H2 closed; probe + criteria are the fixed yardstick for all H3 methods.
Would change my mind: if a trivial re-run showed the cliff at a different n_occ than the
geometry predicts (would mean the mechanism isn't the window), or ceiling≈chance (metric
has no dynamic range). Neither holds.
Spawns: T-004; GOAL H2 update.

## D-013 | 2026-06-12
Context: Merlin flagged that `drift_by_occ` (the curtain-stays-up, all-visible control) is a
bad name — that data has nothing to do with occlusion; it measures ordinary autoregressive
drift at a horizon matched to each occluded condition. The probe is frozen (§8), so any
relabel is a logged decision.
Decision: Rename the control key `drift_by_occ` → `matched_horizon_drift` in the probe code
and the EXP-009 results.json (in-place key rename; **numbers untouched** — preserves the
baseline Merlin approved). Update comments/printout/note text and the README freeze commit.
Re-freeze the probe at the new SHA for all subsequent (H3) runs; annotate EXP-009 provenance
(numbers produced at f1cf860, key migrated for schema consistency).
Alternatives rejected: re-running to regenerate the artifact (rollout noise is unseeded →
numbers would drift off the approved baseline for a cosmetic relabel); leaving EXP-009 with
the old key (schema mismatch vs. future H3 runs that compare against it).
Expected outcome: one clear control name end-to-end; H3 comparison artifacts are schema-
consistent with the baseline.
Would change my mind: if the rename touched any numeric path (it doesn't — pure relabel).
Spawns: probe edit + re-freeze; EXP-009 artifact migration.

## D-014 | 2026-06-12
Context: H2 closed (D-012); H3 entered. FF7 v1 (single-timestep sufficiency) converged with
Merlin (IDEAS.md "Proposed first attempt"); build go-ahead given ("Continue by building v1",
ESC-005). Code-grounding for the build (dynamics_model.py, train_dynamics_model.py read in
full) confirmed the design EXCEPT one claim, corrected here:
- CORRECTION to the converged design: registers do NOT persist across `generate()` steps.
  Every forward re-expands the LEARNED register tokens (dynamics_model.py:282); the only
  state carried between generation steps is the latent sequence (:405), and occluded latents
  are trained toward the color-free curtain encoding. So a training-only change evaluated
  through vanilla `generate()` has NO persistent channel beyond the window. The relay
  requires a param-free INFERENCE addition: carry each frame's final-layer register state
  and inject it as the next step's context register — exactly the interface the FF7 training
  rollout trains. "Change to train_dynamics_model.py ONLY" was too strong; "no architecture
  change / zero new parameters" still holds.
Decision: Build FF7 v1 (T-009, inline — full model context already loaded; worker would
re-derive it) on master, flag-gated default-off:
1. dynamics_model.py, param-free extensions: `forward(..., register_in=None,
   return_registers=False)` (inject per-frame register embeddings in place of the learned
   tokens; optionally return final-layer register states) + `generate_memory()` — sequential
   window-1 rollout carrying register state: denoise frame t+1 from [frame t latent @
   tau_ctx, injected reg_t] with K shortcut steps, then one extra forward to extract
   reg_{t+1} from the generated latent @ tau_ctx. `generate()` dispatches to it when config
   flag `use_register_memory` is set, so the FROZEN probe (5503e75) runs unmodified.
2. train_dynamics_model.py FF7 loss (flag `--ff7 K`): per batch, (a) windowed diffusion
   pass as today with return_registers; (b) one extra rollout forward over (k+1)-frame
   sequences folded into batch — frame t: REAL clean latent noised to tau_ctx=0.9
   (tau_idx 115), reg_t injected; frames t+1..t+k: real latents noised at sampled taus,
   finest d only, flow loss + ramp weight (no bootstrap in the rollout). Total = diffusion
   loss + lambda_ff7 * ff7 loss, lambda_ff7 = 1.0 (v1).
3. Dataset: existing occluded.npy — make_curtain_schedule already varies curtain timing
   (visible runs 2-8, cover runs 1-6; occluded_bouncing.py:96-122). No new data for v1.
4. k as a flag. EXP-010 screens k=1 and k=3, ONE seed each (Merlin removed the 2-seed
   standing order 2026-06-12) + smoke run. Rationale for the k=3 arm: with k=1 the
   "write a re-injectable register downstream of an injected register" interface is never
   trained (training write-side registers come from the windowed pass without upstream
   injection), but inference uses that interface at every hop >= 2; k=3 trains 2 chained
   hops inside the rollout pass. Eval: frozen probe 5503e75 vs T-004 bar
   (color dRGB < ~63 at n_occ in {12,16,24}; baseline EXP-009 at chance ~110 there).
Alternatives rejected: (a) window-W rollout context in the FF7 loss (closer to vanilla
inference, weaker per-step sufficiency pressure — keep the pure converged form for v1);
(b) normalizing/projection layer on injected registers (adds params; raw injection first,
gradient shapes the write side); (c) steganographic-latent route (training never conditions
on own generated latents, so it cannot be learned here); (d) new dataset with longer
occlusions (relay logic says per-step sufficiency does the work; revisit if v1 fails).
Expected outcome: smoke run trains with finite combined loss; k=1 shows recall above chance
for ~1 window past the cliff then decays (untrained chained-write); k=3 sustains recall
further; if either holds dRGB < ~63 at n_occ in {12,16,24}, FF7 is promising -> replicate
seed + present. Window-1 inference may degrade base quality - visible in the probe's own
ceiling/drift controls.
Would change my mind: (1) FF7-model ceiling (n_occ=0) or matched-horizon-drift much worse
than baseline EXP-009 (window-1 inference degrading base dynamics -> window-W variant or
inference rethink before judging the memory claim). (2) k=3 <= k=1 on beyond-window recall
(relay-training rationale wrong -> rethink, not k-sweep). (3) Combined loss diverges or
diffusion loss degrades vs baseline at equal epochs (interference -> lambda_ff7 tuning is
its own decision). (4) Recall at reveal frames whose color evidence is OUTSIDE the training
clip stuck at chance while in-clip ones are fine - expected, not a failure (color is
unknowable there); flagging so it is not misread as relay failure.
Spawns: T-009 (build), EXP-010 (smoke + k=1 + k=3 screening, local 4070).
