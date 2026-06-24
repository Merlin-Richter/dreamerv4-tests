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

## D-015 | 2026-06-13
Context: EXP-010 (ESC-006) showed FF7 v1 supports H3 for hidden COLOR but position is at
chance — including in the matched-horizon DRIFT control (open rollout, curtain UP). Merlin's
read: the position deficit is GENERAL, not memory-specific or FF7-induced (it is in
my_dynamics and EXP-009 too, even curtain-down). Implication: whether ANY memory approach can
retain position/momentum is untestable while the base model cannot track motion in the clear.
He floated candidate causes (ball moves little per step; too few / too-late temporal-mixing
layers — "s,s,s,t" → maybe "s,t,s,s,t…"; never learned the concept of a movable object; or
just needs more training) but is explicitly unsure and asked to understand it before acting.
Greenlit a diagnostic ("yes go").
Decision: Run EXP-011 — a NO-TRAINING diagnostic on existing checkpoints (my_dynamics.pt +
the FF7 arms) + the frozen tokenizer, to (i) confirm/quantify the deficit, (ii) LOCALIZE it
to the tokenizer C vs the dynamics D, and (iii) distinguish failure (a) "never learned motion"
from (b) "learned motion, open-loop rollout chaotically desyncs from the specific GT trajectory"
— before any architecture change or longer training. Components:
  1. GT ball kinematics — per-step displacement and speed read directly from states[:, vx,vy]
     (zero inference). Answers "ball moves little?" and calibrates the copy-last baseline.
  2. Position baselines vs horizon: copy-last-observed-position and chance, vs the model's
     OPEN-loop pos_err. Key test: does the model beat freezing the ball in place?
  3. Closed-loop / teacher-forced 1-step pos_err along the whole trajectory (incl. across
     bounces) vs open-loop multi-step. Good closed-loop + bad open-loop ⇒ (b); bad closed-loop
     ⇒ (a).
  4. Linear probe of FROZEN tokenizer latents → ball (x,y). Decodable ⇒ info is present, deficit
     is in D's propagation; not decodable ⇒ tokenizer C bottlenecks position and no D-side fix
     (incl. FF7) can carry it. Localizes the fix.
  5. Qualitative: eyeball a few open rollouts — plausible-but-desynced motion vs frozen/teleport/blur.
Alternatives rejected: (a) jump straight to the s/t temporal-attention rearrangement — a guess
that needs a retrain and presumes the fix lives in a layer pattern we have not even localized
to C vs D; (b) just train longer — Merlin himself unsure, and we have no evidence it is an
undertraining problem; (c) only swap the probe's position metric (e.g. closed-loop) — premature
before we know (a) vs (b). Diagnostic is hours, local, no training; it picks the lever.
Expected outcome (predictions): I expect the ball to be slow (radius 10 in 64px, bouncing) so
copy-last is a strong baseline; I expect the model's 1-step prediction to BEAT copy-last (the
ceiling 1-step pos_err 1.1px is very tight) and closed-loop to stay good while open-loop
diverges — i.e. lean (b), chaotic open-loop divergence, meaning open-loop GT-matched position
is simply the wrong yardstick and the color memory result stands. I expect position to be
linearly decodable from the tokenizer latents (deficit, if any, in D not C).
Would change my mind: (1) position NOT linearly decodable from frozen tokenizer latents ⇒ the
bottleneck is the TOKENIZER; H3 position-memory work must target C, and FF7-on-D can never
carry position — a major reframing. (2) model 1-step pos_err ≈ copy-last AND closed-loop also
poor ⇒ failure (a), the model genuinely did not learn motion; fixing base dynamics (training/
architecture) becomes the gating task before more H3 method work. (3) ball is genuinely fast
yet model still ≈ copy-last ⇒ strong (a) signal.
Spawns: EXP-011 (local 4070, no training). Built inline (single-threaded local analysis I am
blocked on; consistent with the EXP-008 inline precedent — no worker delegation benefit here).

## D-016 | 2026-06-13
Context: EXP-011 showed the FF7 checkpoints track motion far better than my_dynamics (1-step
teacher-forced 1.0px vs 4.5px), but I could NOT attribute this to the FF7 loss vs the FF7 runs
simply being trained 100 epochs while my_dynamics (old baseline, approximate provenance) was
trained less. EXP-009 (H2 baseline = my_dynamics) and EXP-010 (FF7, fresh 100ep) are therefore
not training-matched. Merlin agreed (ESC-007) to train a budget-matched vanilla baseline.
Checkpoint-config inspection confirmed ff7_k3 and my_dynamics share the IDENTICAL architecture
(DynamicsModelConfig defaults), so the only clean control is a fresh vanilla run at the FF7 budget.
Decision: Run EXP-012 — train a vanilla dynamics model (--ff7 0 --fresh) with the EXACT EXP-010
config (occluded.npy + occluded_actions.npy, trained_autoencoder.pt tokenizer, 100 epochs,
batch 32, lr 3e-4 cosine, seed 0), then evaluate on the frozen probe 5503e75 AND re-run the
EXP-011 diagnostic on it. This (i) retires my_dynamics as the H2/H3 baseline, (ii) attributes the
EXP-011 dynamics gain (FF7 loss vs training budget), (iii) re-anchors the H2 cliff and the H3
comparison on a training-matched baseline. Local 4070 (cluster wrappers T-003 not built; ~2.6h
training based on EXP-010 pace + probe). Config committed at experiments/EXP-012/{config.yaml,run.sh}
(fixes the EXP-010 config.yaml provenance gap).
Alternatives rejected: (a) keep my_dynamics as baseline — confounds every FF7 claim with training
budget; (b) multi-seed baseline now — single seed first to match the single-seed FF7 screen, add
seeds only if the comparison is close (Merlin relaxed the 2-seed order); (c) cluster run — wrappers
aren't built and the 4070 ran EXP-010 fine overnight, so no need.
Expected outcome (predictions): the vanilla baseline reproduces the EXP-009 post-window COLOR
cliff (architectural; better training won't let a sliding window see past its N=8 window) — so H2
stands. On MOTION: I expect the budget-matched vanilla to be substantially better at 1-step
position than my_dynamics (4.5px) — i.e. much of my_dynamics's weakness was undertraining — but
still somewhat worse than FF7 (~1.0px) if the FF7 loss genuinely helps dynamics. If vanilla
1-step ≈ FF7 (~1px), the dynamics gain was training budget, not the FF7 loss.
Would change my mind: (1) vanilla baseline shows BEYOND-window color recall < chance (clearing
the T-004 bar) — would mean EXP-010's color "memory" was a training/inference artifact, not the
register relay (architecturally surprising → halt + rethink H3 attribution). (2) vanilla 1-step
position WORSE than my_dynamics at matched budget — would point to a data/seed/init issue, not
budget. (3) training diverges / val loss >> FF7's 0.0065 at 100ep — config mismatch, diagnose
before comparing.
Spawns: EXP-012 (local 4070; train + probe + EXP-011 diagnostic rerun). Present-then-stop.

## D-017 | 2026-06-13
Context: EXP-012 (budget-matched vanilla baseline, D-016) is training on the 4070 (~2h to go);
per §3/§5 I am waiting on its results and may do only outcome-independent work. Merlin steered
(this session, verbatim-in-substance): "while waiting ... you can do completely separate
independent work. What comes to mind for me is starting careful implementation of KV cache if not
already." KV cache is BOARD task **T-008** (efficiency, NOT a prerequisite for any hypothesis;
its outcome cannot change EXP-012's reconciliation), so it qualifies as the permitted independent
prep. It MUST follow HOWTO/rope_kv_cache_caveat.md (D-011): the current fixed cos/sin table
(size max_temporal_length) re-indexes positions 0..w-1 per window and is cache-incompatible —
cached K/V cannot be re-rotated when the window slides.
Decision: Implement KV caching for the dynamics model D (inference-only; training forward path
left bit-identical), in layered, individually-validated pieces:
  (1) Absolute-position RoPE: temporal Attention computes rotation on the fly from an optional
      `positions` arg (never-reset clock); `positions=None` keeps the exact current table path
      (zero training-path change). Enables both long rollouts past the size-16 table AND
      rotation-stable cross-frame caching.
  (2) KV-cache plumbing through Attention/TransformerBlock/forward (opt-in via default-None args;
      spatial layers are within-frame and never cache).
  (3) Cached `generate`: intra-frame reuse (the K shortcut substeps of one frame share an
      identical context K/V — uncached _denoise_next already freezes ctx_noised across substeps,
      so this is bit-for-bit identical to uncached generate, ~K x fewer temporal-attn FLOPs, no
      RoPE-trap). Cross-frame eviction cache is an opt-in further optimization that freezes the
      per-frame context-noise redraw (a minor, documented deviation) — validated deterministically
      at the forward level, not claimed bit-identical at generate level.
Acceptance gate (the HOWTO's bit-for-bit requirement, applied at the deterministic level):
  - FORWARD-equivalence test: incremental cached forward (frame-by-frame, absolute positions) ==
    full-sequence forward, max|Δ| < 1e-4 on a random model. This is the real correctness proof
    (generate adds RNG; forward is deterministic).
  - GENERATE-equivalence test: seeded uncached generate == intra-frame-cached generate exactly.
  - Existing FF7 smokes (test_ff7_smoke.py) still pass (no training-path regression).
Alternatives rejected: (a) delegate to a cold worker — the RoPE rotation-continuity trap context
is loaded in this session; a cold spawn re-derives it and the Agent guardrail discourages
unrequested spawns (Merlin asked for KV cache, not a worker). (b) cache the FF7 generate_memory
path — it is already window-1 (cheap); the win is in vanilla generate + long-horizon rollouts.
(c) start the next experiment/method instead — forbidden while waiting on EXP-012 (§3).
Expected outcome: a validated, opt-in KV cache that leaves training and current eval numbers
untouched (all gates green), available for the long-horizon memory rollouts H3 will need.
Would change my mind: (1) cannot make cached==uncached within fp tolerance → a real bug in the
absolute-RoPE/cache logic, fix before claiming anything (do NOT ship a cache that silently
diverges — the HOWTO's exact failure mode). (2) the refactor perturbs the training forward or
breaks checkpoint loading → back out and redo opt-in. (3) EXP-012 finishes → STOP this and process
results first (§0.3, finished job takes priority).
Spawns: T-008 (this work). Independent of EXP-012; no present-then-stop gate (not an experiment).

## D-018 | 2026-06-13
Context: ESC-008 resolved. Merlin agreed (1) H2 anchored on the budget-matched baseline + my_dynamics
retired and (2) the confound is resolved (FF7 wins = method, EXP-009/010 retroactively trustworthy).
On (3) he REJECTED the open-loop GT-matched position metric as the way to judge position memory: it
conflates two failures it should separate — (F1) a model that never tracks (predicts center/static)
scores badly, AND (F2) a model that tracks well but suffers early butterfly-effect desync ALSO scores
badly — plus the artifacts already on BOARD (bounded box caps error at chance; the curve turns over
at long horizons because the ball bounces back into prior regions, falsely crediting a desynced
prediction). His proposed direction: measure whether the model's believed (x,y) AND velocity stay
**self-consistent over the occluded timesteps** — at each occluded step read out "what would the
model predict if revealed now" and compare across steps — rather than matching the exact (chaotic) GT
trajectory. This supersedes the BOARD "closed-loop/distributional position metric" framing.
Decision: Design, converge-with-Merlin, verifier-check, then build + FREEZE a **position-memory
consistency metric** that (a) credits a model maintaining a coherent, physically-evolving belief about
the hidden ball (retained position+momentum), (b) does NOT penalize butterfly-effect divergence from
the exact GT trajectory deep in occlusion, (c) DOES penalize forgetting — both the static/center
"forgot it's moving" mode and the wander "lost the object" mode. The metric is the spine of the H3
*position* story (§8): its definition is a logged decision and must be LOCKED before any H3 position
method (e.g. the sequential stop-grad register-relay, IDEAS.md) is run, to keep pre-registration
honest. The design will be put past the `critical-claim-verifier` agent (Merlin-committed) for an
independent measurement-validity audit before freeze.
Working formalization to bring Merlin (not yet locked):
  - Readout of the believed state per occluded step via a velocity-aware **state probe** on the
    dynamics-model hidden state (registers/latents), trained+gated on teacher-forced VISIBLE rollouts
    (cf. EXP-011 latent→xy R²=0.96 on tokenizer C; here it reads the dynamics state). Optional
    cross-check: a counterfactual "reveal-now" readout IF the env exposes a curtain-up control
    (to verify when building).
  - Components: (i) ONSET FIDELITY — believed (x,y,dx,dy) vs GT at occlusion onset & in-window steps
    (fair vs GT; catches F1-static via wrong velocity, catches forgetting once prefix scrolls out);
    (ii) SELF-PHYSICS CONSISTENCY — seed a deterministic billiard rollout from the believed onset
    state and measure residual of the believed trajectory to it over k steps (GT-free, desync-immune;
    low = retained coherent motion; the static-forget degenerate is killed by (i)); (iii) report-only
    GT-TRACKING HORIZON — first step belief–GT exceeds threshold (fair early, informative).
  - Controls: ceiling (teacher-forced visible belief), chance (static-center/shuffled), copy-last
    (frozen ball — must FAIL onset velocity-fidelity), matched-horizon visible (curtain-up) control.
Alternatives rejected: (a) keep the open-loop GT metric — Merlin rejected it (conflates F1/F2 +
artifacts). (b) skip the metric and treat color-only as the H3 result — leaves the more interesting
position half unmeasured; he wants it measured honestly. (c) jump straight to the relay method — would
run a method with no valid yardstick; §8 wants the spine frozen first.
Expected outcome: a frozen, validity-audited position-memory metric on which the budget-matched
vanilla baseline shows poor consistency beyond the window (forgets the moving object) and against
which H3 position methods can be judged honestly.
Would change my mind: (1) the state probe does NOT transfer from visible→occluded hidden states
(low held-out accuracy, or it's decoding the curtain not a belief) → the readout is invalid; redesign
the readout (e.g. reveal-action counterfactual) before trusting any number. (2) the self-physics
consistency score is trivially passable by a degenerate belief that (i) does not catch → tighten/redo
(this is exactly what the verifier audit is for). (3) Merlin redirects the framing (esp. whether to
credit self-consistent-but-GT-diverged beliefs as "memory") → his call, it redefines the metric.
Spawns: design doc + critical-claim-verifier audit → metric build/freeze task (TBD after his sign-off).
No experiment yet; this is instrument design.

## D-019 | 2026-06-13
Context: Independent thread, parallel to the D-018 position-memory metric (which a second orchestrator
is actively building — `src/probe/position_consistency.py`, EXP-013). Picked to NOT touch the metric
spine. ORIENT worry #4 + the EXP-012 "bonus finding" flag an OPEN, unresolved question: FF7 sharpens
the base 1-step teacher-forced dynamics ~4.6× (vanilla_s0 4.66px ≈ my_dynamics 4.51 ≫ ff7 ~1.0px), but
that number was produced through the **register-relay inference path**, not disentangled from the FF7
**loss**. Code-grounded confirmation (this session): `generate()` (dynamics_model.py:528) dispatches
`use_register_memory=True` checkpoints to `generate_memory()`, so EXP-011/012's FF7 1-step numbers ran
via `generate_memory` — a **window-1** relay (last latent + carried register from memory_rollout_init,
:722-728), NOT the ≤N-1=7-frame windowed attention the vanilla_s0 number used. So FF7's ~1px conflates
THREE factors: (i) better weights from the FF7 loss, (ii) the register relay, (iii) window size (1 vs 7).
Decision: Run an analysis-only experiment (EXP-014, NO training, existing checkpoints) that evaluates
1-step teacher-forced pos_err for ff7_k1, ff7_k3, vanilla_s0 through BOTH inference paths on the IDENTICAL
GT window: (a) vanilla windowed path (force `use_register_memory=False` → learned-init scratch registers,
≤7-frame attention) and (b) relay path (`generate_memory`, window-1 + carried register). Reuses
EXP-011/diagnose.py infra + frozen probe env/detector 5503e75 so numbers are comparable to EXP-011/012.
Sanity anchor: relay-path FF7 must reproduce EXP-012's ~0.96–1.02px before any conclusion is drawn.
Alternatives rejected: (a) cross-frame KV-eviction cache (T-008 follow-up) — foundational infra but
serves the relay-training METHOD, which is gated behind the D-018 metric freeze (§8); premature. (b) 2nd
vanilla seed for the motion claim — Merlin didn't ask; lower value. (c) touch the position metric —
owned by the other orchestrator; hands-off.
Expected outcome (prediction): FF7-vanilla-path 1-step lands MUCH better than vanilla_s0's 4.66px (i.e.
closer to ~1–2px), supporting "the FF7 loss is a dynamics regularizer that improves windowed dynamics,
independent of the relay." I expect a residual gap (relay-path slightly better than FF7-vanilla-path)
attributable to window-1 register sufficiency, but the bulk of the 4.6× to be the loss.
Would change my mind: (1) FF7-vanilla-path ≈ vanilla_s0 (~4.5px) while only relay-path ≈ 1px → the
improvement is the RELAY INFERENCE / window-1 sufficiency, NOT better windowed weights; "FF7 is a better
dynamics model" would be FALSE and the EXP-012 bonus claim must be retracted/reframed. (2) relay-path FF7
fails to reproduce EXP-012's ~1px → my harness diverges from the diagnostic; fix before trusting anything.
(3) vanilla_s0 through the (forced) relay path is also ~1px → the relay, not the FF7 loss, carries the
1-step win even for non-FF7 weights (would reframe what the relay does). 
Spawns: EXP-014 (analysis-only; ends present-then-stop per §5, escalated to Merlin). Coordination: no
`git add -A` (other orchestrator has uncommitted position_consistency.py); path-scoped commits only;
ORIENT/BOARD left to the live orchestrator to avoid clobbering its dashboard.

## D-020 | 2026-06-13
Context: ESC-009/ESC-010 resolved ("resolve as whatever; continue"). Merlin reservation: the EXP-013
position-memory metric, as coded, may not be a strong eval instrument — NOT frozen as a spine; position
retention stays open. He directed the next, easily-verifiable work item: **a KV cache for sliding-window
rollouts, as preparation for rollout training** (the eventual sequential register-relay method that
unrolls the dynamics model many steps and needs a cheap, correct rollout substrate).
Current infra (code-grounded this session): `generate()` (dynamics_model.py:518) is the uncached
sliding-window rollout — each step re-noises the last N-1=7 frames and runs K shortcut substeps,
re-encoding the whole window every frame. `generate_cached()` (:603) caches the window's K/V across the
K substeps WITHIN one frame but REBUILDS the cache per frame (bit-identical to generate, ~Kx fewer
temporal FLOPs, no cross-frame reuse). The temporal attention already supports a persistent cache:
K/V are stored ALREADY-RoPE-ROTATED at absolute positions (Attention.forward :160-168, T-008/D-017),
with `cache=`/`commit=`/`positions=` plumbing and `new_kv_cache()`.
Decision: Build a **cross-frame sliding-window KV eviction cache** for rollouts: commit each finalized
frame's K/V into the persistent cache ONCE, and EVICT the oldest time-column when the cache exceeds the
window (N-1). Because cached K/V are pre-rotated at absolute positions, eviction is a pure slice — no
re-rotation (the absolute-RoPE foundation from T-008 is exactly what makes this trivial). Expose it as
reusable primitives mirroring the FF7 relay pattern — `stream_rollout_init(context, action_idx)` /
`stream_rollout_step(state, action_id)` — plus a thin `generate_streaming()` wrapper looping over them,
so the rollout-training method can drive the same primitives. The ONE semantic difference from
`generate()`: each frame's context-noise is drawn ONCE at commit and reused while it sits in the window,
instead of redrawn every step — a documented, defensible deviation (a frame's committed representation
is fixed once generated), and the natural structure for rollout training (context = fixed/detached).
So this is NOT bit-identical to `generate()` at the generate level; it IS bit-identical to a frozen-noise
reference and to a full windowed recompute at the forward level.
Verification (the gate — "easily verifiable", forward-level is the real gate per test_kv_cache.py):
  (1) FORWARD-LEVEL eviction equivalence, NO RNG: streaming cache (commit+evict each frame) == full
      windowed forward over [t-W+1..t] with explicit positions, bit-for-bit (TOL 1e-4), for the new
      frame at every step. This isolates eviction + absolute-RoPE + causal mask from any noise change.
  (2) LONG-rollout past the cos/sin table (T >> max_temporal_length): same equivalence, exercises
      unbounded absolute positions through eviction (the documented RoPE-overflow trap).
  (3) GENERATE-LEVEL: generate_streaming == a frozen-noise reference rollout (same draw order),
      bit-for-bit; AND quantify the deviation from standard generate() (expected tiny) to confirm the
      frozen-noise semantics is benign. With and without action conditioning.
  (4) Speed: streaming is faster than generate_cached at rollout scale (it does NOT rebuild the cache
      per frame) — sanity print, not a gate.
This is implementation infra (NOT an experiment): no present-then-stop gate; the correctness tests are
the artifact. FF7 register-memory path is dispatched unchanged (already window-1, no cache benefit).
Alternatives rejected: (a) jump straight to the relay rollout-training method — Merlin wants the
verifiable substrate first; building training on an unverified rollout cache would conflate cache bugs
with method results. (b) tokenizer-C KV cache (BOARD follow-up b) — not on the rollout-training path.
(c) make it bit-identical to generate() by also caching/redrawing per-step noise — defeats the purpose
(no cross-frame reuse) and the frozen-noise semantics is what training wants anyway.
Expected outcome: a tested `generate_streaming` + init/step primitives, all forward-level equivalence
tests green, measurably faster than generate_cached on long rollouts, ready as the rollout-training base.
Would change my mind: (1) forward-level eviction equivalence FAILS (cached-evicted != full recompute) →
a positional/mask/eviction bug; fix before anything builds on it (this is the whole point of the gate).
(2) the frozen-noise deviation from generate() is NOT small (rollout diverges materially) → re-examine
whether per-step noise redraw is load-bearing; escalate the semantics choice. (3) the primitives can't be
made autograd-friendly for the later training use without redesign → note it now so the relay method
isn't surprised. Spawns: T-012 (this work) — design note tasks/T-012-plan.md + impl + test_stream_cache.

## D-021 | 2026-06-13
Context: Refinement of D-020/T-012 directed by Merlin after reviewing the stream-cache tests. The
generate-level correctness check originally compared generate_streaming against a frozen-noise
reference REIMPLEMENTED inside the test. Merlin's point: a test-local reimplementation can share the
same bug as the implementation, so it may not actually capture divergence. He asked to make the
NON-CACHE version itself able to "draw noise once / based on seed if specified" so the cached rollout
can be compared bit-exactly against a real, independent uncached path.
Decision: Add a deterministic per-frame noise source keyed on the ABSOLUTE frame id and role
(`_make_noise_fn`, role 0 = generation z, role 1 = context-noise) — so noise is content-addressed,
not RNG-call-order-dependent; loop structure and caching cannot perturb which noise a frame gets.
Thread an optional `noise_seed` through generate_streaming / stream_rollout_init/step, and add
`generate_windowed` — the UNCACHED twin (full windowed recompute, independent stepping, same frozen
per-frame context-noise). With a shared `noise_seed`, generate_streaming == generate_windowed
bit-for-bit; the only difference between the two real code paths is the persistent cache, so any
divergence is a cache/eviction/RoPE or bookkeeping bug (not a noise mismatch). Default `noise_seed=None`
preserves the existing global-RNG behavior exactly (no change to the passing default path / KV-cache /
FF7 regression tests).
Verification added (test_stream_cache.py, now 9/9): (i) generate_streaming == generate_windowed under a
shared seed, with and without actions; (ii) `test_seeded_noise_is_reproducible` — seeded rollout is
invariant to global RNG AND a different seed changes the rollout (the comparison isn't trivially equal);
(iii) `test_divergence_is_detectable` — a MUTATION test: disabling eviction makes the cached path
diverge from generate_windowed, proving the comparison actually catches a broken cache. Forward-level
RNG-free eviction equivalence (the primary gate) is unchanged.
Alternatives rejected: (a) keep the test-local reimplementation — Merlin's exact objection (correlated
bugs, may not capture divergence). (b) a single `_rollout(use_cache=...)` flag sharing all stepping
code — guarantees identical noise but the two compared paths would share the rollout bookkeeping, so a
bookkeeping bug wouldn't show; the independent `generate_windowed` cross-checks that too. (c) make
generate() itself seedable/frozen — larger blast radius on the hot default path for no extra test power.
Expected outcome: a test that demonstrably captures cache divergence (mutation test green), and a
reusable deterministic uncached rollout for future debugging/reproducibility.
Would change my mind: (1) the seeded cached vs uncached comparison can't be made bit-exact (some
residual mismatch) → a real cache bug or a hidden nondeterminism; fix before building on it. (2) the
mutation test passes (no divergence on broken eviction) → the comparison is insensitive; strengthen it.
Spawns: none (folds into T-012). No experiment.

## D-022 | 2026-06-13
Context: Merlin wants a basic, reusable perf tool for the rollout KV cache (T-012) — cached vs
no-cache, on the GPU, with a real batch dimension (training-relevant). Wants: rollout-step throughput
("how many rollout steps we are getting"), peak GPU memory, and where time is spent, across different
context-window sizes. Explicitly: "don't go crazy", 60s/config is an upper bound, don't run long perf.
Decision: EXP-015 — a perf benchmark `experiments/EXP-015/perf_rollout.py` (GPU, venv python) comparing
`generate_streaming` (cross-frame KV cache) vs `generate_windowed` (no persistent cache; the matched
uncached twin — same semantics, so the delta is purely the cache). Real model dims (load vanilla_s0
config+weights; perf is weight-agnostic but use realistic shapes + n_actions). Sweep context window
N ∈ {8,16,32,64} at batch B=32 (CLI-overridable). Per config: warmup to fill the window, then a
time-budgeted loop (default ~8s, ≤60) counting completed rollout steps → report steps/s, frames/s
(=B×steps/s), ms/step, peak allocated+reserved MB. Plus a torch.profiler pass on one representative
config per method → top kernels by CUDA time (where time goes; flag compute matmul/SDPA vs memory-bound
elementwise/cat — the cache's concat is the memory-bound overhead). NOTE: exact HBM "memory-stall %"
needs Nsight Compute; the profiler op-breakdown is the practical proxy and will be labeled as such.
Output: results.json + a printed table + a 2-panel plot (steps/s vs N; peak MB vs N) in EXP-015/.
Alternatives rejected: (a) compare against `generate`/`generate_cached` too — more configs, less
apples-to-apples (different noise/semantics); streaming-vs-windowed isolates the cache. (b) Nsight
Compute for true memory stalls — overkill for "basic"; profiler op-time is enough. (c) long runs /
many seeds — Merlin said don't.
Expected outcome: streaming flat-ish ms/step as N grows (O(1) attention/step, only the cache cat grows)
while windowed grows with N (re-encodes the whole window each step) — so streaming's throughput
advantage widens with context length; streaming uses more memory (persistent K/V cache) but bounded by
the window. A basic tool we can rerun at other B/N.
Would change my mind: (1) streaming is NOT faster than windowed at large N (or slower) → the per-step
cat / cache bookkeeping dominates; report honestly and note the regime. (2) streaming peak memory blows
up unboundedly (eviction not actually freeing) → a cache leak; investigate. (3) measurements unstable
across repeats → lengthen budget / fix sync. Spawns: EXP-015 (perf), present-then-stop per §5.

## D-023 | 2026-06-13
Context: ESC-011 — Merlin accepted the EXP-015 perf tool and directed the offered next cut: rerun "with a
significantly higher batch-size … to be close to the GPU memory limit for each approach and compare their
steps/s … see if we get more speedup as we do more parallelism." EXP-015 swept context window N at a fixed
B=32 with BOTH methods at the same batch. The new question is orthogonal: hold N fixed and push BATCH up.
Key asymmetry to surface: cached (`generate_streaming`) has the smaller working set (EXP-015: at N=64,B=32
windowed reserved 4708 MB vs cached ~218 MB alloc), so on this 8 GB laptop 4070 cached should fit a LARGER
batch before OOM than windowed — a double throughput win (faster per step AND more parallelism). "Memory
limit for each approach" therefore means each method at ITS OWN max-fitting batch, not a shared batch.
Decision: EXP-016 — extend the reusable tool `experiments/EXP-015/perf_rollout.py` with a `--batch-sweep`
mode: fix N (default 16, = training context length), sweep an ascending batch list, run each method
independently, catch CUDA OOM and record the largest batch that fits per method (stop escalating that
method on OOM). Report per (method,batch): steps/s, frames/s (=B×steps/s — the real throughput when batch
differs), ms/step, peak alloc/reserved MB. Headline outputs: (a) each method's max-fitting batch + its
steps/s & frames/s there; (b) speedup (cached÷windowed steps/s) vs batch on the shared batches — does it
grow, flatten, or shrink as parallelism rises; (c) frames/s vs batch for both, annotating where windowed
OOMs but cached keeps scaling. Write to `experiments/EXP-016/` via a new `--outdir` (keeps EXP-015 intact).
Run local 4070 (venv python, CUDA). Budget small (≤8s/config); the N-sweep mode is untouched.
Alternatives rejected: (a) a single huge B at fixed N — answers "does cached fit a bigger batch" but not
"does speedup grow with parallelism"; the sweep gives the curve. (b) sweep B and N jointly — too many
configs / muddles the parallelism question; one representative N keeps it clean (can rerun other N if he
wants). (c) auto-double-until-OOM with no list — fine, but an explicit ascending list is more legible and
reproducible; I'll size it from the 8 GB ceiling.
Expected outcome: at fixed N, both methods get faster in frames/s as batch rises until the GPU saturates
(compute-bound), then per-step steps/s falls ~linearly with batch. Two regimes for the speedup ratio: at
small batch cached's per-step win persists (it skips re-encoding the window); as batch grows and both
become compute-bound on the same per-step FLOPs the steps/s ratio may COMPRESS toward ~1 (cached's saving
is re-doing window attention, which is a shrinking fraction of total compute at high B) — so "more speedup
with more parallelism" may be FALSE for the steps/s ratio. BUT cached's decisive win shows up as a higher
max batch (lower memory) → higher peak frames/s overall. I expect the honest headline to be "cached wins
by fitting more parallelism (memory), not by a widening per-step ratio" — I'll report whichever the data
shows.
Would change my mind: (1) cached's speedup ratio actually GROWS with batch → its per-step saving is not
a fixed fraction; report the regime. (2) cached does NOT fit a meaningfully larger batch than windowed →
the EXP-015 memory gap doesn't translate to batch headroom (e.g. cache grows with B too); investigate
where the memory goes. (3) OOM is non-deterministic / fragments unpredictably → fix with empty_cache +
reset between configs, widen spacing, report the fragile boundary honestly rather than a crisp max.
Spawns: EXP-016 (perf, local), present-then-stop per §5.

## D-024 | 2026-06-13
Context: Merlin chose option B (memory-token split + FF9 memory-only sufficiency) over option A (sequential
register relay), color-first (AskUserQuestion). I wrote the design note (tasks/T-013-plan.md) and ran it
past critical-claim-verifier per §4. **V-T013 verdict: REFUTED as specified** — (1) FF9 inherited FF7's
successor setup (own real latents, τ~Uniform, ramp 0.9τ+0.1) so the loss is mostly solvable by local
self-denoising; memory non-load-bearing except in the down-weighted low-τ tail (empirical probe in
experiments/verify-T013/). (2) Even fixed, FF9 v1's single-hop TBPTT-1 gradient trains read+1-hop-write, not
preserve-across-N-hops → predicted to reproduce FF7 (color yes, depth/position no). A (relay) and B
(objective) are COMPLEMENTARY. Escalated ESC-013 (P1/P2/P3). Merlin: "verifier is very correct. This alone
will not fix FF7. But I wanted to do this first so we have a better architectural baseline." → P1 reframed
as the architectural-baseline build, with his own better fix for (1).
Decision: Build the **memory-token architecture + FF9 v2 loss** on the 4070 (T-013), as the H3 memory-token
ARCHITECTURAL BASELINE (NOT expected to beat FF7 on beyond-window depth — the cross-window relay, option A,
is layered on this next). Specifics:
- **Architecture (additive):** distinct MEMORY token type (`n_memory`, learned `memory_tokens`, `memory_in`
  / `return_memory`, `use_full_state_memory`); registers revert to pure scratch (no memory duty). `n_memory=0`
  ⇒ byte-identical to today (smoke-guarded). No `absent_latent` token — withhold via signal level.
- **FF9 v2 loss (Merlin's variable-horizon, pure-noise path — fixes V-T013 finding 1 with NO GT leak):**
  per memory rollout (max lookahead k) sample horizon j∈{1..k}; mini-forward [t..t+j]; frames t..t+j−1 (incl.
  the memory source) at **signal level τ=0** (pure noise → no GT latent anywhere memory could cheat from),
  memory_t injected at t + learned-init memory at t+1..t+j−1 (relayed via the temporal memory channel); the
  **terminal frame t+j at sampled τ** (any level — so a training target exists + memory-conditioned denoiser
  is calibrated); **loss on ALL of t+1..t+j** (Merlin refinement 2026-06-13: intermediates at τ=0 are pure
  memory-sufficiency targets per horizon, terminal at sampled τ — leak-free because the only signal-bearing
  frame is terminal with no successors; j supervised targets/rollout, not 1), **un-ramped** (drop 0.9τ+0.1 so
  low-τ samples aren't down-weighted — the other half of V-T013 finding 1). Knobs: ff9_k, ff9_ramp(off
  default), ff9_tau_last. Impl: fixed (k+1)-frame forward, per-window j∈{1..k}, scatter sampled τ to the
  terminal slot + τ=0 elsewhere, mask the loss to frames 1..j.
- **Mechanism note (recorded):** within one k-window forward, frame t+j attends DIRECTLY to frame t's memory
  tokens → FF9 v2 trains "memory = sufficient attendable full-state object," NOT the cross-window relay
  (preserve after the source leaves the window). That is exactly why it is a baseline, not the depth fix.
- **Measure (reframed, §5):** PRIMARY = within-window memory sufficiency (L(memory)≪L(no-memory) across j) +
  no base-dynamics regression; POSITIONING = frozen probe 5503e75 color at n_occ {12,16,24,32,48} vs
  vanilla_s0 + FF7 (expect ≈FF7; flatter = bonus; worse = investigate); position reported (caveated).
Alternatives rejected: my low-τ-clamp fix (Merlin's variable-horizon pure-noise path is cleaner and trains
multi-horizon sufficiency + guarantees no GT on the path); P2 (isolate low-τ on FF7 — skips the memory-token
arch Merlin wants); P3 (A+B now — bigger/riskier; Merlin wants the baseline first).
Expected outcome: a clean memory-token model; memory-sufficiency probe shows memory beats the prior within
window; no base-dynamics regression; frozen-probe color ≈ FF7 (architectural baseline established, relay next).
Would change my mind: (1) memory-sufficiency probe shows memory NOT load-bearing (L(memory)≈L(no-memory)) →
the τ=0-path loss still has a shortcut or memory collapsed → halt + re-examine before any relay work. (2)
base-dynamics regression vs vanilla_s0 → the τ=0 path or extra tokens hurt the main objective. (3) frozen
color materially WORSE than FF7 → the memory-token swap degraded what FF7's registers achieved.
Spawns: T-013 build (architecture + _ff9_loss + generate_full_state_memory + smokes), then EXP-017
(training run, present-then-stop per §5).

## D-025 | 2026-06-14
Context: Merlin resolved ESC-015 (EXP-017 FF9 v2 accepted). He asked to check that the interactive
viewer `src/D_dynamics_model/play_dynamics_checkpoint.py` supports the new memory rollouts and fix it
if not. Inspection: the viewer dispatches ONLY on `use_register_memory` (FF7 register-carry). An FF9 v2
checkpoint (EXP-017: use_full_state_memory=true, n_memory=4, use_register_memory=false) falls through to
the vanilla sliding-window path — so it would render vanilla rollouts (hallucinated state past the
4-frame window) and even mislabel the mode as `vanilla window=4`. That is NOT the FF9 inference we
evaluated (generate_full_state_memory, A1+B1, V-T013-eval). FF9 also lacks interactive init/step
primitives (FF7 has memory_rollout_init/step); generate_full_state_memory is a monolithic closed loop.
Decision: (1) Add `full_state_rollout_init`/`full_state_rollout_step` to DynamicsModel mirroring the FF7
primitives, and refactor `generate_full_state_memory` into a thin loop over them (single source of truth,
matching the documented generate_memory<->memory_rollout pattern). (2) Dispatch the FF9 v2 case in the
viewer to these primitives, seeding the WRITE from a deeper prefix (up to max_temporal_length-1=15 frames,
not just ROLLOUT_CTX=4) so the snapshot matches the evaluated WRITE; update mode label + header.
Alternatives rejected: inline the FF9 per-step math in the viewer (duplicates the eval inference -> can
drift -> Merlin would see something other than what we evaluated); leave generate_full_state_memory
monolithic and write standalone viewer-only primitives (same drift risk).
Expected outcome: FF9 v2 checkpoint plays through the faithful full-state-memory inference (static color
survives indefinitely past the window); existing FF9 smokes still pass; a new init/step<->generate
equivalence test passes bit-for-bit; FF7/vanilla paths unchanged.
Would change my mind: refactored generate_full_state_memory output differs from the pre-refactor output
on a fixed seed (then the primitive extraction broke RNG order/semantics — revert and inline instead).
Spawns: T-015 (viewer FF9 support + model primitive refactor + equivalence test). No experiment.

## D-026 | 2026-06-14
Context: Merlin redirected focus (his words, this session): "the next thing we actually need to worry
about is predicting ball movement even without curtains." My check of the existing record (EXP-011/012/
013/014) established three regimes on curtain-up (no-occlusion) episodes (ball ~3.2px/step; copy-last=3.2px;
chance~23px): (1) tokenizer ENCODES position fine — linear probe R²=0.96/2.7px (EXP-011); (2) SINGLE-STEP
teacher-forced: vanilla 4.5-4.7px (WORSE than copy-last → barely models motion) vs FF7/FF9-aux-loss models
~1.0px (3× copy-last; the aux loss doubles as a 1-step dynamics regularizer, EXP-014); (3) MULTI-STEP
open-loop: even the best (~1px 1-step) model drifts to chance by ~h12-16 WITH visual feedback (EXP-011),
~1 step blind (EXP-013). So: representation good; single-step good only with aux loss; multi-step
trajectory propagation poor for ALL. The drift signature (fresh ~1px/step compounding to 14.8px@h12, EXP-011
ff7_k3) points to AUTOREGRESSIVE ERROR COMPOUNDING / exposure bias as the dominant remaining deficit — model
trains on near-clean true context, rolls out on its own slightly-wrong predictions. NOT yet method-decided.
Operating mode (Merlin, this session): work autonomously for several hours on motion prediction, "dont block
yourself" (=do NOT halt-and-wait for his review at each experiment gate this session), "dont break anything"
(keep all gates green, additive/config-gated changes only, frozen tokenizer/probe untouched, work on branch
feat/motion-prediction). "You can use the idea agent for inspiration" → method-architect spawned for an
independent mechanistic diagnosis + ranked proposals (writing tasks/T-016-architect-proposal.md).
Decision: (a) Dataset = occluded.npy (n_actions=2), matched to the existing baseline set + the calibrated
probe (OccludedBouncingEnv); measure motion at curtain-up (k=0). (b) Added safe `--max-episodes` train flag
(default=all) for fast subset A/B. (c) Run a budget-matched CONTROL baseline now (vanilla, short budget on a
subset) — method-agnostic, needed regardless of the chosen method → EXP-018. (d) Build a reusable curtain-up
motion eval harness (reusing EXP-011 diagnose.py fns: TF-1step, open-loop curve, copy-last/chance). The METHOD
choice is deferred to D-027 once the architect returns + I verify the design (critical-claim-verifier for any
novel objective). Cap 3 experiments per method decision.
Alternatives rejected: train on bouncing.npy (purer motion but DIFFERENT generator than the probe's
OccludedBouncingEnv → OOD for the calibrated instrument + incomparable to baselines); duplicate 1GB subset
npy files (—max-episodes flag is cleaner/safer); start a method immediately (no confirmed diagnosis/design).
Expected outcome: a short-budget vanilla baseline reproducing the known motion weakness (1-step > copy-last,
open-loop drift) at reduced absolute quality, serving as the A/B control; architect returns a compounding-
targeted proposal (rollout/scheduled-sampling/multi-step or context-distribution-matching family).
Would change my mind: if the architect's diagnosis credibly REFUTES compounding (e.g. shows velocity is
never represented in D's state even teacher-forced multi-step) → re-aim the method at representation, not
rollout. If the control baseline does NOT reproduce the motion weakness at short budget → the subset/budget
is too small to show anything; increase before trusting any A/B.
Spawns: EXP-018 (control baseline), method-architect (T-016), then D-027 (method) + EXP-019.. (treatment).

## D-027 | 2026-06-14
Context: EXP-018 confirmed the multi-step motion deficit = autoregressive error compounding/exposure
bias (teacher-forced pos_err FLAT in horizon for all models; open-loop compounds to chance; τ-sweep
flat → C0 ruled out). method-architect (T-016) ranked C1 (time-axis multi-step prediction loss) #1.
critical-claim-verifier (T-017 verdict, V-T017-C1) audited the C1 design: C-A TRUE-UNDER-CONDITIONS
(mechanism is DAgger/on-policy distribution-correction, NOT a "contraction map" — an anchored-GT loss
recovers the data's true (possibly expanding) gain; helps iff the deficit is off-manifold ACCURACY,
which EXP-018 shows is exactly the ff7/ff9 case); C-C(ii) the TBPTT-1 detach worry REFUTED (detached
step gradient is bit-identical to teacher-forcing the map at the model's own visited state — safe here
because every step has a GT anchor, unlike the T-014 relay which had none); C-D identity-when-off
PROVEN conditional on copying the ff9 guard pattern; C-B one open degenerate mode (prior-emission under
unlearnable drifted context — time-axis analog of the V-T013 shortcut).
Decision: implement C1 = config-gated `multistep_h` (0=identity) loss `_multistep_loss`, additive on
top of the existing diffusion loss, loss-only (no new params, inference/probe/FF7/FF9 untouched). Per
anchor: seed a short real context (~eval prefix length, ≥2 frames for velocity) @ context_signal, then
roll h self-steps — each step predicts the true successor z1[t+j] from a pure-noise (τ=0) target slot
given the model's OWN DETACHED self-generated context (TBPTT-1), finest-d flow loss. Reframe rationale
as DAgger/distribution-correction (per verifier). Mandatory gates: λ_multistep RAMP (warmup) + clean
val/diffusion ≤ 0.003 tripwire. Monitor: per-j multistep loss logged; context-vs-prior gap check in
eval (prior-emission detector). Then A/B on occluded subset: vanilla CONTROL vs C1, same subset/seed/
epochs, eval curtain-up open-loop + TF + val/diffusion.
Alternatives rejected: C0 (no τ-cliff, ruled out by EXP-018 P2); C2 scheduled-sampling fine-tune
(heavier inner-rollout loop; architect says stage AFTER C1 if late horizons still drift); C3 velocity
head (probe R²0.96 says velocity already decodable — likely solves a non-problem).
Expected outcome: C1's open-loop pos_err curve drops BELOW the budget-matched vanilla control
(especially mid-horizons h4–h12), WITHOUT clean val/diffusion regressing past ~0.003 and WITHOUT TF
(per-step map) regressing.
Would change my mind (tripwires): (1) open-loop unchanged vs control despite the multistep term
converging → deficit not off-manifold-accuracy (maybe intrinsic high-gain) → C1 insufficient, consider
C2 / re-diagnose. (2) val/diffusion regresses past ~0.003 → single-frame/multi-step capacity tension
dominates → back off λ / Pareto. (3) per-j multistep loss flattens to a context-independent floor at
large j (prior-emission) → mask/down-weight those j, reduce h.
Spawns: T-018 (C1 implementation + smokes), EXP-019 (vanilla control), EXP-020 (C1 treatment).

## D-028 | 2026-06-16
Context: Preliminary EXP-020 readout (partial C1 ckpt, epoch 17/40) showed a strong DIRECTIONAL win
for C1 over the vanilla control on curtain-up motion: teacher-forced per-step map FLAT+accurate (C1
2.6px h1->h24 vs control ~23px at chance), open-loop resists compounding far longer (C1 crosses chance
at h~20 vs control h=1), collapse monitor PASSES (C1 predDisp 4.6 vs gt 3.2 — coherent, not copy-last).
Caveats: budget mismatch (C1 only 42% trained) + control's TF anomalously weak (23px vs EXP-012
vanilla_s0 4.66px) — the 250-ep subset appears to cripple vanilla motion learning. Merlin's EXP-020
cancel was incidental (needed the GPU), NOT a redirect. He then directed: work continuously+autonomously
on the C1/motion idea overnight, NO escalation gates this session (just produce information), do not
break anything, work within the folders, and use training downtime to (a) extract repeatedly-used
code/evals out of experiments/ into a findable home and (b) clean up the repo (clear places for evals/
environments/models/training/data/readouts).
Decision:
 (1) Launch the FULL 40-ep EXP-020 C1 run (identical command to the partial, --fresh) for the clean
     same-budget A/B vs the EXP-019 control. Preserve the partial ckpt as c1_h4_s0.partial_ep17.pt;
     add experiments/EXP-020/run.sh to fix the provenance gap (no recorded command).
 (2) Quick control sanity-check: confirm EXP-012 vanilla_s0 (full-data) gets ~4.5px TF on THIS probe,
     isolating the 250-ep subset (not an eval-harness bug) as the cause of the control's chance-level TF.
 (3) Refactor during training downtime, SAFE + test-gated: create src/eval/ as the home for reusable,
     NON-frozen eval/rollout/diagnostic code; move experiments/EXP-018/probe_multistep.py ->
     src/eval/motion_probe.py and extract the reusable A/B harness (displacement monitor, cross-chance,
     per-horizon curves) out of EXP-020/ab_eval.py; update the few importers. Repo hygiene: remove
     scratch root test.py, stale src/__pycache__, stray dup src/autoencoder_bouncing.pt (after checking
     unreferenced). Document structure in REPO_MAP.md + sync CLAUDE.md. LEAVE the frozen probe
     (src/probe) and high-fanout core (dynamics_model.py / train_dynamics_model.py) IN PLACE; write a
     ready-to-execute reorg PLAN for the deeper models/training/envs relocation for Merlin to approve
     rather than executing a sweeping, hard-to-verify move while unsupervised.
Alternatives rejected: big-bang reorg of all of src/ overnight (violates "don't break anything";
asymmetric downside if a subtle import breaks while he sleeps); a larger-data A/B before the planned
same-budget one (run the designed experiment first; a larger-data confirmatory run can follow if C1 holds).
Expected outcome: full-budget C1 open-loop pos_err stays below the control across horizons with a
coherent displacement monitor and clean val/diffusion <= ~0.003; every gate test (kv/stream/ff7/ff9/
multistep) stays green through the refactor and probe/ab_eval reproduce prior numbers.
Would change my mind: full-budget C1 fails to beat the control or shows prior-emission collapse
(predDisp -> 0 / per-j floor) -> C1 insufficient, reconsider C2 scheduled-sampling or a larger-data
regime. Any gate test red after a refactor step -> revert that step immediately before proceeding.
Spawns: EXP-020 (full run), src/eval refactor, experiments/EXP-020/run.sh, REPO_MAP.md.

## D-029 | 2026-06-16
Context: EXP-020 full C1 A/B done — C1 SUPPORTED (open-loop below chance through h24; TF flat ~2px;
val 0.00305 < control 0.0082; collapse monitor coherent). Standing caveat: the 250-ep control is a weak
motion model (chance TF; sanity-check) so the A/B conflates "C1 learns a map at all on tiny data" with
"C1 fixes open-loop compounding on a COMPETENT per-step map" (the EXP-018 question). Dataset is 1000
episodes; the competent reference set (vanilla_s0, ff7_k3, ff9v2) was trained on all 1000 and already
probed in EXP-018 (TF ff7/ff9 ~1px, but open-loop compounds to chance ~h12-16). Autonomous session,
no escalation gates (Merlin asleep).
Decision: two next steps.
 (A) Cheap inference-only probe (EXP-022, no training): sweep context_signal at INFERENCE and measure
     the OPEN-LOOP pos_err curve on the existing competent ckpts (vanilla_s0, ff7_k3, c1_h4_s0). Tests
     the IDEAS "uncertainty-aware rollout" lever: does telling the model the context is less reliable
     (lower flat signal) change open-loop compounding? Discriminates whether a trained per-frame
     confidence channel is worth building. ~minutes GPU; reuses src/eval/motion.open_loop_curve.
 (B) EXP-021 (overnight): train C1 on the FULL 1000-ep occluded data (same C1 flags as EXP-020), then
     probe its open-loop+TF curve against the EXISTING competent set (vanilla_s0/ff7_k3/ff9v2). This is
     the confound-free compounding test: among models with a COMPETENT TF map, does C1 have the gentlest
     open-loop compounding? Not epoch-matched to the references' 100ep (infeasible: ~24min/epoch on 1000
     ep), but periodic checkpoints + the reported TF curves control for per-step accuracy. Probe the
     latest checkpoint in the morning; continuable.
Alternatives rejected: a 2nd matched vanilla+C1 larger-data pair (2 long runs, >12h, infeasible
overnight); a seed-1 confirmation of EXP-020 (addresses seed variance, but the CONFOUND is the bigger
caveat — resolve that first); full-data C1 at 100ep to budget-match (infeasible ~45h).
Expected outcome: (A) if the OL curve shifts with context_signal -> the confidence lever has signal
(esp. lowering it helps ff7_k3's compounding); if flat -> inert, drop it. (B) C1-full reaches a
competent TF map (~few px) and shows a gentler open-loop rise than ff7_k3/ff9v2 at matched TF.
Would change my mind: (A) flat OL vs context_signal -> confidence channel inert, do not build a trained
version. (B) C1-full's open-loop compounds as fast as ff7_k3 despite a competent TF map -> the EXP-020
win was the tiny-data regularization effect, NOT a general compounding fix -> re-aim (C2 scheduled
sampling / re-diagnose). C1-full TF fails to reach competence in the budget -> inconclusive, extend.
Spawns: EXP-022 (context_signal probe), EXP-021 (C1 full-data overnight).

## D-030 | 2026-06-16
Context: T-019 repo reorg, rewritten 2026-06-16 + critical-claim-verifier pass (SOUND-WITH-FIXES,
5 blocking fixes folded in). Merlin reviewed and chose **option B** ("Id say move it. Lets get
this right this once") — physically relocate the FROZEN probe spine into src/evals/ rather than the
lower-risk wrap-don't-move (option A). Greenlit the full high-fanout move.
Decision: Execute the full T-019 reorg in 4 gated phases, clean end state (update all callers, no
permanent shims): (P1) env/data split — src/envs/ (BaseEnv ABC + env classes) and src/data/
(generators+loaders), viewers→src/interactive/; (P2) src/evals/ with base.py+registry+_shared, MOVE
the frozen spine in (evals/revisit, evals/position_consistency) + motion/rollout_view; (P3)
models/training/tests split (src/models, src/training, src/tests); (P4) docs (CLAUDE.md, REPO_MAP).
Gates: all 5 dynamics gate tests stay green after each phase; the FROZEN-SPINE BYTE-DIFF gate
(re-run revisit_probe on ff7_k3_s0.pt, diff results.json vs pre-move baseline) runs at the end of
P1 (env dep moves) and again after P2 (spine moves). Per-module commits, each revertible.
Alternatives rejected: option A (wrap-don't-move) — Merlin explicitly wants the spine physically in
evals/, accepting the churn for a clean end state. Big-bang single commit — un-revertible, high blast
radius. Leaving datasets/checkpoints move for a later low-value pass (kept at repo root).
Expected outcome: identical frozen-probe numbers (byte-diff passes), all gate tests green, all 32
importers updated, no broken experiment scripts; cleaner tree that scales to many envs/evals.
Would change my mind: byte-diff MISMATCH after a pure git-mv+import-edit (means a logic change crept
in → STOP, revert, investigate); or a gate test going red that a one-line import fix doesn't resolve.
Spawns: T-019 execution (no new EXP; this is infra). BOARD tracks per-phase.

## D-031 | 2026-06-16
Context: Executing D-030 P2 (move frozen spine into src/evals/). The ~12 historical experiment
scripts import the spine via a non-uniform sys.path-bootstrap + bare-import pattern (two styles:
`from probe_env` and `from probe.revisit_probe`). Rewiring all of them is per-file surgery,
high blast radius, and not cheaply verifiable (probe is stochastic + needs GPU — only
py_compile-checkable). Asked Merlin; he chose "Frozen to their commit."
Decision: Treat experiments/EXP-NNN/ + verify-* scripts as IMMUTABLE provenance records pinned to
the commit they ran at. P2/P3 rewire ONLY live code (src/ internals, gate tests, the new Eval
interface). Historical experiment scripts are left pointing at old paths; to rerun one, check out
its commit (whose src/ matches). NEW experiments use the new structure.
Consequence: experiments/ will NOT import against new src/ at HEAD — this is BY POLICY, not breakage.
Live-importer audit: the only live importers of the moved spine/eval are the spine's own internal
cross-imports (gate tests + play_dynamics import the MODELS, not the probe), so P2's live rewire is
just the moved modules' internals.
Alternatives rejected: rewire all experiments (clean HEAD but large blast radius, subtle-break risk,
not cheaply verifiable) — Merlin judged provenance-freeze cleaner and safer.
Would change my mind: if a CURRENT/live tool (not a historical EXP) turns out to import the old
probe paths, it must be rewired (not frozen).

## D-032 | 2026-06-16
Context: Merlin redirected live: STOP the paused C1/motion research thread (EXP-021 morning eval)
and first build a NEW, "more solid" memory env. Rationale (his): the current bouncing/occluded env
is too fluid — arbitrary continuous positions, arbitrary continuous colors, arbitrary bg colors —
which muddies measurement. He wants a clean, discrete, low-entropy env where recall is crisp.
Decision: Build a GridWorld occluded-memory env: 64x64, 8x8 black grid (2px lines -> 6x6 visible
cells), solid background color from {red,green,blue,pink}, a square of a DIFFERENT one of those four
colors filling one cell's 6x6 interior. Square moves exactly one cell/tick in one of 8 directions
(4 orthogonal + 4 diagonal), reflecting off walls. Same curtain occlusion as OccludedBouncingEnv
(absolute action 0=revealed / 1=occluded; physics runs behind the curtain). Discrete state =>
position recall is an 8x8 classification and color recall is 4-way — both crisply measurable
(aligns with the measurement-validity standing order).
Alternatives rejected: keep tuning the fluid env (Merlin judged it too noisy to measure cleanly);
deferred — kept research thread paused, not dropped (EXP-021 ckpt + findings preserved).
Expected outcome: a BaseEnv subclass + datagen writer + a sample image Merlin approves, before any
training. Discrete recall metrics replace the continuous ΔRGB/position probes for this env line.
Would change my mind: if discretization removes the very failure mode we study (i.e. the model
trivially memorizes 8x8 positions even past the window) — then the env is too easy and needs more
state/entropy. Watch for this at first baseline.
Spawns: T-020 (GridWorld env build) — GATED on Merlin approving the sample look + design questions.

## D-033 | 2026-06-16
Context: Merlin: "write clean eval metrics for [GridWorld] ... making the right decision on what to
score is vital." The discrete env's payoff vs the fluid env: in the fluid env, scoring ANYTHING
needed fuzzy ball localization ("where is the ball, to read its color?"), so the probe fell back to
latent-MSE and POSITION was demoted to a drift-confounded NON-metric (GOAL H2/H3 — the blocked axis).
GridWorld makes localization EXACT and closed-form, which unblocks a clean position metric.
Decision: GridWorld recall eval (src/evals/gridworld/), scored at each REVEAL frame bucketed by
occlusion length k:
  PRIMARY (headline): position recall accuracy (exact 8x8 cell match) vs k  [+ Chebyshev cell-dist
    for graded credit]; color recall (4-way) vs k.
  SECONDARY / confound check: background recall (4-way) vs k.
  DIAGNOSTIC: reflection split (bounced-during-occlusion vs not) — learned the walls vs ballistic;
    readout margin (top1-top2) — flags smeared/hallucinated preds.
  REFERENCE LINES: oracle ceiling (=1.0, instrument self-test), copy-last / no-memory baseline
    (decays as the true square moves — beating it IS "has memory"), chance (1/64 pos, 1/4 color).
  CONTROL (needs the model, lives in the Eval adapter): matched-horizon open-rollout (model run
    curtain-UP same horizon) separates "can't track motion even in the clear" (GOAL's documented
    base-dynamics deficit) from "memory lost past window."
Readout is deterministic (median-cell = bg, max-distance cell = square, nearest-palette = color);
provably exact on true frames, so — unlike the fluid position-consistency metric Merlin declined to
freeze (uncertain strength) — this instrument is self-validating (oracle=1.0).
Rationale for promoting POSITION to headline: it is the ONLY attribute that changes during occlusion,
so it is the one that genuinely requires integrating hidden dynamics (color/bg/identity are static
holds). It is exactly the axis the fluid env couldn't measure cleanly.
Validation (test_gridworld_eval.py, all pass): readout exact on true frames; oracle=1.0; copy-last
decays 0.08@k1->0.00@k29 (square moves every tick -> freezing nearly always wrong -> a CLEAN bar);
random ~ chance (0.017 ~ 1/64).
Status: NOT YET FROZEN. Present to Merlin for sign-off on the scoring choices before locking it as
the GridWorld spine + wiring the model adapter (needs a gridworld-trained tokenizer+dynamics).
Alternatives considered: latent-token MSE (decoder/detection-free, continuity with fluid probe) —
kept as a possible secondary, but pixel readout is now clean and far more interpretable, so it is
the headline. Full-frame exact reconstruction — rejected (decoder-bound, not decomposable).
Would change my mind: if Merlin wants color-first continuity over position-first; or if the model
adapter reveals the decoder is too weak to localize from predicted frames (then add latent-MSE).
Spawns: T-020 eval core (done, validated); model adapter pending gridworld pipeline.

## D-034 | 2026-06-16
Context: Merlin: models are env-specific (don't transfer); flat names like 'dynamics_model.pt' hide
the env and are "incorrect and irritating." Provenance verified (configs + git + EXPERIMENTS/journal):
trained_autoencoder.pt = tokenizer EXP-006 (occluded.npy); my_dynamics.pt = vanilla dynamics EXP-007
(occluded, n_actions=2, retired baseline); dynamics_bouncing.pt = EXP-005 unconditional BOUNCING;
src/autoencoder_bouncing.pt = bouncing tokenizer.
Decision: checkpoints/<env>/<role>.pt. Moved (gitignored artifacts): -> occluded/tokenizer.pt,
occluded/dynamics_vanilla.pt, bouncing/dynamics.pt, bouncing/tokenizer.pt. checkpoints/gridworld/ for
the new pipeline. Repointed all LIVE defaults + frozen-spine argparse default PATHS (probe.py,
consistency.py) — PATH-ONLY, measurement logic byte-unchanged (this is the §8 logged decision for
touching the frozen files). Fixed a latent NameError (train_dynamics --tokenizer default referenced
undefined _TOKENIZER_DIR). Append-only history (DECISIONS/EXPERIMENTS/ESCALATIONS/tasks) left intact
as provenance of what the paths WERE.
Would change my mind: if a frozen experiment rerun needs the old root path, pass it explicitly (the
artifact is external/gitignored; D-031 checkout-to-rerun semantics unaffected by logic).
Spawns: T-021 (done).

## D-035 | 2026-06-18
Context: Merlin: "Its time to work on the cluster interface scripts." This is the long-deferred T-003
(BOARD "Deferred until H3 needs heavy training"; protocol §6 defines 9 wrapper verbs as the ONLY
sanctioned cluster interface). Now needed: the GridWorld pipeline (tokenizer+vanilla dynamics on the
full 6.9GB set) is overnight/OOM territory on the 4070 (~25h local for a 10-ep tokenizer) → cluster.
This implicitly resolves ESC-016 Q2 (compute tier = cluster). Discovery: NO ssh config, NO cluster
host aliases, NO ControlMaster socket, NO checked-in sbatch script, NO lockfile — all cluster access
has been manual by Merlin, so site specifics must come from him.
Decision: Build `scripts/` per §6 — a single connection helper `_common.sh` (ssh over a Merlin-
authenticated ControlMaster socket; machine-parseable first-stderr-line error contract AUTH_DEAD/
QUOTA/BAD_REF/BAD_CONFIG; NEVER re-auth), all site specifics isolated into one `scripts/cluster.env`
(two stanzas), an sbatch template with a venv-by-sha256(requirements.txt) prologue, and the 9 verbs
(sync_code, submit_job, job_status, fetch_logs, wait_for_jobs, pull_results, cancel_job [refuses ids
not in EXPERIMENTS.md], cluster_health [reports BOTH clusters], clean_run [restricted to runs/ subtree]).
Merlin's answers (2026-06-18): NO default cluster — two named clusters **ferranti (H100)** and
**galvani (A100)**, `--cluster` REQUIRED (choice depends on his live fairshare+queue); code sync =
remote git fetch + checkout from origin. Strategy: the config file IS the question — build the whole
scaffold now (touches nothing on the cluster), Merlin fills cluster.env + opens the socket, then test
Phase 1 (read-only verbs) live before anything mutating.
Alternatives rejected: raw ssh/sbatch ad hoc (violates §6, the wrappers ARE the sandboxed interface);
rsync-push code sync (Merlin chose remote fetch+checkout — cleaner SHA provenance); a hardcoded default
cluster (Merlin: pick depends on live load).
Expected outcome: a complete, committed `scripts/` layer; `cluster_health.sh` + `job_status.sh` run
clean against the live socket once Merlin fills cluster.env and authenticates; first real job = the
GridWorld tokenizer on the chosen cluster.
Would change my mind: if the actual login flow is NOT a reusable ControlMaster socket (e.g. a token
that must be minted per-command, or a jump host that breaks multiplexing) — then the connection helper
needs rework and I escalate before building on it.
Spawns: T-003 build (Phase 1 read-only verbs → Phase 2 submit/sync → Phase 3 fetch/wait/pull/cancel/clean).

## D-036 | 2026-06-18
Context: First live test of the T-003 wrappers (D-035) failed AUTH_DEAD even though Merlin had a
master session up. Root cause: the local execution environment was ambiguous. Merlin opened the
ControlMaster socket in **WSL** (Ubuntu); the orchestrator runs the Bash tool as **Git Bash (MSYS2)**.
WSL and Git Bash are separate ssh stacks with separate $HOME and separate unix-socket namespaces, so
a socket opened in one is invisible to the other. PowerShell is unusable for this regardless (can't
run the bash wrappers; Windows-native OpenSSH has no ControlMaster multiplexing → would 2FA every
command). The opener (human) and the wrapper-runner (orchestrator) MUST share one environment.
Decision: **Standardize the cluster wrappers on WSL.** Merlin opens the master in WSL
(`scripts/open_master.sh --cluster ferranti`); the orchestrator invokes every wrapper through WSL
(`wsl.exe -e bash -lc "cd /mnt/c/.../transformer && bash scripts/<verb> ..."`). Rationale: WSL is the
most robust ssh+rsync stack on this machine, it is Merlin's natural environment (so the master socket
won't drift again — the exact failure we hit), and the only cost (invoking via wsl.exe) is on the
orchestrator side, invisible to Merlin. **Deliberate split by concern:** local 4070 *training* stays
in Windows/Git-Bash (CUDA venv `venv/Scripts/python.exe`); *cluster orchestration* lives in WSL. The
wrappers never run training locally, so the split is clean. cluster.env's `~`-relative CONTROL_PATH
resolves correctly in whichever env runs it, so no path edits needed as long as opener==runner==WSL.
Live-verified this session via WSL: cluster_health (real fairshare/queue/disk), job_status (squeue),
submit_job --dry-run (real config), sync_code BAD_REF (remote git + error contract). Not yet tested
live: the submit→sacct→logs→wait→pull mutating pipeline (needs one trivial queued job — gated on Merlin).
Alternatives rejected: Git Bash (works + supports ControlMaster, and would let the orchestrator run
wrappers natively with no wsl prefix — but MSYS2 ssh/rsync has more path/socket edge cases, AND it
creates a recurring human/orchestrator env-mismatch footgun since Merlin prefers WSL); PowerShell
(disqualified above).
Would change my mind: if WSL→cluster networking proves flaky, or rsync over /mnt/c is too slow/buggy
for result pulls — then reconsider Git Bash (orchestrator-native) and require Merlin to open the
master there.
Spawns: HOWTO/cluster.md + scripts/README.md env documentation; cosmetic: none.

## D-037 | 2026-06-18
Context: D-035/036 verified the cluster wrappers' read-only path live. Merlin: run a FULL mini
end-to-end pipeline to "get all the details right" — env setup (venv build), git checkout, data
generation ON the cluster, actual training, W&B integration, and moving the model back to the laptop.
Decision: Run a tiny throwaway pipeline on ferranti (run name `cluster-smoke`): generate a 16-episode
gridworld_mini.npy on the node, train the tokenizer 1 epoch (bs8, MSE, --fresh) with W&B logging to a
throwaway project `cluster-pipeline-test`, save the checkpoint into the run dir, then pull it back with
pull_results --what checkpoints. Exercises every verb + integration: sync_code, submit_job, the venv-
by-hash prologue (first torch install), job_status/sacct, fetch_logs, wait_for_jobs, pull_results, and
clean_run (cleanup). Pre-req fixes folded in: requirements.txt UTF-16→UTF-8 (would have broken pip),
repo-wide LF (.gitattributes), branch pushed to origin (sync_code fetches from GitHub).
Expected outcome: green job; checkpoint lands locally; a W&B run appears under cluster-pipeline-test.
Would change my mind / tripwires: venv build fails (dependency/encoding bug) → fix; W&B doesn't auth on
node (no ~/.netrc) → surface + decide key vs netrc; rsync of the .pt fails → pull_results bug. Any of
these is exactly what this rehearsal exists to catch before the real GridWorld run.
Spawns: run `cluster-smoke` (infra validation row in EXPERIMENTS.md).

## D-038 | 2026-06-18
Context: Merlin flagged a tokenizer-overfit risk in GridWorld: the tokenizer patchifies in 8x8
patches, and the old env used 8x8 cells (6px interior + 2px line, 8px stride) — patch grid aligned
to cell grid, so the tokenizer can trivially overfit one cell per patch and get stuck in a bad local
minimum. He specified a new geometry that breaks the alignment.
Decision: Change GridWorldEnv geometry to **6x6 cells, 10px stride** (8x8 interior + 2px line), with
a **3px black border** each side: 3 + 6*8 + 5*2 + 3 = 64. Cell i interior starts at pixel 3+10*i.
The 10px cell stride is deliberately NOT a multiple of the tokenizer's 8px patch — cells straddle
patch boundaries, so no patch maps to exactly one cell. Recall is now 6x6=36 cells (chance 1/36) +
4-way color; 8 directions unchanged. Overwrote the env in place; regenerated gridworld.npy (3000x200)
on the new env and DELETED all stale-env artifacts (old gridworld.npy + smoke subset + variants,
checkpoints/gridworld/tokenizer.pt [trained on old env], experiments/EXP-023 recon, smoke log).
Updated readout/recall (chance 1/36 derived from GRID_N), tests (geometry + chance assertions; the
small grid's short bounce-recurrence period made the old "largest-k copy-last < 0.5" assertion brittle
on a 1-event bin → switched to a support-weighted large-k average), CLAUDE.md/REPO_MAP.
Validation: geometry pixel-exact (axis runs border3/[cell8,line2]x5/cell8/border3) + visual preview
(experiments/gridworld_v2_preview/geometry.png) + both gate tests green (oracle 1.0, random≈1/36,
copy-last decays 0.12@k1→0.17 weighted).
Note (research): on the smaller 6x6 bouncing grid, copy-last's no-memory floor is a bit higher than
on 8x8 (more position recurrence) — the memory task is marginally easier to fake via periodicity.
Flag to watch when reading the first GridWorld baseline; not a blocker.
Would change my mind: if 6x6 is so periodic that ballistic/periodic extrapolation trivially solves
occluded position (copy-last/no-memory ≈ oracle) — then the env is too easy and needs more cells/state.
Spawns: regenerated dataset; re-runs the (now-invalidated) tokenizer pipeline on the cluster.

## D-039 | 2026-06-18
Context: T-003 cluster pipeline validated (D-037) and the GridWorld env reworked to the anti-overfit
6x6/10px geometry (D-038). Merlin: kick off the REAL GridWorld tokenizer run on ferranti. This is the
ESC-016 Q2 payoff — the first real cluster training, producing the frozen tokenizer backbone for the
GridWorld dynamics stage.
Decision: One ferranti job `gridworld-tok-v2`: (1) regenerate full data ON the node
(generate_gridworld 3000x200, deterministic seed) for self-contained provenance; (2) train the
tokenizer with the PROVEN recipe — LPIPS (vgg) + W&B, --fresh, 40 epochs, bs32, lr3e-4, checkpoint to
runs/gridworld-tok-v2/tokenizer.pt; (3) render recon strips (--save-recon) for the present-then-stop
view. Code @ cf0a8e1 (committed env + datagen + --save-recon). venv reused (requirements unchanged
since the smoke). W&B project transformer-C-tokenizer, run gridworld-tok-v2.
Expected outcome: faithful gridworld reconstruction (crisp grid + colored square in the recon strips,
low val MSE), NO latent collapse (latent_cos well below 0.7, pred_std high). Frozen tokenizer ready
for the vanilla dynamics stage.
Would change my mind / tripwires: latent collapse (latent_cos high / uniform-mean recon) → revisit
MAE-mask / bottleneck; recon drops the square or smears colors → the 10px-vs-8px patch change hurt
recon (would be surprising — it should HELP); val MSE not converging by 40ep → extend or raise lr.
40 epochs is a first solid pass on a simple env; extend if not converged (decide at present-then-stop).
Spawns: ferranti run gridworld-tok-v2 (EXPERIMENTS row); → vanilla dynamics next.

## D-040 | 2026-06-18
Context: Merlin refined the GridWorld eval (supersedes the D-033 proposal's raw-Chebyshev position
credit). Two metrics, build now (model-agnostic; wiring into training/W&B discussed later).
Decision:
  (1) POSITION — graded per reveal event by Chebyshev cell-distance d: exact=full, partial MINIMAL and
      gone by d=3. Default credit POSITION_CREDIT = {0:1.0, 1:0.25, 2:0.0625, >=3:0.0} (each cell of
      distance quarters the credit, then cut to 0). Headline = mean graded `position_score` vs k; the
      strict exact-match `position_acc` (chance 1/36) is reported alongside.
  (2) COLOR — 4-way accuracy of BALL and BACKGROUND, using the dead-simple detection Merlin specified:
      take every cell's mean colour, the MOST DIFFERENT cell (farthest from the median = bg) is the
      ball; nearest-palette gives the class. (Already how readout.py works — kept, clarified.)
  (3) STATISTICAL RELIABILITY — score over MANY reveal events; aggregate reports per-k counts AND a
      standard error per curve so a k-bin's score is a real "chance of keeping position", not luck on a
      few rollouts. A driver evaluates any frame source (oracle/copy-last/model) over a whole dataset.
References unchanged: oracle (graded & exact must be 1.0 — instrument self-test), copy-last (no-memory),
chance (analytic, grid-averaged for the graded score; 1/36 exact). NOT YET wired into training — built as
a reusable scorer + flat per-k dict ready for a periodic W&B eval hook (Merlin: "we'll talk about where").
Expected/validation: oracle position_score==1.0 & color==1.0 at all k; random≈analytic chance; copy-last
decays; on the new 6x6 data the per-k curves have large n with small SE.
Would change my mind: if the {1:0.25,2:0.0625} falloff proves too generous/stingy in practice, it's a
one-line constant change (logged). Spawns: refined src/evals/gridworld/{recall,readout}.py + test.

## D-041 | 2026-06-18
Context: Merlin's W&B dashboard for gridworld-tok-v2 showed the classic GPU-starvation signature on
the H100 — utilization sawtoothing 0<->100%, power ~35% (245/700W), VRAM 32%, temp ~35C. Diagnosed
from our code (not generic): train_tokenizer.py built its DataLoader with all defaults
(`num_workers=0`, no pin_memory/persistent_workers/prefetch) -> every batch loads synchronously on the
main thread (weka memmap read + uint8->float32) while the GPU idles. bf16 autocast was ALREADY on
(not the issue). Compounding: the SLURM job requested only cpu=2, so even adding workers would contend.
Decision: (1) train_tokenizer DataLoader -> num_workers (auto from SLURM_CPUS_PER_TASK/os.cpu_count(),
cap 8; --num-workers override) + pin_memory + persistent_workers + prefetch_factor=4 + drop_last;
(2) submit_job.sh + job_template.sbatch -> `--cpus N` => #SBATCH --cpus-per-task (default 8) + export
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK; (3) torch.set_float32_matmul_precision("high") (free TF32 on
remaining fp32 matmuls). Resubmit gridworld-tok-v2 with --cpus 8, bs64 (VRAM headroom), and watch
it/s + util to confirm. Cancelled 405623 (per Merlin "stop the current run and make a faster one").
Expected outcome: GPU util pinned high, large it/s increase (est. 3-5x from feeding alone) -> the real
tokenizer trains in ~1-1.5h instead of ~5.7h. Benefits ALL future cluster training (dynamics, sweeps).
Would change my mind: if util stays low after the fix, the bottleneck is elsewhere (weka IO, or the
per-step .item() syncs at lines 547/554, or LPIPS-VGG compute) -> next: on-device metric accumulation,
torch.compile, or shard the data. Attention is hand-rolled (QK-norm/soft-cap) so no FlashAttention/SDPA.
Spawns: resubmit EXP-024; deferred follow-ups (per-step sync removal, torch.compile) if still starved.

## D-042 | 2026-06-18
Context: EXP-024 GridWorld tokenizer FAILED — recon drops the ball entirely (background+grid perfect, blue
ball absent in every recon frame, all bg colors; evidence experiments/gridworld-tok-v2/_block0/_block2.png).
Merlin caught it visually; aggregate val MSE 0.00364 was BLIND to it because the ball is ~1/36 cells (~1% of
pixels) so plain mean MSE (train_tokenizer.py:520/574) + LPIPS have near-zero gradient for it → the tokenizer
settled into the trivial background-only local optimum, fast (val MSE plateaued ~ep3). Latents encode no ball
→ unusable backbone. Merlin approved the fix + local-validate-first plan ("yes that's fine").
Decision:
  (1) FOREGROUND-WEIGHTED RECON LOSS in train_tokenizer.py. Per-pixel weight w = 1 + alpha*fg, where fg is
      derived from the VISIBLE input clip (NOT privileged — tokenizer reconstructs fully-visible frames): the
      per-pixel TEMPORAL-MEDIAN over the clip is the static scene (bg+grid+border are constant across a clip;
      the ball is in a different cell each tick so it medians out), and fg = (|frame - temporal_median| summed
      over RGB) > --fg-thresh isolates the moving ball (refinement within Merlin's "deviation-from-background"
      family — temporal-median template isolates the BALL specifically, vs spatial-median which would also
      upweight the static grid/border). Loss = weighted_mse + LPIPS (LPIPS unchanged). New args --fg-weight
      (alpha, default 0.0 = old uniform behavior, opt-in) + --fg-thresh (default 0.1).
  (2) BALL-REGION VALIDITY METRIC to W&B: val/fg_mse and val/bg_mse (recon MSE restricted to fg vs bg pixels)
      + val/fg_frac. fg_mse is the headline guard + mid-training tripwire — if the ball is dropped, fg_mse
      stays high while bg_mse→0 even as aggregate val/mse looks fine. Never trust aggregate MSE for a sparse
      object again (measurement-validity).
  (3) VALIDATE LOCALLY FIRST (4070, small subset / few epochs): confirm the recon visibly keeps the ball and
      val/fg_mse drops, sweeping alpha if needed, BEFORE promoting to a full cluster retrain (§6).
Expected outcome: with adequate alpha the recon shows the ball at the correct cell+color; val/fg_mse drops
substantially (vs the failed run where it would be ~chance); bg stays good. Then a known-good cluster run.
Would change my mind / tripwires: if even high alpha can't bring the ball back, the bottleneck (4 latents x 64)
or MAE-mask is the constraint, not the loss → revisit capacity/MAE (less likely — capacity easily fits
bg(4)+ballcolor(4)+ballcell(36)). If fg mask is dominated by the curtain/occlusion rather than the ball,
tighten thresh or restrict to reveal frames. Spawns: train_tokenizer.py changes + local validation EXP.

## D-043 | 2026-06-23
Context: Merlin reported (from watching the W&B dashboard) that the failed GridWorld tokenizer had a
LOSS EXPLOSION mid-training — loss went really low, then exploded, then recovered to a WORSE final
value. I pulled the EXP-024 W&B curve (run 5d38b2nc) and it confirms + RE-DIAGNOSES the D-042 failure:
  epoch  8: val/mse 0.00226, latent_cos 0.37
  epoch  9: val/mse 0.000613, latent_cos 0.29   <- BEST; latents becoming content-bearing (the ball)
  epoch 10: val/mse 0.0380 (62x), train/mse 0.0387 (17x), pred_std 0.37->0.25   <- EXPLOSION
  epoch 11-12: partial recovery (val back to ~0.004 by ep12)
  epoch 13-29: plateau at val/mse ~0.0037, NEVER returns below the ep9 0.0006  <- recovered worse
The "dropped ball" recon strips were rendered from the POST-explosion ep29 checkpoint. So the EXP-024
read ("pure sparse-target gradient collapse, plateaued ~ep3") was WRONG: plain MSE+LPIPS WAS learning
the ball (val/mse fell 6x below the background-only floor by ep9), an instability destroyed it, and the
single-checkpoint OVERWRITE threw the good ep9 model away. Mechanism (textbook): at the loss minimum the
Adam 2nd-moment v shrinks, so the effective step lr/sqrt(v) grows until one batch lands an oversized
update -> blowup. clip_grad_norm_(1.0) (already on, line 681) does NOT catch this — it clips the
gradient, not the Adam-scaled step. Grounded in code: optimizer was AdamW default betas (0.9, 0.999);
logging was per-EPOCH only (step=epoch) with grad-norm never logged; checkpoint overwritten every epoch.
Decision (fix = stability + observability + safety net; foreground weighting kept but DEMOTED to a nudge):
  (1) STABILITY (mechanistic): AdamW beta2 -> 0.95 (faster-adapting v, the direct fix for the
      low-loss blowup; passed explicitly for this run, arg default left 0.999 so frozen
      bouncing/occluded reproducibility is untouched). eps arg exposed (default 1e-8) as a secondary
      knob if needed.
  (2) STABILITY (backstop): --grad-spike-mult (=5.0 this run): capture the pre-clip grad norm; skip
      opt.step() (zero_grad, advance scheduler) when the norm is non-finite or exceeds 5x its running
      EMA after the warmup grace. Catches the spike-shaped manifestation; beta2 catches the cause.
  (3) SAFETY NET (directly fixes Merlin's pain): best-checkpoint by val/fg_mse (the ball-region recon
      guard, D-042) -> canonical --checkpoint holds the BEST ball-encoding model; a _last.pt holds the
      last. An explosion can never again discard the good model (we'd have kept ep9).
  (4) OBSERVABILITY: per-step W&B logging (--log-every 50) of train/step_mse, train/step_loss,
      train/grad_norm(+ema), cumulative skips — so an intra-epoch explosion is VISIBLE (30 epoch points
      could not show it). Switched tokenizer W&B logging to auto-step with explicit global_step/epoch
      fields (per-step + per-epoch share one monotonic axis).
  (5) FOREGROUND WEIGHTING kept but MODEST (--fg-weight 10, was going to be 30): the ep9 evidence shows
      plain MSE already encoded the ball, so fg-weighting is a nudge toward the 1%-of-pixels ball, not
      the load-bearing fix. val/fg_mse + val/bg_mse logged regardless (validity guard).
  (6) COMPUTE: cluster (ferranti), per Merlin "do even the tests on the cluster, it's faster" + always
      W&B. Same pipeline as EXP-024 (datagen 3000x200 -> tokenizer 30ep bs64 lr3e-4 LPIPS-vgg --fresh
      -> recon strips), + the above flags. EXP-025.
Expected outcome: NO loss explosion (grad_norm stays bounded; if a spike occurs it is skipped and
logged); val/mse continues BELOW the ep9 0.0006 and keeps improving; val/fg_mse drops substantially
(ball reconstructed); recon strips VISIBLY show the colored square at the right cell+colour, distinct
from background. The canonical checkpoint is the best-fg_mse model.
Would change my mind / tripwires:
  - If it explodes AGAIN despite beta2=0.95 + spike-skip -> the cause is not Adam-v (e.g. bf16, a bad
    batch, LPIPS-scale drift); next: raise eps to 1e-6, inspect the logged grad_norm spike + the batch.
  - If NO explosion but the ball is STILL dropped (val/fg_mse stays ~background-only, recon has no ball)
    -> sparse gradient IS the binding constraint after all; raise --fg-weight and/or revisit
    bottleneck(4x64)/MAE-mask. (Less likely given ep9.)
  - If best-fg_mse keeps selecting a very early epoch (fg_mse never really improves) -> fg mask is
    dominated by the curtain not the ball; restrict the metric to revealed frames.
Spawns: train_tokenizer.py stability/logging changes; sync+submit EXP-025 on ferranti; EXP-025 row.

## D-044 | 2026-06-23
Context: EXP-025 produced a usable GridWorld tokenizer (val/mse→4.7e-6, no explosion, no
collapse; recon strips show the colored square at the right cell + colour ≠ bg). Presented
present-then-stop (ESC-017). Merlin: "We now have the working tokenizer." = acceptance.
Decision: FREEZE it as the GridWorld backbone. Copied experiments/gridworld-tok-v3/tokenizer.pt
(best=ep29) → checkpoints/gridworld/tokenizer.pt. Provenance: W&B run 70k76148 @ f1e3d6c (EXP-025).
This is the frozen latent space all GridWorld dynamics / recall work builds on.
Alternatives rejected: train a 2nd seed first (no instability signal warranting it; ball clearly
encoded — defer multi-seed unless a downstream result looks tokenizer-limited).
Expected outcome: dynamics + recall pipeline can now be built against a stable latent space.
Would change my mind: if the tokenizer-roundtrip recall ceiling (next experiment) shows the
latents do NOT preserve enough position/colour to read out at reveal frames → tokenizer is the
bottleneck, revisit (e.g. more latents / bottleneck_dim / fg-weight) before any dynamics run.
Spawns: ESC-017 resolved; next = tokenizer-roundtrip recall ceiling check, then model adapter.

## D-045 | 2026-06-24
Context: Tokenizer frozen (D-044). Merlin: "lets get the eval down" then "Thats fine, you can
continue with that" — approving (a) FREEZE the GridWorld recall eval design as specified, with
per-k judging + off-grid k {3,6,12,16} for the periodic W&B eval, and (b) run the tokenizer-
roundtrip recall ceiling check. Resolves ESC-016 Q1.
Decision: FREEZE the GridWorld recall eval CORE — `src/evals/gridworld/readout.py` (exact
read_square) + the scoring/aggregation/baselines in `recall.py` (score_episode, aggregate, graded
position_score per D-040, oracle/copy-last/chance) — as a result-defining spine. Frozen as of this
commit. Periodicity handling: the 6×6 bounce has period 10 → copy-last spikes to 1.0 at k≡9 (mod 10),
so memory is judged PER-K (beat copy-last at each k), NEVER by a single averaged scalar; the periodic
during-training W&B eval uses OFF-GRID k {3,6,12,16}. MODEL-based frame sources (tokenizer-roundtrip,
dynamics-rollout, matched-horizon control) live in a SEPARATE adapter that imports the frozen core —
they do not edit the frozen scorer. Future changes to the core are logged decisions (§8).
Alternatives rejected: single averaged recall scalar (inflated by the period-10 copy-last spikes);
keeping it unfrozen (method experiments would silently redefine prior results).
Expected outcome: a stable yardstick; the tokenizer-roundtrip ceiling and all dynamics recall curves
are comparable against fixed oracle/copy-last/chance references.
Would change my mind: if a downstream result exposes a scoring flaw (e.g. readout mis-snaps a colour,
or graded credit hides a real effect) → logged decision to amend, re-run affected comparisons.
Spawns: ESC-016 Q1 resolved; EXP-026 tokenizer-roundtrip recall ceiling.

## D-046 | 2026-06-24
Context: GridWorld tokenizer frozen (D-044), recall eval frozen (D-045), latent ceiling == oracle
(EXP-026 → latent not the bottleneck). Merlin approved (ESC-018) proceeding to the vanilla dynamics
baseline. §8: baselines are sacred — the unmodified DreamerV4-style model runs through the identical
frozen probe suite under identical provenance, and future GridWorld memory methods reuse this budget.
Decision: train the UNMODIFIED vanilla dynamics model (no ff7/ff9/multistep; all memory objectives
off) on gridworld.npy with the frozen GridWorld tokenizer, on ferranti. Config: n_actions=2 (curtain
up/down, auto-detected), context-length 16 (= tokenizer window), bs64, lr3e-4, epochs 80, seed 0,
grad-clip max_norm=1.0 (just added — train_dynamics was unclipped; tokenizer/LM already clip). W&B on.
Output → checkpoints/gridworld/dynamics_vanilla.pt + experiments/EXP-027/.
Alternatives rejected: memory model first (must establish the no-memory floor first); local training
(3000×200 too slow on the 4070 — smoke validated the pipeline, cluster does the real run); higher bs
(start bs64 known-safe = tokenizer budget; bump only if util shows headroom — re-profile on run 1).
Expected outcome: stable training (grad-clip prevents the tokenizer-style explosion). On the recall
eval: position recalled WITHIN the 16-frame window, then cliffing toward copy-last/chance for k beyond
the window (vanilla has no memory carrier); colour retained within window, likely lost past it. This is
the baseline curve memory methods must beat per-k.
Would change my mind / tripwires:
  - Vanilla recalls position ≈oracle well PAST its window even OFF-period → env/eval too easy or
    leakage → HALT (also the D-038 too-easy tripwire). Escalate.
  - Vanilla fails position even WITHIN window or in the clear (matched-horizon open-rollout control) →
    a tokenizer-latent or training-budget problem, NOT a memory finding → diagnose before comparisons.
  - Training diverges despite grad-clip → investigate (don't resubmit same config >2×).
Spawns: EXP-027 (vanilla GridWorld dynamics baseline); then dynamics-rollout frame source in adapter.py.
