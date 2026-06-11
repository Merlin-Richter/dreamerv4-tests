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
