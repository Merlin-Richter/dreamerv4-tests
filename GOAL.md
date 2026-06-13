# GOAL.md — Research goal and hypotheses

> **Owner: Merlin.** The orchestrator may propose amendments but applies them only
> after a milestone conversation in which he agreed.
>
> Backfilled 2026-06-11 from journal, git history, and W&B. The H1–H3 framing was
> approved by Merlin in the protocol-adoption exchange (see ESC-001). Success
> criteria for H2/H3 are still to be pre-registered (task T-004) — they MUST be
> written before the experiments that decide them.

## Research idea

World models learned from video (DreamerV4 lineage) compress each frame through a
tokenizer trained with reconstruction-only objectives. Such latents only need to
represent what is *visible in that frame*; nothing forces them to retain
currently-hidden state — inventory contents, off-screen objects, anything behind an
occluder. A dynamics model living in that latent space therefore cannot recall
hidden information, however good its next-frame loss looks (the "look north, turn
around, everything is hallucinated" failure).

We investigate alternative encoding objectives that force the latent state to
retain all information that, under the right actions (open the inventory, lift the
curtain), affects future frames.

Testbed: CurtainsEnv (`src/data_generators/occluded_bouncing.py`) — a bouncing
ball with per-episode fixed color and background, plus an absolute curtain action
that hides the entire frame. A model with ≥1 visible context frame should recall
ball/background color; with ≥2, also position (velocity derivable).

---

## H1: A DreamerV4-style tokenizer + dynamics pipeline can be reproduced at small scale on synthetic video

Status: **supported** (Merlin, milestone 2026-06-12, ESC-003)
Success criteria (written retroactively, hence weak): tokenizer reconstructions
visually faithful without latent collapse; dynamics rollouts preserve per-episode
constants (ball color, background) and approximate ball position over ≥10 frames.
Evidence: EXP-002, EXP-003 (negative→fixed), EXP-004, EXP-005, EXP-006 (tokenizer:
supported), EXP-007 (dynamics: rollouts looked broken) → EXP-008 (D-010: the rollout
failure was an inference bug — context fed ~90% noise; the model was never broken).
With the corrected inference the same checkpoint preserves ball color/position, so the
pipeline reproduces. Known residual: ordinary autoregressive drift late in long
rollouts — minor, tracked separately, not an H1 blocker.

## H2: A sliding-window world model cannot recall hidden state once the evidence leaves its context window

Status: **supported** (Merlin, 2026-06-12: "Yes I agree. This proofs H2.") — EXP-009.

**Corrected mechanism (milestone 2026-06-12, ESC-003 / D-011).** The baseline is a
pure sliding-window transformer with **no persistent state beyond its window**. Each
rollout step it attends over the last N−1 frames, predicts the next latent, drops the
oldest, repeats; RoPE is relative so step 1000 is identical to step N. There is no
"carrying across a boundary" and no recurrent state — information older than N−1
frames is simply *absent from the model*, neither in the input nor stored anywhere.
So once the curtain stays down longer than the window, the last visible frame has
scrolled out and the baseline **cannot** recall ball color/position by construction.
(Note: changing the inference window M<N needs **no retrain** — RoPE is relative.)

This makes H2 nearly architecturally true; the experiment's job is to *calibrate the
decay*: how recall of ball color/position falls toward chance as occlusion length k
crosses the window size N, with proper chance-floor and visible-context-ceiling
controls and a no-occlusion drift control.
Success criteria (PRE-REGISTERED T-004, Merlin-approved 2026-06-12, ESC-004):
- **Headline metric:** ball color ΔRGB at the reveal frame, occluded vs. matched-horizon
  (curtain-up) drift control. Validating secondary: latent-token MSE (r=0.952 with color
  in EXP-009). Position: reported but **drift-confounded**, NOT a success metric.
- **H2 (baseline) claim — MET:** vanilla sliding-window recall = chance once the
  color-carrying prefix scrolls out (here n_occ ≥ 7 at N=8, P=3); no retention past
  the window. EXP-009: color ΔRGB 16.8 (n_occ=6) → 94.4 (7) → 116 (8), vs ceiling 15.9 /
  chance 109.9; cliff at the geometry-predicted frame; drift control rules out ordinary drift.
- **H3 success bar (for later method comparison):** a method "retains hidden state" if at
  n_occ ∈ {12,16,24} (well beyond the window) its color ΔRGB is below halfway between
  ceiling and chance (≈ 63), under the identical frozen probe + matched drift control.
Probe metric (working choice, milestone 2026-06-12): **latent-token MSE** of the
predicted reveal-frame latent vs. the frozen tokenizer's latent of the true frame —
decoder-free *and* detection-free, so it sidesteps the "where is the ball, to read
its color?" problem. To be validated against an interpretable color/position
decomposition (pixel-space, blob-detection); "ball not rendered" tracked as its own
failure mode. Requires the probe suite (T-002) built and frozen first.
Evidence: **EXP-009** (deciding run: the beyond-window cliff, drift-controlled, on the
frozen probe). EXP-007/EXP-008 suggestive but stayed within the window.

## H3 (overarching end-goal, exploratory — not a single pre-registered hypothesis)

**Goal:** force the **encoder and/or the dynamics model** to carry hidden/global
environment state (e.g. occluded ball color/position) in the latent space, so it
survives even after the revealing frames have left the context window. The *how* is
deliberately open — we expect to try many objectives/mechanisms and keep what sticks
(Merlin, milestone 2026-06-12). Framing it as one crisp hypothesis is premature.

Working intuition (refined 2026-06-12, code-grounded): the **latents are pixel-space-bound**
(they decode to the image via the frozen tokenizer), so during occlusion a latent encodes the
curtain, not the hidden ball — latents *cannot* be the carrier. The **register tokens** can:
they're free scratch, and `dynamics_model.py:110-121` shows temporal attention is position-wise,
so each register slot is already its own causal channel through time, with spatial layers
routing info latent↔register within a frame. Retention beyond the window = a **relay**: each
frame re-copies state into its register from the previous frame before the source scrolls out.
The research is finding objectives that make registers store + relay the hidden state.

**Idea registry: `IDEAS.md`** (carriers × forcing-functions × regimes; living, append-only).
First attempt = **FF7 single-timestep-sufficiency** (see IDEAS.md "Proposed first attempt").

Hard constraints (Merlin, non-negotiable): **no privileged data to the model, ever** (only env
obs + reward + env-generated data); **must generalize across environments**. Eval instrumentation
may read sim hidden state to *score* (measurement ≠ model input).

Status: **open — first method (FF7 v1) supports H3 for hidden COLOR; position open & BLOCKED.**
Prereqs cleared: probe frozen (T-002, 5503e75), H2 baseline measured (EXP-009).
Evidence: **EXP-010** (FF7 v1, D-014) — both arms move the post-window color cliff off the
chance floor, clearing the T-004 bar (color ΔRGB < ~63) at n_occ 12 & 16 (k=3 > k=1; no base-
dynamics-degradation tripwires). This supports H3 for **color**.
**Position-tracking blocker (Merlin, 2026-06-13, ESC-006):** the model does not track ball
*position/motion* even in an OPEN rollout (matched-horizon drift control → chance), and this
predates FF7 (my_dynamics, EXP-009 same). So whether any memory method can retain
position/momentum is **unproven** — the base model can't do position in the clear. Diagnosing
this (EXP-011, D-015, no training) before further H3 method work on position. H3 color claim
stands; H3 position claim is gated on the base-dynamics motion deficit.
Note: reconstruction/next-frame loss alone never decides a memory claim — any method
must be judged on the frozen probe, against the H2 baseline (T-004 bar) under identical
provenance discipline (≥2 seeds on promising results; single-seed screening allowed — Merlin
relaxed the standing 2-seed order 2026-06-12).

**Position-memory update (2026-06-13, ESC-009).** A position-memory *consistency* metric was built
and applied (EXP-013, D-018): under TRUE blind occlusion, blind position memory is near-absent —
vanilla_s0 ≈ copy-last (freezes the ball), FF7 only marginally better (esp. k1); the register relay
carries static *color* indefinitely but not dynamic *position/velocity*. **Caveat (Merlin):** he is
not confident this metric, as coded, is a strong evaluation instrument, and chose NOT to freeze it as
the H3 position spine — it is "built, of uncertain strength." So this read is the current best
understanding, NOT a hard pre-registered gate. H3 *position* retention stays **open**; the next method
to attempt it is the **sequential register-relay rollout training** (IDEAS.md), for which efficient
sliding-window rollouts are being prepared (D-020). The metric's strength is revisited if/when a
position method needs a yardstick.
