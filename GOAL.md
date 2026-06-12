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

Status: **open** — this is the next phase (Phase 2).

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
Success criteria: TO BE PRE-REGISTERED (T-004) before the deciding run, informed by
the probe's calibration controls (chance floor, ceiling) but written before the H2
result is read.
Probe metric (working choice, milestone 2026-06-12): **latent-token MSE** of the
predicted reveal-frame latent vs. the frozen tokenizer's latent of the true frame —
decoder-free *and* detection-free, so it sidesteps the "where is the ball, to read
its color?" problem. To be validated against an interpretable color/position
decomposition (pixel-space, blob-detection); "ball not rendered" tracked as its own
failure mode. Requires the probe suite (T-002) built and frozen first.
Evidence: EXP-007/EXP-008 (suggestive: ball identity lost in rollout) — but those
rollouts stayed within the window, so they do not yet isolate the beyond-window case.

## H3 (overarching end-goal, exploratory — not a single pre-registered hypothesis)

**Goal:** force the **encoder and/or the dynamics model** to carry hidden/global
environment state (e.g. occluded ball color/position) in the latent space, so it
survives even after the revealing frames have left the context window. The *how* is
deliberately open — we expect to try many objectives/mechanisms and keep what sticks
(Merlin, milestone 2026-06-12). Framing it as one crisp hypothesis is premature.

Working intuition (D-011, not settled): since per-frame latents also scroll out of
the window, retention can't live in an old latent — the **autoregressive latent chain
itself is the carrier**. If every step's latent is forced to encode the global hidden
state, the always-in-window current latent propagates it forward indefinitely,
independent of window size. The research is finding objectives/architecture that make
that propagation happen and persist.
Status: **open, not started.** Evidence: —
Blocked on: probe suite frozen (T-002), H2 baseline measured.
Note: reconstruction/next-frame loss alone never decides a memory claim — any method
must be judged on the frozen probe, ≥2 seeds, against the H2 baseline under identical
provenance discipline.
