# ORIENT.md

Rewritten: 2026-06-12 (H2 closed; H3 entered; FF7 designed — pre-context-reset checkpoint)

## What we are doing and why

- **H1 — supported** (DreamerV4-style pipeline reproduces; the dynamics "failure" was an
  inference bug, EXP-008 / D-010).
- **H2 — supported** (Merlin, ESC-004; EXP-009): a vanilla sliding-window world model
  cannot recall hidden state once the evidence leaves its window. The frozen probe shows a
  clean cliff — color recall at ceiling for n_occ ≤ 6, chance for n_occ ≥ 7 at N=8/P=3
  (geometry-predicted, drift-controlled, latent-MSE↔color r=0.952).
- **H3 — entered (the real work).** Open-ended end-goal: force the model to carry hidden
  state (ball color/position) so it survives past the context window. Exploratory, high
  failure rate expected — ideas live in **`IDEAS.md`** (carriers × forcing-functions ×
  regimes, append-only; log outcomes tersely).

## Hard constraints (Merlin, non-negotiable) — apply to every H3 idea
- **No privileged data to the model, EVER** — only env obs + reward + env-generated data.
- **Must generalize across environments** — no bouncing-ball-specific hacks.
- Eval instrumentation MAY read sim hidden state to *score* recall (measurement ≠ model input).

## Architecture facts that matter (code-grounded — verify before asserting, see memory)
- M < N inference needs NO retrain (RoPE relative). `generate()` already slides the window.
- **Latents are pixel-space-bound** (decode via frozen tokenizer) → cannot carry off-screen
  state. **Register tokens** can: `dynamics_model.py:110-121` — temporal attention is
  *position-wise*, so each register slot is its own causal time channel; spatial layers route
  latent↔register within a frame. So the carry for H3 already exists; no recurrence wiring.
- KV cache = efficiency only; if/when built obey `HOWTO/rope_kv_cache_caveat.md`.

## In flight / NEXT ACTION (read this)

Two things are **awaiting Merlin** (see ESCALATIONS ESC-005); do NOT start building until
he answers:

1. **FF7 build go-ahead.** The first H3 method is designed, code-grounded, and converged
   with Merlin — full v1 scheme in `IDEAS.md` → "Proposed first attempt — FF7 v1". Summary:
   single-timestep-sufficiency loss (window-1 rollout predicting next k=1 from one frame,
   latents overwritten with real, registers forced to be the carrier), **training-procedure
   change to `train_dynamics_model.py` only — no architecture change**, eval on frozen probe
   (5503e75) against the T-004 bar. NOT yet committed as D-014.
2. **Harness improvement.** Merlin asked how to prevent my two ML-reasoning errors this
   session (I theorized ahead of the code twice). My proposal: (a) a hard "cite the code
   before asserting model behavior" rule in the protocol; (b) optional fresh read-only
   **methods-critic agent** to red-team a method design before its D-NNN is committed.
   Awaiting his pick (rule only, or rule + critic agent). If we add the critic, it should
   review D-014 before commit.

On his go-ahead: write **D-014** (FF7), spawn an implementation worker (one worktree),
smoke-test on the 4070, run as **EXP-010**, present-then-stop (§5) against the probe.

## Current worries
1. **The FF7 loss is the hard part, not the carrier** (Merlin's strong prior): forcing the
   register to actually *store* the right thing — watch for the model gaming the per-frame
   loss by emitting the color prior (= chance). Credit assignment is handled by per-step
   relay (Bellman), not long BPTT.
2. **Position recall is drift-confounded** (EXP-009 confirmed) → color ΔRGB is the headline,
   position is a confounded secondary. Locked in T-004.
3. **val/loss is a poor proxy** (EXP-007 lesson) — the frozen probe is the only trustworthy
   memory signal. Any FF7 run is judged on the probe, ≥2 seeds, vs the H2 baseline.
