# ORIENT.md

Rewritten: 2026-06-12 (post H1-closure milestone, ESC-003 / D-011)

## What we are doing and why

H1 is **complete** (Merlin, 2026-06-12): the DreamerV4-style pipeline reproduces. The
tokenizer was always good (EXP-006); the dynamics model only *looked* broken — EXP-008
proved the EXP-007 rollout failure was an inference bug (context fed ~90% noise at the
default `context_noise=0.1`). With corrected inference the same checkpoint preserves
ball color/position.

We are now in **Phase 2 (H2)**: measure that a sliding-window world model cannot recall
hidden state once the evidence leaves its context window. Then the open-ended end-goal
(H3): force the encoder and/or dynamics model to carry hidden/global state in the latent
space so it survives beyond the window.

## Corrected architecture model (milestone, D-011) — load this, my prior notes were wrong

- **M < N inference needs NO retrain** (RoPE is relative). We pick the inference window.
- **A sliding-window transformer has no persistent state.** No "boundary to carry
  across"; info older than N−1 frames is simply absent from the model. The dynamics
  `generate()` **already slides the window** → the beyond-window regime is reachable
  today, no new architecture, no retrain, just roll out longer than N.
- **KV cache = efficiency, not a prerequisite** for H2. When we build it, obey
  `HOWTO/rope_kv_cache_caveat.md` (cached K/V can't be re-rotated → need a never-reset
  absolute-position clock; the current fixed cos/sin table is cache-incompatible).

## Plan (cheap-signal-first, endorsed by Merlin)

1. **T-007 (in progress):** rename `context_noise`→`context_signal`, default 0.9, fix
   comment. Inference-only cleanup, closes H1's loose end.
2. **T-002:** build & freeze the revisit-consistency probe suite on the existing frozen
   tokenizer + `my_dynamics.pt`. Roll out occlusion length k below→above the window N.
   Primary metric: latent-token MSE (predicted reveal latent vs frozen GT latent),
   validated against a pixel color/position decomposition. Controls: chance floor,
   ceiling, no-occlusion drift. "Ball not rendered" = own failure mode.
3. **T-004:** pre-register H2 criteria (after controls measured, before reading result).
4. Measure H2 baseline → present-then-stop (§5).

T-003 (cluster wrappers) and T-008 (KV cache) deferred to H3 heavy-training time.

## In flight / next action

**H2 is CLOSED — supported** (Merlin, 2026-06-12, ESC-004; EXP-009). The frozen probe is
the calibrated yardstick: vanilla sliding-window recall = chance once the color-carrying
prefix scrolls out of the N=8 window (cliff at n_occ=7, geometry-predicted; drift-controlled;
latent-MSE↔color r=0.952). T-004 success criteria locked. Probe control key renamed
`drift_by_occ`→`matched_horizon_drift` (D-013, re-frozen).

Next action: **enter H3 (the open-ended end-goal)** — force the encoder and/or dynamics to
carry hidden ball color/position in the latent space so it survives past the window. H3 bar
(T-004): color ΔRGB < ~63 at n_occ ∈ {12,16,24}. H3 is exploratory ("try many, keep what
sticks") — first step is to pick a starting mechanism. Proposing a short menu to Merlin for
steering before committing a decision (new phase). Likely candidates: an auxiliary
latent-retention objective on the dynamics model vs. an encoder objective that bakes global
state into per-frame latents. Local 4070 for first cheap probes; cluster (T-003 wrappers,
deferred) only once a mechanism justifies heavy training.

## Current worries

1. **Position recall is drift-confounded** (CONFIRMED in smoke): occluded ≈ matched-drift
   position error at all n_occ. So **color-recall (occluded vs drift) and latent-MSE−drift
   are the clean H2 signals; position is a confounded secondary.** This changes what T-004
   should pre-register — flagged to Merlin.
2. **Metric validation pending at scale**: latent-MSE tracks color qualitatively in the
   smoke run; confirm Pearson r at the full 64-ep run before calling latent-MSE the
   headline (D-011 tripwire).
3. **val/loss is a poor proxy** (EXP-007 lesson) — the probe is now our trustworthy signal.
