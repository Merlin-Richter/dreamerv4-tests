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

Doing T-007 now (the rename), then starting T-002. No cluster jobs (wrappers don't
exist; cluster is manual via Merlin and not needed for Phase 2).

## Current worries

1. **Probe metric is unvalidated.** Latent-MSE is a working guess (Merlin's); if it
   doesn't track the color/position decomposition, the metric is wrong — that
   divergence is a finding to escalate before pre-registering (D-011 tripwire).
2. **Residual autoregressive drift** in long rollouts could confound recall-vs-k. The
   no-occlusion drift control is meant to difference it out; if drift is severe the
   usable k-range shrinks and we revisit (possibly a better/longer dynamics baseline).
3. **val/loss is a poor proxy** (EXP-007 lesson) — until the probe exists we have no
   trustworthy quantitative signal on dynamics memory.
