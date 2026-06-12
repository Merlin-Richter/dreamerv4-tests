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

T-007 done. **T-002 probe is built and validated** (`src/probe/{probe_env,revisit_probe}.py`,
commit fb270e2): detector gate passes on GT; pure-generation rollout at N=8; latent-MSE +
color/position decomposition; chance/ceiling controls + matched-horizon drift curve.
Preliminary 4-ep smoke shows the H2 color cliff at N=8 (see tasks/T-002.md findings).
NOT frozen yet. Next: (1) await Merlin's glance on the drift-confound finding (affects
T-004 metric choice); (2) add the side-by-side sheet (acceptance #5); (3) freeze, then
run the full 64-ep sweep as the H2 baseline experiment (EXP-009) → present-then-stop (§5);
(4) T-004 pre-register criteria. No cluster needed (local on 4070).

## Current worries

1. **Position recall is drift-confounded** (CONFIRMED in smoke): occluded ≈ matched-drift
   position error at all n_occ. So **color-recall (occluded vs drift) and latent-MSE−drift
   are the clean H2 signals; position is a confounded secondary.** This changes what T-004
   should pre-register — flagged to Merlin.
2. **Metric validation pending at scale**: latent-MSE tracks color qualitatively in the
   smoke run; confirm Pearson r at the full 64-ep run before calling latent-MSE the
   headline (D-011 tripwire).
3. **val/loss is a poor proxy** (EXP-007 lesson) — the probe is now our trustworthy signal.
