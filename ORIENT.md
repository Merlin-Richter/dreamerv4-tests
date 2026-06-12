# ORIENT.md

Rewritten: 2026-06-12 (FF7 v1 build started — D-014)

## What we are doing and why

- **H1, H2 — supported** (see GOAL.md; EXP-008/EXP-009). The frozen probe (5503e75) +
  T-004 criteria are the fixed yardstick: H3 bar = color ΔRGB < ~63 at n_occ ∈ {12,16,24}
  (baseline at chance ~110 there).
- **H3 — first method attempt in flight: FF7 v1** (single-timestep sufficiency, D-014).
  Go-ahead given by Merlin ("Continue by building v1"); harness question withdrawn by his
  direct protocol edit (no methods-critic; code-citation rule stays via agent memory;
  ≥2-seed standing order REMOVED — single-seed screening is allowed).

## FF7 v1 in one breath (full design: IDEAS.md + D-014)
Train the register channel to be a 1-step-sufficient statistic: extra rollout forward pass
per batch — frame t with REAL clean latent (@ tau_ctx=0.9) and its windowed-pass register
INJECTED, predict frames t+1..t+k (flow loss, finest d). Latent overwrite kills the latent
color path, so registers must carry hidden state. **D-014 correction to the converged
design:** registers don't persist across `generate()` steps (re-expanded each forward,
dynamics_model.py:282), so inference needs a param-free register-carry rollout
(`generate_memory`), dispatched via a config flag so the frozen probe runs unmodified.

## In flight / NEXT ACTION

**T-009 (in progress, inline on master):**
1. `dynamics_model.py`: `forward(..., register_in, return_registers)` + `generate_memory()`
   + config flag `use_register_memory`. Zero new parameters.
2. `train_dynamics_model.py`: `--ff7 K` flag → combined loss (diffusion + 1.0 × FF7).
3. Smoke on the 4070 (tiny run: finite losses, probe dry-run on the smoke checkpoint).
4. **EXP-010**: k=1 and k=3 arms, one seed each, occluded.npy, then frozen probe →
   present-then-stop (§5). Expect: k=1 relays ~1 window then decays; k=3 further.

## Current worries
1. **Gaming risk** (Merlin's prior): model may satisfy the per-frame loss by emitting the
   color prior (= chance on the probe). The probe, not train loss, is the judge.
2. **Window-1 inference may degrade base dynamics** → watch FF7-model ceiling/drift
   controls vs EXP-009 (D-014 tripwire 1).
3. **Injected registers are out-of-distribution as inputs** (final-layer activations
   replacing learned tokens) — gradient must shape the write side; raw injection in v1.
4. Clips whose color evidence predates the clip start train prior-emission on reveals —
   dilutes the FF7 signal; expected, don't misread as relay failure (D-014 tripwire 4).
