# T-017 — C1 design: time-axis multi-step motion-prediction loss

Status: VERIFIED + IMPLEMENTED (D-027). Verdict: `tasks/T-017-C1-verdict.md` (V-T017-C1).
Origin: D-026 session; method-architect proposal C1 (tasks/T-016-architect-proposal.md).
Scope: improve multi-step ball-motion propagation, curtain-up (no occlusion). Frozen tokenizer.

> **MECHANISM REFRAMED PER VERIFIER (read this first):** the rationale below in places says C1
> "pushes toward a contraction map" — that is **mechanistically wrong** (V-T017-C1 C-A). An
> anchored-GT loss recovers the data's TRUE local gain (possibly expanding), it does not manufacture
> a contraction. The actual, sound mechanism is **on-policy distribution correction (DAgger /
> scheduled sampling)**: it makes the per-step map ACCURATE on the off-trajectory states the rollout
> visits. It therefore helps to the extent the deficit is off-manifold ACCURACY (EXP-018 shows this
> IS the ff7/ff9 case) and will NOT help if the deficit is intrinsic high-gain dynamics. The detach
> (TBPTT-1) is SAFE here (unlike the T-014 relay) precisely because every step has a GT anchor.
> Mandatory gates: λ-ramp warmup + clean-val/diffusion regression tripwire. Open degenerate mode to
> monitor: prior-emission under unlearnable drifted context (per-horizon loss flattening at large j).

## Diagnosis this targets (from T-016, confirmed by reading `dynamics_model.py:399-478`)
`DynamicsModel.loss()` contains NO time-axis successor term: every frame is independently
noised at its own τ and supervised toward its OWN clean latent; the bootstrap distills along
the τ (denoising) axis, not time. The only successor-prediction signal in the file is
`_ff7_loss` / `_ff9_loss`. EXP-014 showed that successor-prediction *loss* (not architecture)
is what yields the 1px single-step map. So vanilla never learns motion; and nothing trains the
model to be right under its OWN multi-step rollout (exposure bias). C1 adds exactly that
gradient.

## The loss (config-gated, identity when off)
New config fields on `DynamicsModelConfig`:
- `multistep_h: int = 0`        # 0 => term absent => byte-identical to pre-C1 (guard like ff9_k=0)
- `lambda_multistep: float = 1.0`

New method `_multistep_loss(self, z1, actions, h)` added to `loss()` behind `if multistep_h > 0`,
mirroring the `_ff9_loss` windowing discipline (fold anchors into the batch dim). For a clip
`z1` of shape (B, T, L, D) and lookahead h, for every anchor frame t that has h successors
(n_t = T - h), build a **self-fed rollout** and put a flow loss on each predicted successor:

For each anchor t (all anchors processed together, folded into batch):
1. Seed the rollout context with the REAL clean latents in the window ending at t, held at
   `context_signal` (the same near-clean level inference uses): `ctx = noise_to_ctx(z1[.., t-w+1 .. t])`
   with w = min(t+1, max_temporal_length-1).
2. For j = 1..h (SEQUENTIAL):
   a. Predict successor t+j from the current context with a **pure-noise target slot (τ=0)** at
      the finest step d (single x-prediction forward):
      `z_hat = self([ctx_detached @ context_signal | pure_noise], tau=[ctx.., 0], d=finest, act)[:, -1:]`
      Grad IS enabled on THIS forward, but the context latents are **detached** (TBPTT-1).
   b. Flow loss `L_j = || z_hat - z1[:, t+j] ||^2` (ramp-free; finest-d plain flow, like `_ff9_loss`).
   c. **Detach** `z_hat`, append to the context window (slide, drop oldest beyond max window),
      and continue. So step j+1's context contains the model's own (detached) prediction of t+j.
3. `_multistep_loss = mean over (anchors, j) of L_j`. Added to total as `lambda_multistep * _multistep_loss`.

The existing per-frame diffusion loss is UNCHANGED and still computed on the clip; C1 is purely
additive on top.

### Why this is the right gradient
- τ=0 target slot ⇒ the predicted position can come ONLY from the context (no ground-truth leak
  into the target), exactly the leak-free principle FF9 v2 used — so it trains motion-from-context.
- Context is the model's OWN detached prediction from step 2 onward ⇒ the model is optimized to be
  correct under the self-generated distribution rollout actually visits ⇒ directly penalizes the
  high-gain extrapolator whose 1-step error compounds (pushes toward a contraction map).
- TBPTT-1 (detach context each step) ⇒ no backprop-through-time explosion; this is the discipline
  V-T014 demanded for relays.

### Cost
Per batch: existing vanilla loss (~3 forwards incl. bootstrap) + h sequential single-finest-d
forwards (each batched over all anchors). For h=4 ≈ 7 forwards (~2.3× vanilla). On the 4070 at
bs32, occluded 250-ep subset, ~30 epochs is ~30-40 min — feasible for an in-session A/B.

## Degenerate modes & guards (must be checked)
- (a) Collapse to copy-last (predict zero motion → bounded ~3.2px/step error). Closed because the
  existing per-frame diffusion loss already rewards correct position, and a copy-last successor
  scores 3.2px/step vs the supervised target. **Monitor:** predicted inter-frame displacement must
  track sim (~3.2px), not 0. (Add to eval.)
- (b) Off-manifold drift: self-generated latents may leave the frozen tokenizer manifold over h
  steps, so the loss chases motion in an invalid region. **Monitor:** decode self-gen latents and
  check reconstruction validity vs j; if dominant, no motion loss fixes it (out of scope, flag).
- (c) Mode-averaging at bounces (sharp bifurcation). Keep h modest (4) first; watch bounce-frame err.
- (d) Single-frame sharpness regression (tension with the diffusion loss). **Monitor:** clean
  val/diffusion must not regress past ~0.003 (EXP-017 anchor ~0.0017). Ramp `lambda_multistep`.

## Identity-when-off (non-negotiable, like n_memory=0 / ff9_k=0)
- `multistep_h == 0` ⇒ `_multistep_loss` never called, no new tensors/params, RNG draw sequence
  unchanged ⇒ a pre-C1 model is byte-identical. Smoke test asserts this (mirror
  `test_n_memory_zero_is_additive_identity`).
- No new nn.Parameters at all (C1 is loss-only; reuses the existing forward). So inference
  (`generate`, `generate_cached`, `generate_memory`, `generate_full_state_memory`, streaming) and
  the frozen probe are entirely untouched — bit-identical. FF7/FF9 paths untouched.

## Falsifiable prediction (the A/B)
Train C1 (multistep_h=4) vs a budget-matched vanilla control (same subset/seed/epochs).
PREDICT: open-loop `pos_err(h4)` drops from ~4.6 toward ~2px and the cross-chance horizon moves
past ~20, WITHOUT clean val/diffusion regressing past ~0.003.
FALSIFIED IF: open-loop error unchanged despite the term converging (⇒ deficit was link-4b or
velocity-state, not link-3) OR val/diffusion regresses materially (⇒ single-frame/multi-step
tension dominates).

## Questions for the verifier
1. Does this loss actually create gradient pressure toward a multi-step *contraction* map, or can
   it be minimized by a degenerate solution the guards above miss?
2. Is the τ=0 pure-noise target slot correct for forcing motion-from-context here (as it was for
   FF9), or does the self-generated detached context reintroduce a leak/shortcut?
3. Is single-finest-d (1 forward/step) self-rollout an acceptable proxy for the K=4 shortcut
   rollout inference uses, or will the train/infer mismatch undermine the exposure-bias fix?
4. Any RNG-order / identity-when-off hazard in splicing this into `loss()`.
