# ORIENT.md

Rewritten: 2026-06-11 (post cold-start, T-001 rescoped + spawned)

## What we are doing and why

We reproduce a DreamerV4-style pipeline (frozen video tokenizer + shortcut-forcing
dynamics model) on CurtainsEnv to then test whether reconstruction-only encoding
objectives fail to retain occluded state (H2) and whether a retention-forcing
objective fixes it (H3). See GOAL.md. The tokenizer is good (EXP-006: VGG-LPIPS,
val/mse 1.41e-4, frozen as `trained_autoencoder.pt`). The dynamics model is the
problem: EXP-007 trained to healthy val/loss (1.93e-3) but rollouts randomize ball
color and position immediately, even with fully visible context. Merlin's verdict
(ESC-001): diagnose this before building the §8 probe suite.

## Strong lead (D-010)

Cold-start code read of `dynamics_model.py` found a likely **inference-only** cause:
the rollout context-noising `ctx_noised = (1-tau_ctx)*noise + tau_ctx*context` with
`tau_ctx=0.1` puts **90% noise on the context** (tau = signal level in this codebase).
That would destroy the ball color/position the rollout is supposed to read from
context → model emits a plausible-but-random ball. Matches every EXP-007 symptom and,
if true, is a one-line fix with NO retraining. Testing it first (D-010), before any
latent-geometry / undertraining work.

## Result (EXP-008, done) — hypothesis confirmed

D-010 SUPPORTED. EXP-008 (inference-only tau_ctx sweep on `my_dynamics.pt`, no
retrain) confirms the EXP-007 rollout failure is an **inference bug**: rollout
context-noising feeds 90% noise on context at the default `context_noise=0.1`
(tau=signal level). At tau_ctx≈0.9–0.99 the SAME checkpoint preserves ball color
through the rollout; gen-MSE −43% (0.0289→0.0165), first generated frame near-perfect
(0.0046). **One-line fix, no retraining needed.** Reframes EXP-007: dynamics model is
much closer to working than its NOTES said. Latent-geometry/undertraining diagnosis
(T-001b) NOT needed. Full read: `experiments/EXP-008/NOTES.md`.

## In flight

- **AWAITING MERLIN (ESC-002).** Present-then-stop per §5. Branch paused — not
  starting the fix, the confirmation rollout, or T-002 until he weighs in. Asked: does
  he agree; fix-default-to-0.9 + confirm-rollout + proceed to probe suite; and the
  `context_noise` semantics (keep "signal level" vs invert to "noise fraction").
- No cluster jobs (wrappers don't exist — T-003; cluster access manual via Merlin).

## Next action (after his verdict)

Likely: patch the rollout context-noise default (~0.9, possibly a quick 0.9/0.95/0.99
pick), re-run the EXP-007 checkpoint rollout to confirm the H1 dynamics baseline, then
T-002 (freeze the revisit-consistency probe suite). Do nothing until ESC-002 answered.

## Current worries

1. **val/loss is a poor proxy** — EXP-007 proves the training loss can look fine
   while rollouts are useless. Until the probe suite (T-002) exists we have no
   trustworthy quantitative signal on dynamics quality.
2. **Provenance debt**: pre-2026-06-09 history is approximate (no logging); EXP-007
   rollout images were never archived; EXP-006's A/B was confounded (commit, batch,
   host). Backfill marks these honestly; going forward the protocol applies.
3. **EXP-005's "plausible rollouts" were only eyeballed** — the unconditional
   baseline may have had the same defect, undetected.
