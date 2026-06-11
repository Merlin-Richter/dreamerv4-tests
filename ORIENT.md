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

## In flight

- **T-001 / EXP-008** — worker building the headless `context_noise` sweep
  diagnostic (`experiments/EXP-008/diagnose_context_noise.py`). Local (4070),
  inference-only on `my_dynamics.pt`. Spec `tasks/T-001.md`. Awaiting worker artifact;
  then orchestrator runs the full tau_ctx∈{0.1,0.5,0.9,0.99} sweep and reconciles.
- No cluster jobs (wrappers don't exist — T-003; cluster access manual via Merlin).

## Next action

Verify the worker's diagnostic artifact (read diff, run acceptance commands myself),
then run the full EXP-008 sweep, reconcile vs D-010's expected outcome + tripwire,
build the GT/rollout side-by-side view, write the decisive read, and escalate for
Merlin's review (every experiment ends in a stop, §5).

## Current worries

1. **val/loss is a poor proxy** — EXP-007 proves the training loss can look fine
   while rollouts are useless. Until the probe suite (T-002) exists we have no
   trustworthy quantitative signal on dynamics quality.
2. **Provenance debt**: pre-2026-06-09 history is approximate (no logging); EXP-007
   rollout images were never archived; EXP-006's A/B was confounded (commit, batch,
   host). Backfill marks these honestly; going forward the protocol applies.
3. **EXP-005's "plausible rollouts" were only eyeballed** — the unconditional
   baseline may have had the same defect, undetected.
