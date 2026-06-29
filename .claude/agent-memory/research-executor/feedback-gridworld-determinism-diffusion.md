---
name: feedback-gridworld-determinism-diffusion
description: Don't conclude "drop diffusion steps / shortcut forcing" from GridWorld results — GridWorld is deterministic and can't test the diffusion sampler's real job (stochastic transitions).
metadata:
  type: feedback
---

When evaluating shortcut-forcing / diffusion-step / x-prediction tradeoffs on GridWorld, do NOT
generalize a "no upside" result into "we can drop diffusion steps / shortcut forcing." Scope the
conclusion to the env's determinism.

**Why:** Merlin pushed back on my conclusion that the shortcut bootstrap was "pointless." GridWorld is
discrete + DETERMINISTIC, so the next-latent distribution is a delta — single-step x-prediction returns
the conditional MEAN, which is exactly correct, so K=1 nails it and the few-step shortcut buys nothing.
That says nothing about stochastic envs: there, one-step x-prediction mode-collapses to the blurred
conditional mean, so you NEED multiple diffusion steps to sample a sharp transition, and the bootstrap's
self-consistency is the mechanism that keeps few-step sampling on the correct ODE trajectory. The whole
point of diffusion steps is resolving stochastic state transitions — GridWorld can neither show that
upside nor test it.

**How to apply:** (1) Diffusion forcing (multi-noise-level denoising = the distribution sampler) is
distinct from the bootstrap/shortcut (a few-step EFFICIENCY distillation). In the mem2mem rollout loss,
"boot off" vs "boot on" toggles only the latter — both keep full diffusion forcing. Don't describe the
ablation as "diffusion on/off." (2) Any conclusion about dropping/keeping diffusion-step machinery must
be scoped "on GridWorld (deterministic)"; validating the real benefit requires a STOCHASTIC / multimodal
env (see `tasks/drafts/harder-grid-env.md`). Related: [[fair-bootstrap-ab-result]] context lives in
`experiments/mem2mem-rollout-boot-fair/NOTES.md`.
