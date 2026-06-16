---
name: probes-dynamics-motion-memory
description: Cheap discriminating probes that localize the broken link for D_dynamics motion/memory deficits (frozen probe, teacher-forced rollout, tau sweep)
metadata:
  type: project
---

Reusable cheap discriminators for D_dynamics_model deficits (all run on a 4070 / CPU with an
existing checkpoint + the frozen probe at commit 5503e75; no training needed):

- **Frozen linear/MLP readout probe (latent -> x,y or color).** Settles representability vs
  identifiability. Position: R²=0.96 / 2.7px (EXP-011) => representation is NOT the problem,
  failures are downstream in D. Use this BEFORE proposing any architecture change.

- **Teacher-forced vs open-loop rollout (the decisive motion probe).** Run the model
  autoregressively but feed GROUND-TRUTH context each step (held at context_signal) instead of
  its own output; compare `pos_err(h)` to open-loop. Teacher-forced stays flat + open-loop
  diverges => failure is **compounding / exposure bias (link 4)**. Both climb => the per-step
  map is a **1-step-only fit (link 3)**, not a real motion model. One run splits the two.

- **tau-context sweep.** Teacher-forced 1-step error while sweeping the context tau the model
  is FED (0.5->1.0). Spike near 0.9 (rollout's value) but flat at trained tau => train/infer
  context-distribution mismatch (link 4b). Retrain-free, just changes eval input.

- **2-frame vs N-frame velocity probe.** Linear-probe (dx,dy) from last-1 / last-2 / full
  window latents to see if velocity needs >1 frame and whether the model's state carries it.

**Why:** these are the calibrated instruments the H3 line already validated; reaching for them
first avoids guessing and avoids building architecture for a downstream (objective) problem.

**How to apply:** log under `experiments/EXP-NNN/`, fixed seed reported. A passing readout
proves info is PRESENT, not that the main objective USES it. See
[[dynamics-loss-singleframe-bottleneck]].
