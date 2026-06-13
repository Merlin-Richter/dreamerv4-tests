---
name: detached-carry-relay-drift
description: Proven failure mode — detached-carry per-step-sufficiency relays preserve state in-window but drift to chance past the trained horizon (V-T014). Plus the synthetic-relay probe pattern that settles it.
metadata:
  type: project
---

**Finding (V-T014, 2026-06-14): a DETACHED-carry, per-step-sufficiency relay does NOT propagate
sufficient hidden state across many hops.** This was T-014 "Mode B" (carry memory tokens across
rollout steps detached; supervise each step only by its own one-step FF9 loss; reuse the T-012
sliding-window eviction cache for the detached context).

**Why (analytical, load-bearing):** the memory carrier is a continuous ACTIVATION with NO per-step
content anchor (no signal level, no noise, not x-predicted, no GT target — confirmed in
`dynamics_model.py` forward: memory read at `mem_start = lat_start + L + n_registers`, injected at
layer 0 via `memory_in`). The per-step loss only asks "be readable by THIS step's reader." With the
carry detached, no gradient connects write_t to loss_{t+1}. So the objective is a *consistency*
condition `write(readable m, x) ≈ readable m'`, NOT a contraction toward a unique target. The Bellman
/ Q-learning analogy the note leaned on is INVALID: Q-learning's TD target is anchored by an observed
scalar reward every step (γ-contraction, unique fixed point); here there is no reward analog, so the
relay is free to slowly rotate/shrink its code (drift/collapse) while still satisfying every step.

**Empirical proof (`experiments/verify-T014/probe_detached_relay_v2.py`, seed 0, CPU):** synthetic
relay — a secret injected at hop 0 and NEVER re-supplied, per-step recover-the-secret loss, GRU
writer; trained to horizon 32, evaluated to 200 hops. Recovery MSE (chance ~0.98):
- IN-WINDOW (≤31): all relay modes ~0 (0.0002–0.0008) → per-step sufficiency learnable, detach looks
  harmless. **THIS IS THE TRAP** — in-window FF9 loss→0 does NOT certify cross-hop preservation.
- BEYOND train depth: detached drifts monotonically to chance — d50 .006 → d100 .149 → d150 .535 →
  **d199 1.077 (≥chance)**. deep-avg detached 0.587 vs full-BPTT 0.007 (84× worse) vs no_relay 0.982.
- tbptt1 (1 hop of grad, IDEAS option B) helps but STILL drifts (deep 0.255). Only full BPTT stays
  flat (d199 0.0135). detached drift d199/d16 = 3589×.

**Corrections (for the build, if pursued):** (1) keep ≥1 hop of grad (tbptt1) — partial only; (2) train
at the DEEPEST eval horizon, no depth extrapolation (the cliff is at the train/eval horizon boundary);
(3) add norm/clamp/projection on the relayed final-layer activation (final-layer→layer-0 over ~200
hops drifts/explodes regardless); (4) gate on a DEEP-HOP sufficiency metric (recover info injected N
hops earlier), never on within-window FF9 loss.

**Probe pattern worth reusing:** to test any "per-step-local credit propagates a relay across many
hops" claim — plant a signal once, never re-supply it, make the per-step loss REQUIRE the relayed
signal, and compare no_relay (lower bound) / detached / tbptt1 / full-BPTT (upper bound) by
recovery-error vs hop depth, training to depth D and EVALUATING past D. The extrapolation gap exposes
drift that in-window metrics hide. See also [[t011-scorer-audit]] (synthetic-belief probe pattern).

**Other T-014 risks confirmed:** cache-under-grad (risk 5) is SOUND iff the committed K/V tensors are
detached (the `_evict_oldest` slice keeps a view; a grad-carrying commit would grow the backward graph
and cause unintended BPTT — add `assert not k.requires_grad`). 50/50 GT split (risk 3) re-opens the
[[V-T013]] shortcut on the noised-GT half (memory non-load-bearing there) — make the strict fraction a
tuned, monitored knob, not hard-coded 0.5.
