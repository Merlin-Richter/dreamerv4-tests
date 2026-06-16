---
name: h3-line-failures
description: Outcomes of the H3 memory line (FF7 register relay, FF9 memory snapshot, detached-carry drift) and taste notes on what worked vs looked sophisticated
metadata:
  type: project
---

H3 = force D to carry hidden/global state past the N-frame context window. Outcomes on record:

- **FF7 (register single-timestep-sufficiency loss, D-014/EXP-010):** registers as memory
  carrier; predict next-k frames from an injected register + near-clean source. Result: replaces
  the post-window color cliff with gentle decay (partial), AND incidentally sharpens base 1-step
  motion ~4.6x (4.5px->1px) — that gain is the LOSS not the relay (EXP-014). Carries static
  COLOR, not dynamic POSITION.

- **FF9 v2 (distinct memory tokens, memory-only sufficiency at tau=0, D-024/EXP-017):** write a
  static full-state snapshot ONCE from the observed prefix, inject frozen. STRICTLY beats FF7 on
  beyond-window static-color retention (flat at ceiling to n_occ=48, 6x window) because a
  written-once snapshot can't drift. Dynamic position still NOT solved (frozen snapshot).
  Correct eval = A1+B1 (tau=0 source, static carry); re-extracting memory each step (B2 / op-3)
  DRIFTS (V-T013-eval).

- **Detached carry across many hops FAILS (V-T014, REFUTED claim).** A per-step-sufficiency
  consistency loss with detached carry is a consistency (not contraction) fixed point with no
  content anchor: in-window loss->0 but BEYOND the trained horizon it drifts monotonically to
  chance (d199 84x worse than full BPTT). **In-window aux-loss=0 does NOT certify cross-hop
  preservation** — that metric misleads. Only keeping >=1 hop of gradient (TBPTT-1) partially
  helps; train at the deepest eval horizon, don't extrapolate.

**Taste notes:**
- The sophisticated-looking "consistency loss + detached relay" was vacuous past the trained
  horizon. The minimal intervention that actually moved the metric was just *adding a
  successor-prediction term* (FF7/FF9) — an objective change, not new machinery.
- Static vs dynamic state are DIFFERENT capabilities: color (static) is carryable by a frozen
  snapshot; position (dynamic) needs an actual trained relay and is still open (T-014/op-3).

**How to apply:** for any persistence/relay proposal, demand a deep-hop sufficiency gate (not
in-window loss) and >=1 gradient hop. See [[dynamics-loss-singleframe-bottleneck]] and
[[probes-dynamics-motion-memory]].
