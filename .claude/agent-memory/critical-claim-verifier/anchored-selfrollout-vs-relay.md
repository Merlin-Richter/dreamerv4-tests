---
name: anchored-selfrollout-vs-relay
description: V-T017-C1 — anchored-GT self-rollout loss (C1) is DAgger not a contraction map; detach is SAFE here (unlike the T-014 relay) because every step has a GT anchor. The gain-probe + detach-equivalence probe that settle it.
metadata:
  type: project
---

**Finding (V-T017-C1, 2026-06-14):** the C1 "time-axis multi-step motion loss" (self-fed rollout,
DETACHED context, **τ=0 pure-noise target slot supervised against the REAL GT successor**, TBPTT-1)
is **on-policy distribution correction (DAgger / scheduled sampling), NOT a "contraction map."** The
design's "pushes toward a contraction map" framing is mechanistically wrong.

**Why (load-bearing):** the per-step target is the *true* `f(state)`; the GT successor of any state
(on- or off-manifold) has the gain fixed by the data. So an anchored-GT loss drives the map toward the
TRUE local Jacobian — it cannot push gain below the data's gain. If the true latent dynamics is
locally expanding (likely here; EXP-013 open-loop chaos), C1 learns an expanding map and will NOT damp
its own drift. It helps ONLY when the deficit is off-manifold *accuracy* (ff7/ff9: 1px flat
teacher-forced but open-loop diverges) — DAgger fixes that. It does NOT fix intrinsic high-gain
dynamics (irreducible).

**Crucial contrast with [[detached-carry-relay-drift]] (T-014, REFUTED):** there the carrier had NO
GT anchor (a self-consistency condition → free to drift/collapse). HERE every step has a GT anchor, so
**detach is SAFE.** Don't reflexively apply the T-014 "detached carry drifts" verdict to C1 — the
anchor is the difference.

**Probes that settle this class (`experiments/verify-T017-C1/`):**
1. *Gain probe* (`probe_c1_gain.py` Part 1): true `f=A·x`, A=1.10, single scalar gain, noiseless. TF
   and C1 BOTH learn g=1.1000 exactly → no spurious contraction. Decisive, analytic.
2. *Detach-equivalence* (inline / EXPERIMENTS V-T017-C1): the detached step-j forward graded against
   GT gives a gradient BIT-IDENTICAL to teacher-forcing the map at the model's own visited state
   (DAgger). Detach removes ONLY the through-context BPTT term, not the distribution-correction signal.
   → "does detach kill the fix?" worry is REFUTED.
3. *Capacity tension* (`probe_c1_gain.py` Part 2): multi-step term steals capacity from the one-step
   map (one-step err 0.024 vs TF 0.012); C1 edges TF in transient (h6) but loses in the saturated tail.
   The "single-frame sharpness regression" guard is REAL — val/diffusion tripwire + λ-ramp are
   mandatory, not optional.

**Un-named degenerate mode C1 admits:** prior-emission under an unlearnable drifted context — when the
detached self-context decoheres, GT is not a function of it, so MSE-optimal = ignore context, emit the
conditional mean (the time-axis analog of the [[V-T013]] non-load-bearing shortcut). Gate it with a
context-vs-prior gap monitor per horizon j; mask horizons where the gap collapses (like FF9's horizon
mask). Keep h small.

**Reusable verdict pattern:** to test "training on self-rollout reduces compounding," (a) check whether
the claimed mechanism is contraction (gain↓) or distribution-correction (accuracy-on-visited↑) — the
scalar-gain probe separates them; (b) check the deficit type (off-manifold accuracy = fixable;
intrinsic gain = not); (c) for any detach claim, compute the detached-step gradient and show it equals
teacher-forcing-at-the-visited-state.
