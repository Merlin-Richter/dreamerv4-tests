---
name: colorfield-comeback-eval-audit
description: ColorField comeback-eval bookkeeping (autoresearch/frozen/eval_comeback.py) audited 7/7 CONFIRMED; reusable independent-reference-tracker + liar-frames probe pattern
metadata:
  type: project
---

# ColorField comeback-eval bookkeeping — audited correct (2026-07-06)

Audited `autoresearch/frozen/eval_comeback.py` (+ adapters.py) — the result-defining
scalar of the autoresearch harness — against `tasks/in-progress/colorfield-env-and-eval.md`.
**All 7 bookkeeping claims CONFIRMED, no defect.** Artifacts: `experiments/colorfield-bookkeeping-audit/`
(5 probes). Complements the earlier `V-colorfield-audit` (env/readout/policies/datagen geometry).

**Proven facts about this eval (re-checkable, don't re-derive):**
- on-screen (readout `ov>=6` x AND y) ⟺ cell center in view, EXACTLY. Reason: offset
  `u = ci*12 - tl` is always ODD (tl=2p−31 odd, cell start even) ⇒ partial-edge overlap
  ∈ {1,3,5,7,9,11}, never 6 ⇒ the fragile boundary never occurs. Holds off-lattice too.
- Comeback fires iff a tracked cell had ≥1 ZERO-overlap frame between visits; partial-only
  dips never fire (the `gap` flag is set only on absence-from-`reads`, reset at visit start).
- age = `v["start"] - v["prev_last"]` = first re-entry frame − last on-screen frame of prev
  visit. **Structural min age = 6** under 2px steps (can't jump center→zero-overlap); so
  bin `[1,16]`'s `[1,6)` sub-range is always empty — labeling quirk, not a bug.
- provenance set at first ON sighting: prefix→"real" (ref = GT map), imag→"imag" (ref =
  own previous recorded color). `phase` excludes re-entries with `start < prefix_len`;
  re-entry exactly at `t=prefix_len` is INCLUDED (imag). Verified the flip.
- Oracle at FROZEN defaults (192/768/8seeds/min30) = composite EXACTLY 1.0, all 12 bins
  qualified acc 1.0, fidelity 1.0. run_eval is deterministic (byte-identical JSON).
- Equal-weight-over-qualified-bins is population-invariant ONLY within a fixed qualified
  set; a shift dropping a bin under `min_events` moves the score (documented caveat).

**Reusable probe pattern (settles comeback/visit/age bookkeeping decisively):**
1. Build an INDEPENDENT reference tracker: own readout (own nearest-palette + overlap),
   own visit segmentation (maximal ON runs), own comeback rule (any OFF frame strictly
   between prev-last-on and this-first-on), own age/provenance/ref/weight. Share only
   geometry+palette CONSTANTS, never logic.
2. Drive both the real `CellTracker` and the reference with the SAME (frame,pos,is_real)
   stream. Use two frame sources: TRUE renders (structure test) AND "liar" frames that
   paint time-varying colors (some ≠ GT, some OUT) so majority-vote / correct / ref /
   OUT-weight code paths are exercised, not just event detection.
3. Compare full event lists field-by-field; COUNT pathology coverage (1-frame visits,
   single-OFF comebacks, partial-return non-events, last-frame/finalize events,
   prefix-boundary events) and report the counts to prove they were actually hit.
4. For the aggregate: re-derive real/imag/composite/gated/fail_reasons from
   `result["events"]` + reported gate flags; must match to ≤1e-12.
5. Oracle self-test MUST be run at the true frozen config (not a shrunk one) — a shrunk
   config can under-populate bins and make a PERFECT oracle read as gated→0.0 for lack of
   qualified bins, masking whether the 1.0 property truly holds.

Gotcha found in my own test (not the code): synthesizing a bin with count 33 at target
acc 0.40 rounds to 13/33≠0.40 — pick bin counts that represent the target accuracy
exactly when testing equal-weight invariance.
