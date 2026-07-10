---
name: frozen-sym-v22-score
description: frozen_sym v2.2-sym continuous headline score (eval_comeback.py) verified 5/5 CONFIRMED; formula, waypoints, and the HEAD-side-by-side + hand-computed-aggregate probe pattern
metadata:
  type: project
---

# frozen_sym v2.2-sym continuous score — verified (2026-07-10)

Audited the gate->continuous rewrite of `autoresearch/frozen_sym/eval_comeback.py`
(SYM tier, "v2.2-sym", Merlin 2026-07-10). Old hard gates (fid>=0.90, KL<=0.2,
score:=0 on fail) replaced by a continuous headline. **All 5 claims CONFIRMED,
no defect.** Artifacts: `experiments/autoresearch-loop-shakedown/verify_v22/`
(probe_all.py 45/45 pass; head_eval.py = HEAD copy with abs imports).

**The formula (verified exact):**
`score = fid * (0.2*ent + 0.8*composite)` where
- `fid = (fid_move + fid_hold)/2`, each an equal-weight mean over the two
  (is_move, ok) pools; empty pool -> 0.0. is_move := (t%5==0 and action!=STAY).
- `ent = clip((0.6 - KL)/0.4, 0, 1)`, KL = first-seen imag-born in-map color
  marginal vs uniform-5; **ent=0 when <20 samples** (not default 1).
- `composite = real_cc * (0.7 + 0.3*consistency_cc)`; composite treated as 0 in
  score when real_cc is None (no qualified in-map real bins).
The **component/bin math is byte-identical to HEAD** (per-bin chance-correction
acc_cc=max(0,(acc-.2)/.8), OUT/border excluded from scored acc, equal-weight over
qualified bins n>=min_events, multiplicative composite) — the edit only swapped
the fidelity aggregation (flat mean -> move/hold pools) and the headline wrapper.

**Verified waypoints (SMALL suite, n_seeds=2, prefix240/imag1280, privileged):**
oracle **exactly 1.0** (float ==); perfect_imaginary (coherent scroller, zero
retention) **0.201** ~= the 0.2 fidelity floor; constant_color **0.046** (fid=0.998
full because a uniform grid is shift-invariant -> ent=0 is the sole killer);
copy_last fid **0.500** exact, score **0.114**; noise_cells 0.0. Monotone
nondecreasing in fid/ent/composite (partials 0.2ent+0.8comp, 0.2fid, 0.8fid all >=0).

**Claim-5 (no stale consumer of deleted keys composite_gated/gates_passed/
fail_reasons/gates):** CONFIRMED. Only the SEALED pixel tier `autoresearch/frozen/`
and stale `experiments/**` probe scripts/output still name them; the real sym
consumer `autoresearch/loop/eval_reduced.py` reads r['score'] + live keys only.
Removed constant FIDELITY_THRESHOLD and kwarg fidelity_threshold: no importer.

**Reusable probe pattern (settles a scorer-rewrite equivalence claim fast):**
1. `git show HEAD:path > head.py`, sed the relative imports to absolute, import it
   side-by-side, and diff real/consistency/composite over ~200 random synthetic
   event sets with a bit-identical (float `==`) deep-compare — proves "component
   math unchanged" without re-deriving it. Component math ignores fidelity, so you
   can feed HEAD the old flat-bool list and NEW the (is_move,ok) pairs.
2. Hand-compute every aggregate facet (build events with exact acc per bin) and
   assert to 1e-12; cover the None branches (real None -> composite 0 in score;
   imag None -> amp floor 0.7) and the ramp edges (<20 samples -> ent 0; mid-KL ->
   independent KL recompute).
3. Differential-test the unchanged helper (fidelity_ok) HEAD-vs-new over random
   inputs to guard against a misread of "byte-identical". NB env action space is
   0..4 (UP/DOWN/LEFT/RIGHT/STAY=4), DELTAS keys 0..4; PHASE_PERIOD=5, OUT_IDX=5.

See also [[colorfield-comeback-eval-audit]] and [[colorfield-eval-goodhart]] (the
PIXEL-tier `autoresearch/frozen/` ancestor this was ported from).
