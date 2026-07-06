---
name: colorfield-eval-goodhart
description: Red-team of autoresearch comeback eval composite_gated — the two Goodhart levers, the reachable-without-retention ceiling, and the reusable exploit-adapter pattern
metadata:
  type: project
---

# ColorField comeback eval (`autoresearch/frozen/eval_comeback.py`) — Goodhart red-team

Red-teamed `run_eval`'s `composite_gated` for exploitability before it is frozen for
an autonomous optimizer. Artifacts: `experiments/colorfield-redteam/`.

**Two free levers (do NOT require genuine long-range retention):**
1. **Consistency term (0.3 weight) is free = 1.0** for any persistent aligned world
   with a deterministic `color = hash(cell)` (ZERO memory). Authors know this
   (`perfect_imaginary`); it's why composite is 0.7-anchored to ground truth.
2. **Real-anchored (0.7) = equal-weight mean over *qualified* age bins, chance floor
   0.2 (not 0).** So "correct up to age~W, chance beyond" is only penalized 1/6 per
   failed bin, floored at 0.2. Plus **OUT tiles weight 0.1 but bin acc is a WEIGHTED
   mean** → painting off-map gray (free geometry) lifts real ~+0.06-0.08.

**Reachable ceilings (measured, frozen-ish config n_seeds=3, all 6 bins qualify):**
- Zero content retention (`perfect_imaginary` / `GeoOutWorld W=0`): **~0.43** — and
  it BEATS honest 16-frame memory (**~0.25**). Metric is non-monotone in memory.
- Bounded **64-frame** buffer, CHANCE for every age>64 (the whole "many steps"
  range): **composite 0.62 (>0.6)**. W=128 → 0.77. So "just use a bigger context
  window" clears the bar without the memory-token mechanism the research targets.

**Gates that HELD (genuinely load-bearing — pure-liar can't reach ~1.0):** fidelity
blocks tight confinement (static walls → `estimate_shift` ambiguous) AND all-OUT
flight (uniform frame → arbitrary shift); entropy blocks color collapse; OUT
down-weight needs >10:1 ratio to dominate (unattainable ~1:1 near boundary). So
bin-starvation and OUT-sea flooding both FAIL.

**Exploit-adapter pattern (reusable):** recover exact registration from
`prefix_frames[0]` OUT bands (`up=31-2pr` pr<=15, `down=2pr-147` pr>=74) + integrate
actions → matches the tracker's path-integral grid EXACTLY (40/40), so you paint any
color at any registration cell with no env peek. `_CellPainter.paint(pos, color_fn)`.

**TRAP that cost me a wrong result first pass:** the tracker records a returning
cell's color at **MAX VISIBILITY** (several frames into the visit). A short-memory
adapter that refreshes its "last seen" the instant a cell re-enters view silently
becomes a FULL-memory model (scored 0.99 "at W=16"). Fix: freeze the belief
**once per visit at entry** (decide remember/forget from the gap to the PREVIOUS
visit). Catch it by dumping paint decisions vs what the tracker read
(`debug_mem2.py`: tracker read GT where I "painted random").

**Top fix:** age-weight the bins or score the oldest-qualified bin / min-over-top-K,
and/or chance-correct per bin `max(0,(acc-0.2)/0.8)`; neutralize free consistency.
This drops `GeoOutWorld(W=64)` 0.62 → <0.4 while leaving a true long-range model at 1.0.

See also [[t011-scorer-audit]] (same family: synthetic-belief probe to test a scorer).
