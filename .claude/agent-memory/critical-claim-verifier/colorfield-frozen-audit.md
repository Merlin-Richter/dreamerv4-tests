---
name: colorfield-frozen-audit
description: autoresearch/frozen/ ColorField env+eval audit — proven-true claims, the estimate_shift degenerate-tie trap, and reusable probe patterns
metadata:
  type: project
---

Audited `autoresearch/frozen/{env,readout,policies,eval_policies,datagen}.py` vs
`tasks/in-progress/colorfield-env-and-eval.md` (2026-07-06). Probes:
`experiments/colorfield-geometry-audit/probe_{geometry,actions,safety,shift,dataset,readout}.py`.

**Geometry cheat-sheet (verified):** 15×15 cells × 12px = 180px world; view 64px; lattice
90×90 (p∈[0,89]); view top-left (map-world) = `2p−31`; padded world 242×242, map at
`[31,211)`; OUT band per side = `max(0,31−2p)` (near) / `max(0,2p−147)` (far), always ODD
{1,3,…,31}; `band=31 ⟺ p∈{0,89} ⟺ outward move invalid`. Palette min pairwise dist 105.5
(blue↔purple), OUT sep ≥156.4 (nearest green).

**Proven TRUE:** geometry (pixel-exact all 8100 pos); invalid-action semantics + no policy
emits invalid (1.28M fuzz); **THE safety claim** — eval policies never return a move with
band≥30 (3.46M adversarial-band fuzz, 0 violations); real-frame `band≥30 ⟺ outward-invalid`
exhaustive; dataset render==step + determinism; read_cells exact incl out-of-map tiles.

**The safety guard is structurally airtight:** `ClosedLoopPolicy.act` (base, not overridden
by any suite member) post-filters `_propose` output to `{STAY} ∪ allowed` where
`allowed={moves with band<30}`. No `_propose` bug can leak an unsafe move. When checking a
guard like this, confirm no subclass overrides the guarded method, then the property is
independent of subclass logic.

**The one defect — estimate_shift degenerate-tie trap** (`readout.py::estimate_shift`): min-MSE
over shifts, ties → scan-order-first `(−3,−3)`. Sign convention `2·DELTAS[a]` is CORRECT and
holds on all textured frames (3820/3820, 0/3000 random). REFUTED as universal: a uniform real
map (all-one-color, legal but ~10⁻³⁴ under sample_map) or straight-band+uniform-map makes the
frame translation-ambiguous → wrong shift. CORNER single-color is RESCUED by the L-shaped OUT
boundary. Docstring acknowledges this ("intended" fidelity mismatch). Reusable trap pattern:
**any min-MSE/argmin shift/registration estimator fails on low-texture inputs — construct the
uniform-input worst case; it's the standard refutation.** Fix = tie-break toward min-magnitude
shift, or have the fidelity gate ABSTAIN on ambiguous (min≈2nd-min MSE) frames.

**Reusable probes:** (1) geometry → dependency-free from-scratch pixel renderer, compare
bit-exact at all lattice positions + corners + band boundaries. (2) safety guard → fuzz with
adversarial synthetic band streams (flicker 0↔64, all-blocked, negative/absurd, mid-pattern
borders) × all suite policies × many seeds, assert the safety predicate on the RETURNED action.
(3) equivalence → exhaustive over ALL positions incl. the far edge p=89 (stride sweeps silently
skip it). Note: `autoresearch/frozen/` is a self-contained package — import `autoresearch.frozen.X`
from repo root; run with `venv/Scripts/python.exe`.
