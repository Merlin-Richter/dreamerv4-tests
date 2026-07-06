# ColorField env + eval-policy audit — verdict report

(Independent adversarial audit, background agent, 2026-07-06. Probes in this directory;
run `venv/Scripts/python.exe -u experiments/colorfield-geometry-audit/probe_<x>.py` from repo root.
Author's own gate tests were deliberately not trusted or used.)

## Scoreboard

| # | Claim | Verdict |
|---|-------|---------|
| 1 | Geometry (12px cells, 180px world, TL=2p−31, band=max(0,31−2p)) | **CONFIRMED** |
| 2 | Invalid-action semantics; `valid_actions` exact; no datagen policy emits invalid | **CONFIRMED** |
| 3 | **SAFETY: eval policies never return a move with band ≥ 30 (arbitrary inputs); real-frame equivalence** | **CONFIRMED** |
| 4 | `estimate_shift` returns 2·DELTAS[a] on real frames | **CONFIRMED (sign + all textured frames) / REFUTED as a universal statement** |
| 5 | Procedural dataset bit-exact vs env; regeneration deterministic | **CONFIRMED** |
| 6 | `read_cells` exact incl. out-of-map; palette separation | **CONFIRMED** |

Net: 5/6 fully confirmed; claim 4 true for its sign convention and every non-degenerate real
frame but false as a universal statement — documented limitation with a concrete fix. No safety
defect found.

## Key evidence

- **Claim 1**: dependency-free pixel renderer, bit-exact on 11,646 renders incl. corners;
  exhaustive over all 8,100 positions; bands odd {1,3,…,31}; far-side formula max(0,2p−147)
  symmetric, verified; 242px world exactly fits the p=89 view (no silent truncation).
- **Claim 2**: valid_actions == truth table at all 8,100 positions; step raises on all 8
  outward-at-edge cases and mutates nothing on raise; 1,279,680 fuzzed policy actions → 0
  invalid; 200-episode dataset scan → 0 outward-at-edge, actions[:,0]==STAY.
- **Claim 3**: analytical structural proof (act() return confined to {STAY} ∪ allowed — hard
  post-filter, no suite member overrides act) + 3,456,000 adversarial-band fuzz actions → 0
  violations, no crashes on all-blocked. Equivalence: real bands odd ⇒ band≥30 ⟺ band=31 ⟺
  p∈{0,89} ⟺ outward invalid — exhaustive, 0 mismatches; real-frame closed-loop drive → 0 raises.
  Band measurement airtight: OUT is ≥156.4 from every map color; band rows 100% OUT vs first map
  row ≤48% (out_frac=0.9 has huge margin) even on worst-case all-green maps.
- **Claim 4 (the one defect)**: `readout.py::estimate_shift` min-MSE ties resolve to scan-order
  first (−3,−3). On translation-ambiguous REAL frames (legal but ~1e−34 probability uniform maps;
  also constructed band+uniform cases) the true shift is not the unique minimizer → spurious
  fidelity mismatches possible. Generic maps: 3,820/3,820 correct incl. borders; 0/3,000 random
  interior errors. Suggested fix (preferred): return an ambiguity flag (min≈second-best or
  variance floor) and have the fidelity gate ABSTAIN on ambiguous frames (a texture-free frame
  cannot demonstrate fidelity either way); alternative: least-motion tie-break.
- **Claim 5**: render_episode == env.step frame-exact on 60 episodes; same-seed regeneration
  identical across all sidecars; different seed differs (non-vacuous).
- **Claim 6**: 105,845 cell reads → 0 misreads; no extended cell straddles the map boundary
  (edges at multiples of CELL_PX); min pairwise palette distance 105.48 (blue↔purple), no-confusion
  radius 52.74; on real frames perturbation is 0 ⇒ readout exact.

## Scope & caveats

Imagined-frame band/shift behaviour is model-dependent by design and out of scope (the safety
guard makes any painted border safe regardless). Claim-3 zero-violation is backed by structural
proof + exhaustive fuzz, not sampling luck. Claim-4 practical impact is confined to the
fidelity gate / oracle self-test on texture-free frames.
