# BUILD_NOTES — autoresearch/frozen_sym (ColorField-SYM frozen layer)

Built 2026-07-07 against the authoritative spec
`tasks/in-progress/colorfield-sym-frozen-layer.md`, with the sealed pixel tier
(`autoresearch/frozen/`, MANIFEST colorfield-frozen-v2.1) as the 1:1 template
wherever the spec is silent. Pure numpy/stdlib, CPU-only. `autoresearch/frozen/`,
`autoresearch/driver/`, and `src/` untouched. Not committed. NOT yet
MANIFEST-sym-sealed (adversarial delta-review + Merlin sign-off pending per the
task's "done when").

## Package contents

| file | role |
|------|------|
| `env.py` | 15x15 board, 5x5 symbolic viewport, phase-5 dilation, `out_bands`, procedural render |
| `policies.py` | the 8-policy datagen zoo on the 15-lattice + phase-forcing `rollout_policy` |
| `datagen.py` | procedural sidecars (maps/starts/actions/policy_ids/ep_seeds/meta.json), CLI, `ColorFieldSymDataset` |
| `eval_policies.py` | 12-entry closed-loop suite driven by OUT bands, band>=2 blocking |
| `eval_comeback.py` | v2.1 comeback eval + `CellTracker` + gates + `aggregate`/`run_eval`/`save_json` |
| `adapters.py` | oracle / perfect_imaginary / noise_cells / constant_color / copy_last + `make_adapter` |
| `tests/` | 4 gate suites (env / policies / datagen / eval), standalone + `-m` runnable |

Shared constants (PALETTE, OUT_IDX, N_CELLS, action ids, `sample_map`) are
IMPORTED from `autoresearch.frozen.env` (read-only), never duplicated.

## Design decisions (spec-ambiguity resolutions; all pixel-tier-analogous)

1. **Tick/phase convention.** Dataset convention kept verbatim from the pixel
   tier: `actions[t]` produces tick `t`, `actions[0] == STAY`; obs[t] =
   (grid, phase = t % 5). The move in `actions[t]` applies iff `t % 5 == 0`.
   Anchor: the spec's fidelity phrasing — "at phase-0 the predicted grid must
   equal the previous grid shifted by the action" — puts the SHIFT on phase-0
   ticks, which forces this reading. Consequence: `env.valid_actions()`
   constrains the NEXT `step()` call (the action producing tick t+1) and is
   `[STAY]` whenever `(t+1) % 5 != 0`; datagen/eval loops consult their policy
   exactly when producing a phase-0 tick.
2. **`positions_from(check=...)` semantics.** `check=True` enforces BOTH board
   bounds and phase discipline (off-phase non-STAY raises — uniform
   invalid-action semantics). Off-phase ticks NEVER move, even with
   `check=False` (env physics, not a validation); `check=False` only allows
   the center to leave the board (imagination registration).
3. **Band blocking is `>= 2`, not `== 2`.** The spec says "band in {0,1,2};
   blocked iff band == 2", which holds on REAL grids; but imagined grids can
   paint wider fully-OUT bands (up to 5), and `==` would walk into them. `>=`
   is the exact analogue of the pixel tier's `band >= 30` rule and coincides
   with `==` on real grids. `out_bands` returns the raw run length (0..5),
   unclipped. A fully-OUT line = all 5 cells OUT (the analogue of the pixel
   >=90%-of-row rule; 4/5 = 0.8 < 0.9 there too).
4. **Visit read = the RETURN-tick grid value**, used both as the scored
   comeback color and as the record the next comeback's consistency ref reads
   (spec: "read color = the grid value, exact — no majority voting"). The
   pixel tier's majority-over-visit collapses to a single-tick read here; the
   return tick is the moment memory is tested (before re-observation can
   refresh). Mid-visit repaints therefore don't update the record — same
   information the pixel majority would usually pick, and exact.
5. **Tracker simplifications (spec-mandated).** No partial-overlap machinery:
   on-screen := cell is one of the 25 viewport slots at the registered center;
   a visit ends only via a fully-absent tick, so the inter-visit absence is a
   single contiguous run and v2.1 max-gap age == that gap. The max-run
   bookkeeping is still implemented verbatim (and cross-checked by an
   independent brute-force reimplementation from position traces).
   `finalize()` is a no-op kept for pixel API parity — events fire at visit
   start, nothing is pending at episode end. All 25 slots are tracked,
   including off-board (OUT) coords: OUT-referenced events are excluded from
   the score and surface as `border_recall` (identical to pixel v2.1).
6. **Fidelity gate (exact on symbols).** Off-phase tick or STAY: predicted
   grid must equal the previous grid EXACTLY. Phase-0 move: equality on the
   shifted 4x5/5x4 overlap; the newly revealed line is unconstrained ("handle
   border/OUT fill" = don't gate it — recalled/imagined content there is the
   eval's business). Pooled fraction over all imagination ticks, threshold
   0.90. Note the ~80% off-phase ticks are a free pass for any
   copy-consistent model — but phase-0 moves are 20%, so copy_last pins at
   ~0.81 < 0.90 and is still killed (measured 0.810).
7. **constant_color PASSES fidelity in this tier** (a uniform grid is
   shift-invariant; in the pixel tier the shift-estimator tie-break failed
   it). The ENTROPY gate is the load-bearing killer (KL = log 5 >> 0.2), as
   the pixel design already argued. Asserted explicitly in the gate test.
8. **perfect_imaginary needs no privileged peek.** The pixel version peeked
   `env.pos` to phase-align its 12px cell grid with the tracker registration;
   symbols have no sub-cell phase — from ANY internal origin its world is a
   constant translation of the tracker's coordinates (it integrates the same
   action stream), so consistency is exactly 1.0 without env. It still
   accepts (and ignores) `env` for factory-signature parity.
9. **noise_cells = fresh iid (5,5) grid every tick** (spec: keep the analogue
   that fails fidelity). It dies on the off-phase-unchanged clause (fid 0.000).
10. **Fidelity transition artifact (inherited from pixel).** The first
    imagination tick is compared against the last REAL grid, so a liar that
    ignores the prefix eats exactly one honest miss per episode:
    perfect_imaginary/constant_color measure fid ~0.999, not 1.0. Oracle is
    exactly 1.0. Intended: a real model must continue the real prefix.
11. **Amplitude scaling (pixel range / 6, in EFFECTIVE MOVES).** Datagen:
    out-and-back 2..10, box sides 2..8, lawnmower lane 1..2, dwell 4..16
    moves, dart threshold manhattan >= 7, anchor radius 1. Eval suite:
    oab (2,5)/(5,9)/(9,14), box (2,4)x10 / (5,9)x6, sweep lanes (1,2)/(2,4),
    idiot p 0.7/0.92 (probabilities unscaled), retrace (5,10)/(10,18),
    dwell_dart dash (4,10) dwell (3,9). Eval-policy counters decrement per
    consult = per effective move.
12. **corner_start = within 2 cells of a random corner** (spec), vs 10 lattice
    steps in the pixel tier.
13. **Bin edges kept at (1,17,33,65,129,257), ages in TICKS.** The spec's
    "expect bin1 sparse" concern does NOT materialize: at spec defaults
    (192/768, 12 policies x 2 seeds) the oracle logs 1623 real-provenance
    events in [1,16] and ALL SIX bins qualify in both provenances at
    min_events=30. No edge adjustment needed at freeze time on this evidence.

## Deviations from the pixel tier (all spec-driven)

- No `readout.py`: readout is identity by construction; `out_bands` (the only
  readout-like utility) lives in `env.py`; `estimate_shift`/`label_pixels`/
  `read_cells` have no analogue (exact symbol equality replaces them).
- New off-phase-unchanged fidelity clause (the spec's "new, free gate").
- `border_drift_px` -> `border_drift_cells` (same diagnostic, cell units,
  computed against the true board geometry at the path-integral center,
  correct for fully-off-board centers).
- Bounded-memory fence recalibrated (test-only, `tests/test_eval.py`): with
  the spec-mandated W in TICKS (16/80/huge) the W/episode ratio is 16x smaller
  than the pixel config (80/1280 ticks vs 64/256 moves), so young-bin
  self-refresh pollution is much heavier — bin [1,16] measures ~0.41 for W=80
  (pixel analogue ~0.88). The fence therefore checks: full memory ~1.0
  (>=0.99), strict monotonicity, W=16 <= 0.15, W=80 in [0.20, 0.55] (~ its
  covered-bin fraction; measured 0.34), fully-covered bins [17,32],[33,64]
  >= 0.50 (measured 0.77/0.71), beyond-window bins (lo>=129) <= 0.15
  (measured 0.0), young bin flagged in [0.20, 0.65]. Measured landscape
  (SMALL config, real-anchored acc_cc per bin):
  W=16: .11 .00 .06 .00 .03 .02 -> 0.036 | W=80: .41 .77 .71 .16 .00 .00 ->
  0.340 | W=huge: all 1.0 -> 1.000.

## Baseline table (full 12-policy suite, spec defaults 192/768, n_seeds=2, privileged)

| adapter | composite_gated | fidelity | gates | note |
|---|---|---|---|---|
| oracle | **1.0000** (exact) | 1.000 | pass | drift 0.0; all 6 bins qualified, both provenances |
| perfect_imaginary | 0.0056 | 0.999 | pass | consistency 1.0; chance-clamped real |
| copy_last | 0.0000 | 0.810 | FAIL fidelity | phase-0 moves kill it |
| constant_color | 0.0000 | 0.999 | FAIL entropy | passes fidelity (shift-invariant) by design |
| noise_cells | 0.0000 | 0.000 | FAIL fidelity | off-phase-unchanged clause |

## Test transcript (venv/Scripts/python.exe, CPU, 2026-07-07; total ~12 s)

```
$ venv/Scripts/python.exe -u autoresearch/frozen_sym/tests/test_env.py
PASS test_determinism
PASS test_geometry_constants_shared_with_pixel_tier
PASS test_invalid_action_semantics_at_phase0
PASS test_out_bands_exact_on_real_grids
PASS test_phase_forced_stay_semantics
PASS test_positions_from_phase_discipline
PASS test_render_episode_matches_env_steps
PASS test_render_matches_reference
test_env: ALL PASS

$ venv/Scripts/python.exe -u autoresearch/frozen_sym/tests/test_policies.py
PASS test_allowed_moves_helper
PASS test_datagen_policies_valid_and_covering
PASS test_datagen_rollout_convention
PASS test_eval_policies_never_push_banded_sides
PASS test_eval_policies_valid_on_real_env
test_policies: ALL PASS

$ venv/Scripts/python.exe -u autoresearch/frozen_sym/tests/test_datagen.py
PASS test_generate_load_replay_determinism
test_datagen: ALL PASS

$ venv/Scripts/python.exe -u autoresearch/frozen_sym/tests/test_eval.py
PASS test_aggregate_math
PASS test_bounded_window_monotone_and_capped
PASS test_constant_color_is_gated_to_zero
PASS test_copy_last_is_gated_to_zero
PASS test_noise_cells_fails_fidelity
PASS test_oracle_scores_exactly_one
PASS test_perfect_imaginary_liar_scores_zero
PASS test_tracker_matches_brute_force
PASS test_unprivileged_factories_get_none
test_eval: ALL PASS
```

Both invocation modes verified: direct file execution (path bootstrap) and
`python -m autoresearch.frozen_sym.tests.test_<suite>`. Datagen CLI smoke:
`python -m autoresearch.frozen_sym.datagen --out <tmp> --n-episodes 4 --T 1024`
generated + loaded + replayed clean ((1024,5,5) uint8 grids, off-phase all
STAY).
