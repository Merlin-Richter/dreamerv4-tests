# ColorField comeback-eval — independent bookkeeping audit (verdict report)

(Independent adversarial audit, background agent, 2026-07-06. Probes in this directory;
reproduce: `PYTHONPATH=. venv/Scripts/python.exe -u experiments/colorfield-bookkeeping-audit/<probe>.py`.
Author's gate tests deliberately not trusted; all evidence from independent reimplementations.)

## Verdicts — 7/7 CONFIRMED, no bookkeeping defect found in the scored path

| # | Claim | Verdict |
|---|---|---|
| 1 | on-screen (ov≥6px x AND y) ⟺ center in view; no parity break | CONFIRMED |
| 2 | comeback = on → ≥1 zero-overlap frame → center back; partial-return never fires; scored once | CONFIRMED |
| 3 | age = first_on(re-entry) − last_on(prev visit); no off-by-one | CONFIRMED |
| 4 | provenance prefix→GT / imag→own prev read; re-entry-in-prefix excluded; boundary exact at t=prefix_len | CONFIRMED |
| 5 | bins/weights/equal-weight mean/composite/gating reproduce from raw events (≤1e−12) | CONFIRMED |
| 6 | oracle == 1.0 exactly AT FULL FROZEN DEFAULTS (192/768/8 seeds/min_events 30; 39,722 real + 11,694 imag events, all 12 bins acc 1.0); equal-weight rule age-distribution-invariant while qualified set fixed | CONFIRMED |
| 7 | run_eval determinism (byte-identical JSON) | CONFIRMED |

Key evidence: parity proof that overlap==6 / center-at-boundary cases are unreachable (offsets
always odd); independent reference tracker driven by the same pixel streams, 120 runs incl.
liar-colored frames — 0 field-level disagreements across (cell, provenance, t, age, color, ref,
correct, weight, phase); 6,966 partial-return situations all correctly non-firing; 329
single-OFF-frame comebacks (tightest case) all fire; prefix boundary flips exactly at
prefix_len; equal-weight invariance to 12 dp under population shifts with fixed qualified set.

## Caveats (non-defects, flagged for the write-up)

- **N1**: structural minimum comeback age under 2px steps is 6 (center-in-view → zero-overlap
  needs ≥3 outbound steps); bin [1,16]'s [1,6) sub-range is unreachable. Labeling remark only.
- **N2**: the equal-weight protection is CONDITIONAL — a shift that pushes a bin below
  min_events changes the qualified set and moves the score. Consistent with spec intent.
- **C1** (out of scope, flagged): all proofs use exact flat-color renders; a real model emitting
  noisy/gradient cells could be misread at partial visibility. Readout-robustness question
  (nearest-palette-of-mean is the mitigation), not a bookkeeping bug. Dovetails with the
  geometry audit's estimate_shift ambiguity finding on texture-free frames.
