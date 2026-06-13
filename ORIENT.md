# ORIENT.md

Rewritten: 2026-06-13 ~06:30 (EXP-010 BOTH arms done overnight; reconciled; ESC-006 open)

## What we are doing and why
- **H1, H2 — supported.** Frozen probe 5503e75 is the yardstick; T-004 H3 bar =
  color ΔRGB < ~63 at n_occ ∈ {12,16,24} (EXP-009 baseline at chance ~110 there).
- **H3 — first method (FF7 v1, D-014) screened. EXP-010 SUPPORTS H3 (color-only).**
  Both arms replace the baseline post-window cliff with a gentle decay; clear the T-004
  bar at n_occ 12 & 16, miss at 24; k=3 > k=1 (relay holds). Caveat: color retained,
  **position at chance** → latent-MSE (secondary) doesn't corroborate (position confound,
  as T-004 anticipated). No D-014 tripwires fired.

## In flight
**NOTHING running. 4070 idle.** We are at a **present-then-stop gate (ESC-006)** awaiting
Merlin's verdict on EXP-010. Per §5 the §3 prep allowance does NOT apply here — do not
start the next decision, seeds, or any FF7 variant until he answers.

## NEXT ACTION
Wait for Merlin's ESC-006 verdict. His three questions: (1) agree with the color-only read?
(2) is color-retained/position-at-chance enough to credit H3 progress? (3) next direction —
my rec: replicate k=3 at a 2nd seed AND start designing a position/motion-carrying FF7
variant (color-only is the easy half). On his answer: write the next decision, then act.

## Access points for his review (ESC-006)
- `experiments/EXP-010/headline.png` (the one chart that says it all)
- `experiments/EXP-010/comparison.{md,html}`, `k1/sheet.png`, `k3/sheet.png`
- Full reconciliation in `experiments/EXP-010/NOTES.md`; W&B 82klng1c (k1) / 17u810q2 (k3)

## Current worries
1. **Color-only win.** Position unretained — the harder, more interesting half of hidden
   state is not solved. Is partial retention "H3 progress" in Merlin's eyes? (ESC-006 Q2.)
2. **Single seed.** k=3's 2-pt miss at n_occ 24 and the k=1>expected surprise both want a
   replication seed before over-reading. (Standing ≥2-seed order was removed; replicate on
   promise — this looks promising enough.)
3. **latent-MSE non-corroboration** is benign (position confound, pre-flagged) but means the
   detection-free headline we hoped for is, for THIS method, carried entirely by the color
   decomposition. Worth keeping in mind for the eventual writeup.
