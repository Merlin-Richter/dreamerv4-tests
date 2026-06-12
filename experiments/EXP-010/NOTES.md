# EXP-010 — FF7 v1 screening: k=1 vs k=3 (single seed each)

Decision: D-014. Hypothesis: H3 (FF7 attempt #1). Probe frozen at 5503e75. Local 4070.

## Setup
- Method: FF7 single-timestep sufficiency (D-014; IDEAS.md FF7 v1 + build-time correction):
  combined loss = shortcut-forcing diffusion + 1.0 × FF7 rollout term; inference =
  `generate_memory` register-carry rollout (param-free), auto-dispatched via checkpoint flag.
- Data: occluded.npy (+ actions), frozen tokenizer trained_autoencoder.pt.
- Arms: `--ff7 1 --seed 0` and `--ff7 3 --seed 0`, training budget matched to each other
  (and roughly to the EXP-009 baseline's my_dynamics.pt; see provenance below).
- Eval: frozen probe, full grid (N=8, P=3, n_occ {2,4,6,7,8,9,12,16,24}, 64 eps/n_occ)
  vs T-004 bar: color ΔRGB < ~63 at n_occ ∈ {12,16,24} (EXP-009 baseline: chance ~110 there,
  ceiling ~16, drift 17–40).

## Expected (pre-registered in D-014, written before any result)
- Smoke: finite combined loss; both components decrease.
- k=1: recall above chance for ~1 window past the cliff (n_occ 7–9), decaying toward chance
  by n_occ 12+ (the chained write-after-read interface is untrained at k=1).
- k=3: recall sustained further than k=1 (in-pass relay trained over 2 hops).
- Tripwires (D-014): FF7 ceiling/drift much worse than EXP-009 → window-1 inference degrades
  base dynamics, judge memory claim only after addressing; k=3 ≤ k=1 beyond-window → relay
  rationale wrong; loss divergence or diffusion-component degradation → interference.

## Provenance
- Code: master @ <SHA after T-009 commit> ; probe 5503e75 (unmodified).
- Runs: local (4070). W&B project transformer-D-dynamics, runs: <fill>.

## Observed
<fill after runs>

## Reconciliation
<fill — Expected / Observed / Surprise / Hypothesis impact / Tripwires checked / Next>
