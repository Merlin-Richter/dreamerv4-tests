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
- Code: master @ ec45dc1 (T-009 commit) ; probe 5503e75 (unmodified).
- Runs: local (4070). W&B project transformer-D-dynamics.
  k=1 = run 82klng1c (exp010-ff7k1-s0); k=3 = run 17u810q2 (exp010-ff7k3-s0).
- Training: 100 ep each, budget-matched. Final val: k1 0.00650, k3 0.00725 (both healthy,
  no divergence). Checkpoints k1/ff7_k1_s0.pt, k3/ff7_k3_s0.pt.

## Observed
Headline = color ΔRGB at reveal (lower = better recall). T-004 bar < ~63 at n_occ {12,16,24}.
Baseline (EXP-009) is at chance (~100–120) for all n_occ ≥ 7.

| n_occ                | 6  | 7    | 8     | 9    | 12   | 16   | 24   |
|----------------------|----|------|-------|------|------|------|------|
| baseline occluded    |16.8| 94.4 | 116.0 |113.9 |108.4 |100.5 |120.3 |
| **FF7 k=1 occluded** |27.1| 31.7 | 37.9  | 39.0 | 52.1 | 59.0 | 79.8 |
| **FF7 k=3 occluded** |24.3| 27.7 | 31.6  | 32.2 | 39.8 | 55.1 | 65.1 |
| k=1 drift-ctrl       |20.1| 23.0 | 24.5  | 24.6 | 27.9 | 28.2 | 32.2 |
| k=3 drift-ctrl       |20.1| 23.3 | 25.4  | 24.7 | 24.5 | 31.8 | 36.5 |

T-004 bar (<63 at 12/16/24): k=1 → 52.1✓ / 59.0✓ / 79.8✗ ; k=3 → 39.8✓ / 55.1✓ / 65.1✗.
Both arms clear the bar at n_occ 12 and 16; both narrowly miss at n_occ 24 (k=3 by 2 pts).

Controls (own rollout path): ceiling ΔRGB k1 9.3 / k3 9.0 (baseline 15.9 — FF7 BETTER, not
worse); chance ~100–107; ball_lost_rate 0.00 everywhere; detector gate PASS both.

**Position is at chance, not retained.** pos_err_px occluded k1 ~19–21, k3 ~21–28 vs chance
~21–22. So color (a static attribute) is carried through occlusion; exact position is not.

**latent-MSE (secondary) does NOT corroborate the headline.** FF7 occluded latent-MSE
(~0.74–0.80) sits near chance MSE (~0.90) and barely separates from its own drift control —
because latent-MSE is dominated by the at-chance position (pearson latentMSE↔posErr 0.97/0.92,
vs latentMSE↔color only 0.81/0.69). This is the position-confound T-004 anticipated when it
made color the headline and latent-MSE merely validating; here the two metrics dissociate
because color is retained while position is not.

## Reconciliation
**Expected** (D-014, pre-registered): k=1 above chance for ~1 window past the cliff (n_occ
7–9), decaying *toward chance* by n_occ 12+ (chained relay untrained at k=1); k=3 sustained
further than k=1.

**Observed:** k=1 did NOT decay toward chance by 12+ — it held far below chance through
n_occ 16 (52.1, 59.0) and was still well under chance at 24 (79.8 vs 100). k=3 was uniformly
better than k=1 beyond the window (39.8/55.1/65.1 vs 52.1/59.0/79.8). The baseline cliff
(→chance at n_occ 7) is replaced in both arms by a smooth, gentle decay that stays below the
T-004 bar out to n_occ 16.

**Surprise: mild-to-high (favorable).** Two surprises: (1) k=1 substantially exceeded its
own pre-registered expectation — the "untrained chained interface" reasoning was too
pessimistic; even single-frame sufficiency + param-free register-carry relays color far past
one window. (2) A clean color/position dissociation: color carried, position at chance, and
consequently the secondary latent-MSE stays flat near chance. The headline metric and the
secondary metric disagree — honestly, latent-MSE alone would read this as a near-null result;
color decomposition is what reveals the retention. (Position-at-chance is broadly consistent
with it being dynamics-dependent and pre-flagged drift-confounded, but the completeness of the
position null is notable.)

**Hypothesis impact (H3):** Supporting. This is the first method to move the post-window cliff
off the chance floor on the frozen probe — exactly the H3 target. Scope of the claim:
**hidden COLOR retention, not full hidden-state retention** (position not retained). Clears the
pre-registered bar at 2 of 3 test points for both arms; misses n_occ 24.

**Tripwires checked (D-014):** NONE triggered, all in the favorable direction. (a) Base-quality
degradation from window-1 inference: ceiling/drift are equal-or-better than EXP-009, not worse
→ clear. (b) k=3 ≤ k=1 beyond window (relay rationale wrong): FALSE, k=3 > k=1 everywhere
beyond window → relay rationale holds. (c) loss divergence / diffusion-component degradation:
both arms trained to healthy val loss, ceiling control near-perfect → no interference.

**Next: ESCALATE — present-then-stop (§5).** Strong supporting result with one honest caveat
(color-only; latent-MSE non-corroborating due to position confound) and one near-miss
(n_occ 24). No next decision until Merlin's verdict. → ESC-006.
