---
name: t011-scorer-audit
description: Audit findings + reusable probe pattern for the T-011 position-memory consistency metric (billiard-residual + onset-anchor)
metadata:
  type: project
---

T-011 proposes a position-memory metric: read belief via 1-frame counterfactual curtain-UP
reveal → gated detect_ball → believed (x,y); score = onset GT-anchor (part 1) + best-fit
constant-speed billiard residual (part 2, headline, GT-free). Audited pre-freeze for D-018.

**Probe pattern (reusable, no trained model needed):** the scorer takes a sequence of believed
(x,y) only, so test its VALIDITY with synthetic belief trajectories: gt, f2_bounce (true physics
+1.5px start offset → bounce desync), hallucinated (coherent billiard, made-up init), frozen,
shuffled, smooth_drift (heading random-walk). Run the proposed scorer on each; check
GT≈floor / forgetting≫floor / F2≈floor separation NUMERICALLY. Add detector noise ~0.65px.
Artifacts: `experiments/verify-T011-scorer/` (scorer_probe.py, adversary_c2.py, c4_markov_check.py).

**What's PROVEN:**
- Headline residual separates remember from forget at n_occ≥8 (gt/f2/halluc ≈0.77 floor;
  frozen 6.6, shuffled 10.8, smooth-drift 4.9 @ n_occ16, noise 0.65, seeds 1000+).
- **DEGRADES at small n_occ:** 3-param billiard fit (x0,y0,θ0) to ~4 points is over-flexible —
  smooth_drift resid 0.88 ≈ GT floor @ n_occ4. Needs an n_occ floor (≥~8) before the residual
  is trustworthy, OR a residual gate that ignores the first few steps.
- Fixed-speed-S constraint does real work: a coherent ball at 0.7×/1.3× env speed scores
  resid ~3.4 ≫ floor. Do NOT free the speed.
- Onset anchor catches NAIVE hallucination: random made-up billiard onset_pos min 4.6px over
  60 seeds ≫ gt 0.8px floor. C2 holds against naive hallucination.

**THE BLIND SPOT (highest-value finding):** an "anchor-matched-then-diverge" belief — equals GT
for the first n_anchor in-window steps, then a *different arbitrary* physical billiard — PASSES
both parts (resid 1.33, onset_pos 0.85). The metric cannot distinguish genuine dead-reckoning
from "track for n_anchor steps then confabulate a physical path." This is the literal content of
the spec's "Open question for Merlin" and is the boundary of what the metric can claim. Onset
anchor only constrains the in-window steps; everything after is free as long as it's physical.

**Recommended fixes before freeze:** (1) set an n_occ floor (≥8) for the residual headline;
(2) add a minimum-residual / fit-quality gate so a degenerate short trajectory can't trivially
fit; (3) keep speed fixed at env S (verified load-bearing); (4) document that the metric scores
"coherent physical belief anchored to the real ball at onset," NOT "correct dead-reckoning" —
late confabulation is credited by design (Merlin's call per D-018).

C4 env-mechanics PASS: curtain action is absolute/Markov (see [[occluded-env-physics]]); a
1-frame reveal is in-distribution; generate()/generate_cached() accept arbitrary per-frame
action_idx (dynamics_model.py ~L518, L603). Model-side readout (ceiling control) still unverified
— needs a trained model.
