---
name: occluded-env-physics
description: Exact physics/render rules of OccludedBouncingEnv — the ground-truth a position-memory scorer must replicate
metadata:
  type: reference
---

`src/data_generators/occluded_bouncing.py` — `OccludedBouncingEnv`. Rules a scorer/probe must
replicate exactly (verified by reading `_advance_physics` / `_render`):

- img_size=64, radius=10. Walls: ball center clamped to [10, 53] (= radius, img-1-radius).
- Step order per frame: advance (x+=vx, y+=vy) THEN reflect+clamp. So saved state[t] is the
  POST-advance position. Per-axis reflection: x-wall flips vx & clamps x; y-wall flips vy & clamps
  y; corner flips both. At a bounce frame the finite-diff displacement SHRINKS (clamp), not just
  sign-flips — a scorer must allow a magnitude dip at bounce frames.
- Speed: vx,vy ~ U(1.5,3.0) with random sign, FIXED within an episode. Episode speed
  S=hypot(vx,vy) ∈ ~[2.1,4.2]. Read S from saved states; it is a fair per-episode reference.
- **Curtain action is ABSOLUTE/Markov** (verified empirically, c4_markov_check.py): physics
  advances every frame regardless of action; render at frame t depends ONLY on action[t] and
  current physics state. Two episodes with same seed but different curtain histories produce
  byte-identical frames wherever both choose up. → a 1-frame counterfactual reveal mid-occlusion
  is in-distribution and needs no retrain.
- Seed-fixed → physics trajectory identical across different curtain schedules / reveal horizons
  (the T-011 "shared-prefix" readout construction is sound on the env side).
- Detector (revisit_probe.detect_ball): value>=150 threshold, min area 5px; GT detector gate
  requires pos p99<1.5px, dRGB p99<10, miss<=1%. Believed-pos detector noise ~0.65px.
