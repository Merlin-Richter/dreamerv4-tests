# ORIENT.md

Rewritten: 2026-06-13 ~02:15 (EXP-010 k1 arm DONE, k3 arm training)

## What we are doing and why

- **H1, H2 — supported.** Frozen probe 5503e75 + T-004 criteria are the yardstick:
  H3 bar = color ΔRGB < ~63 at n_occ ∈ {12,16,24} (EXP-009 baseline: chance ~110 there).
- **H3 — FF7 v1 built (T-009 done, commit ec45dc1), EXP-010 screening RUNNING.**
  D-014 has the full design + the build-time correction (registers don't persist across
  vanilla generate() steps → param-free `generate_memory` register-carry rollout added;
  probe runs unmodified via checkpoint flag dispatch).

## In flight (this is the thing to check on cold start)

**EXP-010** (local 4070, background bash chain, started 2026-06-12):
1. **k=1 arm DONE** (2026-06-13 02:10): 100 epochs (train 0.00635 / val 0.00651),
   probe complete, detector gate green. `experiments/EXP-010/k1/{results.json,sheet.png}`.
   Early sanity read (NOT reconciliation): color ΔRGB @ n_occ 12/16/24 = 52.1/59.0/79.8
   vs chance ~101 and T-004 H3 bar <63 → k=1 clears the bar at 12 & 16. Ceiling 9.3 /
   drift 14–32: no inference-degradation tripwire visible.
2. **k=3 arm RUNNING** (started 02:10, ~2.85 it/s ≈ same pace; ETA ~05:30 incl. probe)
   → `experiments/EXP-010/k3/`
W&B: exp010-ff7k1-s0 / exp010-ff7k3-s0 (project transformer-D-dynamics).
Chain aborts at first failure (&&-chained).
If a cold start finds the chain dead mid-way: check the k*/train.log tail and W&B,
diagnose, do NOT silently relaunch (3-same-failure rule → escalate).

## NEXT ACTION when EXP-010 finishes
Reconcile per §5 in `experiments/EXP-010/NOTES.md` (expectations pre-registered there +
D-014 tripwires: ceiling/drift degradation; k=3 ≤ k=1; loss interference; out-of-clip
reveals at chance is EXPECTED, not relay failure). Build comparison view (FF7 arms vs
EXP-009 curves + sheets), decisive read, ESC-006, **present-then-stop** — no next decision
before Merlin's verdict.

## Current worries
1. Model may game the per-frame loss by emitting the color prior (= chance on probe).
2. Window-1 `generate_memory` inference may degrade base dynamics → judge via the probe's
   own ceiling/drift controls vs EXP-009.
3. Chained register re-injection at inference is only approximately trained (k=1 not at
   all beyond hop 1; k=3 in-pass only) — the central empirical question of EXP-010.
4. Background-task timeout risk: the harness may cap the chain (~10 min?); verify the
   trainings are actually alive past that mark (W&B heartbeat / log mtime) before idling.
