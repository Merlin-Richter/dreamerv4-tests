# GridWorldV2: action-driven square (7 actions) — env + datagen + recall eval

Requested by Merlin 2026-07-04. Motivation: a trustworthy, fast, deterministic testbed where the
hidden state depends on the ACTION STREAM — under occlusion the belief must integrate actions
t..t+n−1 (with wall clamping = nonlinear), not just extrapolate ballistics. This is the env the
sparse-memory design (tasks/drafts/sparse-memory-spatial-inject.md) needs: memmaze lacks a good
eval; GridWorld v1's square moves autonomously.

Actions: 0=reveal, 1=hide (curtain latch; square does NOT move on toggle ticks),
2=up, 3=down, 4=left, 5=right (clamped at walls), 6=stay. Deterministic given (seed, actions).

## Done means
`src/envs/gridworldv2.py` + `src/datagen/generate_gridworldv2.py` + `src/evals/gridworldv2/recall.py`
(+ DRAFT specs for each, per Merlin's standing note: new specs allowed, declared draft) +
`src/tests/test_gridworldv2.py` gates green; v1 readout verified exact on v2 frames; recall
instrument self-test (oracle == 1.0) green; small local dataset generated; CLAUDE.md updated.
Training arms on v2 = follow-up tasks (Merlin's call).

## Provenance
- (filled at close)
