# gridworldv2.py — action-driven GridWorld (7 actions)

> **STATUS: DRAFT** (agent-written 2026-07-04 from Merlin's verbal design; Merlin has not signed
> off the wording). Decision points the agent chose, flagged for review: wall behavior = CLAMP
> (blocked move = stay put; no velocity, so no reflection), curtain = LATCHED state set by
> actions 0/1, the square does NOT move on a curtain-toggle tick, action numbering below.

The action-conditioned successor of GridWorld (v1). Same 64×64 geometry, palette, rendering, and
readout compatibility — but the square has NO autonomous physics: it moves ONLY when commanded.
Purpose: a deterministic, exactly-probeable memory env where the hidden state depends on the
ACTION STREAM — under occlusion, predicting the reveal requires integrating every movement action
(with wall clamping, a nonlinear function of the sequence), not extrapolating ballistics.

## Actions (n_actions = 7)

| id | name | effect |
|----|------|--------|
| 0 | reveal | curtain := up (revealed). Square does not move this tick. |
| 1 | hide | curtain := down (occluded). Square does not move this tick. |
| 2 | up | row := max(row−1, 0) |
| 3 | down | row := min(row+1, GRID_N−1) |
| 4 | left | col := max(col−1, 0) |
| 5 | right | col := min(col+1, GRID_N−1) |
| 6 | stay | no movement |

- The curtain is a LATCHED state (unlike v1's per-frame absolute action): movement/stay actions
  leave it unchanged, so physics runs invisibly behind a latched-down curtain. Ids 0/1 keep v1's
  reveal/occlude semantics so eval conventions transfer.
- Movement into a wall is BLOCKED (clamp): the square stays. There is no velocity anywhere.
- Deterministic: (reset seed, action sequence) fully determines every frame.

## Episode constants / state

- Per-episode (seeded at `reset`): background color + square color (distinct, from v1's 4-color
  PALETTE), random start cell, curtain up. Colors exposed measurement-only exactly like v1.
- `step(action) -> (frame uint8 HWC BGR, state float32[3])` with state `[col, row, curtain]`.
- `hidden_state() -> float32[3]` measurement-only, never a model input.
- `render_revealed() -> frame` measurement-only: the revealed render at the CURRENT position
  regardless of the curtain latch — the oracle frame source for the recall eval (v1's eval got
  this by passing action 0; v2's toggle-tick semantics require an explicit measurement hook).

## Rendering

Identical to v1 by construction (imports v1's geometry + drawing helpers): 3px border + 6×8px
cells + 2px lines = 64; curtain = flat CURTAIN_COLOR gray. Consequence: the v1 closed-form
readout (`evals/gridworld/readout.read_square`) is exact on v2 frames unchanged.

## Shared movement policy

`sample_moves(rng, n, run_max=4) -> list[int]`: movement-action stream in {2..6} drawn as
direction RUNS (uniform action, run length ~ U{1..run_max}) — used by datagen AND the recall
eval so train/eval action statistics match; runs give real displacement (a pure uniform walk
barely moves).

## Invariants

- BGR end-to-end; BaseEnv contract (`reset`/`step`/measurement-only `hidden_state`).
- Geometry constants imported from v1 — the two envs can never drift apart visually.
- Toggle ticks never move the square; movement ticks never change the curtain.
