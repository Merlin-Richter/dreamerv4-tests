# generate_gridworldv2.py — GridWorldV2 dataset writer + preview

> **STATUS: DRAFT** (agent-written 2026-07-04; Merlin has not signed off). Agent-chosen decision
> points flagged for review: the alternating revealed/occluded RUN schedule and its length
> ranges, and the movement-run policy (shared `sample_moves`, run_max=4).

Drives `GridWorldV2Env` to write the action-conditioned memory dataset. Env (physics+rendering)
lives in `src/envs/gridworldv2.py`; this is the datagen layer.

## Outputs (under data/, gitignored)

- `data/gridworldv2.npy` (N, T, 64, 64, 3) uint8 BGR frames
- `data/gridworldv2_actions.npy` (N, T) uint8 in 0..6 (the trainer auto-detects n_actions=7)
- `data/gridworldv2_states.npy` (N, T, 3) float32 [col, row, curtain]
- `data/gridworldv2_colors.npy` (N, 2) uint8 [bg_idx, square_idx] (PALETTE order)

## Action schedule (per episode, seeded from the env's rng)

Alternating runs — every episode contains in-window prediction, occluded belief-tracking, and
reveals after occlusion:

- Start with `start_visible` (default 3) revealed movement ticks (curtain starts up at reset).
- Then alternate: an OCCLUDED run = one `hide` tick + (O−1) movement ticks, O ~ U{2..12}; a
  REVEALED run = one `reveal` tick + (R−1) movement ticks, R ~ U{3..10}.
- All movement ticks are drawn from one per-episode `sample_moves` stream (direction runs,
  run_max=4) so displacement statistics match the recall eval.

Convention: `action[t]` describes frame `t` (identical to v1).

## Interface

- `generate_episode(n_frames=200, seed) -> (frames, actions, states, colors)`.
- CLI: `--n_episodes` (writer; default output `data/gridworldv2.npy`), `--n_frames` (200),
  `--out`, `--debug` (cv2 preview of one scheduled episode), `--frames F --episode i`
  (play back a saved episode). Episode i uses seed `seed0 + i` (`--seed0`, default 0).
