# generate_gridworld.py — write the GridWorld training dataset (+ a viewer). Drives GridWorldEnv.

The TRAINING data writer (tokenizer + dynamics train on it). The eval is env-direct and needs no
dataset. Streams the big frames array straight to a disk memmap (a 3000×200×64×64×3 set is ~7.4 GB).

## Interface
- `make_curtain_schedule(rng, n_frames, ...) -> uint8[n_frames]` — the per-frame curtain (0/1) policy.
- `generate_episode(n_frames=200, img_size=64, seed) -> (frames(T,H,W,3)u8, actions(T)u8,
  states(T,5)f32, colors(2)u8=[bg_idx,sq_idx])`.
- `generate_dataset(args)` — writes `<out>.npy` (memmap) + `_actions/_states/_colors.npy`.
- CLI: default = generate; `--play` interactive curtain control; `--debug` preview; `--frames X --episode i`
  play a saved episode. (Viewers are cv2 conveniences — optional / trimmable.)

## Behavior
- **Curtain schedule (block sampler, Merlin's policy):** at each step draw one of three blocks —
  `p_single=0.90` (one frame, occluded with prob 0.5), `p_run_visible=0.05` (8 revealed in a row),
  `p_run_occluded=0.05` (8 occluded in a row). A `start_visible=2` prefix is forced revealed so the
  square's direction is observable before any occlusion. So the data has the common 1-step mix AND long
  runs of either scenario.
- **Episode**: reset env (seeded), build a schedule, step the env through it collecting frames+states;
  store the per-episode `colors=[bg_idx, sq_idx]` (constant within an episode, PALETTE order) so evals
  score 4-way colour without re-deriving from pixels.
- **Dataset**: per-episode seeds from a meta-rng; frames → disk memmap (never held whole in RAM); small
  arrays saved normally; print occluded fraction.

## Invariants
- Outputs: `frames (N,T,H,W,3)u8 BGR`, `actions (N,T)u8` (0=up/revealed,1=down/occluded), `states (N,T,5)
  f32 [col,row,dcol,drow,curtain]`, `colors (N,2)u8 [bg_idx,sq_idx]` PALETTE order. Datasets gitignored,
  written under `data/`.
- `action[t]` is the curtain at frame t (matches env convention). Physics is action-independent.
