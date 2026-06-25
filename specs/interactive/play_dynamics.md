# play_dynamics.py — interactive single-frame viewer for a trained dynamics checkpoint.

Eyeball memory behaviour: seed a few real frames, then generate frames one keypress at a time, choosing
the curtain each step. Lets you watch whether the square is remembered through a manual occlusion.

## Interface
- CLI: `--checkpoint <dynamics.pt> --tokenizer <frozen.pt> --frames data/gridworld.npy`.
- Keys: `0` = generate next frame with action 0 (revealed), `1` = action 1 (occluded), `r` reset, `q` quit.

## Behavior
- Load the dynamics checkpoint + frozen tokenizer. Reset = take ~4 real seed frames (from the dataset or
  the env), encode to latents. Each keypress: append the chosen action and call `model.generate(...)` for
  ONE more frame (the carrying rollout for memory models, plain for vanilla), decode it, display it
  (scaled, red border when occluded). Rollout continues indefinitely until reset.

## Invariants
- Uses the single `model.generate` path (carrying for `n_memory>0`); no FF7/snapshot special branches.
- Display only converts BGR→RGB for the window; the model/eval pipeline stays BGR. Inference-only (no grad).
