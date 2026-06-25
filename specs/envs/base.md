# base.py — BaseEnv: the shared interface every world-model environment implements.

A new env subclasses `BaseEnv` and implements `reset` + `step`; then every eval and the datagen can
drive it. Carries the two project-wide contracts.

## Interface
- `class BaseEnv(ABC)`: class attrs `n_actions` (0=unconditioned), `img_size` (default 64).
  - `reset(seed) -> self` — seed + init one episode, deterministic given seed.
  - `step(action=0) -> (frame (H,W,3)u8 BGR, state np.ndarray)` — advance one frame.
  - `hidden_state() -> np.ndarray` — measurement-only view of the sim's hidden state.

## Behavior
- Single-episode, seeded, deterministic. Subclasses expose any hidden quantities evals need to score
  (e.g. `.color`). The `state` vector's width/semantics are PER-ENV (typed as a plain ndarray; each
  subclass documents its layout — GridWorld's is `[col,row,dcol,drow,curtain]`).

## Invariants
- **Channel-order contract:** frames are BGR end-to-end (cv2 native); the pipeline treats the channel
  axis opaquely; RGB↔BGR conversion happens ONLY for on-screen display. Evals compare colours WITHOUT a swap.
- **Privileged-state contract:** the model sees only the rendered frame (+ action). `hidden_state()` /
  `.color` are exposed for MEASUREMENT (evals scoring recall) and are NEVER fed to the model as input.
