# gridworld.py — the discrete 6x6 occluded-memory environment (env only; datagen lives elsewhere).

A square occupies one cell of a 6x6 grid and steps one cell each tick, reflecting off walls. A curtain can
hide everything. Discreteness makes recall exact: position = which of 36 cells, colour = which of 4. The
hidden state to retain across an occlusion is: square cell (6x6), square colour (4), direction (8).

## Interface
- Geometry constants: `IMG_SIZE=64`, `GRID_N=6`, `CELL=10` (stride px), `VIS=8` (interior px),
  `BORDER=3`, `BLACK=(0,0,0)`, `CURTAIN_COLOR=(128,128,128)`.
- `PALETTE: dict[name->BGR]` = red/green/blue/pink (ordered → stable class idx); `COLOR_NAMES=tuple`.
- `DIRECTIONS` = the 8 (dcol,drow) in {-1,0,1}^2 \ (0,0).
- `cell_origin(idx) -> int` : top-left interior pixel of cell idx along one axis = `BORDER + CELL*idx`.
- `interior_axis_mask(img_size=64) -> bool[img_size]` : True on cell interiors, False on lines/border.
- `make_grid_background(bg_color, img_size=64) -> uint8 HWC BGR` : solid bg + black grid, no square.
- `stamp_square(frame, col, row, color)` : in-place fill cell (col,row) 8x8 interior.
- `class GridWorldEnv(BaseEnv)`: `n_actions=2`; `reset(seed=None)->self` (default random seed); `step(action=0)->(frame uint8
  HWC BGR, state float32[5])`; `hidden_state()->float32[5]`. Measurement attrs after reset:
  `col,row,dcol,drow,curtain`, `color/color_name` (square BGR), `bg_color/bg_name`.

## Behavior
- Geometry: `3 + 6*VIS(8) + 5*line(2) + 3 = 64`; cell i interior at `cell_origin(i)=3+10i`, spans 8px;
  every cell interior is a uniform 8x8 block, everything else black line/border.
- reset(seed): pick distinct bg & square colours from PALETTE; random start cell + 1-of-8 direction.
- step(action): FIRST advance physics (1 cell/tick, wall-reflect per axis), THEN set curtain=action and
  render. Returns frame + `state=[col,row,dcol,drow,curtain]`.
- Render: action 0 = revealed (grid template + stamped square); action 1 = occluded (whole frame filled
  with CURTAIN_COLOR gray). Physics runs regardless, so the square keeps moving behind the curtain.
- Action convention: `action[t]` is the action applied at step t; its ONLY effect is frame t's curtain
  (0=revealed, 1=occluded) — it does NOT affect the square's physics (autonomous). So action[t] is both
  "the action at t" and, equivalently, frame t's curtain state.

## Invariants
- **BGR everywhere** (envs/base.py contract). Geometry FIXED at 64px/6-cell (reject other img_size).
- `CELL=10` stride is deliberately NOT a multiple of the tokenizer's 8px patch (anti-overfit, D-038).
- **Revealed frames contain black-line pixels; the occluded frame is flat gray (CURTAIN_COLOR), zero
  black pixels.** (This is what readout.py's `is_occluded` keys on — keep it true.)
- bg colour != square colour. `hidden_state()` is measurement-only — NEVER fed to a model.
- CURTAIN_COLOR (128,128,128) is distinct from black and from all 4 palette colours.
