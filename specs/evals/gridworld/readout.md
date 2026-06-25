# readout.py — read GridWorld full env state (square cell + colours) out of one frame, exactly.

Closed-form, needs no ground truth, so it works identically on predicted and true frames. This is the
point of the discrete env: the readout is exact and this allows predictions to be accurately compared against ground truth state.

## Interface
- `cell_mean_colors(frame) -> (GRID_N, GRID_N, 3)` : mean interior colour of every cell (row, col).
- `nearest_palette_idx(color) -> int` : index (PALETTE order) of the closest of the 4 palette colours.
- `read_square(frame) -> dict{col,row,color_idx,bg_idx,margin,is_occluded}` : the readout.

## Behavior
- Background colour = the **median** cell colour (robust to the 1 outlier square among 36 cells).
- Square cell = the cell whose interior colour is **farthest** from that background.
- square/bg colour = nearest of the 4 PALETTE colours to the square-cell / background colours.
- `margin` = (top1 − top2) distance-from-bg across cells = confidence (crisp = large, smeared = small).
- `is_occluded` = No black pixels = There exists no pixel on screen that has R<25 AND B<25 AND G<25 (because a revealed frame has black lines)

## Invariants
- All colours are **BGR** (env channel-order contract) — do not RGB-flip.
- Geometry comes from `envs.gridworld` (`GRID_N`, `VIS`, `cell_origin`) — read cell interiors via
  `cell_origin(i)` for x0/y0 and a `VIS`-sized window; never hard-code pixel offsets here.
- Pure numpy, no torch, no ground-truth input. On a true frame it recovers exact (col,row,colour).
