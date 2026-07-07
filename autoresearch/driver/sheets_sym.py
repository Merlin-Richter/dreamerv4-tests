"""Qualitative rollout sheets for ColorField-SYM (driver tooling — illustrates,
never decides). Mirrors driver/sheets.py: snake prefix (teacher-forced real grids)
then a revisit imagination leg back through seen territory. TOP = ground truth,
BOTTOM = imagination; per-column full-viewport cell accuracy (25 cells, chance 0.2).
Actions are per-tick with phase-5 discipline (moves only at phase 0, STAY forced
off-phase); columns sample every effective move (5 ticks).

Usage:
  venv/Scripts/python.exe -u -m autoresearch.driver.sheets_sym \
    --checkpoint autoresearch/runs/sym20/dynamics.pt --out autoresearch/runs/sym20
"""

import argparse
import os

import cv2
import numpy as np

from ..editable.adapter_sym import make_adapter
from ..frozen.env import PALETTE
from ..frozen_sym.env import (DOWN, LEFT, RIGHT, STAY, UP, ColorFieldSymEnv)

CELL_PX = 24


def _ticks(moves):
    """Effective-move list -> per-tick actions (move at phase 0, STAY off-phase)."""
    out = []
    for m in moves:
        out += [m] + [STAY] * 4
    return out


def script_actions():
    """Start (11,2). Prefix snake R8 U5 L8 D5 R8 D3 (37 moves); revisit L8 U5 R6
    (19 moves) — recrosses the band seen 90-180 ticks earlier."""
    prefix_moves = [RIGHT] * 8 + [UP] * 5 + [LEFT] * 8 + [DOWN] * 5 + [RIGHT] * 8 + [DOWN] * 3
    imag_moves = [LEFT] * 8 + [UP] * 5 + [RIGHT] * 6
    return [STAY] + _ticks(prefix_moves), _ticks(imag_moves)


def grid_img(grid, border):
    img = PALETTE[np.asarray(grid, dtype=np.int64)]
    img = np.repeat(np.repeat(img, CELL_PX, 0), CELL_PX, 1)
    return cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=border)


def make_sheet(ckpt, out_dir, map_seed=5, device="cuda"):
    os.makedirs(out_dir, exist_ok=True)
    prefix_actions, imag_actions = script_actions()

    env = ColorFieldSymEnv()
    grid, _ = env.reset(seed=map_seed, start=(11, 2))
    prefix_grids = [grid]
    for a in prefix_actions[1:]:
        g, _ = env.step(a)
        prefix_grids.append(g)
    gt = []
    for a in imag_actions:
        g, _ = env.step(a)
        gt.append(g)

    adapter = make_adapter(str(ckpt), device)(None)
    adapter.begin(np.stack(prefix_grids), np.asarray(prefix_actions, dtype=np.int64))
    img = [adapter.step(int(a)) for a in imag_actions]

    cols_t, cols_b, labels, accs = [], [], [], []
    for j in (len(prefix_grids) - 2, len(prefix_grids) - 1):
        cols_t.append(grid_img(prefix_grids[j], (0, 200, 0)))
        cols_b.append(grid_img(prefix_grids[j], (0, 200, 0)))
        labels.append(f"ctx {j}")
        accs.append(None)
    for i in range(4, len(imag_actions), 5):          # each effective move
        cols_t.append(grid_img(gt[i], (30, 30, 30)))
        cols_b.append(grid_img(img[i], (30, 30, 30)))
        labels.append(f"+{i + 1}")
        accs.append(float((np.asarray(img[i]) == np.asarray(gt[i])).mean()))

    th, tw = cols_t[0].shape[:2]
    header = 24
    sheet = np.full((header + 2 * th + 22, tw * len(cols_t), 3), 15, np.uint8)
    for c, (ct, cb, lab, acc) in enumerate(zip(cols_t, cols_b, labels, accs)):
        x = c * tw
        sheet[header:header + th, x:x + tw] = ct
        sheet[header + th:header + 2 * th, x:x + tw] = cb
        cv2.putText(sheet, lab, (x + 5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (230, 230, 230), 1, cv2.LINE_AA)
        if acc is not None:
            cv2.putText(sheet, f"{acc:.2f}", (x + 5, header + 2 * th + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 255), 1, cv2.LINE_AA)
    out_path = os.path.join(out_dir, f"sheet_sym_revisit_seed{map_seed}.png")
    cv2.imwrite(out_path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    scored = [a for a in accs if a is not None]
    print(f"[sheet] {out_path}")
    print(f"[sheet] viewport cell acc per effective move: first {scored[0]:.3f} "
          f"mean {np.mean(scored):.3f} last {scored[-1]:.3f}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--map-seeds", type=int, nargs="+", default=[5, 6])
    args = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for s in args.map_seeds:
        make_sheet(args.checkpoint, args.out, map_seed=s, device=device)


if __name__ == "__main__":
    main()
