"""Teacher-forced 1-step in-window probe for the loop (NOT agent-editable).

Real context grids[0:t0] committed via adapter.begin (the eval inference path), then ONE
rollout_step predicts grid[t0]; per-cell accuracy split by what predicting the cell needs:
  shift  = visible in the previous frame (pure copy; floor for "learned dynamics" = ~1.0)
  window = not visible at t0-1 but seen within the last W=16 ticks
  past   = seen only beyond the window (carried memory)
  unseen = never seen (chance floor 0.2)
OUT-of-map cells excluded. Prints grep-able `inwindow_*:` lines.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from autoresearch.editable.adapter_sym import make_adapter            # noqa: E402
from autoresearch.editable.train_sym import load_split, render_grid   # noqa: E402
from autoresearch.frozen_sym.env import BOARD, VIEW_HALF              # noqa: E402

W = 16


def cell_class(r, c, t0, positions):
    pr, pc = positions[t0 - 1]
    if abs(r - pr) <= VIEW_HALF and abs(c - pc) <= VIEW_HALF:
        return "shift"
    for t in range(t0 - 2, -1, -1):
        pr, pc = positions[t]
        if abs(r - pr) <= VIEW_HALF and abs(c - pc) <= VIEW_HALF:
            return "window" if t >= t0 - W else "past"
    return "unseen"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val", default="data/colorfield_sym_val")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--cuts", type=int, nargs="+", default=[50, 100, 150, 190])
    args = ap.parse_args()

    maps, positions, actions = load_split(Path(args.val))
    actions = actions.numpy() if hasattr(actions, "numpy") else np.asarray(actions)
    factory = make_adapter(args.checkpoint)

    hits, tot = defaultdict(int), defaultdict(int)
    for ep in range(args.episodes):
        m, pos, act = maps[ep], positions[ep], actions[ep]
        grids = np.stack([render_grid(m, pos[t]) for t in range(max(args.cuts) + 1)])
        for t0 in args.cuts:
            adapter = factory(None)
            adapter.begin(grids[:t0], act[:t0].astype(np.int64))
            pred = adapter.step(int(act[t0]))
            gt = grids[t0]
            pr, pc = pos[t0]
            for dr in range(-VIEW_HALF, VIEW_HALF + 1):
                for dc in range(-VIEW_HALF, VIEW_HALF + 1):
                    r, c = pr + dr, pc + dc
                    if not (0 <= r < BOARD and 0 <= c < BOARD):
                        continue
                    k = cell_class(r, c, t0, pos)
                    tot[k] += 1
                    hits[k] += int(pred[dr + VIEW_HALF, dc + VIEW_HALF]
                                   == gt[dr + VIEW_HALF, dc + VIEW_HALF])

    for k in ["shift", "window", "past", "unseen"]:
        acc = hits[k] / tot[k] if tot[k] else float("nan")
        print(f"inwindow_{k}:   {acc:.4f}  (n={tot[k]})")


if __name__ == "__main__":
    main()
