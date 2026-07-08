"""Teacher-forced 1-step in-window probe (sym tier).

Real context grids[0:t0] committed via adapter.begin (the eval inference path), then ONE
rollout_step predicts grid[t0]; per-cell accuracy split by what predicting the cell needs:

  shift    cell was visible in the PREVIOUS frame (t0-1)      -> pure copy/shift, no memory
  window   not visible at t0-1 but seen within the last W=16  -> in-window attention
  past     seen only > W ticks ago                            -> carried memory
  unseen   never seen in context                              -> chance floor (0.2)

OUT-of-map cells are excluded (statically predictable). A model that merely learned to
shift the previous frame scores 1.0 / chance / chance / chance.

Run: venv/Scripts/python.exe -u experiments/colorfield-symprobe/probe_inwindow.py \
       --checkpoint experiments/colorfield-symprobe/dynamics_sym.pt
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from autoresearch.editable.adapter_sym import make_adapter          # noqa: E402
from autoresearch.editable.train_sym import load_split, render_grid  # noqa: E402
from autoresearch.frozen_sym.env import BOARD as BOARD_N                      # noqa: E402

W = 16
VIEW_R = 2  # 5x5 viewport half-extent


def cell_class(r, c, t0, positions):
    """Classify world cell (r,c) by when it was last visible before t0."""
    pr, pc = positions[t0 - 1]
    if abs(r - pr) <= VIEW_R and abs(c - pc) <= VIEW_R:
        return "shift"
    for t in range(t0 - 2, -1, -1):
        pr, pc = positions[t]
        if abs(r - pr) <= VIEW_R and abs(c - pc) <= VIEW_R:
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

    hits = defaultdict(int)
    tot = defaultdict(int)
    for ep in range(args.episodes):
        m, pos, act = maps[ep], positions[ep], actions[ep]
        grids = np.stack([render_grid(m, pos[t]) for t in range(max(args.cuts) + 1)])
        for t0 in args.cuts:
            adapter = factory(None)
            adapter.begin(grids[:t0], act[:t0].astype(np.int64))
            pred = adapter.step(int(act[t0]))                 # (5,5) ids 0..5
            gt = grids[t0]
            pr, pc = pos[t0]
            for dr in range(-VIEW_R, VIEW_R + 1):
                for dc in range(-VIEW_R, VIEW_R + 1):
                    r, c = pr + dr, pc + dc
                    if not (0 <= r < BOARD_N and 0 <= c < BOARD_N):
                        continue                              # OUT cell: excluded
                    k = cell_class(r, c, t0, pos)
                    tot[k] += 1
                    hits[k] += int(pred[dr + VIEW_R, dc + VIEW_R] == gt[dr + VIEW_R, dc + VIEW_R])

    print(f"ckpt={args.checkpoint}  episodes={args.episodes} cuts={args.cuts}  (chance 0.2)")
    for k in ["shift", "window", "past", "unseen"]:
        n = tot[k]
        print(f"  {k:7s} acc {hits[k] / n:.3f}  (n={n})" if n else f"  {k:7s} n=0")


if __name__ == "__main__":
    main()
