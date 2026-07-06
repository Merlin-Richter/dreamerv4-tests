"""Qualitative rollout sheets for ColorField (driver tooling — illustrates, never
decides; the frozen comeback eval decides).

One deterministic scripted episode per sheet: a SNAKE prefix (192 real frames,
teacher-forced into the model via the adapter) covering a patch of the map, then a
REVISIT imagination leg (96 steps) that walks back through seen territory — the
directest possible eyeball test of "does it repaint what it saw".

Sheet layout: sampled columns, TOP row = ground truth (env stepped with the same
actions), BOTTOM row = model imagination. Prefix columns (green border) show the
real context tail. Prints per-sample on-screen cell accuracy vs GT (the readout,
same as the eval uses) so the eyeball has numbers next to it.

Usage:
  venv/Scripts/python.exe -u -m autoresearch.driver.sheets \
    --checkpoint autoresearch/runs/cal20/dynamics.pt --out autoresearch/runs/cal20
"""

import argparse
import os

import cv2
import numpy as np

from ..editable.adapter import make_adapter
from ..frozen.env import (DOWN, LEFT, RIGHT, STAY, UP, ColorFieldEnv)
from ..frozen.readout import read_cells

SCALE = 3
PREFIX_TAIL_COLS = 3   # how many prefix (real) columns to show
SAMPLE_EVERY = 6       # imagination sampling stride


def script_actions():
    """Prefix (192 frames incl. leading STAY) + imagination (96). Start (60,12).
    Prefix snake: R40 U24 L40 D24 R40 D23 -> ends (83,52), all in-lattice.
    Imagination revisit: L40 U24 R32 -> re-crosses the prefix band, ages ~60-190."""
    prefix = [STAY] + [RIGHT] * 40 + [UP] * 24 + [LEFT] * 40 + [DOWN] * 24 \
        + [RIGHT] * 40 + [DOWN] * 23
    imag = [LEFT] * 40 + [UP] * 24 + [RIGHT] * 32
    return prefix, imag


def cell_acc(img_frame, gt_frame, pos):
    """On-screen cell agreement between imagined and GT frame (nearest-palette)."""
    ri = read_cells(img_frame, pos)
    rg = read_cells(gt_frame, pos)
    keys = [k for k, r in rg.items() if r.on_screen]
    if not keys:
        return None
    return sum(ri[k].color == rg[k].color for k in keys) / len(keys)


def _tile(frame_rgb, border=None):
    img = cv2.resize(frame_rgb, None, fx=SCALE, fy=SCALE,
                     interpolation=cv2.INTER_NEAREST)
    if border is not None:
        img = cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=border)
    else:
        img = cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    return img


def make_sheet(ckpt, tokenizer, out_dir, map_seed=5, device="cuda"):
    os.makedirs(out_dir, exist_ok=True)
    prefix_actions, imag_actions = script_actions()

    env = ColorFieldEnv()
    frame = env.reset(seed=map_seed, start=(60, 12))
    prefix_frames = [frame]
    for a in prefix_actions[1:]:
        prefix_frames.append(env.step(a))
    gt_frames, gt_pos = [], []
    for a in imag_actions:
        gt_frames.append(env.step(a))
        gt_pos.append(env.pos)

    factory = make_adapter(str(ckpt), str(tokenizer), device)
    adapter = factory(None)
    adapter.begin(np.stack(prefix_frames), np.asarray(prefix_actions, dtype=np.int64))
    img_frames = [adapter.step(int(a)) for a in imag_actions]

    # columns: prefix tail (real, green) + sampled imagination
    cols_top, cols_bot, labels, accs = [], [], [], []
    for j in range(PREFIX_TAIL_COLS):
        t = len(prefix_frames) - PREFIX_TAIL_COLS + j
        cols_top.append(_tile(prefix_frames[t], border=(0, 200, 0)))
        cols_bot.append(_tile(prefix_frames[t], border=(0, 200, 0)))
        labels.append(f"ctx {t}")
        accs.append(None)
    for i in range(SAMPLE_EVERY - 1, len(imag_actions), SAMPLE_EVERY):
        cols_top.append(_tile(gt_frames[i]))
        cols_bot.append(_tile(img_frames[i]))
        labels.append(f"+{i + 1}")
        acc = cell_acc(img_frames[i], gt_frames[i], gt_pos[i])
        accs.append(acc)

    tile_h = cols_top[0].shape[0]
    tile_w = cols_top[0].shape[1]
    header = 26
    sheet = np.full((header + 2 * tile_h + 24, tile_w * len(cols_top), 3), 15, np.uint8)
    for c, (ct, cb, lab, acc) in enumerate(zip(cols_top, cols_bot, labels, accs)):
        x = c * tile_w
        sheet[header:header + tile_h, x:x + tile_w] = ct
        sheet[header + tile_h:header + 2 * tile_h, x:x + tile_w] = cb
        cv2.putText(sheet, lab, (x + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (230, 230, 230), 1, cv2.LINE_AA)
        if acc is not None:
            cv2.putText(sheet, f"{acc:.2f}", (x + 6, header + 2 * tile_h + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1, cv2.LINE_AA)
    cv2.putText(sheet, "TOP: ground truth | BOTTOM: imagination (green = real context tail)"
                       " | numbers = on-screen cell acc",
                (max(0, sheet.shape[1] - 980), sheet.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)

    out_path = os.path.join(out_dir, f"sheet_revisit_seed{map_seed}.png")
    cv2.imwrite(out_path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    scored = [a for a in accs if a is not None]
    print(f"[sheet] {out_path}")
    print(f"[sheet] mean on-screen cell acc over {len(scored)} sampled imag frames: "
          f"{np.mean(scored):.3f} (first {scored[0]:.3f} last {scored[-1]:.3f})")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="checkpoints/colorfield/tokenizer.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--map-seeds", type=int, nargs="+", default=[5, 6])
    args = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for s in args.map_seeds:
        make_sheet(args.checkpoint, args.tokenizer, args.out, map_seed=s, device=device)


if __name__ == "__main__":
    main()
