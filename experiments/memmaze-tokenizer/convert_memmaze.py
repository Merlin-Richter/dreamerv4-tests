#!/usr/bin/env python3
"""Convert Memory Maze per-trajectory .npz files into one mmappable .npy for train_tokenizer.py.

The 9x9 offline dataset unzips to many per-trajectory .npz files, each holding `image` (T, 64, 64, 3)
uint8 plus labels we ignore here (the tokenizer only autoencodes frames). This walks every .npz under
--raw, stacks the `image` arrays into a preallocated np.lib.format.open_memmap of shape (N, T, H, W, 3)
uint8 written row-by-row, so RAM stays flat regardless of N. Channels are kept AS-IS (RGB, untouched).

Output is what train_tokenizer.py mmaps via np.load(..., mmap_mode='r'); it does its own train/val split.

Run with -u. Example:
    python -u experiments/memmaze-tokenizer/convert_memmaze.py \
        --raw data/memmaze9x9_raw --out data/memmaze9x9.npy
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True,
                    help="Dir holding the unzipped per-trajectory .npz files (searched recursively).")
    ap.add_argument("--out", type=Path, required=True, help="Output single .npy (N, T, H, W, 3) uint8.")
    ap.add_argument("--key", type=str, default="image", help="NPZ array key holding the frames.")
    ap.add_argument("--limit", type=int, default=None, help="Only convert the first N trajectories (testing).")
    args = ap.parse_args()

    files = sorted(args.raw.rglob("*.npz"))
    if not files:
        sys.exit(f"No .npz files under {args.raw}")
    if args.limit is not None:
        files = files[: args.limit]
    n = len(files)

    with np.load(files[0]) as z:
        if args.key not in z:
            sys.exit(f"{files[0]} has no '{args.key}' (keys: {list(z.keys())})")
        shape0, dtype0 = z[args.key].shape, z[args.key].dtype
    if len(shape0) != 4 or shape0[-1] != 3:
        sys.exit(f"Unexpected image shape {shape0} in {files[0]} (want (T, H, W, 3)).")
    t, h, w, c = shape0
    gb = n * t * h * w * c / 1e9
    print(f"{n} trajectories | per-traj '{args.key}' {shape0} {dtype0} "
          f"-> out {(n, t, h, w, c)} uint8 ({gb:.1f} GB) at {args.out}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.uint8, shape=(n, t, h, w, c))

    t0 = time.time()
    for i, f in enumerate(files):
        with np.load(f) as z:
            img = z[args.key]
            if img.shape != (t, h, w, c):
                sys.exit(f"Shape mismatch at {f}: {img.shape} != {(t, h, w, c)} "
                         f"(non-uniform trajectory lengths break a single rectangular .npy).")
            out[i] = img if img.dtype == np.uint8 else img.astype(np.uint8)
        if (i + 1) % 200 == 0 or i + 1 == n:
            dt = time.time() - t0
            print(f"  {i + 1}/{n}  ({(i + 1) / max(dt, 1e-6):.1f} traj/s, {dt:.0f}s elapsed)", flush=True)
    out.flush()
    del out
    print(f"DONE -> {args.out} ({n} trajectories, {t} frames each)", flush=True)


if __name__ == "__main__":
    main()
