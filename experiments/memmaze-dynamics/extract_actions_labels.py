#!/usr/bin/env python3
"""Extract actions + eval labels from the raw Memory Maze .npz trajectories.

The tokenizer converter (experiments/memmaze-tokenizer/convert_memmaze.py) kept only 'image'.
Dynamics training needs per-frame actions, and the future memmaze recall/probe eval needs the
privileged labels (agent_pos, maze_layout, target_*...). This walks the SAME sorted rglob order as
the converter (so episode indices align with data/memmaze9x9.npy) and writes one array per key:

  - 'action' -> <frames_stem>_actions.npy (N, T) int64 (argmax'd if stored one-hot) — the exact
    sidecar name train_dynamics.py auto-detects.
  - every other non-image key with a uniform per-episode shape -> <frames_stem>_<key>.npy (N, ...).
    Non-uniform keys are reported and skipped (can be revisited when the eval needs them).

Run with -u. Example (cluster):
    python -u experiments/memmaze-dynamics/extract_actions_labels.py \
        --raw data/memmaze9x9_raw --frames data/memmaze9x9.npy
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True, help="Dir with per-trajectory .npz (recursive).")
    ap.add_argument("--frames", type=Path, required=True,
                    help="The converted frames .npy these labels must align with (name + N,T checks).")
    ap.add_argument("--limit", type=int, default=None, help="First N trajectories only (testing).")
    args = ap.parse_args()

    files = sorted(args.raw.rglob("*.npz"))
    if not files:
        sys.exit(f"No .npz files under {args.raw}")
    if args.limit is not None:
        files = files[: args.limit]
    n = len(files)

    frames = np.load(args.frames, mmap_mode="r")
    if frames.shape[0] != n:
        sys.exit(f"Episode count mismatch: {n} npz files vs frames N={frames.shape[0]} — "
                 f"raw dir and frames npy are out of sync, refusing to write misaligned labels.")
    t_frames = frames.shape[1]

    with np.load(files[0]) as z:
        keys = sorted(z.keys())
        print(f"{n} trajectories | npz keys: "
              + ", ".join(f"{k}{z[k].shape}:{z[k].dtype}" for k in keys), flush=True)

    # Plan outputs: probe episode 0 for shapes.
    outs = {}       # key -> np.memmap
    skipped = []
    with np.load(files[0]) as z:
        for k in keys:
            if k == "image":
                continue
            arr = z[k]
            if k == "action":
                # per-frame action; accept (T,) ints or (T, A) one-hot -> argmax int64
                if arr.ndim == 2:
                    shape, dtype = (n, arr.shape[0]), np.int64
                elif arr.ndim == 1:
                    shape, dtype = (n, arr.shape[0]), np.int64
                else:
                    sys.exit(f"Unexpected 'action' shape {arr.shape}")
                if arr.shape[0] != t_frames:
                    print(f"  NOTE: action T={arr.shape[0]} != frames T={t_frames} "
                          f"(will store as-is; alignment handled downstream)", flush=True)
                    shape = (n, arr.shape[0])
                path = args.frames.with_name(args.frames.stem + "_actions.npy")
            else:
                shape, dtype = (n,) + arr.shape, arr.dtype
                path = args.frames.with_name(f"{args.frames.stem}_{k}.npy")
            outs[k] = (np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape), path)

    t0 = time.time()
    for i, f in enumerate(files):
        with np.load(f) as z:
            for k, (mm, _) in outs.items():
                arr = z[k]
                if k == "action" and arr.ndim == 2:
                    arr = arr.argmax(axis=-1)
                if mm[i].shape != arr.shape:
                    sys.exit(f"Non-uniform shape for '{k}' at {f}: {arr.shape} != {mm[i].shape}")
                mm[i] = arr
        if (i + 1) % 200 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n}  ({(i + 1) / max(time.time() - t0, 1e-6):.1f} traj/s)", flush=True)

    for k, (mm, path) in outs.items():
        mm.flush()
        print(f"  '{k}' -> {path}  {mm.shape} {mm.dtype}", flush=True)
    if "action" in outs:
        acts = outs["action"][0]
        print(f"  action stats: min {acts[:].min()} max {acts[:].max()} "
              f"(n_actions = {int(acts[:].max()) + 1})", flush=True)
    if skipped:
        print(f"  skipped non-uniform keys: {skipped}", flush=True)
    print("EXTRACT DONE", flush=True)


if __name__ == "__main__":
    main()
