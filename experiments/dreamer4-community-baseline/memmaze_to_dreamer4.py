#!/usr/bin/env python3
"""Convert raw Memory-Maze .npz trajectories into the community Dreamer4 repo's data format.

Target repo: github.com/nicklashansen/dreamer4 (see the task file for the pinned commit). That repo's
training reads TWO paired directory trees per task (task name must be in its `task_set.TASK_SET`):

  frame shards (for the tokenizer AND as the dynamics frame source), consumed by ShardedFrameDataset:
      <frames_root>/<task>/<task>_shard0000.pt , ...   each = {"frames": (S, 3, H, W) uint8}   (CHW!)

  demo file (per-frame action/reward/episode, for --use_actions dynamics), consumed by WMDataset:
      <demos_root>/<task>.pt = {"episode": (N,) int64, "action": (N, A) float32, "reward": (N,) float32}

The demo arrays are indexed 1:1 with the GLOBAL frame order across ALL trajectories (the exact order the
frames are written into shards). This script builds BOTH in lockstep so they stay aligned.

Memory-Maze raw .npz (memory-maze README): keys `image (1001,64,64,3) uint8 RGB`, `action (1001,6) binary
one-hot` = "last action" (the action that PRODUCED that frame; action[0] is a dummy no-op before the first
obs), `reward (1001,) float`. Discrete(6) = (noop, forward, left, right, forward_left, forward_right).

Action alignment: the community WMDataset ultimately pairs frame[g] with demo_action[g], and its train
loop comment intends demo_action[g] = "the action that produced obs[g]". Memory-Maze's raw action[g] is
already exactly that ("last action" arriving at image[g]), so NO time-shift is applied by default. The
one-hot is placed in the first 6 of A=16 columns (the repo hardcodes ActionEncoder action_dim=16;
columns 6..15 stay 0). Confirm empirically on a smoke run via the repo's `stats/action_shuffle_loss_ratio`
(should climb clearly above 1); `--action-shift` is provided only to A/B that if ever needed.

Channels stay RGB end-to-end (the community repo is RGB-native, torchvision read_image order). Frames are
kept at native 64x64 by default (--target-size to bilinear-resize, e.g. 128 to match the repo default).

Run with -u. Example (on the cluster, after download_memmaze.py --unzip):
    python -u memmaze_to_dreamer4.py --raw data/memmaze9x9_raw/train --out-dir data/d4_memmaze/train \
        --task memmaze --shard-size 2048
    python -u memmaze_to_dreamer4.py --raw data/memmaze9x9_raw/eval  --out-dir data/d4_memmaze/eval \
        --task memmaze --shard-size 2048
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ACTION_DIM = 16   # the community repo hardcodes ActionEncoder(action_dim=16); one-hot lives in [:n_act]


def _atomic_torch_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _hash_array(hasher, array: np.ndarray) -> None:
    """Hash an array without materializing a second full-size ``bytes`` object."""
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    hasher.update(memoryview(array).cast("B"))


def _load_traj(npz_path: Path):
    """Return raw arrays and a content fingerprint without duplicating the frames."""
    with np.load(npz_path) as z:
        img = np.asarray(z["image"])                      # (T,64,64,3) uint8 RGB
        act = np.asarray(z["action"])                     # (T,6) one-hot OR (T,) int
        rew = np.asarray(z["reward"]).astype(np.float32) if "reward" in z else None
    content_hash = hashlib.sha256()
    _hash_array(content_hash, img)
    _hash_array(content_hash, act)
    if rew is not None:
        _hash_array(content_hash, rew)
    return img, act, rew, content_hash.hexdigest()


def _trajectory_length(npz_path: Path) -> int:
    """Read only the small action member during the allocation pre-pass."""
    with np.load(npz_path) as z:
        return int(z["action"].shape[0])


def _peak_rss_gib() -> float | None:
    """Return process peak RSS on the Linux cluster without adding a dependency."""
    if not sys.platform.startswith("linux"):
        return None
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def main():
    ap = argparse.ArgumentParser(description="Memory-Maze .npz -> community Dreamer4 shards + demo.")
    ap.add_argument("--raw", type=Path, required=True, help="Dir with per-trajectory .npz (recursive).")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output root (creates shards/ and demos/).")
    ap.add_argument("--task", type=str, default="memmaze", help="Task name (MUST be in the repo TASK_SET).")
    ap.add_argument("--shard-size", type=int, default=2048, help="Frames per shard (repo default 2048).")
    ap.add_argument("--target-size", type=int, default=None,
                    help="Bilinear-resize frames to this square size (default: keep native 64).")
    ap.add_argument("--action-shift", type=int, default=0, choices=[-1, 0, 1],
                    help="Shift demo_action vs frames (default 0 = raw MM alignment). See module docstring.")
    ap.add_argument("--limit", type=int, default=None, help="First N trajectories only (smoke tests).")
    args = ap.parse_args()

    files = sorted(args.raw.rglob("*.npz"))
    if not files:
        sys.exit(f"No .npz under {args.raw}")
    if args.limit is not None:
        files = files[: args.limit]
    n = len(files)

    shards_dir = args.out_dir / "shards" / args.task
    demos_dir = args.out_dir / "demos"
    manifest_path = args.out_dir / "conversion_manifest.json"
    existing = list(shards_dir.glob("*.pt")) if shards_dir.is_dir() else []
    if existing or (demos_dir / f"{args.task}.pt").exists() or manifest_path.exists():
        sys.exit(
            f"Refusing to mix with an existing conversion under {args.out_dir}. "
            "Use a fresh --out-dir (or deliberately remove the incomplete conversion first)."
        )
    shards_dir.mkdir(parents=True, exist_ok=True)
    demos_dir.mkdir(parents=True, exist_ok=True)

    # Preallocate the final demo arrays. The old converter kept thousands of tensors in lists and used
    # repeated torch.cat calls for frames; allocator-retained temporaries made RSS grow by roughly one
    # full trajectory per iteration. This pre-pass touches only the small action member in each archive.
    lengths = [_trajectory_length(f) for f in files]
    total_frames = sum(lengths)
    episode = torch.empty((total_frames,), dtype=torch.int64)
    action = torch.zeros((total_frames, ACTION_DIM), dtype=torch.float32)
    reward = torch.empty((total_frames,), dtype=torch.float32)

    shard_buf: torch.Tensor | None = None
    shard_fill = 0
    shard_idx = 0
    demo_pos = 0
    n_act_seen = set()
    content_hashes = []
    t0 = time.time()

    def flush_shard(frames: torch.Tensor) -> None:
        nonlocal shard_idx
        out = shards_dir / f"{args.task}_shard{shard_idx:04d}.pt"
        _atomic_torch_save({"frames": frames}, out)
        shard_idx += 1

    for i, f in enumerate(files):
        img, act, rew, content_hash = _load_traj(f)
        if img.dtype != np.uint8 or img.ndim != 4 or img.shape[-1] != 3:
            raise ValueError(f"{f}: expected image (T,H,W,3) uint8, got {img.shape} {img.dtype}")
        T = int(img.shape[0])
        if T != lengths[i] or act.shape[0] != T or (rew is not None and rew.shape != (T,)):
            raise ValueError(f"{f}: inconsistent time dimensions image={T}, action={act.shape}, reward="
                             f"{None if rew is None else rew.shape}")

        if act.ndim == 2:                                 # one-hot (T, n_act)
            n_act = int(act.shape[1])
            idx = act.argmax(axis=1).astype(np.int64, copy=False)
        elif act.ndim == 1:                               # int (T,)
            n_act = int(act.max()) + 1
            idx = act.astype(np.int64, copy=False)
        else:
            raise ValueError(f"{f}: expected action (T,A) or (T,), got {act.shape}")
        if n_act > ACTION_DIM or idx.min() < 0 or idx.max() >= ACTION_DIM:
            raise ValueError(f"{f}: action indices do not fit stored dimension {ACTION_DIM}")

        rew_values = rew if rew is not None else np.zeros((T,), dtype=np.float32)
        if args.action_shift != 0:                        # optional A/B only; default keeps raw alignment
            idx = np.roll(idx, args.action_shift)
            rew_values = np.roll(rew_values, args.action_shift)

        dst = slice(demo_pos, demo_pos + T)
        episode[dst] = i
        rows = torch.arange(demo_pos, demo_pos + T)
        action[rows, torch.from_numpy(idx)] = 1.0
        reward[dst] = torch.from_numpy(np.asarray(rew_values, dtype=np.float32))
        demo_pos += T

        n_act_seen.add(n_act)
        content_hashes.append(content_hash)

        out_h = args.target_size or int(img.shape[1])
        out_w = args.target_size or int(img.shape[2])
        if shard_buf is None:
            shard_buf = torch.empty((args.shard_size, 3, out_h, out_w), dtype=torch.uint8)
        elif tuple(shard_buf.shape[2:]) != (out_h, out_w):
            raise ValueError(f"{f}: frame size changed to {(out_h, out_w)}")

        src_pos = 0
        while src_pos < T:
            take = min(args.shard_size - shard_fill, T - src_pos)
            src = torch.from_numpy(img[src_pos:src_pos + take]).permute(0, 3, 1, 2)
            if tuple(src.shape[2:]) != (out_h, out_w):
                resized = F.interpolate(src.float(), size=(out_h, out_w), mode="bilinear",
                                        align_corners=False)
                src = resized.clamp_(0, 255).to(torch.uint8)
            shard_buf[shard_fill:shard_fill + take].copy_(src)
            shard_fill += take
            src_pos += take
            if shard_fill == args.shard_size:
                flush_shard(shard_buf)
                shard_fill = 0

        if (i + 1) % 200 == 0 or i + 1 == n:
            peak_rss = _peak_rss_gib()
            rss_text = "" if peak_rss is None else f", peak_rss={peak_rss:.2f} GiB"
            print(f"  {i + 1}/{n} traj  ({(i + 1) / max(time.time() - t0, 1e-6):.1f}/s, "
                  f"{shard_idx} shards{rss_text})", flush=True)

    if shard_fill > 0:                                    # clone: do not serialize unused backing storage
        assert shard_buf is not None
        flush_shard(shard_buf[:shard_fill].clone())

    assert demo_pos == total_frames
    N = total_frames
    _atomic_torch_save(
        {"episode": episode, "action": action, "reward": reward},
        demos_dir / f"{args.task}.pt",
    )

    rel_names = [str(f.relative_to(args.raw)).replace("\\", "/") for f in files]
    names_sha256 = hashlib.sha256("\n".join(rel_names).encode("utf-8")).hexdigest()
    manifest = {
        "format": "dreamer4-community-memmaze-v2-bounded-memory",
        "task": args.task,
        "raw_root": str(args.raw.resolve()),
        "trajectory_count": n,
        "frame_count": N,
        "trajectory_names_sha256": names_sha256,
        "trajectory_names": rel_names,
        "trajectory_content_sha256": content_hashes,
        "shard_count": shard_idx,
        "shard_size": args.shard_size,
        "target_size": args.target_size,
        "action_shift": args.action_shift,
        "action_dim_source": sorted(n_act_seen),
        "action_dim_stored": ACTION_DIM,
        "action_convention": "raw action[t] produced raw image[t]; no shift by default",
        "channel_order": "RGB",
    }
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_tmp.replace(manifest_path)

    if len(n_act_seen) != 1:
        print(f"  WARNING: inconsistent n_actions across trajectories: {sorted(n_act_seen)}", flush=True)
    print(f"DONE task={args.task}: {n} trajectories, N={N} frames, {shard_idx} shards "
          f"(shard_size={args.shard_size}), n_actions={sorted(n_act_seen)}", flush=True)
    print(f"  frames -> {shards_dir}", flush=True)
    print(f"  demo   -> {demos_dir / (args.task + '.pt')}  "
          f"(episode/action(A={ACTION_DIM})/reward, N={N})", flush=True)


if __name__ == "__main__":
    main()
