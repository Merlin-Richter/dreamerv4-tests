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
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ACTION_DIM = 16   # the community repo hardcodes ActionEncoder(action_dim=16); one-hot lives in [:n_act]


def _load_traj(npz_path: Path, target_size: int | None):
    """Return (frames_u8 (T,3,H,W) uint8 RGB, action_onehot (T,ACTION_DIM) f32, reward (T,) f32, n_act)."""
    with np.load(npz_path) as z:
        img = np.asarray(z["image"])                      # (T,64,64,3) uint8 RGB
        act = np.asarray(z["action"])                     # (T,6) one-hot OR (T,) int
        rew = np.asarray(z["reward"]).astype(np.float32) if "reward" in z else None
    T = img.shape[0]
    frames = torch.from_numpy(img).permute(0, 3, 1, 2).contiguous()  # (T,3,H,W) uint8
    if target_size is not None and frames.shape[-1] != target_size:
        f = frames.float() / 255.0
        f = F.interpolate(f, size=(target_size, target_size), mode="bilinear", align_corners=False)
        frames = (f.clamp(0, 1) * 255.0).to(torch.uint8)

    if act.ndim == 2:                                     # one-hot (T, n_act)
        n_act = act.shape[1]
        idx = act.argmax(axis=1)
    else:                                                 # int (T,)
        n_act = int(act.max()) + 1
        idx = act.astype(np.int64)
    onehot = np.zeros((T, ACTION_DIM), dtype=np.float32)
    onehot[np.arange(T), idx] = 1.0
    reward = rew if rew is not None else np.zeros((T,), dtype=np.float32)
    return frames, torch.from_numpy(onehot), torch.from_numpy(reward.astype(np.float32)), n_act


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
    shards_dir.mkdir(parents=True, exist_ok=True)
    demos_dir.mkdir(parents=True, exist_ok=True)

    buf: list[torch.Tensor] = []          # rolling frame buffer (each (t,3,H,W))
    buf_n = 0
    shard_idx = 0
    ep_all, act_all, rew_all = [], [], []
    n_act_seen = set()
    t0 = time.time()

    def flush_full_shards():
        nonlocal buf, buf_n, shard_idx
        while buf_n >= args.shard_size:
            concat = torch.cat(buf, dim=0)
            to_save, remainder = concat[: args.shard_size], concat[args.shard_size:]
            out = shards_dir / f"{args.task}_shard{shard_idx:04d}.pt"
            torch.save({"frames": to_save.contiguous()}, out)
            buf = [remainder] if remainder.shape[0] > 0 else []
            buf_n = remainder.shape[0]
            shard_idx += 1

    for i, f in enumerate(files):
        frames, onehot, reward, n_act = _load_traj(f, args.target_size)
        n_act_seen.add(n_act)
        T = frames.shape[0]
        if args.action_shift != 0:                        # optional A/B only; default keeps raw alignment
            onehot = torch.roll(onehot, shifts=args.action_shift, dims=0)
            reward = torch.roll(reward, shifts=args.action_shift, dims=0)
        buf.append(frames)
        buf_n += T
        ep_all.append(torch.full((T,), i, dtype=torch.int64))
        act_all.append(onehot)
        rew_all.append(reward)
        flush_full_shards()
        if (i + 1) % 200 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n} traj  ({(i + 1) / max(time.time() - t0, 1e-6):.1f}/s, "
                  f"{shard_idx} shards)", flush=True)

    if buf_n > 0:                                         # trailing partial shard
        out = shards_dir / f"{args.task}_shard{shard_idx:04d}.pt"
        torch.save({"frames": torch.cat(buf, dim=0).contiguous()}, out)
        shard_idx += 1

    episode = torch.cat(ep_all)
    action = torch.cat(act_all)
    reward = torch.cat(rew_all)
    N = episode.shape[0]
    total_shard_frames = (shard_idx - 1) * args.shard_size if shard_idx else 0
    torch.save({"episode": episode, "action": action, "reward": reward},
               demos_dir / f"{args.task}.pt")

    if len(n_act_seen) != 1:
        print(f"  WARNING: inconsistent n_actions across trajectories: {sorted(n_act_seen)}", flush=True)
    print(f"DONE task={args.task}: {n} trajectories, N={N} frames, {shard_idx} shards "
          f"(shard_size={args.shard_size}), n_actions={sorted(n_act_seen)}", flush=True)
    print(f"  frames -> {shards_dir}", flush=True)
    print(f"  demo   -> {demos_dir / (args.task + '.pt')}  "
          f"(episode/action(A={ACTION_DIM})/reward, N must be >= shard total {total_shard_frames})", flush=True)


if __name__ == "__main__":
    main()
