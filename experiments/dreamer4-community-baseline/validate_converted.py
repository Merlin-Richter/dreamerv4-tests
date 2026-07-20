#!/usr/bin/env python3
"""Validate a real Memory Maze conversion and optionally prove split disjointness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_manifest(root: Path) -> dict:
    path = root / "conversion_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate(root: Path, task: str) -> dict:
    manifest = load_manifest(root)
    shard_paths = sorted((root / "shards" / task).glob("*.pt"))
    if not shard_paths:
        raise AssertionError(f"no shards under {root}")
    demo_path = root / "demos" / f"{task}.pt"
    demo = torch.load(demo_path, map_location="cpu", weights_only=False)
    episode = demo["episode"]
    action = demo["action"]
    reward = demo["reward"]
    n = int(episode.shape[0])
    assert action.shape == (n, 16) and reward.shape == (n,)
    assert episode.dtype == torch.int64 and action.dtype == torch.float32
    assert torch.isfinite(action).all() and torch.isfinite(reward).all()
    assert torch.allclose(action.sum(-1), torch.ones(n)), "actions are not one-hot"
    assert action[:, 6:].abs().max().item() == 0.0, "nonzero action outside Memory Maze's six dims"

    shard_size = int(manifest["shard_size"])
    frame_count = 0
    for i, path in enumerate(shard_paths):
        frames = torch.load(path, map_location="cpu", weights_only=False)["frames"]
        assert frames.ndim == 4 and frames.shape[1:] == (3, 64, 64), (path, frames.shape)
        assert frames.dtype == torch.uint8
        if i + 1 < len(shard_paths):
            assert frames.shape[0] == shard_size, (path, frames.shape[0], shard_size)
        frame_count += int(frames.shape[0])
    assert frame_count == n == int(manifest["frame_count"])
    assert len(shard_paths) == int(manifest["shard_count"])
    assert int(torch.unique_consecutive(episode).numel()) == int(manifest["trajectory_count"])

    summary = {
        "root": str(root.resolve()),
        "task": task,
        "trajectories": int(manifest["trajectory_count"]),
        "frames": frame_count,
        "shards": len(shard_paths),
        "trajectory_names_sha256": manifest["trajectory_names_sha256"],
    }
    print(json.dumps(summary, indent=2))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--task", default="memmaze")
    ap.add_argument("--compare-other", type=Path, default=None,
                    help="Another converted root; assert raw trajectory names are disjoint.")
    args = ap.parse_args()
    first = validate(args.root, args.task)
    if args.compare_other is not None:
        second = validate(args.compare_other, args.task)
        overlap = set(first["trajectory_content_sha256"]) & set(second["trajectory_content_sha256"])
        assert not overlap, f"split leakage: {len(overlap)} identical raw trajectories"
        print("SPLIT DISJOINTNESS PASSED")
    print("CONVERSION VALIDATION PASSED")


if __name__ == "__main__":
    main()
