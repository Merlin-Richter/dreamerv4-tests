#!/usr/bin/env python3
"""Identity/alignment gate for the locked community Memory Maze data."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

EXPECTED_TOKENIZER_SHA256 = "347052fae0212ea2c6b943ae7c28a886298ce551d4155b882084d63a3ea48797"
EXPECTED_TRAIN_MANIFEST_SHA256 = "834c9b29e4436614694635826d570d3695542058e020487500691e8954ab673c"
EXPECTED_EVAL_MANIFEST_SHA256 = "3739484c11a87dca14c714b3b491e24e923f9a0a0c48c11cc4bf2e6950e62d20"


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(root: Path):
    path = root / "conversion_manifest.json"
    manifest = json.loads(path.read_text())
    assert manifest["format"] == "dreamer4-community-memmaze-v2-bounded-memory"
    assert manifest["task"] == "memmaze"
    # The approved conversion kept Memory Maze's native 64x64 resolution, so
    # target_size is null rather than an explicit resize request. The baseline
    # validator independently checks every stored frame is actually 64x64.
    assert manifest["target_size"] is None
    assert manifest["action_shift"] == 0
    assert manifest["action_dim_source"] == [6]
    assert manifest["action_dim_stored"] == 16
    assert manifest["action_convention"] == "raw action[t] produced raw image[t]; no shift by default"
    assert manifest["channel_order"] == "RGB"
    assert len(manifest["trajectory_content_sha256"]) == manifest["trajectory_count"]
    return manifest, sha256(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dreamer4", type=Path, required=True)
    ap.add_argument("--train-root", type=Path, required=True)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    train, train_manifest_sha = load_manifest(args.train_root)
    heldout, eval_manifest_sha = load_manifest(args.eval_root)
    assert train_manifest_sha == EXPECTED_TRAIN_MANIFEST_SHA256, train_manifest_sha
    assert eval_manifest_sha == EXPECTED_EVAL_MANIFEST_SHA256, eval_manifest_sha
    overlap = set(train["trajectory_content_sha256"]) & set(heldout["trajectory_content_sha256"])
    assert not overlap, f"train/eval leakage: {len(overlap)} trajectories"

    tokenizer_sha = sha256(args.tokenizer)
    assert tokenizer_sha == EXPECTED_TOKENIZER_SHA256, tokenizer_sha

    source = args.dreamer4.resolve() / "dreamer4"
    sys.path.insert(0, str(source))
    from model import pack_bottleneck_to_spatial, temporal_patchify
    from train_dynamics import load_frozen_tokenizer_from_pt_ckpt
    from wm_dataset import WMDataset

    dataset = WMDataset(
        data_dir=str(args.train_root / "demos"),
        frames_dir=str(args.train_root / "shards"),
        seq_len=32,
        img_size=64,
        action_dim=16,
        shard_size=int(train["shard_size"]),
        cache_mb=16,
        tasks_json="__none__",
        tasks=["memmaze"],
        strict_tasks=True,
        verbose=False,
    )
    task_idx, start = dataset._lookup(0)
    sample = dataset[0]
    aligned = torch.zeros_like(sample["act"])
    aligned[1:] = sample["act"][:-1]
    aligned_mask = torch.zeros_like(sample["act_mask"])
    aligned_mask[1:] = sample["act_mask"][:-1]
    raw_demo = dataset.act[task_idx]
    assert torch.equal(aligned[1:], raw_demo[start + 1:start + 32])
    assert aligned[0].abs().max() == 0 and aligned_mask[0].abs().max() == 0
    assert aligned[:, 6:].abs().max() == 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, _, tok_args = load_frozen_tokenizer_from_pt_ckpt(
        str(args.tokenizer), device=device,
        override={"H": 64, "W": 64, "C": 3, "patch": 4},
    )
    frames = sample["obs"][:-1].unsqueeze(0).to(device).float().div_(255.0)
    with torch.no_grad():
        first, _ = encoder(temporal_patchify(frames, 4))
        transport_copy, _ = encoder(temporal_patchify(frames.clone(), 4))
    packed_first = pack_bottleneck_to_spatial(first, n_spatial=8, k=2)
    packed_copy = pack_bottleneck_to_spatial(transport_copy, n_spatial=8, k=2)
    encoding_max_abs = float((packed_first - packed_copy).abs().max())
    assert encoding_max_abs == 0.0
    assert first.shape == (1, 32, 16, 32)

    report = {
        "tokenizer_sha256": tokenizer_sha,
        "train_manifest_sha256": train_manifest_sha,
        "eval_manifest_sha256": eval_manifest_sha,
        "train_trajectory_names_sha256": train["trajectory_names_sha256"],
        "eval_trajectory_names_sha256": heldout["trajectory_names_sha256"],
        "train_trajectories": train["trajectory_count"],
        "eval_trajectories": heldout["trajectory_count"],
        "content_overlap": 0,
        "channel_order": "RGB",
        "action_alignment": "raw action[t] produced raw image[t]",
        "window": 32,
        "encoding_strategy": "online exact (episode,start,W) windows; no whole-clip cache",
        "repeated_window_encoding_max_abs": encoding_max_abs,
        "tokenizer_latent_shape": list(first.shape),
        "tokenizer_config_n_latents": int(tok_args.get("n_latents", 16)),
        "tokenizer_config_d_bottleneck": int(tok_args.get("d_bottleneck", 32)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("DATA IDENTITY VALIDATION PASSED")


if __name__ == "__main__":
    main()
