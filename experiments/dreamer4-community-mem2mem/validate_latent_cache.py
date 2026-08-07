#!/usr/bin/env python3
"""Gate cache identity, full hashes, online equality, and long-clip lookup alignment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from latent_cache import CachedLatentClipDataset, WindowLatentCache, load_manifest, sha256
from train_mem2mem import EXPECTED_TOKENIZER_SHA256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dreamer4", type=Path, required=True)
    ap.add_argument("--data-dirs", nargs="+", required=True)
    ap.add_argument("--frame-dirs", nargs="+", required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--train-manifest", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--full-hash", action="store_true")
    args = ap.parse_args()

    manifest, manifest_sha = load_manifest(args.cache)
    assert manifest["tokenizer_sha256"] == sha256(args.tokenizer) == EXPECTED_TOKENIZER_SHA256
    assert manifest["train_manifest_sha256"] == sha256(args.train_manifest)
    assert manifest["dtype"] == "float32" and int(manifest["window"]) == 32
    if args.full_hash:
        assert sha256(args.cache / manifest["latents_file"]) == manifest["latents_sha256"]
        assert sha256(args.cache / manifest["row_by_start_file"]) == manifest["row_by_start_sha256"]

    source = args.dreamer4.resolve() / "dreamer4"
    sys.path.insert(0, str(source))
    from model import pack_bottleneck_to_spatial, temporal_patchify
    from train_dynamics import load_frozen_tokenizer_from_pt_ckpt
    from wm_dataset import WMDataset

    common = dict(
        data_dir=args.data_dirs, frames_dir=args.frame_dirs, img_size=64, action_dim=16,
        shard_size=2048, cache_mb=128, tasks_json="__none__", tasks=["memmaze"],
        strict_tasks=True, verbose=False,
    )
    windows = WMDataset(seq_len=32, **common)
    clips = WMDataset(seq_len=128, **common)
    cache = WindowLatentCache(args.cache)
    assert len(windows) == int(manifest["window_count"])
    assert int(windows.ep[0].numel()) == int(manifest["frame_count"])

    rows = sorted(set([0, 1, len(windows) // 7, len(windows) // 2, len(windows) - 2, len(windows) - 1]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, _, _ = load_frozen_tokenizer_from_pt_ckpt(
        str(args.tokenizer), device=device,
        override={"H": 64, "W": 64, "C": 3, "patch": 4},
    )
    max_abs = 0.0
    with torch.inference_mode():
        for row in rows:
            task_idx, start = windows._lookup(row)
            frames = windows._get_frames(task_idx, start, 32).unsqueeze(0)
            frames = frames.to(device).float().div_(255.0)
            latent, _ = encoder(temporal_patchify(frames, 4))
            packed = pack_bottleneck_to_spatial(latent, n_spatial=8, k=2)[0].cpu()
            cached_row = cache.rows_for_starts(np.array([start]))[0]
            assert int(cached_row) == row
            cached = torch.from_numpy(np.array(cache.latents[cached_row], copy=True))
            max_abs = max(max_abs, float((packed - cached).abs().max()))
            assert torch.equal(packed, cached)

    cached_clips = CachedLatentClipDataset(clips, args.cache, window=32, clip_length=128)
    for clip_index in (0, len(clips) // 3, len(clips) - 1):
        sample = cached_clips[clip_index]
        _, start = clips._lookup(clip_index)
        expected_rows = cache.rows_for_starts(start + np.arange(0, 97, 16))
        expected = torch.from_numpy(np.array(cache.latents[expected_rows], copy=True))
        assert torch.equal(sample["latents"], expected)
        assert int(sample["_global_start"]) == start

    report = {
        "cache_manifest_sha256": manifest_sha,
        "cache_shape": manifest["shape"],
        "dtype": manifest["dtype"],
        "full_hash_checked": args.full_hash,
        "online_vs_cached_max_abs": max_abs,
        "sampled_online_windows": rows,
        "sampled_long_clips": [0, len(clips) // 3, len(clips) - 1],
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "train_manifest_sha256": manifest["train_manifest_sha256"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("EXACT COMMUNITY LATENT CACHE VALIDATION PASSED")


if __name__ == "__main__":
    main()
