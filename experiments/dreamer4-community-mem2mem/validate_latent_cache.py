#!/usr/bin/env python3
"""Gate cache identity, numerical encoder equivalence, and long-clip lookup alignment."""
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
    ap.add_argument("--reference-batch-size", type=int, default=64)
    ap.add_argument("--comparison-batch-sizes", type=int, nargs="*", default=())
    ap.add_argument("--require-bit-exact-comparison-batches", type=int, nargs="*", default=())
    ap.add_argument("--max-singleton-abs", type=float, default=float("inf"))
    ap.add_argument("--max-replay-abs", type=float, default=0.0)
    ap.add_argument("--max-comparison-abs", type=float, default=float("inf"))
    ap.add_argument("--max-comparison-relative-l2", type=float, default=float("inf"))
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
    def error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict:
        actual = actual.float()
        expected = expected.float()
        delta = actual - expected
        expected_norm = float(expected.norm())
        return {
            "max_abs": float(delta.abs().max()),
            "mean_abs": float(delta.abs().mean()),
            "rmse": float(delta.square().mean().sqrt()),
            "relative_l2": float(delta.norm()) / max(expected_norm, 1e-30),
            "cosine": float(torch.nn.functional.cosine_similarity(
                actual.flatten(), expected.flatten(), dim=0
            )),
            "bit_equal": bool(torch.equal(actual, expected)),
        }

    def compare_at_batch_size(batch_size: int) -> dict:
        if batch_size <= 0:
            raise ValueError("comparison batch sizes must be positive")
        groups = {}
        group_starts = sorted(set((row // batch_size) * batch_size for row in rows))
        with torch.inference_mode():
            for group_start in group_starts:
                group_rows = list(range(group_start, min(group_start + batch_size, len(windows))))
                frames = []
                starts = []
                for row in group_rows:
                    task_idx, start = windows._lookup(row)
                    frames.append(windows._get_frames(task_idx, start, 32))
                    starts.append(start)
                frame_batch = torch.stack(frames).to(device).float().div_(255.0)
                latent, _ = encoder(temporal_patchify(frame_batch, 4))
                packed = pack_bottleneck_to_spatial(latent, n_spatial=8, k=2).cpu()
                cached_rows = cache.rows_for_starts(np.asarray(starts, dtype=np.int64))
                expected = torch.from_numpy(np.array(cache.latents[cached_rows], copy=True))
                groups[str(group_start)] = error_stats(packed, expected)
        return {
            "batch_size": batch_size,
            "max_abs": max(item["max_abs"] for item in groups.values()),
            "max_relative_l2": max(item["relative_l2"] for item in groups.values()),
            "min_cosine": min(item["cosine"] for item in groups.values()),
            "all_bit_equal": all(item["bit_equal"] for item in groups.values()),
            "groups": groups,
        }

    singleton_rows = {}
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
            singleton_rows[str(row)] = error_stats(packed, cached)

    # Reproduce the builder's sequential batch shape. This distinguishes a bad
    # cache write from harmless floating-point kernel differences caused by
    # comparing batch-64 construction against singleton online encoding.
    batch_size = int(args.reference_batch_size)
    replay = compare_at_batch_size(batch_size)
    comparisons = {
        str(comparison_batch_size): compare_at_batch_size(comparison_batch_size)
        for comparison_batch_size in args.comparison_batch_sizes
        if comparison_batch_size != batch_size
    }

    cached_clips = CachedLatentClipDataset(clips, args.cache, window=32, clip_length=128)
    for clip_index in (0, len(clips) // 3, len(clips) - 1):
        sample = cached_clips[clip_index]
        _, start = clips._lookup(clip_index)
        expected_rows = cache.rows_for_starts(start + np.arange(0, 97, 16))
        expected = torch.from_numpy(np.array(cache.latents[expected_rows], copy=True))
        assert torch.equal(sample["latents"], expected)
        assert int(sample["_global_start"]) == start

    singleton_max_abs = max(item["max_abs"] for item in singleton_rows.values())
    replay_max_abs = replay["max_abs"]
    report = {
        "cache_manifest_sha256": manifest_sha,
        "cache_shape": manifest["shape"],
        "dtype": manifest["dtype"],
        "full_hash_checked": args.full_hash,
        "singleton_online_vs_cached": {
            "max_abs": singleton_max_abs,
            "rows": singleton_rows,
        },
        "builder_batch_replay_vs_cached": {
            **replay,
            "reference_batch_size": batch_size,
        },
        "comparison_batch_shapes_vs_cached": comparisons,
        "sampled_online_windows": rows,
        "sampled_long_clips": [0, len(clips) // 3, len(clips) - 1],
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "train_manifest_sha256": manifest["train_manifest_sha256"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if singleton_max_abs > args.max_singleton_abs:
        raise RuntimeError(
            f"singleton online/cache max_abs {singleton_max_abs} > {args.max_singleton_abs}"
        )
    if replay_max_abs > args.max_replay_abs:
        raise RuntimeError(
            f"builder-batch replay/cache max_abs {replay_max_abs} > {args.max_replay_abs}"
        )
    for required_batch in args.require_bit_exact_comparison_batches:
        comparison = comparisons.get(str(required_batch))
        if comparison is None:
            raise RuntimeError(f"required comparison batch {required_batch} was not evaluated")
        if not comparison["all_bit_equal"]:
            raise RuntimeError(f"required comparison batch {required_batch} is not bit-exact")
    for comparison_batch, comparison in comparisons.items():
        if comparison["max_abs"] > args.max_comparison_abs:
            raise RuntimeError(
                f"batch-{comparison_batch} max_abs {comparison['max_abs']} "
                f"> {args.max_comparison_abs}"
            )
        if comparison["max_relative_l2"] > args.max_comparison_relative_l2:
            raise RuntimeError(
                f"batch-{comparison_batch} relative_l2 {comparison['max_relative_l2']} "
                f"> {args.max_comparison_relative_l2}"
            )
    print("EXACT COMMUNITY LATENT CACHE VALIDATION PASSED")


if __name__ == "__main__":
    main()
