#!/usr/bin/env python3
"""Build every exact (episode, window_start, W) community-tokenizer latent once."""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from latent_cache import CACHE_FORMAT, sha256
from train_mem2mem import EXPECTED_TOKENIZER_SHA256


class WindowDataset(Dataset):
    def __init__(self, base, window: int):
        self.base = base
        self.window = int(window)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        task_idx, start = self.base._lookup(int(index))
        return {
            "frames": self.base._get_frames(task_idx, start, self.window),
            "row": torch.tensor(index, dtype=torch.long),
            "start": torch.tensor(start, dtype=torch.long),
        }


def atomic_json(payload: dict, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dreamer4", type=Path, required=True)
    ap.add_argument("--data-dirs", nargs="+", required=True)
    ap.add_argument("--frame-dirs", nargs="+", required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--train-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--packing-factor", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--cache-mb", type=int, default=128)
    ap.add_argument("--shard-size", type=int, default=2048)
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        print(f"LATENT CACHE ALREADY COMPLETE: {manifest_path}")
        return

    tokenizer_sha = sha256(args.tokenizer)
    if tokenizer_sha != EXPECTED_TOKENIZER_SHA256:
        raise RuntimeError(f"unapproved tokenizer {tokenizer_sha}")
    train_manifest_sha = sha256(args.train_manifest)

    source = args.dreamer4.resolve() / "dreamer4"
    sys.path.insert(0, str(source))
    from model import pack_bottleneck_to_spatial, temporal_patchify
    from train_dynamics import load_frozen_tokenizer_from_pt_ckpt
    from wm_dataset import WMDataset

    base = WMDataset(
        data_dir=args.data_dirs,
        frames_dir=args.frame_dirs,
        seq_len=args.window,
        img_size=64,
        action_dim=16,
        shard_size=args.shard_size,
        cache_mb=args.cache_mb,
        tasks_json="__none__",
        tasks=["memmaze"],
        strict_tasks=True,
        verbose=True,
    )
    if len(base.tasks) != 1:
        raise RuntimeError("cache builder expects exactly one memmaze task")
    starts = base.valid_starts[0].numpy().astype(np.int64, copy=False)
    if starts.size != len(base) or np.any(starts[1:] <= starts[:-1]):
        raise RuntimeError("valid cache starts must be strictly increasing")

    row_path = args.out / "row-by-start.npy"
    if not row_path.exists():
        row_partial = args.out / "row-by-start.partial.npy"
        row_map = np.lib.format.open_memmap(
            row_partial, mode="w+", dtype=np.int32, shape=(int(base.ep[0].numel()),)
        )
        row_map[:] = -1
        row_map[starts] = np.arange(starts.size, dtype=np.int32)
        row_map.flush()
        del row_map
        os.replace(row_partial, row_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, _, tok_args = load_frozen_tokenizer_from_pt_ckpt(
        str(args.tokenizer), device=device,
        override={"H": 64, "W": 64, "C": 3, "patch": 4},
    )
    n_latents = int(tok_args.get("n_latents", 16))
    d_bottleneck = int(tok_args.get("d_bottleneck", 32))
    if n_latents % args.packing_factor:
        raise RuntimeError("tokenizer latents are not divisible by packing factor")
    n_spatial = n_latents // args.packing_factor
    d_spatial = d_bottleneck * args.packing_factor
    shape = (len(base), args.window, n_spatial, d_spatial)

    partial_path = args.out / "latents.partial.npy"
    final_latents = args.out / "latents.npy"
    progress_path = args.out / "progress.json"
    if partial_path.exists() and final_latents.exists():
        raise RuntimeError("both partial and final latent arrays exist; refusing ambiguous resume")
    if final_latents.exists():
        finalized = np.load(final_latents, mmap_mode="r")
        if finalized.shape != shape or finalized.dtype != np.float32:
            raise RuntimeError(
                f"final latent array mismatch: {finalized.shape}/{finalized.dtype}, "
                f"expected {shape}/float32"
            )
        del finalized
        completed = len(base)
        out = None
        print("FINAL LATENT ARRAY EXISTS; resuming hashes/manifest only", flush=True)
    elif partial_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        expected = {
            "format": CACHE_FORMAT,
            "shape": list(shape),
            "dtype": "float32",
            "tokenizer_sha256": tokenizer_sha,
            "train_manifest_sha256": train_manifest_sha,
        }
        for key, value in expected.items():
            if progress.get(key) != value:
                raise RuntimeError(f"partial cache metadata mismatch for {key}")
        completed = int(progress["completed_rows"])
        out = np.lib.format.open_memmap(partial_path, mode="r+")
    else:
        completed = 0
        out = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float32, shape=shape)
        atomic_json({
            "format": CACHE_FORMAT,
            "shape": list(shape),
            "dtype": "float32",
            "tokenizer_sha256": tokenizer_sha,
            "train_manifest_sha256": train_manifest_sha,
            "completed_rows": 0,
        }, progress_path)

    stop = False

    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)

    if out is not None:
        subset = torch.utils.data.Subset(WindowDataset(base, args.window), range(completed, len(base)))
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        t0 = time.time()
        last_completed = completed
        with torch.inference_mode():
            for batch_idx, batch in enumerate(loader, start=1):
                rows = batch["row"].numpy().astype(np.int64, copy=False)
                if rows[0] != last_completed or np.any(rows != np.arange(rows[0], rows[0] + rows.size)):
                    raise RuntimeError("cache builder lost sequential row order")
                frames = batch["frames"].to(device, non_blocking=True).float().div_(255.0)
                latent, _ = encoder(temporal_patchify(frames, 4))
                packed = pack_bottleneck_to_spatial(
                    latent, n_spatial=n_spatial, k=args.packing_factor
                )
                out[rows] = packed.cpu().numpy()
                last_completed = int(rows[-1]) + 1
                if batch_idx % args.log_every == 0 or last_completed == len(base) or stop:
                    out.flush()
                    atomic_json({
                        "format": CACHE_FORMAT,
                        "shape": list(shape),
                        "dtype": "float32",
                        "tokenizer_sha256": tokenizer_sha,
                        "train_manifest_sha256": train_manifest_sha,
                        "completed_rows": last_completed,
                    }, progress_path)
                    rate = (last_completed - completed) / max(time.time() - t0, 1e-9)
                    print(
                        f"cache rows {last_completed:,}/{len(base):,} ({rate:.1f} windows/s)",
                        flush=True,
                    )
                if stop:
                    print("CACHE BUILD INTERRUPTED CLEANLY; resume with the same command", flush=True)
                    return

        out.flush()
        del out
        os.replace(partial_path, final_latents)
    progress_path.unlink(missing_ok=True)
    manifest = {
        "format": CACHE_FORMAT,
        "window": args.window,
        "packing_factor": args.packing_factor,
        "shape": list(shape),
        "dtype": "float32",
        "latents_file": final_latents.name,
        "latents_sha256": sha256(final_latents),
        "row_by_start_file": row_path.name,
        "row_by_start_sha256": sha256(row_path),
        "tokenizer_sha256": tokenizer_sha,
        "train_manifest_sha256": train_manifest_sha,
        "n_latents": n_latents,
        "d_bottleneck": d_bottleneck,
        "n_spatial": n_spatial,
        "d_spatial": d_spatial,
        "frame_count": int(base.ep[0].numel()),
        "window_count": len(base),
        "encoding": "exact independent causal W-frame windows; FP32 outputs",
    }
    atomic_json(manifest, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("EXACT COMMUNITY LATENT CACHE BUILD PASSED")


if __name__ == "__main__":
    main()
