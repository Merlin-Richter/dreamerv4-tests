#!/usr/bin/env python3
"""Disk-constrained Memory Maze prep (vast box, 32 GB): stream npz -> latent cache, NO frames npy.

The ferranti pipeline (unzip -> convert_memmaze.py -> 35.7 GB memmaze9x9.npy -> ensure_latent_cache)
cannot fit on the vast rental's 32 GB container disk. Training itself never reads pixels — it runs
off the fp16 latent cache; `train_archive.py` / `ensure_latent_cache` touch the frames npy only for
its SHAPE (mmap header). So this builds exactly the artifacts training needs, straight from the
per-trajectory .npz files:

  1. `<frames>.latents-<tokhash>.npy` (N, T, n_latents, bottleneck_dim) fp16 + meta json —
     format-identical to ensure_latent_cache's output: same sorted-rglob episode order as
     convert_memmaze.py, channels AS-IS (RGB, untouched), same window/batch/bf16-autocast encode.
  2. `<frames>_actions.npy` (N, T) int64 (argmax if one-hot) — extract_actions_labels.py semantics,
     same episode order as (1) by construction.
  3. `<frames>` itself as a SPARSE PLACEHOLDER: a real npy header claiming (N, T, H, W, 3) uint8
     with no data blocks behind it (~0 real disk). DO NOT rsync/copy it (that materializes 35.7 GB
     of zeros) and DO NOT read pixels from it — a sibling marker file spells this out on the box.

Run with -u. Example (vast, inside the venv):
    python -u experiments/hierarchical-archive-memory/prep_vast.py \
        --raw data/memmaze9x9_raw --frames data/memmaze9x9.npy \
        --tokenizer checkpoints/memmaze/tokenizer.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from training.train_dynamics import (  # noqa: E402
    _file_sha256, encode_frames, ensure_latent_cache, latent_cache_paths, load_tokenizer,
)

PLACEHOLDER_MARKER = ".SPARSE-PLACEHOLDER.txt"


def gpu_smoke(allow_cpu: bool) -> str:
    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
          f"available={torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        if allow_cpu:
            return "cpu"
        sys.exit("CUDA unavailable — encoding ~2900 eps on CPU is not viable. "
                 "(--allow-cpu to override; check the venv's torch build vs the GPU arch.)")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    x = torch.randn(512, 512, device="cuda")
    y = float((x @ x).sum())
    assert np.isfinite(y)
    print(f"GPU OK: {name} (CC {cap[0]}.{cap[1]}), smoke matmul finite", flush=True)
    return "cuda"


def write_sparse_placeholder(frames_path: Path, shape: tuple) -> None:
    """A real npy header claiming `shape` uint8, ftruncated to full size with no data blocks."""
    if frames_path.exists():
        blocks_gb = os.stat(frames_path).st_blocks * 512 / 1e9
        old = np.load(frames_path, mmap_mode="r")
        if old.shape == shape and blocks_gb < 0.05:
            print(f"[placeholder] already present ({blocks_gb:.3f} GB real) — keeping", flush=True)
            return
        sys.exit(f"{frames_path} exists (shape {old.shape}, {blocks_gb:.1f} GB real blocks) — "
                 f"refusing to overwrite what may be a real dataset.")
    mm = np.lib.format.open_memmap(frames_path, mode="w+", dtype=np.uint8, shape=shape)
    del mm  # header written, file ftruncated to full size; no pages dirtied => sparse
    st = os.stat(frames_path)
    real_gb = st.st_blocks * 512 / 1e9
    print(f"[placeholder] {frames_path.name}: apparent {st.st_size / 1e9:.1f} GB, "
          f"real {real_gb:.3f} GB", flush=True)
    if real_gb > 0.5:
        frames_path.unlink()
        sys.exit("placeholder MATERIALIZED (filesystem does not keep it sparse) — aborting "
                 "before the disk fills; this box cannot hold a real frames npy.")
    marker = frames_path.with_name(frames_path.name + PLACEHOLDER_MARKER)
    marker.write_text(
        f"{frames_path.name} is a SPARSE PLACEHOLDER written by prep_vast.py.\n"
        f"It holds ONLY a valid npy header (shape {shape} uint8) so shape checks in\n"
        f"train_archive.py / ensure_latent_cache pass; every pixel reads as 0.\n"
        f"Training runs off the fp16 latent cache next to it. Do NOT rsync/copy this file\n"
        f"(it materializes {st.st_size / 1e9:.1f} GB of zeros) and do NOT train pixels from it.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True,
                    help="Dir holding the unzipped per-trajectory .npz files (searched recursively).")
    ap.add_argument("--frames", type=Path, required=True,
                    help="Placeholder frames npy path — names the latent cache/actions sidecars.")
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--batch-episodes", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="First N trajectories only (testing).")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    device = gpu_smoke(args.allow_cpu)

    files = sorted(args.raw.rglob("*.npz"))  # SAME order as convert_memmaze/extract_actions_labels
    if not files:
        sys.exit(f"No .npz files under {args.raw}")
    if args.limit is not None:
        files = files[: args.limit]
    n = len(files)

    with np.load(files[0]) as z:
        keys = sorted(z.keys())
        img0, act0 = z["image"], z["action"]
        print(f"{n} trajectories | keys: " + ", ".join(f"{k}{z[k].shape}" for k in keys), flush=True)
    if img0.ndim != 4 or img0.shape[-1] != 3:
        sys.exit(f"Unexpected image shape {img0.shape} (want (T, H, W, 3)).")
    t, h, w, _ = img0.shape
    act_t = act0.shape[0]
    if act_t != t:
        sys.exit(f"action T={act_t} != image T={t} — alignment assumption broken, refusing.")
    one_hot = act0.ndim == 2

    payload = torch.load(args.tokenizer, map_location="cpu", weights_only=False)
    tok_T = int(payload["config"]["max_temporal_length"])
    n_lat = int(payload["config"]["n_latents"])
    bdim = int(payload["config"]["bottleneck_dim"])
    del payload
    tokenizer = load_tokenizer(args.tokenizer, device)
    lat_npy, meta_p = latent_cache_paths(args.frames, args.tokenizer)
    use_amp = device == "cuda"
    print(f"[prep] encoding {n} eps x {t} frames in windows of {tok_T} "
          f"-> {lat_npy.name} (fp16, ({n}, {t}, {n_lat}, {bdim}))", flush=True)

    args.frames.parent.mkdir(parents=True, exist_ok=True)
    lat_tmp = lat_npy.with_name(lat_npy.name + f".tmp{os.getpid()}")
    out = np.lib.format.open_memmap(lat_tmp, mode="w+", dtype=np.float16, shape=(n, t, n_lat, bdim))
    acts_path = args.frames.with_name(args.frames.stem + "_actions.npy")
    acts_tmp = acts_path.with_name(acts_path.name + f".tmp{os.getpid()}")
    acts_out = np.lib.format.open_memmap(acts_tmp, mode="w+", dtype=np.int64, shape=(n, t))

    t0 = time.perf_counter()
    bs = args.batch_episodes
    with torch.no_grad():
        for e0 in range(0, n, bs):
            e1 = min(n, e0 + bs)
            imgs = np.empty((e1 - e0, t, h, w, 3), dtype=np.uint8)
            for j, f in enumerate(files[e0:e1]):
                with np.load(f) as z:
                    img = z["image"]
                    if img.shape != (t, h, w, 3):
                        sys.exit(f"Shape mismatch at {f}: {img.shape} != {(t, h, w, 3)}")
                    imgs[j] = img  # channels AS-IS (RGB), matching convert_memmaze.py
                    a = z["action"]
                    acts_out[e0 + j] = a.argmax(axis=-1) if one_hot else a
            clip = torch.from_numpy(imgs.astype(np.float32) / 255.0).to(device)
            zs = []
            for w0 in range(0, t, tok_T):
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                    zt = encode_frames(tokenizer, clip[:, w0:w0 + tok_T])
                zs.append(zt.float().cpu())
            out[e0:e1] = torch.cat(zs, dim=1).numpy().astype(np.float16)
            if e1 % max(bs * 25, 1) < bs or e1 == n:
                dt = time.perf_counter() - t0
                print(f"[prep]   {e1}/{n} eps  ({e1 / max(dt, 1e-9):.1f} eps/s)", flush=True)
    out.flush(); del out
    acts_out.flush()
    n_actions = int(acts_out[:].max()) + 1
    del acts_out

    sample = np.load(lat_tmp, mmap_mode="r")[0].astype(np.float32)
    assert np.isfinite(sample).all(), "non-finite latents in episode 0"
    print(f"[prep] latents ep0: mean {sample.mean():.4f} std {sample.std():.4f} (finite)", flush=True)

    meta_p.write_text(json.dumps({
        "frames": str(args.frames.name), "frames_shape": [n, t, h, w, 3],
        "tokenizer": str(args.tokenizer), "tokenizer_sha256_12": _file_sha256(args.tokenizer),
        "window": tok_T, "dtype": "float16",
        "latents_shape": [n, t, n_lat, bdim],
        "built_by": "prep_vast.py (streamed from npz; frames npy is a sparse placeholder)",
    }, indent=2))
    os.replace(lat_tmp, lat_npy)
    os.replace(acts_tmp, acts_path)
    del tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"[prep] BUILT {lat_npy.name} in {time.perf_counter() - t0:.0f}s | "
          f"actions {acts_path.name} (n_actions={n_actions})", flush=True)

    write_sparse_placeholder(args.frames, (n, t, h, w, 3))

    # End-to-end check: the trainer's own cache lookup must HIT (never loads the tokenizer).
    got = ensure_latent_cache(args.frames, args.tokenizer, "cpu")
    assert Path(got) == lat_npy, f"cache lookup returned {got}, expected {lat_npy}"
    print(f"PREP DONE: n={n} t={t} n_actions={n_actions} latents={lat_npy.name}", flush=True)


if __name__ == "__main__":
    main()
