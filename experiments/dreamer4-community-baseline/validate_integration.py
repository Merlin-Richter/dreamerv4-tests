#!/usr/bin/env python3
"""Cluster-free validation of the Memory-Maze -> community-Dreamer4 data integration.

Synthesizes a handful of memory-maze-format .npz trajectories, runs memmaze_to_dreamer4.py on them,
then loads the output with the ACTUAL community dataset classes (ShardedFrameDataset for the tokenizer,
WMDataset for the action-conditioned dynamics) and asserts shapes/one-hotness. Also runs a tiny 64x64
forward through the community Encoder/Decoder/Dynamics to confirm the architecture accepts the resolution.

This needs NO GPU, NO real data, NO cluster — only torch + a checkout of the community repo. It is the
regression test for the converter and the fastest way to re-confirm the integration after any change.

    python -u validate_integration.py --dreamer4 /path/to/dreamer4     # repo root (contains dreamer4/)

Passing exit code 0 + "VALIDATION PASSED" means the converted format is accepted end-to-end.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CONVERTER = HERE / "memmaze_to_dreamer4.py"


def synth_npz(raw: Path, n_traj=5, T=60):
    rng = np.random.default_rng(0)
    for k in range(n_traj):
        img = rng.integers(0, 256, (T, 64, 64, 3), dtype=np.uint8)
        idx = rng.integers(0, 6, (T,))
        act = np.zeros((T, 6), np.float32)
        act[np.arange(T), idx] = 1.0                       # one-hot, like the real .npz
        np.savez(raw / f"traj_{k:03d}.npz", image=img, action=act,
                 reward=np.zeros((T,), np.float32))
    return n_traj, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dreamer4", type=Path, required=True,
                    help="Community dreamer4 repo root (the dir whose child 'dreamer4/' holds model.py).")
    ap.add_argument("--shard-size", type=int, default=32, help="Small so multiple shards are exercised.")
    args = ap.parse_args()

    pkg = args.dreamer4 / "dreamer4"
    assert (pkg / "wm_dataset.py").exists(), f"{pkg} is not the community package dir"
    sys.path.insert(0, str(pkg))

    work = Path(tempfile.mkdtemp(prefix="d4val_"))
    raw = work / "raw"; raw.mkdir()
    n_traj, T = synth_npz(raw)
    print(f"[synth] {n_traj} npz of T={T} -> {raw}")

    out = work / "d4"
    r = subprocess.run([sys.executable, "-u", str(CONVERTER), "--raw", str(raw), "--out-dir", str(out),
                        "--task", "memmaze", "--shard-size", str(args.shard_size)],
                       capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr[-2000:]); sys.exit("converter failed")

    from sharded_frame_dataset import ShardedFrameDataset
    from wm_dataset import WMDataset, collate_batch
    from torch.utils.data import DataLoader

    frames_root, demos_root = out / "shards", out / "demos"

    # tokenizer path
    tok_ds = ShardedFrameDataset(outdirs=[str(frames_root)], tasks=["memmaze"], seq_len=8)
    xb = tok_ds[0]
    assert xb.shape == (8, 3, 64, 64) and xb.dtype == torch.float32 and 0 <= xb.min() and xb.max() <= 1
    print(f"[tokenizer] ShardedFrameDataset ok: sample={tuple(xb.shape)} range=[{xb.min():.2f},{xb.max():.2f}]")

    # dynamics path — WMDataset shard_size MUST match the converter's shard-size
    wm = WMDataset(data_dir=[str(demos_root)], frames_dir=[str(frames_root)], seq_len=16, img_size=64,
                   action_dim=16, shard_size=args.shard_size, tasks=["memmaze"], tasks_json="__none__",
                   strict_tasks=False, verbose=True)
    b = next(iter(DataLoader(wm, batch_size=4, shuffle=True, collate_fn=collate_batch)))
    assert b["obs"].shape == (4, 17, 3, 64, 64), b["obs"].shape        # T+1 frames
    assert b["act"].shape == (4, 16, 16), b["act"].shape               # (B,T,A=16)
    a = b["act"]
    assert a.sum(-1).allclose(torch.ones_like(a.sum(-1))), "act not one-hot per step"
    assert a[..., 6:].abs().max() == 0, "nonzero beyond 6 action dims"
    print(f"[dynamics] WMDataset ok: obs={tuple(b['obs'].shape)} act={tuple(b['act'].shape)} (clean one-hot in first 6)")

    # architecture forward at 64x64
    from model import Encoder, Decoder, Tokenizer, Dynamics, temporal_patchify, pack_bottleneck_to_spatial
    H = W = 64; patch = 4; n_patches = (H // patch) ** 2; d_patch = patch * patch * 3
    n_lat, d_b, pack, k_max = 16, 32, 2, 8
    enc = Encoder(patch_dim=d_patch, d_model=256, n_latents=n_lat, n_patches=n_patches, n_heads=4,
                  depth=2, d_bottleneck=d_b, dropout=0.0, mlp_ratio=4.0, time_every=1, mae_p_min=0.0,
                  mae_p_max=0.9, scale_pos_embeds=False)
    dec = Decoder(d_bottleneck=d_b, d_model=256, n_heads=4, depth=2, n_latents=n_lat, n_patches=n_patches,
                  d_patch=d_patch, dropout=0.0, mlp_ratio=4.0, time_every=1, scale_pos_embeds=False)
    tok = Tokenizer(enc, dec).eval()
    x = torch.rand(2, 6, 3, H, W)
    with torch.no_grad():
        pred, _, _ = tok(temporal_patchify(x, patch))
        z, _ = enc(temporal_patchify(x, patch))
    assert z.shape == (2, 6, n_lat, d_b) and pred.shape == (2, 6, n_patches, d_patch)
    dyn = Dynamics(d_model=128, d_bottleneck=d_b, d_spatial=d_b * pack, n_spatial=n_lat // pack,
                   n_register=4, n_agent=1, n_heads=4, depth=2, k_max=k_max).eval()
    z1 = pack_bottleneck_to_spatial(z, n_spatial=n_lat // pack, k=pack)
    si = torch.full((2, 6), int(round(math.log2(k_max))), dtype=torch.long)
    sg = torch.randint(0, k_max, (2, 6))
    act = torch.zeros(2, 6, 16); act[:, :, 1] = 1.0
    am = torch.zeros(2, 6, 16); am[:, :, :6] = 1.0
    with torch.no_grad():
        y, _ = dyn(act, si, sg, z1, act_mask=am)
    assert y.shape == z1.shape
    print(f"[model] 64x64 forward ok: z={tuple(z.shape)} z1_packed={tuple(z1.shape)} dyn_out={tuple(y.shape)} "
          "(output is 0 on untrained model: flow_x_head is zero-init)")

    print("\nVALIDATION PASSED - community WMDataset + ShardedFrameDataset + model accept the format at 64x64.")


if __name__ == "__main__":
    main()
