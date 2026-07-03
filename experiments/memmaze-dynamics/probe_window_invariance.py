#!/usr/bin/env python3
"""Measure how much a frame's latent depends on WHERE its encode window started.

Context: the latent disk cache (train_dynamics.ensure_latent_cache) encodes episodes in fixed
non-overlapping tokenizer windows, but dynamics training slices those latents at arbitrary clip
offsets. The tokenizer encoder is temporally CAUSAL within its window, so a frame's latent is
architecturally a function of the window start. Merlin's claim (2026-07-03): per-frame-reconstruction
training gives no pressure to use temporal context, so latents are ~window-invariant. This probe
measures it.

Method: encode frames[0:W] (window start 0) and frames[off:off+W] (window start off) for E episodes;
on the overlapping frames [off, W) compare:
  - latent cosine similarity + relative L2 (per frame, flattened latents),
  - decoded-recon consistency: pixel MSE between the two decodes, each vs ground truth,
  - fp16 cast error (cache storage precision), for reference.
Prints a table + writes JSON next to this script.

Run (either env; GridWorld locally, memmaze on cluster):
  python -u experiments/memmaze-dynamics/probe_window_invariance.py \
      --frames data/gridworld.npy --tokenizer checkpoints/gridworld/tokenizer.pt --offset 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from training.train_dynamics import load_tokenizer, encode_frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--n-episodes", type=int, default=8)
    ap.add_argument("--offset", type=int, default=None,
                    help="Second window's start offset (default: tokenizer window // 2).")
    ap.add_argument("--out", type=Path, default=None,
                    help="JSON output (default: probe_window_invariance_<framesstem>.json here).")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(args.tokenizer, map_location="cpu", weights_only=False)
    W = int(payload["config"]["max_temporal_length"])
    del payload
    off = args.offset if args.offset is not None else W // 2
    assert 0 < off < W, f"offset must be in (0, {W})"

    tok = load_tokenizer(args.tokenizer, device)
    raw = np.load(args.frames, mmap_mode="r")
    E = min(args.n_episodes, raw.shape[0])
    assert raw.shape[1] >= off + W, f"episodes too short: T={raw.shape[1]} < off+W={off + W}"

    frames = torch.from_numpy(
        np.asarray(raw[:E, :off + W]).astype(np.float32) / 255.0).to(device)

    with torch.no_grad():
        z_a = encode_frames(tok, frames[:, :W])            # window start 0   (fp32 weights, no amp)
        z_b = encode_frames(tok, frames[:, off:off + W])   # window start off
        # overlap: absolute frames [off, W)
        a = z_a[:, off:]                                   # positions off..W-1 in window A
        b = z_b[:, :W - off]                               # positions 0..W-1-off in window B
        # per-frame flattened latent vectors
        af = a.reshape(E, W - off, -1)
        bf = b.reshape(E, W - off, -1)
        cos = torch.nn.functional.cosine_similarity(af, bf, dim=-1)          # (E, overlap)
        rel_l2 = (af - bf).norm(dim=-1) / af.norm(dim=-1).clamp_min(1e-12)   # (E, overlap)
        # decode both and compare in pixel space (what the dynamics' consumer ultimately sees)
        ra = tok.decoder(z_a)[:, off:]
        rb = tok.decoder(z_b)[:, :W - off]
        gt = frames[:, off:W]
        mse_ab = ((ra - rb) ** 2).mean().item()
        mse_a_gt = ((ra - gt) ** 2).mean().item()
        mse_b_gt = ((rb - gt) ** 2).mean().item()
        # fp16 storage error, for scale reference
        fp16_err = (a - a.half().float()).norm(dim=-1).reshape(E, W - off, -1).mean().item()
        lat_scale = af.norm(dim=-1).mean().item()

    # per-overlap-position profile (does divergence grow with distance from the window start?)
    cos_by_pos = cos.mean(dim=0).cpu().tolist()
    result = {
        "frames": str(args.frames), "tokenizer": str(args.tokenizer),
        "window": W, "offset": off, "n_episodes": E, "n_overlap_frames": W - off,
        "latent_cos_mean": cos.mean().item(), "latent_cos_min": cos.min().item(),
        "latent_rel_l2_mean": rel_l2.mean().item(), "latent_rel_l2_max": rel_l2.max().item(),
        "latent_norm_mean": lat_scale, "fp16_cast_err_norm": fp16_err,
        "recon_mse_between_windows": mse_ab,
        "recon_mse_winA_vs_gt": mse_a_gt, "recon_mse_winB_vs_gt": mse_b_gt,
        "latent_cos_by_overlap_pos": cos_by_pos,
    }
    print(f"window={W} offset={off} episodes={E} overlap={W - off} frames")
    print(f"  latent cos-sim   mean {result['latent_cos_mean']:.6f}  min {result['latent_cos_min']:.6f}")
    print(f"  latent rel-L2    mean {result['latent_rel_l2_mean']:.6f}  max {result['latent_rel_l2_max']:.6f}")
    print(f"  recon MSE: A-vs-B {mse_ab:.3e} | A-vs-GT {mse_a_gt:.3e} | B-vs-GT {mse_b_gt:.3e}")
    print(f"  (fp16 cast err norm {fp16_err:.3e} vs latent norm {lat_scale:.3f})")
    out = args.out or Path(__file__).parent / f"probe_window_invariance_{args.frames.stem}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
