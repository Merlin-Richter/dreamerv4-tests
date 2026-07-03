#!/usr/bin/env python3
"""Batch-size / throughput search for the memmaze dynamics arms (synthetic latents).

Times a REAL optimizer step (forward + backward + AdamW) for:
  - vanilla:  DynamicsModel.loss on (bs, W, 32, 16) latents,
  - mem2mem:  mem2mem_rollout_loss on (bs, CLIP, 32, 16) latents (rollout-only winner config,
              worst-case n_ctx = W).
over a bs ladder, reporting s/step, clips/s and peak VRAM, catching OOM. Synthetic latents make it
independent of the prep job (timing is shape-driven; loader IO on the 3GB fp16 cache is negligible).

Config mirrors the campaign choice (512/12/16, W=32, clip 128, n_memory 8, ff9 3, n_actions 6).
Run on the cluster:
  python -u experiments/memmaze-dynamics/bs_search.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "experiments" / "mem2mem"))

from models.dynamics_model import DynamicsModel, DynamicsModelConfig  # noqa: E402
from rollout import mem2mem_rollout_loss                              # noqa: E402


def time_steps(step_fn, n_warm=3, n_time=8):
    for _ in range(n_warm):
        step_fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_time):
        step_fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding-dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--n-heads", type=int, default=16)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--clip-len", type=int, default=128)
    ap.add_argument("--n-memory", type=int, default=8)
    ap.add_argument("--n-actions", type=int, default=6)
    ap.add_argument("--bs", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    args = ap.parse_args()

    device = "cuda"
    assert torch.cuda.is_available()
    print(torch.cuda.get_device_name(0))
    torch.set_float32_matmul_precision("high")

    for arm in ("vanilla", "mem2mem"):
        n_mem = 0 if arm == "vanilla" else args.n_memory
        ff9 = 0 if arm == "vanilla" else 3
        cfg = DynamicsModelConfig(
            embedding_dim=args.embedding_dim, depth=args.depth, n_heads=args.n_heads,
            max_temporal_length=args.window, n_latents=32, bottleneck_dim=16,
            n_actions=args.n_actions, n_memory=n_mem, ff9_k=ff9)
        T = args.window if arm == "vanilla" else args.clip_len
        print(f"== {arm}: dim{args.embedding_dim}/d{args.depth}/h{args.n_heads} W={args.window} "
              f"T_clip={T} n_memory={n_mem} ==", flush=True)
        for bs in args.bs:
            torch.manual_seed(0)
            model = DynamicsModel(cfg).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
            nparams = sum(p.numel() for p in model.parameters())
            z1 = torch.randn(bs, T, 32, 16, device=device)
            acts = torch.randint(0, args.n_actions, (bs, T), device=device)
            gen = torch.Generator(device=device).manual_seed(0)

            def step():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if arm == "vanilla":
                        loss = model.loss(z1, acts)
                    else:
                        loss, _ = mem2mem_rollout_loss(
                            model, z1, acts, n_ctx=args.window, device=device, gen=gen,
                            bootstrap=False, n_d_unlocked=1, use_ff9=True)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            try:
                torch.cuda.reset_peak_memory_stats()
                s = time_steps(step)
                mem = torch.cuda.max_memory_allocated() / 2**30
                print(f"  bs {bs:4d}: {s:7.3f} s/step  {bs / s:7.1f} clips/s  "
                      f"peak {mem:5.1f} GB  ({nparams / 1e6:.1f}M params)", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"  bs {bs:4d}: OOM", flush=True)
                del model, opt, z1
                torch.cuda.empty_cache()
                break
            del model, opt, z1
            torch.cuda.empty_cache()
    print("BS-SEARCH DONE", flush=True)


if __name__ == "__main__":
    main()
