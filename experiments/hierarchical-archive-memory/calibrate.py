"""Synthetic production-checkpoint memory/time calibration for archive rollout training."""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import fields
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import ArchiveDynamicsConfig, DynamicsModelArchive  # noqa: E402
from rollout import archive_rollout_backward                   # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--frames", type=int, default=64)
    p.add_argument("--dense-tbptt-frames", type=int, default=64)
    p.add_argument("--archive-interval", type=int, default=16)
    p.add_argument("--archive-per-memory", type=int, default=1)
    p.add_argument("--fast-memory-hide-frac", type=float, default=0.0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(args.resume, map_location="cpu", weights_only=False)
    allowed = {f.name for f in fields(ArchiveDynamicsConfig)}
    raw = {k: v for k, v in payload["config"].items() if k in allowed}
    raw.update(ff9_k=0, archive_interval=args.archive_interval,
               archive_per_memory=args.archive_per_memory,
               archive_compressor_depth=1, archive_compressor_mlp_ratio=2.0,
               archive_max_sets=0, archive_gate_init=1e-3)
    cfg = ArchiveDynamicsConfig(**raw)
    model = DynamicsModelArchive(cfg).to(device).train()
    missing = model.load_state_dict(payload["model_state_dict"], strict=False).missing_keys
    bad = [k for k in missing if not k.startswith(
        ("archive_compressor.", "archive_readers.", "archive_norms.", "archive_gates."))]
    assert not bad, bad

    B, T = args.batch_size, args.frames
    z = torch.randn(B, T, cfg.n_latents, cfg.bottleneck_dim, device=device)
    actions = (torch.randint(0, cfg.n_actions, (B, T), device=device)
               if cfg.n_actions > 0 else None)
    model.zero_grad(set_to_none=True)
    if device == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
        stats = archive_rollout_backward(
            model, z, actions, device=device,
            gen=torch.Generator(device=device).manual_seed(123),
            dense_tbptt_frames=args.dense_tbptt_frames, max_frames=T,
            bootstrap=False, n_d_unlocked=1,
            fast_memory_hide_frac=args.fast_memory_hide_frac)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    comp_grad = sum(float(q.grad.abs().sum()) for q in model.archive_compressor.parameters()
                    if q.grad is not None)
    print(f"device={device} B={B} T={T} dt={dt:.2f}s stats={stats}")
    print(f"compressor_grad_l1={comp_grad:.4e}")
    if device == "cuda":
        print(f"cuda_peak_alloc={torch.cuda.max_memory_allocated()/2**30:.2f}GiB "
              f"peak_reserved={torch.cuda.max_memory_reserved()/2**30:.2f}GiB")


if __name__ == "__main__":
    main()
