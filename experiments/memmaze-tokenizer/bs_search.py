#!/usr/bin/env python3
"""Find the largest tokenizer batch size that fits (+ its throughput) on this GPU.

Builds the AutoEncoder at the Memory-Maze LOCKED config (overridable) and runs a few REAL
fwd+bwd+optimizer steps on synthetic clips at increasing batch sizes, catching CUDA OOM. Reports the
max batch size that fits and clips/s, so the train job's --batch-size is grounded rather than guessed.
No dataset needed -- random uint8 frames reproduce the activation-memory profile. Run with -u on the H100.

    python -u experiments/memmaze-tokenizer/bs_search.py --lpips
"""
import argparse
import sys
import time
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.tokenizer import AutoEncoder, AutoEncoderConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding-dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--n-heads", type=int, default=16)
    ap.add_argument("--n-latents", type=int, default=32)
    ap.add_argument("--bottleneck-dim", type=int, default=16)
    ap.add_argument("--context-length", type=int, default=64)
    ap.add_argument("--lpips", action="store_true", help="Include the LPIPS-VGG term (matches the train config).")
    ap.add_argument("--batch-sizes", type=int, nargs="+",
                    default=[1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128])
    ap.add_argument("--steps", type=int, default=6, help="fwd+bwd steps per batch size (step 0 is warmup).")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no CUDA device -- run this on the H100.")
    device = "cuda"
    torch.set_float32_matmul_precision("high")

    cfg = AutoEncoderConfig(
        embedding_dim=args.embedding_dim, depth=args.depth, n_heads=args.n_heads,
        n_latents=args.n_latents, bottleneck_dim=args.bottleneck_dim,
        max_temporal_length=args.context_length, img_input_H=64, img_input_W=64,
    )
    L = cfg.max_temporal_length
    props = torch.cuda.get_device_properties(0)
    print(f"GPU {props.name} ({props.total_memory / 1e9:.0f} GB) | cfg dim={cfg.embedding_dim} "
          f"depth={cfg.depth} heads={cfg.n_heads} n_lat={cfg.n_latents} bneck={cfg.bottleneck_dim} "
          f"L={L} lpips={args.lpips}", flush=True)

    lpips_fn = None
    if args.lpips:
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
        for p in lpips_fn.parameters():
            p.requires_grad_(False)

    best = None
    for bs in args.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model = AutoEncoder(cfg).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
            x = torch.randint(0, 256, (bs, L, 64, 64, 3), dtype=torch.uint8, device=device).float() / 255.0
            t0 = None
            for s in range(args.steps):
                if s == 1:
                    torch.cuda.synchronize()
                    t0 = time.time()
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    pred = model(x)
                    loss = ((pred - x) ** 2).mean()
                    if lpips_fn is not None:
                        lp = lpips_fn(
                            pred.reshape(bs * L, 64, 64, 3).permute(0, 3, 1, 2),
                            x.reshape(bs * L, 64, 64, 3).permute(0, 3, 1, 2),
                            normalize=True,
                        ).mean()
                        loss = loss + 0.2 * lp
                opt.zero_grad()
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            dt = time.time() - t0
            its = (args.steps - 1) / dt
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"  bs={bs:4d}  OK   peak {peak:5.1f} GB   {its:5.2f} it/s   {its * bs:7.1f} clips/s",
                  flush=True)
            best = bs
            del model, opt, x, pred, loss
        except torch.cuda.OutOfMemoryError:
            print(f"  bs={bs:4d}  OOM", flush=True)
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  bs={bs:4d}  OOM (RuntimeError)", flush=True)
                break
            raise

    print(f"\nMAX batch size that fits: {best}", flush=True)
    if best is not None:
        print("Pick the train --batch-size at or just below this (LPIPS-VGG saturates util, so a "
              "larger batch only *fits* more, it does not speed up -- see HOWTO/cluster.md).", flush=True)


if __name__ == "__main__":
    main()
