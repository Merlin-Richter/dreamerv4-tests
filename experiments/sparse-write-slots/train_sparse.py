"""Train the sparse write-slots dynamics model (fork of experiments/mem2mem/train_mem2mem.py,
rollout-only, no FF9/bootstrap/curriculum — the no-bootstrap winner recipe with the sparse loss).

Run:
  python -u experiments/sparse-write-slots/train_sparse.py \
    --frames data/gridworldv2.npy --tokenizer checkpoints/gridworld/tokenizer.pt \
    --checkpoint checkpoints/gridworldv2/dynamics_sparse_n8.pt --epochs 50 --batch-size 64 --clip-len 64
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.dynamics_model import DynamicsModelConfig                 # noqa: E402
from training.train_dynamics import (ChunkClipDataset, _split_batch,  # noqa: E402
                                     ensure_latent_cache)
import wlog                                                            # noqa: E402
from model import DynamicsModelSparseWS                                # noqa: E402
from rollout_sparse import sparse_rollout_loss                         # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clip-len", type=int, default=64)
    p.add_argument("--n-memory", type=int, default=4)
    p.add_argument("--tbptt-frames", type=int, default=None, help="Default 2*max_temporal_length.")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    wlog.add_args(p)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    gen = torch.Generator(device=device).manual_seed(args.seed)

    raw = np.load(args.frames, mmap_mode="r")
    n_ep = raw.shape[0] if args.max_episodes is None else min(raw.shape[0], args.max_episodes)
    cand = args.frames.with_name(args.frames.stem + "_actions.npy")
    actions_np = np.load(cand) if cand.is_file() else None
    actions = torch.from_numpy(actions_np).long() if actions_np is not None else None
    n_actions = int(actions_np.max()) + 1 if actions_np is not None else 0

    torch.manual_seed(0)
    perm = torch.randperm(n_ep).numpy()
    torch.manual_seed(args.seed)
    n_val = max(1, int(args.val_fraction * n_ep))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    cache = ensure_latent_cache(args.frames, args.tokenizer, device)
    lat = np.load(cache, mmap_mode="r")
    cfg = DynamicsModelConfig(n_latents=int(lat.shape[2]), bottleneck_dim=int(lat.shape[3]),
                              n_actions=n_actions, n_memory=args.n_memory, ff9_k=0)
    N = cfg.max_temporal_length
    clip_len = max(args.clip_len, N)
    model = DynamicsModelSparseWS(cfg).to(device)
    nparams = sum(q.numel() for q in model.parameters())
    print(f"device={device} params={nparams/1e6:.2f}M n_actions={n_actions} clip_len={clip_len} "
          f"n_sparse={model.SPARSE_N} W={2*model.SPARSE_N} tbptt={args.tbptt_frames or 2*N}")

    train_ds = ChunkClipDataset(lat, train_idx, clip_len, actions=actions)
    val_ds = ChunkClipDataset(lat, val_idx, clip_len, actions=actions)
    lk = dict(num_workers=args.num_workers, pin_memory=(device == "cuda"))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **lk)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **lk)

    wlog.init(args, cfg, project="transformer-mem2mem")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup = max(200, int(0.05 * total_steps)); decay_start = int(0.8 * total_steps)
    emr = 1e-6 / args.lr

    def lr_lambda(s):
        if s < warmup:
            return (s + 1) / warmup
        if s < decay_start:
            return 1.0
        q = (s - decay_start) / max(1, total_steps - decay_start)
        return emr + (1 - emr) * 0.5 * (1 + np.cos(np.pi * q))

    sched = LambdaLR(opt, lr_lambda)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        agg = {"loss": 0.0, "flow": 0.0, "flow_norm": 0.0, "n": 0}
        for batch in train_loader:
            z1, acts = _split_batch(batch, device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                loss, parts = sparse_rollout_loss(model, z1, acts, device=device, gen=gen,
                                                  tbptt_frames=args.tbptt_frames,
                                                  max_frames=args.max_frames)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            agg["loss"] += float(loss.detach()); agg["flow"] += parts["flow"]
            agg["flow_norm"] += parts["flow_norm"]; agg["n"] += 1

        model.eval()
        vloss, nb = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                z1, acts = _split_batch(batch, device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                    a = acts[:, :N] if acts is not None else None
                    vloss += float(model.loss(z1[:, :N], a)); nb += 1
        vloss /= max(1, nb)
        nn_ = max(1, agg["n"])
        print(f"Epoch {epoch+1}/{args.epochs} | val(normal): {vloss:.5f} | "
              f"train sparse: {agg['loss']/nn_:.5f} (flow {agg['flow']/nn_:.4f} "
              f"flow_norm {agg['flow_norm']/nn_:.4f}) | lr {opt.param_groups[0]['lr']:.2e}")
        wlog.log({"val/loss_normal": vloss, "train/sparse": agg["loss"]/nn_,
                  "train/sparse_flow": agg["flow"]/nn_, "train/sparse_flow_norm": agg["flow_norm"]/nn_,
                  "lr": opt.param_groups[0]["lr"]}, step=epoch)
        torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg)}, args.checkpoint)
    print(f"saved -> {args.checkpoint}")


if __name__ == "__main__":
    main()
