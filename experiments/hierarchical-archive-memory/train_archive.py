"""Continue a trained Memory Maze fast-memory model with hierarchical sparse archive memory.

Example (cluster/local paths as appropriate):

    python -u experiments/hierarchical-archive-memory/train_archive.py \
      --frames data/memmaze9x9.npy \
      --tokenizer checkpoints/memmaze/tokenizer.pt \
      --resume checkpoints/memmaze/dynamics_mem2mem_noff9.pt \
      --checkpoint checkpoints/memmaze/dynamics_archive.pt \
      --epochs 50 --batch-size 4 --clip-len 512 --dense-tbptt-frames 64

The optimizer step is one complete long clip: dense graphs are backwarded/freed in bounded blocks,
then accumulated archive-proxy gradients are applied through a deferred compressor VJP.
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import wlog  # noqa: E402
from model import ArchiveDynamicsConfig, DynamicsModelArchive  # noqa: E402
from rollout import archive_rollout_backward                   # noqa: E402
from training.train_dynamics import (                          # noqa: E402
    ChunkClipDataset, _split_batch, ensure_latent_cache,
)


def _cfg_from_resume(payload: dict, args, *, n_actions: int, n_latents: int,
                     bottleneck_dim: int) -> ArchiveDynamicsConfig:
    allowed = {f.name for f in fields(ArchiveDynamicsConfig)}
    raw = {k: v for k, v in payload["config"].items() if k in allowed}
    is_archive = "archive_per_memory" in payload["config"]

    def pick(cli, key, default):
        if cli is not None:
            if is_archive and key in raw and cli != raw[key]:
                raise ValueError(f"cannot change {key} while resuming archive checkpoint "
                                 f"({raw[key]} -> {cli})")
            return cli
        return raw.get(key, default)

    raw.update(
        archive_interval=pick(args.archive_interval, "archive_interval", 16),
        archive_per_memory=pick(args.archive_per_memory, "archive_per_memory", 1),
        archive_compressor_depth=pick(
            args.compressor_depth, "archive_compressor_depth", 1),
        archive_compressor_mlp_ratio=pick(
            args.compressor_mlp_ratio, "archive_compressor_mlp_ratio", 2.0),
        archive_max_sets=pick(args.archive_max_sets, "archive_max_sets", 0),
        archive_gate_init=pick(args.archive_gate_init, "archive_gate_init", 1e-3),
    )
    # This experiment uses rollout-only memory training; no FF9 auxiliary term.
    raw["ff9_k"] = 0
    cfg = ArchiveDynamicsConfig(**raw)
    assert cfg.n_memory > 0, "archive continuation requires a fast-memory checkpoint"
    assert cfg.max_temporal_length == 32, (
        f"version-one archive training is fixed at W=32; checkpoint has {cfg.max_temporal_length}")
    assert cfg.n_actions == n_actions, f"checkpoint n_actions={cfg.n_actions}, dataset={n_actions}"
    assert cfg.n_latents == n_latents and cfg.bottleneck_dim == bottleneck_dim, (
        "tokenizer latent shape does not match dynamics checkpoint")
    return cfg


def _load_weights(model: DynamicsModelArchive, payload: dict) -> str:
    state = payload["model_state_dict"]
    is_archive = any(k.startswith(("archive_compressor.", "archive_readers.",
                                   "archive_norms.", "archive_gates.")) for k in state)
    if is_archive:
        model.load_state_dict(state, strict=True)
        return "archive-resume"

    incompatible = model.load_state_dict(state, strict=False)
    allowed = ("archive_compressor.", "archive_readers.", "archive_norms.", "archive_gates.")
    bad_missing = [k for k in incompatible.missing_keys if not k.startswith(allowed)]
    if bad_missing or incompatible.unexpected_keys:
        raise RuntimeError(f"warm-start mismatch: missing={bad_missing}, "
                           f"unexpected={incompatible.unexpected_keys}")
    return f"base-warm-start ({len(incompatible.missing_keys)} new archive tensors)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--resume", type=Path, required=True,
                   help="Fast-memory or archive checkpoint to continue from.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clip-len", type=int, default=512)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--dense-tbptt-frames", type=int, default=64)
    p.add_argument("--archive-interval", type=int, default=None)
    p.add_argument("--archive-per-memory", type=int, default=None)
    p.add_argument("--compressor-depth", type=int, default=None)
    p.add_argument("--compressor-mlp-ratio", type=float, default=None)
    p.add_argument("--archive-max-sets", type=int, default=None)
    p.add_argument("--archive-gate-init", type=float, default=None)
    p.add_argument("--bootstrap", action="store_true",
                   help="Enable shortcut bootstrap; default is rollout-only winner's pure flow.")
    p.add_argument("--n-d-unlocked", type=int, default=1,
                   help="Number of finest shortcut step sizes sampled (default 1 = d_min only).")
    p.add_argument("--fast-memory-hide-frac", type=float, default=0.0,
                   help="Eligible examples that hide carried old-half fast memory.")
    p.add_argument("--hide-latents-frac", type=float, default=0.5,
                   help="Conditional fraction of fast-memory-hiding examples that also hide latents.")
    p.add_argument("--archive-drop-frac", type=float, default=0.0,
                   help="Optional archive ablation fraction; pathless examples are forbidden.")
    p.add_argument("--relay-grad-clip", type=float, default=None)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    wlog.add_args(p)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    gen = torch.Generator(device=device).manual_seed(args.seed)

    raw_frames = np.load(args.frames, mmap_mode="r")
    n_ep = raw_frames.shape[0] if args.max_episodes is None else min(
        raw_frames.shape[0], args.max_episodes)
    action_path = args.frames.with_name(args.frames.stem + "_actions.npy")
    actions_np = np.load(action_path) if action_path.is_file() else None
    actions = torch.from_numpy(actions_np).long() if actions_np is not None else None
    n_actions = int(actions_np.max()) + 1 if actions_np is not None else 0

    torch.manual_seed(0)
    perm = torch.randperm(n_ep).numpy()
    torch.manual_seed(args.seed)
    n_val = max(1, int(args.val_fraction * n_ep))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    latent_path = ensure_latent_cache(args.frames, args.tokenizer, device)
    lat = np.load(latent_path, mmap_mode="r")
    payload = torch.load(args.resume, map_location="cpu", weights_only=False)
    cfg = _cfg_from_resume(payload, args, n_actions=n_actions,
                           n_latents=int(lat.shape[2]), bottleneck_dim=int(lat.shape[3]))
    model = DynamicsModelArchive(cfg).to(device)
    load_mode = _load_weights(model, payload)

    clip_len = max(args.clip_len, cfg.max_temporal_length + cfg.max_temporal_length // 2)
    if clip_len > lat.shape[1]:
        raise ValueError(f"clip_len={clip_len} exceeds episode length {lat.shape[1]}")
    if args.max_frames is not None and args.max_frames > clip_len:
        raise ValueError("--max-frames cannot exceed --clip-len")
    half = cfg.max_temporal_length // 2
    if args.dense_tbptt_frames % half:
        raise ValueError(f"dense TBPTT must be a multiple of half-window {half}")

    train_ds = ChunkClipDataset(lat, train_idx, clip_len, actions=actions)
    val_ds = ChunkClipDataset(lat, val_idx, clip_len, actions=actions)
    loader_kw = dict(num_workers=args.num_workers, pin_memory=(device == "cuda"))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            **loader_kw)
    if len(train_loader) == 0:
        raise ValueError("empty training loader; reduce batch size or clip length")

    nparams = sum(q.numel() for q in model.parameters())
    archive_params = sum(q.numel() for n, q in model.named_parameters() if n.startswith("archive_"))
    print(f"device={device} load={load_mode} params={nparams/1e6:.2f}M "
          f"archive={archive_params/1e6:.2f}M W={cfg.max_temporal_length} "
          f"N={cfg.archive_interval} M={cfg.n_memory} R={cfg.archive_per_memory} "
          f"clip={clip_len} dense_tbptt={args.dense_tbptt_frames} bs={args.batch_size} "
          f"fast_hide={args.fast_memory_hide_frac} hide_lat={args.hide_latents_frac} "
          f"archive_drop={args.archive_drop_frac}")

    wlog.init(args, cfg, project="transformer-archive-memory")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup = max(200, int(0.05 * total_steps))
    decay_start = int(0.8 * total_steps)
    eta_min_ratio = 1e-6 / args.lr

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        if step < decay_start:
            return 1.0
        q = (step - decay_start) / max(1, total_steps - decay_start)
        return eta_min_ratio + (1 - eta_min_ratio) * 0.5 * (1 + np.cos(np.pi * q))

    sched = LambdaLR(opt, lr_lambda)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        keys = ("loss", "flow", "flow_norm", "n_slides", "n_archives", "n_archives_used",
                "clean_frac", "noise_frac", "fast_hide_frac", "hide_latents_frac",
                "archive_drop_frac", "relay_clip_frac")
        agg = {k: 0.0 for k in keys}; nb = 0
        for batch in train_loader:
            z1, acts = _split_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                stats = archive_rollout_backward(
                    model, z1, acts, device=device, gen=gen,
                    dense_tbptt_frames=args.dense_tbptt_frames,
                    max_frames=args.max_frames, bootstrap=args.bootstrap,
                    n_d_unlocked=args.n_d_unlocked,
                    fast_memory_hide_frac=args.fast_memory_hide_frac,
                    hide_latents_frac=args.hide_latents_frac,
                    archive_drop_frac=args.archive_drop_frac,
                    relay_grad_clip=args.relay_grad_clip)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); global_step += 1
            for k in keys:
                agg[k] += stats[k]
            nb += 1
            wlog.log({f"train/{k}": stats[k] for k in keys} |
                     {"train/grad_norm": float(grad_norm), "lr": opt.param_groups[0]["lr"]},
                     step=global_step)

        # Fixed-window local monitor.  ff9_k is zero in the archive config, so this is diffusion only.
        model.eval()
        val_loss = 0.0; nvb = 0
        with torch.no_grad():
            for batch in val_loader:
                z1, acts = _split_batch(batch, device)
                W = cfg.max_temporal_length
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                    val_loss += float(model.loss(z1[:, :W], acts[:, :W] if acts is not None else None))
                nvb += 1
                if nvb >= args.val_batches:
                    break
        val_loss /= max(1, nvb)
        den = max(1, nb)
        means = {k: agg[k] / den for k in keys}
        print(f"Epoch {epoch+1}/{args.epochs} | val(local) {val_loss:.5f} | "
              f"archive rollout {means['loss']:.5f} | archives {means['n_archives']:.1f} "
              f"used {means['n_archives_used']:.1f} | hide {means['fast_hide_frac']:.3f}/"
              f"{means['hide_latents_frac']:.3f} | relay_clip {means['relay_clip_frac']:.3f} "
              f"| lr {opt.param_groups[0]['lr']:.2e}")
        wlog.log({"val/loss_local": val_loss} |
                 {f"epoch/{k}": v for k, v in means.items()}, step=global_step)
        torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg)}, args.checkpoint)

    print(f"saved -> {args.checkpoint}")


if __name__ == "__main__":
    main()
