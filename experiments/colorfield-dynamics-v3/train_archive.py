"""Matched-fork ColorField trainer for W=16 hierarchical archive memory.

Warm-starts the shared rollout-only/no-FF9 fast-memory checkpoint, adds the
experiment-local compressor/readers from ``hierarchical-archive-memory``, and
continues for an exact optimizer-step count.  Every rollout window is W=16 and
advances by eight frames; smaller sampled contexts are forbidden.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = ROOT / "experiments" / "hierarchical-archive-memory"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ARCHIVE_DIR))
sys.path.insert(0, str(ROOT))

from model import ArchiveDynamicsConfig, DynamicsModelArchive  # noqa: E402
from rollout import archive_rollout_backward                   # noqa: E402
from autoresearch.editable.train import RandomClipDataset, load_split  # noqa: E402

W_PIN = 16
N_ACTIONS = 5


def config_from_base(payload: dict, args) -> ArchiveDynamicsConfig:
    allowed = {f.name for f in fields(ArchiveDynamicsConfig)}
    raw = {k: v for k, v in payload["config"].items() if k in allowed and k != "dtype"}
    raw.update(
        ff9_k=0,
        archive_interval=args.archive_interval,
        archive_per_memory=args.archive_per_memory,
        archive_compressor_depth=args.compressor_depth,
        archive_compressor_mlp_ratio=args.compressor_mlp_ratio,
        archive_max_sets=args.archive_max_sets,
        archive_gate_init=args.archive_gate_init,
    )
    cfg = ArchiveDynamicsConfig(**raw)
    assert cfg.max_temporal_length == W_PIN, (
        f"ColorField archive window must be W=16, got {cfg.max_temporal_length}")
    assert cfg.archive_interval == W_PIN, (
        f"matched v3 recipe compresses each complete 16-frame memory segment, got "
        f"N={cfg.archive_interval}")
    assert cfg.n_actions == N_ACTIONS, (cfg.n_actions, N_ACTIONS)
    assert cfg.n_memory > 0 and cfg.ff9_k == 0
    return cfg


def load_base_weights(model: DynamicsModelArchive, payload: dict) -> None:
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
    prefixes = ("archive_compressor.", "archive_readers.", "archive_norms.", "archive_gates.")
    bad_missing = [k for k in incompatible.missing_keys if not k.startswith(prefixes)]
    if bad_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"archive warm-start mismatch: missing={bad_missing}, "
            f"unexpected={incompatible.unexpected_keys}")
    print(f"[archive] warm-started base; initialized {len(incompatible.missing_keys)} archive tensors",
          flush=True)


def main() -> None:
    t0 = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/colorfield"))
    p.add_argument("--val", type=Path, default=Path("data/colorfield_val"))
    p.add_argument("--tokenizer", type=Path, default=Path("checkpoints/colorfield/tokenizer.pt"))
    p.add_argument("--resume", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--budget-s", type=float, required=True)
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--sched-steps", type=int, default=None,
                   help="LR-schedule horizon; max-steps remains an independent safety cap.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--clip-len", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dense-tbptt-frames", type=int, default=32)
    p.add_argument("--archive-interval", type=int, default=16)
    p.add_argument("--archive-per-memory", type=int, default=1)
    p.add_argument("--compressor-depth", type=int, default=1)
    p.add_argument("--compressor-mlp-ratio", type=float, default=2.0)
    p.add_argument("--archive-max-sets", type=int, default=0)
    p.add_argument("--archive-gate-init", type=float, default=1e-3)
    p.add_argument("--fast-memory-hide-frac", type=float, default=0.25)
    p.add_argument("--hide-latents-frac", type=float, default=0.5)
    p.add_argument("--archive-drop-frac", type=float, default=0.0)
    p.add_argument("--relay-grad-clip", type=float, default=None)
    p.add_argument("--val-batches", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-every", type=int, default=1000)
    args = p.parse_args()

    assert args.max_steps > 0
    assert args.clip_len >= 2 * W_PIN and args.clip_len % (W_PIN // 2) == 0
    assert args.dense_tbptt_frames % (W_PIN // 2) == 0
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    gen = torch.Generator(device=device).manual_seed(args.seed)

    lat, actions, cache = load_split(args.data, args.tokenizer)
    val_lat, val_actions, _ = load_split(args.val, args.tokenizer)
    payload = torch.load(args.resume, map_location="cpu", weights_only=False)
    cfg = config_from_base(payload, args)
    model = DynamicsModelArchive(cfg).to(device)
    load_base_weights(model, payload)

    train_ds = RandomClipDataset(lat, actions, args.clip_len, random_offsets=True)
    val_ds = RandomClipDataset(val_lat, val_actions, args.clip_len, random_offsets=False)
    loader_kw = dict(num_workers=args.num_workers, pin_memory=use_amp)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=len(train_ds) >= args.batch_size, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)
    assert len(train_loader) > 0

    nparams = sum(q.numel() for q in model.parameters())
    archive_params = sum(q.numel() for n, q in model.named_parameters() if n.startswith("archive_"))
    print(
        f"device={device} params={nparams/1e6:.2f}M archive={archive_params/1e6:.2f}M "
        f"cache={cache.name} fixed_n_ctx=[16] W=16 slide=8 N={cfg.archive_interval} "
        f"clip={args.clip_len} bs={args.batch_size} ff9=OFF "
        f"fast_hide={args.fast_memory_hide_frac}/{args.hide_latents_frac} "
        f"max_steps={args.max_steps} budget_s={args.budget_s}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched_steps = args.sched_steps or args.max_steps
    warmup = max(10, min(200, int(0.1 * sched_steps)))
    decay_start = int(0.8 * sched_steps)
    eta_min_ratio = 1e-6 / args.lr

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if step < decay_start:
            return 1.0
        q = min(1.0, (step - decay_start) / max(1, sched_steps - decay_start))
        return eta_min_ratio + (1 - eta_min_ratio) * 0.5 * (1 + np.cos(np.pi * q))

    sched = LambdaLR(opt, lr_lambda)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg)}, args.checkpoint)

    step = 0
    budget_hit = False
    epoch = 0
    keys = ("loss", "flow", "flow_norm", "n_slides", "n_archives", "n_archives_used",
            "clean_frac", "noise_frac", "fast_hide_frac", "hide_latents_frac",
            "archive_drop_frac", "relay_clip_frac")
    while step < args.max_steps and not budget_hit:
        epoch += 1
        model.train()
        agg = {k: 0.0 for k in keys}; nb = 0
        for z1, acts in train_loader:
            z1, acts = z1.to(device), acts.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                stats = archive_rollout_backward(
                    model, z1, acts, device=device, gen=gen,
                    dense_tbptt_frames=args.dense_tbptt_frames,
                    bootstrap=False, n_d_unlocked=1,
                    fast_memory_hide_frac=args.fast_memory_hide_frac,
                    hide_latents_frac=args.hide_latents_frac,
                    archive_drop_frac=args.archive_drop_frac,
                    relay_grad_clip=args.relay_grad_clip)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            for k in keys:
                agg[k] += stats[k]
            nb += 1
            if args.save_every > 0 and step % args.save_every == 0:
                save()
                print(f"[archive] step={step} elapsed={time.perf_counter()-t0:.1f}s "
                      f"loss={stats['loss']:.5f} archives={stats['n_archives']:.1f}/"
                      f"{stats['n_archives_used']:.1f}", flush=True)
            if step >= args.max_steps:
                break
            if time.perf_counter() - t0 >= args.budget_s:
                budget_hit = True
                break

        vloss = float("nan")
        if not budget_hit:
            model.eval(); vsum = 0.0; nvb = 0
            with torch.no_grad():
                for z1, acts in val_loader:
                    z1, acts = z1.to(device), acts.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                        vsum += float(model.loss(z1[:, :W_PIN], acts[:, :W_PIN]))
                    nvb += 1
                    if nvb >= args.val_batches:
                        break
            vloss = vsum / max(1, nvb)
        den = max(1, nb)
        print(f"Epoch {epoch} | steps {step} | elapsed {time.perf_counter()-t0:.1f}s | "
              f"val(local) {vloss:.5f} | archive rollout {agg['loss']/den:.5f} | "
              f"archives {agg['n_archives']/den:.1f}/{agg['n_archives_used']/den:.1f} | "
              f"lr {opt.param_groups[0]['lr']:.2e}", flush=True)
        save()

    save()
    elapsed = time.perf_counter() - t0
    if budget_hit:
        print(f"BUDGET_STOP step={step} elapsed={elapsed:.1f}", flush=True)
    else:
        print(f"MAX_STEPS_DONE step={step} elapsed={elapsed:.1f}", flush=True)
    print(f"saved -> {args.checkpoint}", flush=True)


if __name__ == "__main__":
    main()
