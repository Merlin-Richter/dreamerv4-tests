"""Train the dynamics model with the mem->mem rollout signal (task: test-new-memory-training).

Standalone trainer (src/ untouched): imports the unmodified DynamicsModel + train_dynamics' data /
tokenizer helpers, and runs a 50/50 mix of (a) the normal shortcut-forcing loss on a window and
(b) the mem->mem sliding rollout (experiments/mem2mem/rollout.py). The mem->mem rollout is what
teaches the model to construct memory tokens from prior memory tokens (verified by test_autograd.py).

Run (CUDA via repo venv), e.g.:
  venv/Scripts/python.exe -u experiments/mem2mem/train_mem2mem.py \
    --frames data/gridworld.npy --tokenizer checkpoints/gridworld/tokenizer.pt \
    --checkpoint checkpoints/gridworld/dynamics_mem2mem.pt --epochs 50 --batch-size 64 --clip-len 64
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataclasses import asdict                                                  # noqa: E402
from models.dynamics_model import DynamicsModel, DynamicsModelConfig            # noqa: E402
from training.train_dynamics import (ChunkClipDataset, _split_batch,            # noqa: E402
                                     ensure_latent_cache)
import wlog                                                                     # noqa: E402
from rollout import mem2mem_rollout_loss                                        # noqa: E402


def step_curriculum(p, n_d, warmup, add_every):
    """Number of FINEST step sizes unlocked at training fraction p in [0,1). 1 (only d_min) for the
    warmup, then +1 every ``add_every`` of training, capped at n_d. Finest-first so a coarse step's
    bootstrap target (a one-finer step) is always already trained."""
    if p < warmup:
        return 1
    return min(n_d, 2 + int((p - warmup) / add_every))


def wallclock_curriculum(hours, *, warmup_hours, full_hours, max_unlocked):
    """Finest-first unlock count for a wall-clock curriculum.

    ``max_unlocked-1`` coarse targets are introduced evenly from ``warmup_hours`` (first unlock)
    through ``full_hours`` (final unlock). For the Memory Maze K=4 continuation this produces
    1/128 only for hour 0..1, then unlocks 1/64, 1/32, 1/16, 1/8, and finally 1/4 at hour 6.
    """
    if max_unlocked <= 1 or hours < warmup_hours:
        return 1
    if hours >= full_hours:
        return max_unlocked
    if max_unlocked == 2:
        return 2
    # There are max_unlocked-1 coarse targets, hence max_unlocked-2 intervals between the first
    # coarse unlock at warmup_hours and the final coarse unlock at full_hours.
    interval = (full_hours - warmup_hours) / (max_unlocked - 2)
    return min(max_unlocked, 2 + int((hours - warmup_hours) / interval))


def valid_n_ctx(N, clip_len):
    """Powers of two in [4, N] that fit at least one slide of the clip (need >= 1.5*n_ctx frames)."""
    out, w = [], 4
    while w <= N:
        if w + w // 2 <= clip_len:
            out.append(w)
        w *= 2
    return out or [min(4, N)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--resume", type=Path, default=None,
                   help="Load model weights to continue training FROM (chained long runs). Config "
                        "comes from the CLI (must match the checkpoint's architecture); optimizer/LR "
                        "state restarts.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clip-len", type=int, default=64, help="Long-clip length fed to the rollout.")
    p.add_argument("--context-length", type=int, default=None,
                   help="Model temporal window (max_temporal_length). Default: dataclass default.")
    p.add_argument("--embedding-dim", type=int, default=None,
                   help="Transformer width (default: dataclass default; env-dependent).")
    p.add_argument("--depth", type=int, default=None, help="Depth; multiple of 3.")
    p.add_argument("--n-heads", type=int, default=None, help="Attention heads.")
    p.add_argument("--n-registers", type=int, default=None, help="Scratch register tokens/frame.")
    p.add_argument("--n-memory", type=int, default=4)
    p.add_argument("--ff9", type=int, default=3, metavar="K")
    p.add_argument("--mem2mem-frac", type=float, default=0.5, help="P(batch uses mem->mem vs normal).")
    p.add_argument("--no-bootstrap", action="store_true",
                   help="Disable the shortcut bootstrap distillation in the rollout new-half loss "
                        "(finest-step flow only). Default: bootstrap ON (matches the normal diffusion loss). "
                        "Also forces d_min-only sampling (uniform tau) — the rollout-only winner config.")
    p.add_argument("--boot-loss-off", action="store_true",
                   help="CONTROL ARM for the bootstrap A/B: keep the curriculum d-sampling (snapped-tau "
                        "grid, coarse steps present) but disable the bootstrap LOSS term (coarse-d tokens "
                        "get flow MSE). Holds the tau distribution identical to the bootstrap run so the "
                        "ONLY difference is the bootstrap gradient. Unlike --no-bootstrap, does NOT force "
                        "d_min-only. Pair with --ff9-norm-flow.")
    p.add_argument("--ff9-norm-flow", action="store_true",
                   help="Normalize the FF9 term by the pure d_min FLOW magnitude (not the mixed "
                        "flow+bootstrap diffusion mean), keeping FF9's effective weight invariant to the "
                        "bootstrap. Required for a fair bootstrap A/B; default keeps the mixed mean "
                        "(faithful to model.loss).")
    p.add_argument("--no-ff9", action="store_true",
                   help="Drop the FF9 sufficiency term: memory is trained ONLY by the rollout flow loss "
                        "(50/50 clean/noise; the noise-mode flow loss is the memory signal). Ablation.")
    p.add_argument("--relay-grad-clip", type=float, default=None, metavar="C",
                   help="Per-hop relay GRADIENT normalizer: scale each carried memory tensor's gradient "
                        "DOWN per batch element so ||grad_b|| <= C (scale-down only). Combats the backward "
                        "explosion through the mem relay (~2-3x/hop at init, catastrophic for small "
                        "windows). Default None = OFF (byte-identical). Training-only; forward/inference "
                        "unchanged. Logs the per-epoch clip fraction.")
    p.add_argument("--no-curriculum", action="store_true",
                   help="Disable the step-size curriculum (sample every supported K>=K_min step from "
                        "step 0). "
                        "Default: ramp d finest-first (only d_min for --curr-warmup, then +1 step every "
                        "--curr-add-every of training).")
    p.add_argument("--curr-warmup", type=float, default=0.15,
                   help="Fraction of training with ONLY d_min (pure flow, no bootstrap) before unlocking.")
    p.add_argument("--curr-add-every", type=float, default=0.025,
                   help="After warmup, unlock the next coarser step every this fraction of training.")
    p.add_argument("--wallclock-hours", type=float, default=0.0,
                   help="Positive value enables a wall-clock-bounded continuation. Training stops after "
                        "this many active optimizer-step hours; LR and the optional wall-clock shortcut "
                        "curriculum use the same clock. Use a generous outer SLURM allocation.")
    p.add_argument("--curr-warmup-hours", type=float, default=1.0,
                   help="Wall-clock curriculum: active hours with d_min only; first coarse d unlocks here.")
    p.add_argument("--curr-full-hours", type=float, default=6.0,
                   help="Wall-clock curriculum: active hour at which --curr-max-unlocked is reached.")
    p.add_argument("--curr-max-unlocked", type=int, default=None,
                   help="Maximum number of finest supported d targets to unlock. Default: every supported "
                        "target (K>=min_sampling_steps; six targets for K_max=128, K_min=4).")
    p.add_argument("--checkpoint-every-hours", type=float, default=1.0,
                   help="During wall-clock training, overwrite the continuation checkpoint at this active-"
                        "hour cadence, in addition to epoch-end and unlock-boundary saves.")
    p.add_argument("--max-frames", type=int, default=None, help="Cap rollout length (memory/footprint).")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--max-episodes", type=int, default=None, help="Use only the first N episodes (smoke).")
    p.add_argument("--num-workers", type=int, default=4)
    wlog.add_args(p)
    args = p.parse_args()
    if args.wallclock_hours < 0:
        p.error("--wallclock-hours must be non-negative")
    if args.wallclock_hours > 0:
        if not (0 <= args.curr_warmup_hours < args.curr_full_hours <= args.wallclock_hours):
            p.error("wall-clock curriculum requires 0 <= --curr-warmup-hours < "
                    "--curr-full-hours <= --wallclock-hours")
        if args.checkpoint_every_hours <= 0:
            p.error("--checkpoint-every-hours must be positive")

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

    # Latent disk cache (train_dynamics.ensure_latent_cache): tokenizer encodes the dataset ONCE per
    # (frames, tokenizer) combo; training streams mmapped fp16 latents and never holds the tokenizer.
    cache = ensure_latent_cache(args.frames, args.tokenizer, device)
    lat = np.load(cache, mmap_mode="r")  # (N, T, n_latents, bottleneck_dim) fp16
    dims = {k: v for k, v in dict(max_temporal_length=args.context_length,
                                  embedding_dim=args.embedding_dim, depth=args.depth,
                                  n_heads=args.n_heads, n_registers=args.n_registers).items()
            if v is not None}
    cfg = DynamicsModelConfig(n_latents=int(lat.shape[2]), bottleneck_dim=int(lat.shape[3]),
                              n_actions=n_actions, n_memory=args.n_memory, ff9_k=args.ff9, **dims)
    assert cfg.n_memory > 0 and cfg.ff9_k > 0, "mem2mem requires n_memory>0 and ff9>0"
    N = cfg.max_temporal_length
    clip_len = max(args.clip_len, N)
    model = DynamicsModel(cfg).to(device)
    if args.resume is not None:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        print(f"[resume] loaded weights from {args.resume}")
    nparams = sum(p.numel() for p in model.parameters())
    ncts = valid_n_ctx(N, clip_len)
    max_supported = getattr(model, "n_train_d", model.n_d)
    max_unlocked = max_supported if args.curr_max_unlocked is None else args.curr_max_unlocked
    if not (1 <= max_unlocked <= max_supported):
        p.error(f"--curr-max-unlocked must be in [1, {max_supported}], got {max_unlocked}")
    print(f"device={device} params={nparams/1e6:.2f}M n_actions={n_actions} clip_len={clip_len} "
          f"n_ctx choices={ncts} mem2mem_frac={args.mem2mem_frac} "
          f"bootstrap={not (args.no_bootstrap or args.boot_loss_off)} "
          f"(boot_loss_off={args.boot_loss_off}) ff9_norm_flow={args.ff9_norm_flow} "
          f"use_ff9={not args.no_ff9} relay_grad_clip={args.relay_grad_clip}")
    if args.wallclock_hours > 0:
        print(f"[wallclock] train={args.wallclock_hours:g}h d_min_only={args.curr_warmup_hours:g}h "
              f"full_unlock={args.curr_full_hours:g}h max_unlocked={max_unlocked}/{max_supported} "
              f"checkpoint_every={args.checkpoint_every_hours:g}h")

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

    sched = None if args.wallclock_hours > 0 else LambdaLR(opt, lr_lambda)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    active_seconds = 0.0
    next_checkpoint_hour = args.checkpoint_every_hours

    def save_checkpoint(unlocked):
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "continuation_state": {
                "active_hours": active_seconds / 3600.0,
                "optimizer_steps": gstep,
                "n_d_unlocked": unlocked,
            },
        }, args.checkpoint)

    def set_wallclock_lr(hours):
        # Rebuild fresh AdamW moments gently during the d_min-only hour, hold peak LR for the bulk,
        # then cosine-cool over the final 20% of active training.
        if hours < args.curr_warmup_hours:
            q = hours / max(args.curr_warmup_hours, 1e-12)
            scale = emr + (1 - emr) * q
        else:
            decay_hour = 0.8 * args.wallclock_hours
            if hours < decay_hour:
                scale = 1.0
            else:
                q = (hours - decay_hour) / max(args.wallclock_hours - decay_hour, 1e-12)
                q = min(1.0, max(0.0, q))
                scale = emr + (1 - emr) * 0.5 * (1 + np.cos(np.pi * q))
        for group in opt.param_groups:
            group["lr"] = args.lr * scale

    gstep = 0
    last_unlocked = 1
    stop = False
    for epoch in range(args.epochs):
        model.train()
        agg = {"normal": 0.0, "mem2mem": 0.0, "flow": 0.0, "flow_norm": 0.0, "ff9": 0.0,
               "n_m": 0, "n_n": 0, "relay_clipped": 0.0, "relay_hooks": 0.0}
        for batch in train_loader:
            step_started = time.monotonic()
            active_hours = active_seconds / 3600.0
            if args.wallclock_hours > 0:
                if active_hours >= args.wallclock_hours:
                    stop = True
                    break
                set_wallclock_lr(active_hours)
            z1, acts = _split_batch(batch, device)   # batch IS fp32 latents (from the cache)
            # step-size curriculum: only d_min while no_bootstrap (winner repro); ramp finest-first
            # otherwise. --boot-loss-off keeps the curriculum (coarse d sampled) but drops the boot term.
            if args.no_bootstrap:
                n_unlocked = 1
            elif args.wallclock_hours > 0:
                n_unlocked = wallclock_curriculum(
                    active_hours, warmup_hours=args.curr_warmup_hours,
                    full_hours=args.curr_full_hours, max_unlocked=max_unlocked)
            elif args.no_curriculum:
                n_unlocked = None
            else:
                n_unlocked = step_curriculum(gstep / total_steps, max_supported,
                                             args.curr_warmup, args.curr_add_every)
            current_unlocked = n_unlocked if n_unlocked is not None else max_supported
            if current_unlocked != last_unlocked:
                save_checkpoint(last_unlocked)
                print(f"[curriculum] active={active_hours:.3f}h unlock "
                      f"{last_unlocked}->{current_unlocked}/{max_supported}")
            last_unlocked = current_unlocked
            use_boot = not (args.no_bootstrap or args.boot_loss_off)
            use_m2m = torch.rand(1, generator=gen, device=device).item() < args.mem2mem_frac
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                if use_m2m:
                    W = ncts[torch.randint(len(ncts), (1,), generator=gen, device=device).item()]
                    loss, parts = mem2mem_rollout_loss(model, z1, acts, n_ctx=W, device=device,
                                                       gen=gen, max_frames=args.max_frames,
                                                       bootstrap=use_boot,
                                                       n_d_unlocked=n_unlocked,
                                                       use_ff9=not args.no_ff9,
                                                       ff9_norm_flow=args.ff9_norm_flow,
                                                       relay_grad_clip=args.relay_grad_clip)
                    agg["mem2mem"] += float(loss.detach()); agg["n_m"] += 1
                    agg["flow"] += parts["flow"]; agg["ff9"] += parts["ff9"]
                    agg["flow_norm"] += parts["flow_norm"]
                else:
                    off = int(torch.randint(0, clip_len - N + 1, (1,), generator=gen, device=device))
                    a = acts[:, off:off + N] if acts is not None else None
                    loss = model.loss(z1[:, off:off + N], a)
                    agg["normal"] += float(loss.detach()); agg["n_n"] += 1
            opt.zero_grad()
            loss.backward()
            if use_m2m and args.relay_grad_clip is not None:
                st = getattr(model, "_relay_clip_stats", None)
                if st:
                    agg["relay_clipped"] += st["clipped"]; agg["relay_hooks"] += st["hooks"]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
            gstep += 1
            active_seconds += time.monotonic() - step_started
            if args.wallclock_hours > 0:
                active_hours = active_seconds / 3600.0
                if active_hours >= next_checkpoint_hour:
                    save_checkpoint(last_unlocked)
                    print(f"[checkpoint] active={active_hours:.3f}h -> {args.checkpoint}")
                    while next_checkpoint_hour <= active_hours:
                        next_checkpoint_hour += args.checkpoint_every_hours
                if active_hours >= args.wallclock_hours:
                    stop = True
                    break

        # --- light val: normal shortcut-forcing loss on a fixed window (monitor) ---
        model.eval()
        vloss, nb = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                z1, acts = _split_batch(batch, device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                    a = acts[:, :N] if acts is not None else None
                    vloss += float(model.loss(z1[:, :N], a)); nb += 1
        vloss /= max(1, nb)
        nm, nn = max(1, agg["n_m"]), max(1, agg["n_n"])
        relay_clip_frac = agg["relay_clipped"] / max(1.0, agg["relay_hooks"])
        clip_str = f" | relay_clip {relay_clip_frac:.3f}" if args.relay_grad_clip is not None else ""
        print(f"Epoch {epoch+1}/{args.epochs} | val(normal): {vloss:.5f} | "
              f"train mem2mem: {agg['mem2mem']/nm:.5f} (flow {agg['flow']/nm:.4f} "
              f"flow_norm {agg['flow_norm']/nm:.4f} ff9 {agg['ff9']/nm:.4f}) "
              f"| train normal: {agg['normal']/nn:.5f} | d_unlocked {last_unlocked}/{max_supported} "
              f"| lr {opt.param_groups[0]['lr']:.2e}{clip_str}")
        wlog.log({"val/loss_normal": vloss, "train/mem2mem": agg["mem2mem"]/nm,
                  "train/mem2mem_flow": agg["flow"]/nm, "train/mem2mem_flow_norm": agg["flow_norm"]/nm,
                  "train/mem2mem_ff9": agg["ff9"]/nm, "train/relay_clip_frac": relay_clip_frac,
                  "train/normal": agg["normal"]/nn, "train/d_unlocked": last_unlocked,
                  "lr": opt.param_groups[0]["lr"]}, step=epoch)
        save_checkpoint(last_unlocked)
        if stop:
            break
    print(f"saved -> {args.checkpoint}")


if __name__ == "__main__":
    main()
