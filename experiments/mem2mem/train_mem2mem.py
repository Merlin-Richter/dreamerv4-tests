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
                                     load_tokenizer, encode_frames)
import wlog                                                                     # noqa: E402
from rollout import mem2mem_rollout_loss                                        # noqa: E402


def step_curriculum(p, n_d, warmup, add_every):
    """Number of FINEST step sizes unlocked at training fraction p in [0,1). 1 (only d_min) for the
    warmup, then +1 every ``add_every`` of training, capped at n_d. Finest-first so a coarse step's
    bootstrap target (a one-finer step) is always already trained."""
    if p < warmup:
        return 1
    return min(n_d, 2 + int((p - warmup) / add_every))


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
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clip-len", type=int, default=64, help="Long-clip length fed to the rollout.")
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
                   help="Disable the step-size curriculum (sample all d steps from step 0). "
                        "Default: ramp d finest-first (only d_min for --curr-warmup, then +1 step every "
                        "--curr-add-every of training).")
    p.add_argument("--curr-warmup", type=float, default=0.15,
                   help="Fraction of training with ONLY d_min (pure flow, no bootstrap) before unlocking.")
    p.add_argument("--curr-add-every", type=float, default=0.025,
                   help="After warmup, unlock the next coarser step every this fraction of training.")
    p.add_argument("--max-frames", type=int, default=None, help="Cap rollout length (memory/footprint).")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--max-episodes", type=int, default=None, help="Use only the first N episodes (smoke).")
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

    tok = load_tokenizer(args.tokenizer, device)
    cfg = DynamicsModelConfig(n_actions=n_actions, n_memory=args.n_memory, ff9_k=args.ff9)
    assert cfg.n_memory > 0 and cfg.ff9_k > 0, "mem2mem requires n_memory>0 and ff9>0"
    N = cfg.max_temporal_length
    clip_len = max(args.clip_len, N)
    model = DynamicsModel(cfg).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    ncts = valid_n_ctx(N, clip_len)
    print(f"device={device} params={nparams/1e6:.2f}M n_actions={n_actions} clip_len={clip_len} "
          f"n_ctx choices={ncts} mem2mem_frac={args.mem2mem_frac} "
          f"bootstrap={not (args.no_bootstrap or args.boot_loss_off)} "
          f"(boot_loss_off={args.boot_loss_off}) ff9_norm_flow={args.ff9_norm_flow} "
          f"use_ff9={not args.no_ff9} relay_grad_clip={args.relay_grad_clip}")

    train_ds = ChunkClipDataset(raw, train_idx, clip_len, actions=actions)
    val_ds = ChunkClipDataset(raw, val_idx, clip_len, actions=actions)
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

    tok_T = int(getattr(tok, "config", cfg).max_temporal_length)  # tokenizer temporal window (RoPE table)

    def encode(frames):
        # The tokenizer's temporal RoPE table only spans tok_T frames, so encode the long clip in
        # non-overlapping tok_T-frame blocks (each block gets the same <=tok_T temporal context the
        # tokenizer was trained with) and concatenate the per-frame latents.
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
            outs = [encode_frames(tok, frames[:, i:i + tok_T]) for i in range(0, frames.shape[1], tok_T)]
        return torch.cat(outs, dim=1).float()  # (B, clip_len, n_latents, bottleneck) fp32 for the rollout

    gstep = 0
    last_unlocked = 1
    for epoch in range(args.epochs):
        model.train()
        agg = {"normal": 0.0, "mem2mem": 0.0, "flow": 0.0, "flow_norm": 0.0, "ff9": 0.0,
               "n_m": 0, "n_n": 0, "relay_clipped": 0.0, "relay_hooks": 0.0}
        for batch in train_loader:
            frames, acts = _split_batch(batch, device)
            z1 = encode(frames)
            # step-size curriculum: only d_min while no_bootstrap (winner repro); ramp finest-first
            # otherwise. --boot-loss-off keeps the curriculum (coarse d sampled) but drops the boot term.
            if args.no_bootstrap:
                n_unlocked = 1
            elif args.no_curriculum:
                n_unlocked = None
            else:
                n_unlocked = step_curriculum(gstep / total_steps, model.n_d,
                                             args.curr_warmup, args.curr_add_every)
            last_unlocked = n_unlocked if n_unlocked is not None else model.n_d
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
            opt.step(); sched.step(); gstep += 1

        # --- light val: normal shortcut-forcing loss on a fixed window (monitor) ---
        model.eval()
        vloss, nb = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                frames, acts = _split_batch(batch, device)
                z1 = encode(frames)
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
              f"| train normal: {agg['normal']/nn:.5f} | d_unlocked {last_unlocked}/{model.n_d} "
              f"| lr {opt.param_groups[0]['lr']:.2e}{clip_str}")
        wlog.log({"val/loss_normal": vloss, "train/mem2mem": agg["mem2mem"]/nm,
                  "train/mem2mem_flow": agg["flow"]/nm, "train/mem2mem_flow_norm": agg["flow_norm"]/nm,
                  "train/mem2mem_ff9": agg["ff9"]/nm, "train/relay_clip_frac": relay_clip_frac,
                  "train/normal": agg["normal"]/nn, "train/d_unlocked": last_unlocked,
                  "lr": opt.param_groups[0]["lr"]}, step=epoch)
        torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg)}, args.checkpoint)
    print(f"saved -> {args.checkpoint}")


if __name__ == "__main__":
    main()
