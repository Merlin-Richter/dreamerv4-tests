#!/usr/bin/env python3
"""48-effective-hour rollout-only mem2mem trainer for community Dreamer 4."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Sampler

HERE = Path(__file__).resolve().parent
EXPECTED_TOKENIZER_SHA256 = "347052fae0212ea2c6b943ae7c28a886298ce551d4155b882084d63a3ea48797"
PRODUCTION_LOCK = {
    "batch_size": 24,
    "num_workers": 4,
    "cache_mb": 128,
    "shard_size": 2_048,
    "window": 32,
    "clip_length": 128,
    "tbptt_frames": 64,
    "n_memory": 8,
    "k_max": 8,
    "bootstrap_start": 5_000,
    "self_fraction": 0.25,
    "d_model": 512,
    "depth": 8,
    "n_heads": 4,
    "n_register": 4,
    "n_agent": 1,
    "time_every": 1,
    "packing_factor": 2,
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "grad_clip": 1.0,
}


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CounterBatchSampler(Sampler[list[int]]):
    """Stateless, resume-exact random batches keyed by optimizer step."""

    def __init__(self, size: int, batch_size: int, start_step: int, stop_step: int, seed: int):
        self.size = int(size)
        self.batch_size = int(batch_size)
        self.start_step = int(start_step)
        self.stop_step = int(stop_step)
        self.seed = int(seed)
        if self.size <= 0 or self.batch_size <= 0:
            raise ValueError("dataset and batch size must be positive")

    def __iter__(self):
        for step in range(self.start_step, self.stop_step):
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed, step])))
            yield rng.integers(0, self.size, size=self.batch_size).tolist()

    def __len__(self):
        return max(0, self.stop_step - self.start_step)


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        sample["_source_window_index"] = torch.tensor(index, dtype=torch.long)
        return sample


def rng_state(generator: torch.Generator):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "rollout": generator.get_state(),
    }


def restore_rng(state, generator: torch.Generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Loading the full checkpoint with map_location="cuda" also moves these
    # serialized ByteTensors to CUDA. Generator state APIs require CPU bytes,
    # including for CUDA generators.
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
    generator.set_state(state["rollout"].cpu())


def atomic_save(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def resolved_config(args):
    return {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "cache_mb": args.cache_mb,
        "shard_size": args.shard_size,
        "window": args.window,
        "clip_length": args.clip_length,
        "tbptt_frames": args.tbptt_frames,
        "n_memory": args.n_memory,
        "k_max": args.k_max,
        "bootstrap_start": args.bootstrap_start,
        "self_fraction": args.self_fraction,
        "d_model": args.d_model,
        "depth": args.depth,
        "n_heads": args.n_heads,
        "n_register": args.n_register,
        "n_agent": args.n_agent,
        "time_every": args.time_every,
        "packing_factor": args.packing_factor,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "optimizer_step": "one per complete 128-frame clip batch",
        "loss_normalization": "each of 6 scored 16-frame new halves weighted 1/6",
        "tbptt_boundary": "after 4 slides (64 newly committed frames), then after final 2 slides",
        "prediction_modes": "per-sequence 50/50 latent-present vs memory-load-bearing",
        "ff9": False,
        "archive": False,
    }


def assert_production_lock(args):
    if args.allow_nonproduction_config:
        return
    actual = resolved_config(args)
    wrong = {key: (actual[key], expected) for key, expected in PRODUCTION_LOCK.items()
             if actual[key] != expected}
    if wrong:
        raise ValueError(f"production-locked fields changed: {wrong}")
    if args.seed != 0:
        raise ValueError("production seed is locked to 0")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dreamer4", type=Path, required=True)
    p.add_argument("--data-dirs", nargs="+", required=True)
    p.add_argument("--frame-dirs", nargs="+", required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--tasks-json", default="__none__")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--resolved-config", type=Path, default=None)
    p.add_argument("--active-ledger", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cache-mb", type=int, default=128)
    p.add_argument("--shard-size", type=int, default=2048)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--clip-length", type=int, default=128)
    p.add_argument("--tbptt-frames", type=int, default=64)
    p.add_argument("--n-memory", type=int, default=8)
    p.add_argument("--k-max", type=int, default=8)
    p.add_argument("--bootstrap-start", type=int, default=5_000)
    p.add_argument("--self-fraction", type=float, default=0.25)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-register", type=int, default=4)
    p.add_argument("--n-agent", type=int, default=1)
    p.add_argument("--time-every", type=int, default=1)
    p.add_argument("--packing-factor", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-hours", type=float, default=48.0)
    p.add_argument("--max-steps", type=int, default=100_000_000)
    p.add_argument("--save-every-hours", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    p.add_argument("--wandb-project", default="dreamer4-memmaze-community")
    p.add_argument("--wandb-run-name", default="memmaze-community-d4-mem2mem")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--allow-nonproduction-config", action="store_true")
    return p


def main():
    args = parser().parse_args()
    assert_production_lock(args)
    if args.max_hours <= 0 or args.save_every_hours <= 0:
        raise ValueError("max-hours and save-every-hours must be positive")
    if args.window * 4 != args.clip_length or args.tbptt_frames != 2 * args.window:
        raise ValueError("locked rollout geometry requires L=4W and TBPTT=2W")

    dreamer4 = args.dreamer4.resolve()
    sys.path.insert(0, str(dreamer4 / "dreamer4"))
    sys.path.insert(0, str(HERE))
    from model import Dynamics, pack_bottleneck_to_spatial, temporal_patchify
    from train_dynamics import load_frozen_tokenizer_from_pt_ckpt
    from wm_dataset import WMDataset, collate_batch
    from rollout import mem2mem_rollout

    tokenizer_sha = sha256(args.tokenizer)
    if tokenizer_sha != EXPECTED_TOKENIZER_SHA256:
        raise RuntimeError(f"unapproved tokenizer {tokenizer_sha}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_everything(args.seed)
    rollout_generator = torch.Generator(device=device).manual_seed(args.seed)

    dataset = WMDataset(
        data_dir=args.data_dirs,
        frames_dir=args.frame_dirs,
        seq_len=args.clip_length,
        img_size=64,
        action_dim=16,
        shard_size=args.shard_size,
        cache_mb=args.cache_mb,
        tasks_json=args.tasks_json,
        tasks=["memmaze"],
        strict_tasks=True,
        verbose=True,
    )
    indexed_dataset = IndexedDataset(dataset)
    seen_source_windows = torch.zeros(len(dataset), dtype=torch.bool)

    encoder, _, tok_args = load_frozen_tokenizer_from_pt_ckpt(
        str(args.tokenizer), device=device, override={"H": 64, "W": 64, "C": 3, "patch": 4}
    )
    n_latents = int(tok_args.get("n_latents", 16))
    d_bottleneck = int(tok_args.get("d_bottleneck", 32))
    if n_latents % args.packing_factor:
        raise ValueError("tokenizer latent count is not divisible by packing factor")
    n_spatial = n_latents // args.packing_factor
    d_spatial = d_bottleneck * args.packing_factor

    # Memory is constructed after all shared parameters by the patched Dynamics,
    # so seed-0 shared initialization is identical to the vanilla arm.
    dynamics = Dynamics(
        d_model=args.d_model,
        d_bottleneck=d_bottleneck,
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        n_register=args.n_register,
        n_agent=args.n_agent,
        n_heads=args.n_heads,
        depth=args.depth,
        k_max=args.k_max,
        n_memory=args.n_memory,
        dropout=0.0,
        mlp_ratio=4.0,
        time_every=args.time_every,
        space_mode="wm_agent_isolated",
        scale_pos_embeds=False,
    ).to(device)
    optimizer = torch.optim.AdamW(
        dynamics.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )
    scaler = GradScaler(device="cuda", enabled=device.type == "cuda")

    step = 0
    active_seconds = 0.0
    wandb_id = None
    if args.resume is not None:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        if payload["resolved_config"] != resolved_config(args):
            raise RuntimeError("resume configuration differs from checkpoint")
        if payload["tokenizer_sha256"] != tokenizer_sha:
            raise RuntimeError("resume tokenizer hash differs")
        dynamics.load_state_dict(payload["dynamics"], strict=True)
        optimizer.load_state_dict(payload["opt"])
        scaler.load_state_dict(payload["scaler"])
        step = int(payload["step"])
        active_seconds = float(payload["active_seconds"])
        wandb_id = payload.get("wandb_id")
        seen_source_windows.copy_(payload["seen_source_windows"].to(torch.bool))
        restore_rng(payload["rng"], rollout_generator)
        print(f"[resume] step={step} active_hours={active_seconds / 3600.0:.6f}", flush=True)

    resolved = resolved_config(args)
    resolved_path = args.resolved_config or args.checkpoint.with_name("resolved-config.json")
    active_ledger = args.active_ledger or args.checkpoint.with_name("active-clock.jsonl")
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    if active_ledger.exists():
        rows = [json.loads(line) for line in active_ledger.read_text().splitlines() if line.strip()]
        if args.resume is None and rows:
            raise RuntimeError(f"active ledger already exists without --resume: {active_ledger}")
        kept = [row for row in rows if int(row["step"]) <= step]
        if kept:
            if int(kept[-1]["step"]) != step:
                raise RuntimeError("active ledger does not reach the resumed optimizer step")
            if abs(float(kept[-1]["active_cumulative_s"]) - active_seconds) > 1e-6:
                raise RuntimeError("active ledger and checkpoint clocks disagree")
        active_ledger.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in kept), encoding="utf-8"
        )
    resolved_path.write_text(json.dumps({
        **resolved,
        "tokenizer_sha256": tokenizer_sha,
        "data_dirs": args.data_dirs,
        "frame_dirs": args.frame_dirs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "active_budget_seconds": args.max_hours * 3600.0,
    }, indent=2, sort_keys=True) + "\n")

    def checkpoint_payload():
        checkpoint_args = vars(args).copy()
        checkpoint_args.update({
            "d_model_dyn": args.d_model,
            "dyn_depth": args.depth,
            "seq_len": args.window,
            "use_actions": True,
            "scale_pos_embeds": False,
        })
        return {
            "step": step,
            "epoch": 0,
            "active_seconds": active_seconds,
            "elapsed_train_s": active_seconds,
            "dynamics": dynamics.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": checkpoint_args,
            "resolved_config": resolved,
            "tokenizer_sha256": tokenizer_sha,
            "wandb_id": wandb_id,
            "rng": rng_state(rollout_generator),
            "seen_source_windows": seen_source_windows,
            "exposure": {
                "optimizer_steps": step,
                "unique_source_windows": int(seen_source_windows.sum()),
                "source_frames_read": step * args.batch_size * (args.clip_length + 1),
                "scored_frames": step * args.batch_size * (args.clip_length - args.window),
                "scored_slides_per_step": 6,
                "target_frames_per_step": args.batch_size * (args.clip_length - args.window),
            },
        }

    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        print(f"[signal] {signum}; checkpointing after the current complete optimizer step", flush=True)
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)

    sampler = CounterBatchSampler(len(dataset), args.batch_size, step, args.max_steps, args.seed)
    loader = DataLoader(
        indexed_dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_batch,
    )

    wandb = None
    if args.wandb_mode != "disabled":
        import wandb as wandb_module
        wandb = wandb_module
        if wandb_id is None:
            wandb_id = wandb.util.generate_id()
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            id=wandb_id,
            resume="allow",
            config={**vars(args), **resolved},
        )

    budget_seconds = args.max_hours * 3600.0
    next_save = (math.floor(active_seconds / (args.save_every_hours * 3600.0)) + 1) * (
        args.save_every_hours * 3600.0
    )
    print(
        f"device={device} params={sum(p.numel() for p in dynamics.parameters()):,} "
        f"dataset={len(dataset):,} batch={args.batch_size} start_step={step} "
        f"active={active_seconds:.2f}/{budget_seconds:.2f}s",
        flush=True,
    )

    for batch in loader:
        if active_seconds >= budget_seconds or stop_requested:
            break
        source_indices = batch.pop("_source_window_index")
        seen_source_windows[source_indices.long()] = True
        frames = batch["obs"][:, :-1].to(device, non_blocking=True).float().div_(255.0)
        raw_actions = batch["act"].to(device, non_blocking=True).clamp(-1, 1)
        raw_masks = batch["act_mask"].to(device, non_blocking=True)
        actions = torch.zeros_like(raw_actions)
        action_masks = torch.zeros_like(raw_masks)
        actions[:, 1:] = raw_actions[:, :-1]
        action_masks[:, 1:] = raw_masks[:, :-1]
        actions.mul_(action_masks)

        # Encode every exact W-frame key independently, matching the accepted
        # vanilla encoder's temporal window convention.  No whole-clip cache.
        encoded_windows = {}
        with torch.no_grad():
            for start in range(0, args.clip_length - args.window + 1, args.window // 2):
                end = start + args.window
                patches = temporal_patchify(frames[:, start:end], 4)
                latent, _ = encoder(patches)
                encoded_windows[start] = pack_bottleneck_to_spatial(
                    latent, n_spatial=n_spatial, k=args.packing_factor
                )

        def window_source(start, end):
            if end - start != args.window:
                raise ValueError("rollout requested a non-W encoder key")
            return encoded_windows[start]

        B = frames.shape[0]
        B_self = int(round(args.self_fraction * B))
        B_self = max(0, min(B - 1, B_self))
        progress = min(1.0, active_seconds / budget_seconds)
        lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        active_wall_start = time.time()
        active_start = time.monotonic()
        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            result = mem2mem_rollout(
                dynamics,
                window_source,
                actions,
                action_masks,
                window=args.window,
                clip_length=args.clip_length,
                tbptt_frames=args.tbptt_frames,
                k_max=args.k_max,
                B_self=B_self,
                step=step,
                bootstrap_start=args.bootstrap_start,
                generator=rollout_generator,
                backward_fn=lambda segment: scaler.scale(segment).backward(),
                detach_boundaries=True,
            )
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(dynamics.parameters(), args.grad_clip)
        else:
            grad_norm = torch.zeros((), device=device)
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        active_duration = time.monotonic() - active_start
        active_wall_end = time.time()
        active_seconds += active_duration
        step += 1
        active_ledger.parent.mkdir(parents=True, exist_ok=True)
        with active_ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "step": step,
                "wall_start_epoch_s": active_wall_start,
                "wall_end_epoch_s": active_wall_end,
                "active_duration_s": active_duration,
                "active_cumulative_s": active_seconds,
            }, sort_keys=True) + "\n")

        if step % args.log_every == 0 or step == 1:
            memory_std = float(result.final_memory.detach().float().std())
            try:
                import resource
                host_max_rss_kib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            except (ImportError, AttributeError):
                host_max_rss_kib = 0.0
            metrics = {
                "loss/total": result.mean_loss,
                "loss/flow_mse": result.flow_mse,
                "loss/bootstrap_mse": result.bootstrap_mse,
                "stats/memory_std": memory_std,
                "stats/memory_only_fraction": result.memory_only_fraction,
                "stats/grad_norm": float(grad_norm),
                "resources/gpu_allocated_gib": (
                    torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0
                ),
                "resources/gpu_reserved_gib": (
                    torch.cuda.max_memory_reserved(device) / (1024 ** 3) if device.type == "cuda" else 0.0
                ),
                "resources/host_max_rss_kib": host_max_rss_kib,
                "stats/scored_frames": step * args.batch_size * (args.clip_length - args.window),
                "stats/unique_source_windows": int(seen_source_windows.sum()),
                "stats/source_frames_read": step * args.batch_size * (args.clip_length + 1),
                "time/active_hours": active_seconds / 3600.0,
                "lr": lr,
            }
            print(
                f"step={step} active_h={active_seconds / 3600.0:.6f} "
                f"loss={result.mean_loss:.6f} flow={result.flow_mse:.6f} "
                f"boot={result.bootstrap_mse:.6f} mem_std={memory_std:.4f} "
                f"grad={float(grad_norm):.4f} lr={lr:.3e}",
                flush=True,
            )
            if wandb is not None:
                wandb.log(metrics, step=step)

        if active_seconds >= next_save or stop_requested:
            payload = checkpoint_payload()
            atomic_save(payload, args.checkpoint)
            snapshot = args.checkpoint.with_name(
                f"step-{step:08d}-active-{active_seconds:012.2f}s.pt"
            )
            atomic_save(payload, snapshot)
            print(f"[checkpoint] {args.checkpoint} step={step} active_s={active_seconds:.2f}", flush=True)
            while next_save <= active_seconds:
                next_save += args.save_every_hours * 3600.0
        if active_seconds >= budget_seconds or stop_requested:
            break

    atomic_save(checkpoint_payload(), args.checkpoint)
    print(
        f"TRAINING STOP step={step} active_seconds={active_seconds:.6f} "
        f"budget_seconds={budget_seconds:.6f} checkpoint={args.checkpoint}",
        flush=True,
    )
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
