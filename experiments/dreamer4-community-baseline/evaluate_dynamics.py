#!/usr/bin/env python3
"""Held-out autoregressive rollout evaluation for the community Dreamer 4 baseline."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def _seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _filmstrip(frames: np.ndarray, label: str, ctx: int) -> np.ndarray:
    strip = np.ascontiguousarray(np.concatenate(list(frames), axis=1))
    boundary = ctx * frames.shape[2]
    cv2.line(strip, (boundary, 0), (boundary, strip.shape[0] - 1), (255, 255, 0), 2)
    cv2.rectangle(strip, (0, 0), (min(strip.shape[1] - 1, 260), 17), (0, 0, 0), -1)
    cv2.putText(strip, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    return strip


def _to_u8(frames: torch.Tensor) -> np.ndarray:
    return (frames.permute(0, 1, 3, 4, 2).clamp(0, 1) * 255.0).byte().cpu().numpy()


def _mse_per_t(pred: torch.Tensor, target: torch.Tensor, ctx: int, horizon: int) -> torch.Tensor:
    delta = pred[:, ctx:ctx + horizon].float() - target[:, ctx:ctx + horizon].float()
    return delta.square().mean(dim=(0, 2, 3, 4))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dreamer4", type=Path, required=True)
    ap.add_argument("--dynamics-checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer-checkpoint", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-sequences", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, str(args.dreamer4 / "dreamer4"))
    from model import Dynamics, pack_bottleneck_to_spatial, temporal_patchify
    from train_dynamics import (
        decode_packed_to_frames,
        load_frozen_tokenizer_from_pt_ckpt,
        make_tau_schedule,
        sample_autoregressive_packed_sequence,
    )
    from wm_dataset import WMDataset, collate_batch

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.dynamics_checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(checkpoint["args"])
    if not bool(cfg.get("use_actions", False)):
        raise AssertionError("dynamics checkpoint is not action-conditioned")

    encoder, decoder, tok_cfg = load_frozen_tokenizer_from_pt_ckpt(
        str(args.tokenizer_checkpoint), device=device
    )
    H = int(tok_cfg.get("H", 128))
    W = int(tok_cfg.get("W", 128))
    C = int(tok_cfg.get("C", 3))
    patch = int(tok_cfg.get("patch", 4))
    n_latents = int(tok_cfg.get("n_latents", 16))
    d_bottleneck = int(tok_cfg.get("d_bottleneck", 32))
    packing = int(cfg.get("packing_factor", 2))
    if (H, W, C) != (64, 64, 3):
        raise AssertionError(f"expected native Memory Maze 64x64 RGB, got {(H, W, C)}")
    if n_latents % packing:
        raise AssertionError("tokenizer latents are not divisible by packing factor")

    n_spatial = n_latents // packing
    dynamics = Dynamics(
        d_model=int(cfg.get("d_model_dyn", 512)),
        d_bottleneck=d_bottleneck,
        d_spatial=d_bottleneck * packing,
        n_spatial=n_spatial,
        n_register=int(cfg.get("n_register", 4)),
        n_agent=int(cfg.get("n_agent", 1)),
        n_heads=int(cfg.get("n_heads", 4)),
        depth=int(cfg.get("dyn_depth", 8)),
        k_max=int(cfg.get("k_max", 8)),
        dropout=float(cfg.get("dropout", 0.0)),
        mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
        time_every=int(cfg.get("time_every", 1)),
        space_mode=str(cfg.get("space_mode", "wm_agent_isolated")),
        scale_pos_embeds=bool(cfg.get("scale_pos_embeds", False)),
    ).to(device)
    dynamics.load_state_dict(checkpoint["dynamics"], strict=True)
    dynamics.eval()

    length = args.ctx + args.horizon
    train_seq_len = int(cfg.get("seq_len", 32))
    if length > train_seq_len:
        raise AssertionError(f"requested {length} frames exceeds trained sequence length {train_seq_len}")
    dataset = WMDataset(
        data_dir=str(args.data_dir),
        frames_dir=str(args.frames_dir),
        seq_len=train_seq_len,
        img_size=H,
        action_dim=16,
        cache_mb=128,
        tasks_json="__none__",
        tasks=["memmaze"],
        verbose=True,
    )
    if len(dataset) < args.n_sequences:
        raise AssertionError(f"held-out dataset only has {len(dataset)} sequences")
    indices = [int((i + 0.5) * len(dataset) / args.n_sequences) for i in range(args.n_sequences)]
    batch = collate_batch([dataset[i] for i in indices])

    obs_u8 = batch["obs"].to(device, non_blocking=True)
    act = batch["act"].to(device, non_blocking=True)
    mask = batch["act_mask"].to(device, non_blocking=True)
    frames = obs_u8[:, :-1].float() / 255.0
    actions = torch.zeros_like(act)
    actions[:, 1:] = act[:, :-1]
    act_mask = torch.zeros_like(mask)
    act_mask[:, 1:] = mask[:, :-1]

    frames = frames[:, :length]
    actions = actions[:, :length]
    act_mask = act_mask[:, :length]
    wrong_actions = actions.clone()
    wrong_actions[:, args.ctx:] = actions.roll(shifts=1, dims=0)[:, args.ctx:]
    wrong_fraction = float(
        (wrong_actions[:, args.ctx:].argmax(dim=-1) != actions[:, args.ctx:].argmax(dim=-1))
        .float().mean().item()
    )
    if wrong_fraction <= 0:
        raise AssertionError("wrong-action control did not change any rollout actions")

    schedule = make_tau_schedule(
        k_max=int(cfg.get("k_max", 8)), schedule="shortcut", d=0.25
    )
    with torch.inference_mode():
        latent, _ = encoder(temporal_patchify(frames, patch))
        packed = pack_bottleneck_to_spatial(latent, n_spatial=n_spatial, k=packing)

        _seed(args.seed, device)
        pred_correct_z = sample_autoregressive_packed_sequence(
            dynamics, z_gt_packed=packed, ctx_length=args.ctx, horizon=args.horizon,
            k_max=int(cfg.get("k_max", 8)), sched=schedule,
            actions=actions, act_mask=act_mask,
        )
        _seed(args.seed, device)
        pred_wrong_z = sample_autoregressive_packed_sequence(
            dynamics, z_gt_packed=packed, ctx_length=args.ctx, horizon=args.horizon,
            k_max=int(cfg.get("k_max", 8)), sched=schedule,
            actions=wrong_actions, act_mask=act_mask,
        )
        pred_correct = decode_packed_to_frames(
            decoder, z_packed=pred_correct_z, H=H, W=W, C=C, patch=patch,
            packing_factor=packing,
        )
        pred_wrong = decode_packed_to_frames(
            decoder, z_packed=pred_wrong_z, H=H, W=W, C=C, patch=patch,
            packing_factor=packing,
        )

    copy_last = frames.clone()
    copy_last[:, args.ctx:] = frames[:, args.ctx - 1:args.ctx].expand(
        -1, args.horizon, -1, -1, -1
    )
    mse_correct_t = _mse_per_t(pred_correct, frames, args.ctx, args.horizon)
    mse_wrong_t = _mse_per_t(pred_wrong, frames, args.ctx, args.horizon)
    mse_copy_t = _mse_per_t(copy_last, frames, args.ctx, args.horizon)
    mse_correct = float(mse_correct_t.mean().item())
    mse_wrong = float(mse_wrong_t.mean().item())
    mse_copy = float(mse_copy_t.mean().item())
    pred_delta = float(
        (pred_correct[:, args.ctx:] - pred_wrong[:, args.ctx:]).float().square().mean().item()
    )

    gt_u8 = _to_u8(frames)
    correct_u8 = _to_u8(pred_correct)
    wrong_u8 = _to_u8(pred_wrong)
    copy_u8 = _to_u8(copy_last)
    rows = []
    for i in range(args.n_sequences):
        rows.extend([
            _filmstrip(gt_u8[i], f"SEQ {i} - HELD-OUT GT", args.ctx),
            _filmstrip(correct_u8[i], "MODEL - CORRECT ACTIONS", args.ctx),
            _filmstrip(wrong_u8[i], "MODEL - WRONG FUTURE ACTIONS", args.ctx),
            _filmstrip(copy_u8[i], "COPY-LAST BASELINE", args.ctx),
            np.zeros((6, length * W, 3), dtype=np.uint8),
        ])
    sheet = np.concatenate(rows[:-1], axis=0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = args.out_dir / "heldout_rollout_sheet.png"
    if not cv2.imwrite(str(sheet_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write {sheet_path}")
    metrics = {
        "dynamics_checkpoint": str(args.dynamics_checkpoint.resolve()),
        "tokenizer_checkpoint": str(args.tokenizer_checkpoint.resolve()),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_elapsed_train_s": float(checkpoint.get("elapsed_train_s", 0.0)),
        "device": str(device),
        "dataset_size": len(dataset),
        "dataset_indices": indices,
        "context_frames": args.ctx,
        "rollout_frames": args.horizon,
        "shortcut_steps": int(schedule["K"]),
        "wrong_action_fraction": wrong_fraction,
        "mse_correct_actions": mse_correct,
        "mse_wrong_actions": mse_wrong,
        "mse_copy_last": mse_copy,
        "correct_over_wrong_mse": mse_correct / max(mse_wrong, 1e-12),
        "correct_over_copy_mse": mse_correct / max(mse_copy, 1e-12),
        "correct_vs_wrong_prediction_mse": pred_delta,
        "psnr_correct_actions_db": -10.0 * math.log10(max(mse_correct, 1e-12)),
        "mse_correct_per_rollout_step": [float(x) for x in mse_correct_t.cpu()],
        "mse_wrong_per_rollout_step": [float(x) for x in mse_wrong_t.cpu()],
        "mse_copy_per_rollout_step": [float(x) for x in mse_copy_t.cpu()],
        "sheet_layout": (
            "each sequence: held-out GT / autoregressive model with correct actions / "
            "matched-noise model with cyclically wrong future actions / copy-last baseline; "
            "cyan line separates 8-frame context from 16-frame rollout"
        ),
    }
    metrics_path = args.out_dir / "heldout_rollout_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"SAVED {sheet_path} {metrics_path}")


if __name__ == "__main__":
    main()
