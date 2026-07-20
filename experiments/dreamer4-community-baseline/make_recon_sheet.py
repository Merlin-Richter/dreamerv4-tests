#!/usr/bin/env python3
"""Render held-out Memory Maze target/reconstruction/error filmstrips."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def filmstrip(frames: np.ndarray) -> np.ndarray:
    return np.concatenate(list(frames), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dreamer4", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--raw-eval", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-sequences", type=int, default=4)
    ap.add_argument("--length", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, str(args.dreamer4 / "dreamer4"))
    from model import temporal_patchify, temporal_unpatchify
    from train_dynamics import load_frozen_tokenizer_from_pt_ckpt

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoder, decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(
        str(args.checkpoint), device=device
    )
    H = int(tok_args.get("H", 128))
    W = int(tok_args.get("W", 128))
    C = int(tok_args.get("C", 3))
    patch = int(tok_args.get("patch", 4))
    if (H, W, C) != (64, 64, 3):
        raise AssertionError(f"expected native Memory Maze 64x64 RGB, got {(H, W, C)}")

    files = sorted(args.raw_eval.rglob("*.npz"))
    if len(files) < args.n_sequences:
        raise SystemExit(f"need {args.n_sequences} eval trajectories, found {len(files)}")

    rows = []
    mses = []
    chosen = []
    for i, path in enumerate(files[:args.n_sequences]):
        with np.load(path) as z:
            images = np.asarray(z["image"])
        max_start = images.shape[0] - args.length
        start = (37 + 251 * i) % max(1, max_start + 1)
        gt_u8 = images[start:start + args.length]
        x = torch.from_numpy(gt_u8).to(device=device, dtype=torch.float32)
        x = x.permute(0, 3, 1, 2).unsqueeze(0) / 255.0
        with torch.inference_mode():
            z, _ = encoder(temporal_patchify(x, patch))
            pred_patches = decoder(z)
            pred = temporal_unpatchify(pred_patches, H, W, C, patch)
        pred_u8 = (pred[0].permute(0, 2, 3, 1).clamp(0, 1) * 255.0).byte().cpu().numpy()
        err = np.abs(pred_u8.astype(np.int16) - gt_u8.astype(np.int16)).astype(np.uint8)
        err_vis = np.clip(err.astype(np.int16) * 4, 0, 255).astype(np.uint8)
        mse = float(np.mean((pred_u8.astype(np.float32) / 255.0 - gt_u8.astype(np.float32) / 255.0) ** 2))
        mses.append(mse)
        chosen.append({"trajectory": str(path.relative_to(args.raw_eval)), "start": start, "mse": mse})

        gt_strip = filmstrip(gt_u8)
        pred_strip = filmstrip(pred_u8)
        err_strip = filmstrip(err_vis)
        cv2.putText(gt_strip, "GT", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(pred_strip, "RECON", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(err_strip, "ABS ERR x4", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        rows.extend([gt_strip, pred_strip, err_strip, np.zeros((4, gt_strip.shape[1], 3), np.uint8)])

    sheet = np.concatenate(rows[:-1], axis=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write {args.out}")
    metrics = {
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "mean_mse": float(np.mean(mses)),
        "mean_psnr_db": float(-10.0 * np.log10(max(float(np.mean(mses)), 1e-12))),
        "sequences": chosen,
        "layout": "for each sequence: GT / RECON / absolute RGB error x4",
    }
    metrics_path = args.out.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"SAVED {args.out} {metrics_path}")


if __name__ == "__main__":
    main()
