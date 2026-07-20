#!/usr/bin/env python3
"""Print stable JSON provenance/throughput fields from a community checkpoint."""
import argparse
import json
from pathlib import Path

import torch


ap = argparse.ArgumentParser()
ap.add_argument("checkpoint", type=Path)
ap.add_argument("--out", type=Path, default=None)
args = ap.parse_args()
ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
elapsed = float(ckpt.get("elapsed_train_s", 0.0))
step = int(ckpt.get("step", 0))
summary = {
    "checkpoint": str(args.checkpoint.resolve()),
    "step": step,
    "epoch": int(ckpt.get("epoch", 0)),
    "elapsed_train_s": elapsed,
    "optimizer_steps_per_s": step / elapsed if elapsed > 0 else None,
    "args": ckpt.get("args", {}),
}
text = json.dumps(summary, indent=2) + "\n"
print(text, end="")
if args.out is not None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
