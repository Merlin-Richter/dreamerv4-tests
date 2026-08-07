#!/usr/bin/env python3
"""Hard gate that distinguishes a resumable partial checkpoint from the final 48h artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from train_mem2mem import EXPECTED_TOKENIZER_SHA256, PRODUCTION_LOCK


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--expected-training-seconds", type=float, default=172_800.0)
    ap.add_argument("--expected-latent-cache-manifest-sha256", required=True)
    ap.add_argument("--max-overshoot-seconds", type=float, default=10.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    resolved = payload["resolved_config"]
    wrong = {
        key: (resolved.get(key), expected)
        for key, expected in PRODUCTION_LOCK.items()
        if resolved.get(key) != expected
    }
    assert not wrong, f"checkpoint violates frozen production config: {wrong}"
    assert resolved["ff9"] is False and resolved["archive"] is False
    assert payload["tokenizer_sha256"] == EXPECTED_TOKENIZER_SHA256
    assert payload["latent_cache_manifest_sha256"] == args.expected_latent_cache_manifest_sha256

    step = int(payload["step"])
    elapsed = float(payload["elapsed_train_s"])
    assert args.expected_training_seconds <= elapsed, (elapsed, args.expected_training_seconds)
    assert elapsed <= args.expected_training_seconds + args.max_overshoot_seconds, elapsed

    exposure = payload["exposure"]
    batch = int(PRODUCTION_LOCK["batch_size"])
    clip = int(PRODUCTION_LOCK["clip_length"])
    window = int(PRODUCTION_LOCK["window"])
    assert int(exposure["optimizer_steps"]) == step
    assert int(exposure["source_frames_read"]) == 0
    assert int(exposure["source_latent_windows_read"]) == step * batch * 7
    assert int(exposure["source_latent_frames_read"]) == step * batch * 7 * window
    assert int(exposure["scored_frames"]) == step * batch * (clip - window)
    assert int(exposure["scored_slides_per_step"]) == 6
    assert int(exposure["target_frames_per_step"]) == batch * (clip - window)
    assert int(exposure["unique_source_windows"]) == int(payload["seen_source_windows"].sum())

    memory = payload["dynamics"]["memory_tokens"]
    assert tuple(memory.shape) == (1, 1, 8, 512), tuple(memory.shape)
    for key in ("opt", "scaler", "rng", "dynamics", "seen_source_windows"):
        assert key in payload

    report = {
        "elapsed_train_s": elapsed,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "exposure": exposure,
        "memory_tokens_shape": list(memory.shape),
        "production_config": resolved,
        "step": step,
        "tokenizer_sha256": payload["tokenizer_sha256"],
        "latent_cache_manifest_sha256": payload["latent_cache_manifest_sha256"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("FINAL 48-TRAINING-WALL-HOUR CHECKPOINT GATE PASSED")


if __name__ == "__main__":
    main()
