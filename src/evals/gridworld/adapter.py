"""Model-based frame sources for the GridWorld recall eval.

The recall CORE (`readout.py` + the scoring/baselines in `recall.py`) is FROZEN (D-045) and pure
numpy. Anything that needs a torch model lives HERE and imports the frozen core — it never edits it.

Frame sources provided:
  * tokenizer_roundtrip(model, frames) -> recon frames. The CEILING the frozen latent imposes: it
    encodes→decodes the TRUE frames, so a reveal frame (curtain up, square visible) is autoencoded.
    Reading the square out of the recon measures whether the latent space can even REPRESENT the
    square's cell+colour — the upper bound on what any dynamics model on this tokenizer can recall
    (it predicts latents that get decoded the same way). NOT a memory test (the input is the answer).

  * (TODO) dynamics_rollout(...) — the real memory test; needs a trained GridWorld dynamics model.

All frames are uint8 BGR (env channel-order contract; gridworld.npy is BGR — verified, the
train_tokenizer "RGB" labels are mislabeled). Position readout is channel-order-independent; colour
readout needs BGR, which the dataset already is, so no flip.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/


@torch.no_grad()
def tokenizer_roundtrip(model, frames_u8: np.ndarray, L: int, device: str) -> np.ndarray:
    """encode->decode a whole episode through the frozen tokenizer, window by window.

    `frames_u8`: (T, H, W, 3) uint8 BGR. Processed in consecutive length-L windows (the tokenizer's
    max_temporal_length); the final window is clamped to [T-L, T] so the tail is covered (overlap is
    harmless — last write wins, and each reveal frame's recon depends only on its own window). Mirrors
    train_tokenizer's test path exactly: float32 input /255, model as loaded (bf16 weights handled
    internally), clamp[0,1], back to uint8. Returns (T, H, W, 3) uint8 BGR.
    """
    T = len(frames_u8)
    if T < L:
        raise ValueError(f"episode length {T} < tokenizer window {L}")
    out = np.empty_like(frames_u8)
    for s in range(0, T, L):
        s = min(s, T - L)
        clip = frames_u8[s:s + L].astype(np.float32) / 255.0
        x = torch.from_numpy(clip).unsqueeze(0).to(device)
        rec = model.decoder(model.encoder(x))
        rec01 = rec[0].clamp(0, 1).cpu().float().numpy()  # (L, H, W, 3)
        out[s:s + L] = (rec01 * 255.0).round().astype(np.uint8)
    return out


def load_tokenizer(checkpoint: str, device: str):
    """Load a frozen tokenizer checkpoint (mirrors train_tokenizer). Returns (model, L)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dataclasses import fields
    from models.tokenizer import AutoEncoder, AutoEncoderConfig
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    allowed = {f.name for f in fields(AutoEncoderConfig)}
    cfg = AutoEncoderConfig(**{k: v for k, v in payload["config"].items() if k in allowed})
    model = AutoEncoder(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, cfg.max_temporal_length
