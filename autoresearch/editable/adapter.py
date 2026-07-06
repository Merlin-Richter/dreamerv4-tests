"""Eval bridge: dynamics checkpoint -> frozen comeback-eval adapter. EDITABLE LAYER.

Exports ``make_adapter(ckpt_path, tokenizer_path, device) -> factory`` usable as

    from autoresearch.frozen.eval_comeback import run_eval
    run_eval(make_adapter(ckpt, tok, "cuda"), privileged=False, ...)

Candidate models run privileged=False, so ``factory(env_or_none)`` receives None —
the returned adapter must work from the prefix alone (no env peeking):

  begin(prefix_frames, prefix_actions):
      (P,64,64,3) uint8 RGB [0,255] real frames + (P,) int actions (actions[0]=STAY).
      Frames -> float [0,1] -> FROZEN tokenizer encoder in 16-frame chunks (the cache's
      encoding convention) -> the model's carrying rollout via rollout_init. The prefix
      (192 frames) exceeds the pinned window W=16: rollout_init's long-context prefill
      commits the first window in one pass, then teacher-forces each remaining TRUE
      frame through the sliding window with written-memory relay — exactly what it
      exists for, so a memory model absorbs the whole prefix into its memory tokens.
  step(action) -> (64,64,3) uint8 RGB:
      rollout_step(commit=True): K shortcut denoising steps (K = config.inference_steps,
      typically 4) + the near-clean commit pass with the written memory token, then
      decode the committed clean latents through the frozen tokenizer decoder (T=1).

Everything runs torch.no_grad, eval-mode (dropout/MAE off), bf16 autocast on cuda.
The model + tokenizer are loaded ONCE per make_adapter; each factory call returns a
fresh adapter (per-episode rollout state), sharing the loaded modules.

Loop agent: you MAY edit this file (your model may need a different inference path);
the frozen eval + the driver's window probe keep it honest.
"""
from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

try:  # package import (the driver's path)
    from .model import DynamicsModel, DynamicsModelConfig
    from ..frozen.tokenizer_model import AutoEncoder, AutoEncoderConfig
except ImportError:  # run as a script: python autoresearch/editable/adapter.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from autoresearch.editable.model import DynamicsModel, DynamicsModelConfig
    from autoresearch.frozen.tokenizer_model import AutoEncoder, AutoEncoderConfig

TOK_CHUNK = 16  # frozen tokenizer temporal window == the latent cache's encoding chunk


def _cfg_from_dict(d: dict, cls):
    """Rebuild a dataclass config from a checkpoint dict; drop 'dtype' (torch.dtype does
    not survive JSON round-trips) and unknown keys (forward compat)."""
    allowed = {f.name for f in fields(cls)} - {"dtype"}
    return cls(**{k: v for k, v in d.items() if k in allowed})


def load_dynamics(ckpt_path, device: str) -> DynamicsModel:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = _cfg_from_dict(payload["config"], DynamicsModelConfig)
    model = DynamicsModel(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    return model


def load_tokenizer(tokenizer_path, device: str) -> AutoEncoder:
    payload = torch.load(tokenizer_path, map_location=device, weights_only=False)
    cfg = _cfg_from_dict(payload["config"], AutoEncoderConfig)
    model = AutoEncoder(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    return model


class DynamicsAdapter:
    """begin/step world-model adapter around the carrying rollout (see module docstring)."""

    def __init__(self, model: DynamicsModel, tokenizer: AutoEncoder, device: str,
                 K: int = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.K = K  # None -> model.config.inference_steps
        self.state = None

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                              enabled=self.device.startswith("cuda"))

    @torch.no_grad()
    def begin(self, prefix_frames: np.ndarray, prefix_actions: np.ndarray) -> None:
        P = len(prefix_frames)
        x = torch.from_numpy(np.ascontiguousarray(prefix_frames)).to(self.device)
        x = x.to(torch.float32).div_(255.0).unsqueeze(0)          # (1, P, 64, 64, 3) in [0,1]
        acts = torch.as_tensor(np.asarray(prefix_actions), dtype=torch.long,
                               device=self.device).unsqueeze(0)   # (1, P)

        # Encode in TOK_CHUNK-frame windows, batching full chunks along B (identical to
        # sequential chunk encodes — the encoder is windowed-causal). Trailing partial
        # chunk (P % TOK_CHUNK) is encoded as-is (RoPE table slices to T).
        n_full = P // TOK_CHUNK
        zs = []
        with self._autocast():
            if n_full > 0:
                xw = x[0, :n_full * TOK_CHUNK].reshape(n_full, TOK_CHUNK, *x.shape[2:])
                for s in range(0, n_full, 8):
                    z = self.tokenizer.encoder(xw[s:s + 8])       # (b, 16, L, D)
                    zs.append(z.float().reshape(1, -1, *z.shape[2:]))
            if P % TOK_CHUNK:
                zs.append(self.tokenizer.encoder(x[:, n_full * TOK_CHUNK:]).float())
        context = torch.cat(zs, dim=1)                            # (1, P, n_latents, bottleneck)

        with self._autocast():
            self.state = self.model.rollout_init(context, acts, K=self.K)

    @torch.no_grad()
    def step(self, action: int) -> np.ndarray:
        assert self.state is not None, "step() before begin()"
        a = torch.tensor([[int(action)]], dtype=torch.long, device=self.device)
        with self._autocast():
            z = self.model.rollout_step(self.state, a, commit=True)  # (1, 1, L, D)
            frame = self.tokenizer.decoder(z.float())                # (1, 1, 64, 64, 3) in [0,1]
        frame = frame.float().clamp_(0.0, 1.0)[0, 0]
        return (frame * 255.0).round_().to(torch.uint8).cpu().numpy()


def make_adapter(ckpt_path, tokenizer_path, device: str = None, K: int = None):
    """adapter_factory for the frozen eval. Loads the dynamics checkpoint
    ({"model_state_dict", "config"}) and the FROZEN tokenizer once; the factory ignores
    its env argument (candidate models are unprivileged — it receives None)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dynamics(ckpt_path, device)
    tokenizer = load_tokenizer(tokenizer_path, device)

    def factory(env_or_none):
        return DynamicsAdapter(model, tokenizer, device, K=K)

    return factory


if __name__ == "__main__":  # smoke: one frozen-eval episode end-to-end
    import argparse

    ap = argparse.ArgumentParser(description="Run ONE comeback-eval episode (smoke).")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, default=Path("checkpoints/colorfield/tokenizer.pt"))
    ap.add_argument("--prefix-len", type=int, default=48)
    ap.add_argument("--imag-len", type=int, default=64)
    args = ap.parse_args()

    from autoresearch.frozen.eval_comeback import run_episode
    from autoresearch.frozen.eval_policies import EvalOutAndBack

    factory = make_adapter(args.checkpoint, args.tokenizer)
    events, fidelity, first_imag_colors, band_err, positions = run_episode(
        factory, EvalOutAndBack(20, 30), map_seed=1, ep_seed=2,
        prefix_len=args.prefix_len, imag_len=args.imag_len, privileged=False)
    print(f"episode OK: {len(events)} events, fidelity={float(np.mean(fidelity)):.3f}, "
          f"{len(first_imag_colors)} imagination-born cells, "
          f"band_err={float(np.mean(band_err)):.2f}px, {len(positions)} positions", flush=True)
