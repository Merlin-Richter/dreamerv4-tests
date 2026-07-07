"""Eval bridge: dynamics checkpoint -> frozen_sym comeback-eval adapter.
EDITABLE LAYER — SYMBOLIC tier (no tokenizer anywhere).

Exports ``make_adapter(ckpt_path, device) -> factory`` usable as

    from autoresearch.frozen_sym.eval_comeback import run_eval
    run_eval(make_adapter(ckpt, "cpu"), privileged=False, ...)

Candidate models run privileged=False, so ``factory(env_or_none)`` receives None —
the returned adapter must work from the prefix alone (no env peeking):

  begin(prefix_grids, prefix_actions):
      (P,5,5) uint8 symbolic grids + (P,) int per-TICK actions (actions[0]=STAY).
      Grids + phases (the prefix starts at episode tick 0, so phase[t] = t % 5)
      -> one-hot "latents" (P, 5, 35) via ``encode_latents`` (THE codec, shared
      with train_sym.py) -> the model's carrying rollout via rollout_init. The
      prefix (192 ticks) exceeds the pinned window W=16: rollout_init's
      long-context prefill commits the first window in one pass, then
      teacher-forces each remaining TRUE frame through the sliding window with
      written-memory relay — so a memory model absorbs the whole prefix into its
      memory tokens.
  step(action) -> (5,5) uint8:
      rollout_step(commit=True): K shortcut denoising steps (K =
      config.inference_steps, typically 4) + the near-clean commit pass with the
      written memory token, then ARGMAX-decode the committed latents' first 30
      dims per row back to a (5,5) grid of cell ids 0..5 (phase dims ignored).

THE CODEC (the sym tier's tokenizer replacement — pure one-hot, exact):
  latent row r of frame t = [ onehot6(grid[t, r, 0]) | ... | onehot6(grid[t, r, 4])
                              | onehot5(t % 5) ]                    -> 35 dims
  n_latents = 5 (one token per viewport ROW), bottleneck_dim = 35. The phase
  block is appended to EVERY row; the x-prediction target includes it (trivially
  predictable, harmless). Decode = per-cell argmax over the 6 one-hot classes.

Everything runs torch.no_grad, eval-mode (dropout off), bf16 autocast on cuda
only (CPU stays fp32). The model is loaded ONCE per make_adapter; each factory
call returns a fresh adapter (per-episode rollout state) sharing the module.

Loop agent: you MAY edit this file (your model may need a different inference
path); the frozen_sym eval + the driver's window probe keep it honest.
"""
from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

try:  # package import (the driver's path)
    from .model import DynamicsModel, DynamicsModelConfig
    from ..frozen_sym.env import OUT_IDX, PHASE_PERIOD, VIEW_CELLS
except ImportError:  # run as a script: python autoresearch/editable/adapter_sym.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from autoresearch.editable.model import DynamicsModel, DynamicsModelConfig
    from autoresearch.frozen_sym.env import OUT_IDX, PHASE_PERIOD, VIEW_CELLS

# --- codec geometry (derived from the frozen_sym env; = 6, 30, 35) -------------
N_CELL_CLASSES = OUT_IDX + 1                       # 5 palette colors + OUT = 6
CELL_DIMS = VIEW_CELLS * N_CELL_CLASSES            # 5 cells x 6 one-hot = 30
BOTTLENECK_DIM = CELL_DIMS + PHASE_PERIOD          # + 5 phase one-hot dims = 35
N_LATENTS = VIEW_CELLS                             # one token per viewport row = 5


def encode_latents(grids: np.ndarray, ticks: np.ndarray) -> np.ndarray:
    """(T,5,5) uint8 grids of cell ids 0..5 + (T,) ABSOLUTE episode tick indices
    -> (T, 5, 35) float32 one-hot latents (see THE CODEC in the module header).
    Ticks are absolute because the phase block encodes t % PHASE_PERIOD."""
    g = np.asarray(grids)
    t = np.asarray(ticks)
    assert g.ndim == 3 and g.shape[1:] == (VIEW_CELLS, VIEW_CELLS), g.shape
    assert t.shape == (g.shape[0],), (t.shape, g.shape)
    assert int(g.max()) < N_CELL_CLASSES, f"cell id {int(g.max())} >= {N_CELL_CLASSES}"
    z = np.empty((g.shape[0], N_LATENTS, BOTTLENECK_DIM), dtype=np.float32)
    cells = np.eye(N_CELL_CLASSES, dtype=np.float32)[g]        # (T, 5, 5, 6)
    z[:, :, :CELL_DIMS] = cells.reshape(g.shape[0], N_LATENTS, CELL_DIMS)
    phase = np.eye(PHASE_PERIOD, dtype=np.float32)[t % PHASE_PERIOD]  # (T, 5)
    z[:, :, CELL_DIMS:] = phase[:, None, :]                    # same block, EVERY row
    return z


def decode_latents(z: np.ndarray) -> np.ndarray:
    """(..., 5, 35) predicted latents -> (..., 5, 5) uint8 grid of cell ids 0..5
    via per-cell ARGMAX over the 6 one-hot classes (phase dims ignored)."""
    z = np.asarray(z, dtype=np.float32)
    assert z.shape[-2:] == (N_LATENTS, BOTTLENECK_DIM), z.shape
    cells = z[..., :CELL_DIMS].reshape(*z.shape[:-1], VIEW_CELLS, N_CELL_CLASSES)
    return cells.argmax(axis=-1).astype(np.uint8)


def _cfg_from_dict(d: dict, cls):
    """Rebuild a dataclass config from a checkpoint dict; drop 'dtype' (torch.dtype does
    not survive JSON round-trips) and unknown keys (forward compat)."""
    allowed = {f.name for f in fields(cls)} - {"dtype"}
    return cls(**{k: v for k, v in d.items() if k in allowed})


def load_dynamics(ckpt_path, device: str) -> DynamicsModel:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = _cfg_from_dict(payload["config"], DynamicsModelConfig)
    assert cfg.n_latents == N_LATENTS and cfg.bottleneck_dim == BOTTLENECK_DIM, (
        f"checkpoint dims (n_latents={cfg.n_latents}, bottleneck={cfg.bottleneck_dim}) "
        f"!= sym codec ({N_LATENTS}, {BOTTLENECK_DIM}) — is this a pixel-tier checkpoint?")
    model = DynamicsModel(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    return model


class SymDynamicsAdapter:
    """begin/step world-model adapter around the carrying rollout (see module docstring)."""

    def __init__(self, model: DynamicsModel, device: str, K: int = None):
        self.model = model
        self.device = device
        self.K = K  # None -> model.config.inference_steps
        self.state = None

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                              enabled=self.device.startswith("cuda"))

    @torch.no_grad()
    def begin(self, prefix_grids: np.ndarray, prefix_actions: np.ndarray) -> None:
        P = len(prefix_grids)
        # The eval prefix is ticks 0..P-1 of the episode, so absolute ticks = arange(P).
        z = encode_latents(np.ascontiguousarray(prefix_grids), np.arange(P))
        context = torch.from_numpy(z).unsqueeze(0).to(self.device)   # (1, P, 5, 35)
        acts = torch.as_tensor(np.asarray(prefix_actions), dtype=torch.long,
                               device=self.device).unsqueeze(0)      # (1, P)
        with self._autocast():
            self.state = self.model.rollout_init(context, acts, K=self.K)

    @torch.no_grad()
    def step(self, action: int) -> np.ndarray:
        assert self.state is not None, "step() before begin()"
        a = torch.tensor([[int(action)]], dtype=torch.long, device=self.device)
        with self._autocast():
            z = self.model.rollout_step(self.state, a, commit=True)  # (1, 1, 5, 35)
        return decode_latents(z.float().cpu().numpy()[0, 0])


def make_adapter(ckpt_path, device: str = None, K: int = None):
    """adapter_factory for the frozen_sym eval. Loads the dynamics checkpoint
    ({"model_state_dict", "config"}) once — NO tokenizer (symbolic tier); the factory
    ignores its env argument (candidate models are unprivileged — it receives None)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dynamics(ckpt_path, device)

    def factory(env_or_none):
        return SymDynamicsAdapter(model, device, K=K)

    return factory


if __name__ == "__main__":  # smoke: one frozen_sym-eval episode end-to-end
    import argparse

    ap = argparse.ArgumentParser(description="Run ONE sym comeback-eval episode (smoke).")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--prefix-len", type=int, default=48)
    ap.add_argument("--imag-len", type=int, default=64)
    args = ap.parse_args()

    from autoresearch.frozen_sym.eval_comeback import run_episode
    from autoresearch.frozen_sym.eval_policies import EvalOutAndBack

    factory = make_adapter(args.checkpoint, device=args.device)
    events, fidelity, first_imag_colors, band_err, positions = run_episode(
        factory, EvalOutAndBack(2, 5), map_seed=1, ep_seed=2,
        prefix_len=args.prefix_len, imag_len=args.imag_len, privileged=False)
    print(f"episode OK: {len(events)} events, fidelity={float(np.mean(fidelity)):.3f}, "
          f"{len(first_imag_colors)} imagination-born cells, "
          f"band_err={float(np.mean(band_err)):.2f} cells, {len(positions)} positions",
          flush=True)
