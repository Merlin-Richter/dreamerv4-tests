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


def load_dynamics(checkpoint: str, device: str):
    """Load a trained dynamics model. Returns (model, config)."""
    from dataclasses import fields
    from models.dynamics_model import DynamicsModel, DynamicsModelConfig
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    allowed = {f.name for f in fields(DynamicsModelConfig)}
    cfg = DynamicsModelConfig(**{k: v for k, v in payload["config"].items() if k in allowed})
    model = DynamicsModel(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, cfg


@torch.no_grad()
def dynamics_rollout_frames(model, tokenizer, frames_u8: np.ndarray, curtain: np.ndarray,
                            device: str, K: int = None, control_curtain_up: bool = False,
                            inference: str = "auto") -> np.ndarray:
    """Faithful per-reveal-event rollout (see experiments/EXP-027/recall_design.md).

    For each reveal event (lv = last curtain-up, k occluded, reveal_t = lv+k+1): context = TRUE latents
    ending at lv; autoregressively generate (reveal_t - lv) frames feeding the true curtain action ids;
    decode the predicted reveal latent inside a temporal window; place that frame at reveal_t. Returns a
    (T,H,W,3) uint8 BGR array that is the true frames with ONLY the reveal frames replaced by predictions
    (score_episode reads exactly those indices). control_curtain_up=True forces curtain UP for the
    generated frames (matched-horizon free-run-in-the-clear control).
    """
    from evals.gridworld.recall import find_reveal_events
    max_T = model.config.max_temporal_length  # tokenizer & dynamics share this window (RoPE table size)
    out = frames_u8.copy()
    cur_np = np.asarray(curtain).astype(np.int64)

    for ev in find_reveal_events(curtain):
        lv, k, t = ev["last_visible_t"], ev["k"], ev["reveal_t"]
        a = max(0, lv - (max_T - 1))                 # encode a <=max_T window of TRUE frames ending at lv
        wf = frames_u8[a:lv + 1].astype(np.float32) / 255.0
        fx = torch.from_numpy(wf).unsqueeze(0).to(device)        # (1, T_ctx, H, W, 3)
        ctx = tokenizer.encoder(fx)                              # (1, T_ctx, n_lat, dim)
        n_gen = t - lv                                           # k occluded + 1 reveal
        act = torch.from_numpy(cur_np[a:t + 1]).unsqueeze(0).to(device).clone()  # (1, T_ctx + n_gen)
        if control_curtain_up:
            act[:, ctx.shape[1]:] = 0                            # matched-horizon control: curtain UP
        # FF9 inference = the NORMAL sliding-window rollout with memory tokens carried in the window via
        # temporal attention (generate_cached plain=True bypasses the frozen-snapshot dispatch; per-frame
        # memory tokens are still present and chained across frames). This is the one intended FF9
        # inference. "snapshot" = the generate_full_state_memory special case (kept only for reference).
        if inference == "relay":
            # FF9 rollout-training inference (D-048): UPDATING memory carry (op-3 / B2). The trained
            # cross-window relay — memory re-written each step from a pure-noise source (A1).
            gen = model.generate_updating_memory(ctx, n_gen, K=K, action_idx=act)
        elif inference == "snapshot":
            # FF9 v2 frozen snapshot (B1): write memory once, carry it unchanged (static state only).
            gen = model.generate_full_state_memory(ctx, n_gen, K=K, action_idx=act)
        else:  # "windowed"/"auto"/default — normal rollout (vanilla: no memory tokens; FF9: carried)
            gen = model.generate_cached(ctx, n_gen, K=K, action_idx=act, plain=True)
        full = torch.cat((ctx, gen), dim=1)
        win = full[:, -max_T:]                                   # temporal window ending at the reveal
        dec = tokenizer.decoder(win)[0].clamp(0, 1).cpu().float().numpy()  # (w, H, W, 3)
        out[t] = (dec[-1] * 255.0).round().astype(np.uint8)
    return out


def gen_recall_episode(seed: int, n_ctx: int, k: int):
    """ENV-DIRECT controlled recall scenario (Merlin: evals drive the env, not the dataset). n_ctx
    revealed context frames (model observes the square move) → exactly k occluded frames → 1 reveal.
    Returns (frames[n_ctx+k+1], states, colors[bg_idx,sq_idx] PALETTE-order, curtain). The single reveal
    event has occlusion length k with last_visible = n_ctx-1, so the frozen scorer + dynamics_rollout_
    frames pick it up unchanged."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    from envs.gridworld import PALETTE, GridWorldEnv
    env = GridWorldEnv().reset(seed)
    frames, states = [], []
    sched = [0] * n_ctx + [1] * k + [0]
    for a in sched:
        f, s = env.step(a)
        frames.append(f); states.append(s)
    names = list(PALETTE.keys())
    colors = np.array([names.index(env.bg_name), names.index(env.color_name)], dtype=np.int64)
    return np.array(frames), np.array(states, np.float32), colors, np.array(sched, np.int64)


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
