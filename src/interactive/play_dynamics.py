"""
Interactive single-frame viewer for a trained dynamics checkpoint.

Seed a few real frames, then generate frames one keypress at a time, choosing the curtain
each step, to eyeball whether the square is remembered through a manual occlusion.

  0  generate next frame with action 0 (revealed)
  1  generate next frame with action 1 (occluded)
  r  reset (new random seed clip)
  q  quit

Each keypress appends the chosen action and calls the single `model.generate` path for ONE
more frame (the carrying rollout when the model has memory tokens, plain otherwise), decodes
it and displays it (scaled, red border when occluded). Rollouts continue until reset.

Run from repo root:
    python src/interactive/play_dynamics.py --checkpoint <dyn.pt> --tokenizer <tok.pt> --frames data/gridworld.npy
"""

import argparse
import random
import sys
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]   # .../src (the package root)
_ROOT = _SRC.parent                          # repo root
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.dynamics_model import DynamicsModel, DynamicsModelConfig
from models.tokenizer import AutoEncoder, AutoEncoderConfig

SEED_FRAMES = 4
WINDOW_TITLE = "Dynamics interactive rollout"


def _config_from_checkpoint(cfg_dict: dict, cls):
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in cfg_dict.items() if k in allowed})


def _tensor01_to_bgr(img_t: torch.Tensor) -> np.ndarray:
    x = img_t.detach().cpu().float().clamp(0.0, 1.0).numpy()
    return cv2.cvtColor((x * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


class InteractiveRollout:
    """Single-frame interactive rollout driven by the unified `model.generate` path."""

    def __init__(self, *, raw, actions_raw, tokenizer, model, device, h, w):
        self.raw = raw
        self.actions_raw = actions_raw
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.n_eps = raw.shape[0]
        self.n_frames = raw.shape[1]
        self.scale = max(2, 512 // max(h, w))

        self.latents = None          # (1, T, n_latents, bottleneck_dim)
        self.actions = []            # one action id per latent frame
        self.ep = 0
        self.frame_base = 0
        self.current_bgr = None
        self.is_seed = True

    @torch.no_grad()
    def reset(self):
        if self.n_frames < SEED_FRAMES:
            raise ValueError(f"Episode length {self.n_frames} < seed frames {SEED_FRAMES}.")
        self.ep = random.randrange(self.n_eps)
        self.frame_base = random.randrange(self.n_frames - SEED_FRAMES + 1)

        clip = self.raw[self.ep, self.frame_base:self.frame_base + SEED_FRAMES]
        x = torch.from_numpy(clip.astype(np.float32) / 255.0).unsqueeze(0).to(self.device)
        self.latents = self.tokenizer.encoder(x)

        if self.actions_raw is not None:
            self.actions = [int(self.actions_raw[self.ep, self.frame_base + t])
                            for t in range(SEED_FRAMES)]
        else:
            self.actions = [0] * SEED_FRAMES

        self.current_bgr = _tensor01_to_bgr(x[0, -1])
        self.is_seed = True

    @torch.no_grad()
    def step(self, action: int):
        action_idx = None
        if self.model.n_actions > 0:
            ids = self.actions + [action]
            action_idx = torch.tensor([ids], device=self.device, dtype=torch.long)

        nxt = self.model.generate(self.latents, n_generate=1, action_idx=action_idx)
        self.latents = torch.cat((self.latents, nxt), dim=1)
        self.actions.append(int(action))

        recon = self.tokenizer.decoder(nxt)[0, 0]
        self.current_bgr = _tensor01_to_bgr(recon)
        self.is_seed = False

    def render(self) -> np.ndarray:
        img = self.current_bgr.copy()
        h, w = img.shape[:2]
        last_action = self.actions[-1]
        if last_action:  # occluded -> red border
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 2)

        disp = cv2.resize(img, (w * self.scale, h * self.scale), interpolation=cv2.INTER_NEAREST)
        frame_idx = self.frame_base + len(self.actions) - 1
        kind = "seed" if self.is_seed else "generated"
        lines = [
            f"ep {self.ep}  frame {frame_idx}  ({kind})  action={last_action}",
            "0=revealed  1=occluded  r=reset  q=quit",
        ]
        y = 12
        for line in lines:
            cv2.putText(disp, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (180, 180, 180), 1, cv2.LINE_AA)
            y += 12
        return disp


def load_tokenizer(checkpoint: Path, device: str) -> AutoEncoder:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = _config_from_checkpoint(payload["config"], AutoEncoderConfig)
    model = AutoEncoder(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_dynamics(checkpoint: Path, device: str) -> DynamicsModel:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = _config_from_checkpoint(payload["config"], DynamicsModelConfig)
    model = DynamicsModel(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive single-frame dynamics rollout.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Dynamics model checkpoint ({config, model_state_dict}).")
    parser.add_argument("--tokenizer", type=Path, required=True,
                        help="Frozen tokenizer checkpoint (must match the dynamics env).")
    parser.add_argument("--frames", type=Path, default=_ROOT / "data" / "gridworld.npy",
                        help="Frames .npy (N, T, H, W, 3) uint8.")
    parser.add_argument("--actions", type=Path, default=None,
                        help="Actions .npy (N, T). Default: '<frames>_actions.npy' if present.")
    args = parser.parse_args()

    raw = np.load(args.frames)
    if raw.ndim != 5 or raw.shape[-1] != 3:
        raise ValueError(f"Expected frames (N, T, H, W, 3), got {raw.shape}")
    _, _, h, w, _ = raw.shape

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_dynamics(args.checkpoint, device)
    tokenizer = load_tokenizer(args.tokenizer, device)

    actions_raw = None
    if model.n_actions > 0:
        actions_path = args.actions
        if actions_path is None:
            cand = args.frames.with_name(args.frames.stem + "_actions.npy")
            actions_path = cand if cand.is_file() else None
        if actions_path is None:
            raise FileNotFoundError(
                f"Checkpoint is action-conditioned (n_actions={model.n_actions}) but no actions "
                f"file was found next to {args.frames}. Pass --actions."
            )
        actions_raw = np.load(actions_path)

    rollout = InteractiveRollout(raw=raw, actions_raw=actions_raw, tokenizer=tokenizer,
                                 model=model, device=device, h=h, w=w)

    print("Interactive dynamics rollout  (0=revealed  1=occluded  r=reset  q=quit)")
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
    rollout.reset()
    cv2.imshow(WINDOW_TITLE, rollout.render())

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            rollout.reset()
        elif key == ord("0"):
            rollout.step(0)
        elif key == ord("1"):
            rollout.step(1)
        else:
            continue
        cv2.imshow(WINDOW_TITLE, rollout.render())

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
