"""
Interactive dynamics-model checkpoint viewer for memory research.

Shows a single frame at a time. After reset, four ground-truth buffer frames seed the
rollout (tokenizer may encode a longer prefix for better latents). Each key press
generates exactly one new frame:

  0  action 0 (curtain up / revealed)
  1  action 1 (curtain down / occluded)

The dynamics model always sees a sliding temporal window of four latents when denoising
the next frame (frame 5 uses latents 1–4, frame 6 uses 2–5, etc.). Rollouts continue
indefinitely until reset.

Run from repo root:
    python src/D_dynamics_model/play_dynamics_checkpoint.py

Or from this folder:
    python play_dynamics_checkpoint.py
"""

import argparse
import random
import sys
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np
import torch

_SRC = Path(__file__).resolve().parent
_TOKENIZER_DIR = _SRC.parent / "C_multi_image_auto_encoder"
for p in (_SRC, _TOKENIZER_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dynamics_model import DynamicsModel, DynamicsModelConfig
from video_auto_encoder import AutoEncoder, AutoEncoderConfig

ROLLOUT_CTX = 4
WINDOW_TITLE = "Dynamics interactive rollout"


def _config_from_checkpoint(cfg_dict: dict, cls):
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in cfg_dict.items() if k in allowed})


def load_tokenizer(checkpoint: Path, device: str) -> AutoEncoder:
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Tokenizer checkpoint not found: {checkpoint}. Train C first or pass --tokenizer."
        )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = _config_from_checkpoint(payload["config"], AutoEncoderConfig)
    model = AutoEncoder(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def encode_frames(tokenizer: AutoEncoder, frames: torch.Tensor) -> torch.Tensor:
    """frames: (B, T, H, W, 3) in [0,1] -> latents (B, T, n_latents, bottleneck_dim)."""
    return tokenizer.encoder(frames)


def tensor01_to_bgr(img_t: torch.Tensor) -> np.ndarray:
    x = img_t.detach().cpu().float().clamp(0.0, 1.0).numpy()
    return cv2.cvtColor((x * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


def decode_latent_frame(tokenizer: AutoEncoder, latent: torch.Tensor) -> np.ndarray:
    """latent: (1, 1, n_latents, dim) -> BGR uint8 image."""
    recon = tokenizer.decoder(latent)[0, 0]
    return tensor01_to_bgr(recon)


def wait_key_down() -> int:
    """Block until a handled key is pressed; for 0/1, wait for release before returning."""
    handled = {ord("0"), ord("1"), ord(" "), ord("r"), ord("R"), ord("q"), ord("Q"), 27}
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key not in handled:
            continue
        if key in (ord("0"), ord("1")):
            while True:
                up = cv2.waitKey(30) & 0xFF
                if up != key:
                    break
        return key


class InteractiveRollout:
    """Single-frame interactive rollout with fixed four-frame dynamics context."""

    def __init__(
        self,
        *,
        raw: np.ndarray,
        actions_raw: np.ndarray | None,
        tokenizer: AutoEncoder,
        model: DynamicsModel,
        device: str,
        tokenizer_context: int,
        inference_steps: int,
        h: int,
        w: int,
    ) -> None:
        self.raw = raw
        self.actions_raw = actions_raw
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.tokenizer_context = max(ROLLOUT_CTX, int(tokenizer_context))
        self.inference_steps = inference_steps
        self.n_eps = raw.shape[0]
        self.n_frames = raw.shape[1]
        self.scale = max(2, 512 // max(h, w))

        self.latents: torch.Tensor | None = None
        self.actions: list[int] = []
        self.ep: int = 0
        self.frame_base: int = 0
        self.n_generated: int = 0
        self.current_bgr: np.ndarray | None = None
        self.current_is_gt: bool = True

    def reset(self) -> None:
        tok_len = self.tokenizer_context
        if self.n_frames < tok_len:
            raise ValueError(
                f"Episode length {self.n_frames} < tokenizer context {tok_len}."
            )

        self.ep = random.randrange(self.n_eps)
        tok_start = random.randrange(self.n_frames - tok_len + 1)
        self.frame_base = tok_start + tok_len - ROLLOUT_CTX

        clip = self.raw[self.ep, tok_start:tok_start + tok_len].astype(np.float32) / 255.0
        x = torch.from_numpy(clip).unsqueeze(0).to(self.device)

        with torch.no_grad():
            latents_full = encode_frames(self.tokenizer, x)
        self.latents = latents_full[:, -ROLLOUT_CTX:].contiguous()

        if self.actions_raw is not None:
            self.actions = [
                int(self.actions_raw[self.ep, self.frame_base + t])
                for t in range(ROLLOUT_CTX)
            ]
        else:
            self.actions = [0] * ROLLOUT_CTX

        ctx_idx = ROLLOUT_CTX - 1
        gt = clip[-1]
        self.current_bgr = tensor01_to_bgr(torch.from_numpy(gt))
        self.current_is_gt = True
        self.n_generated = 0

    @torch.no_grad()
    def step(self, action: int) -> None:
        assert self.latents is not None

        window = self.latents[:, -ROLLOUT_CTX:]
        # Action ids for the ROLLOUT_CTX context frames plus the frame being generated.
        act_ids = self.actions[-ROLLOUT_CTX:] + [action]

        act_window = None
        if self.model.n_actions > 0:
            act_idx = torch.tensor([act_ids], device=self.device, dtype=torch.long)
            act_window = self.model.action_features(act_idx)

        nxt = self.model._denoise_next(window, self.inference_steps, act_window)
        self.latents = torch.cat((self.latents, nxt), dim=1)
        self.actions.append(int(action))
        self.n_generated += 1

        self.current_bgr = decode_latent_frame(self.tokenizer, nxt)
        self.current_is_gt = False

    def render(self) -> np.ndarray:
        assert self.current_bgr is not None
        img = self.current_bgr.copy()
        h, w = img.shape[:2]

        last_action = self.actions[-1]
        if last_action:
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 2)

        frame_idx = self.frame_base + len(self.actions) - 1
        kind = "GT ctx" if self.current_is_gt else "generated"
        lines = [
            f"ep {self.ep}  frame {frame_idx}  ({kind})  action={last_action}",
            f"ctx={ROLLOUT_CTX}  tok_ctx={self.tokenizer_context}  "
            f"gen_steps={self.n_generated}  K={self.inference_steps}",
            "0=revealed  1=occluded  space/r=reset  q/Esc=quit",
        ]

        disp = cv2.resize(
            img,
            (w * self.scale, h * self.scale),
            interpolation=cv2.INTER_NEAREST,
        )

        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
        y = 10
        for line in lines:
            cv2.putText(disp, line, (4, y), font, scale, (180, 180, 180), thick, cv2.LINE_AA)
            y += 11

        return disp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive single-frame dynamics rollout (4-frame temporal context)."
    )
    parser.add_argument("--frames", type=Path, default=_SRC.parent.parent / "bouncing.npy",
                        help="Path to frames .npy (N, T, H, W, C) uint8.")
    parser.add_argument("--actions", type=Path, default=None,
                        help="Actions .npy (N, T). Default: '<frames>_actions.npy' if present.")
    parser.add_argument("--tokenizer", type=Path,
                        default=_TOKENIZER_DIR / "autoencoder_bouncing.pt",
                        help="Frozen C tokenizer checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=_SRC / "dynamics_bouncing.pt",
                        help="Dynamics model checkpoint.")
    parser.add_argument("--tokenizer-context", type=int, default=None,
                        help="Frames fed to the tokenizer on reset (>= 4). "
                             "Default: tokenizer max_temporal_length.")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    raw = np.load(args.frames)
    if raw.ndim != 5 or raw.shape[-1] != 3:
        raise ValueError(f"Expected (N, T, H, W, 3), got {raw.shape}")
    _, _, h, w, _ = raw.shape

    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = _config_from_checkpoint(payload["config"], DynamicsModelConfig)
    model = DynamicsModel(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    actions_raw = None
    if cfg.n_actions > 0:
        actions_path = args.actions
        if actions_path is None:
            cand = args.frames.with_name(args.frames.stem + "_actions.npy")
            actions_path = cand if cand.is_file() else None
        if actions_path is None:
            raise FileNotFoundError(
                f"Checkpoint is action-conditioned (n_actions={cfg.n_actions}) but no actions "
                f"file was found next to {args.frames}. Pass --actions."
            )
        actions_raw = np.load(actions_path)

    tok_payload = torch.load(args.tokenizer, map_location=device, weights_only=False)
    tok_cfg = _config_from_checkpoint(tok_payload["config"], AutoEncoderConfig)
    tokenizer = load_tokenizer(args.tokenizer, device)
    tok_ctx = args.tokenizer_context or tok_cfg.max_temporal_length

    rollout = InteractiveRollout(
        raw=raw,
        actions_raw=actions_raw,
        tokenizer=tokenizer,
        model=model,
        device=device,
        tokenizer_context=tok_ctx,
        inference_steps=cfg.inference_steps,
        h=h,
        w=w,
    )

    print("Interactive dynamics rollout")
    print(f"  dynamics temporal context: {ROLLOUT_CTX} frames (fixed)")
    print(f"  tokenizer encode length:   {max(ROLLOUT_CTX, tok_ctx)} frames on reset")
    print("  0 = curtain up   1 = curtain down   space/r = new context   q/Esc = quit")

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
    rollout.reset()
    cv2.imshow(WINDOW_TITLE, rollout.render())

    while True:
        key = wait_key_down()
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord(" "), ord("r"), ord("R")):
            rollout.reset()
        elif key == ord("0"):
            rollout.step(0)
        elif key == ord("1"):
            rollout.step(1)
        cv2.imshow(WINDOW_TITLE, rollout.render())

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
