"""Headless check of the play_dynamics_checkpoint FF7 register-carry fix (T-010).

Drives the real InteractiveRollout (no cv2 window) through Merlin's scenario on ff7_k3:
a ball-visible context -> N curtain-down frames -> reveal, and reads the revealed ball's
COLOR (episode-constant) from the carried latent. Compares memory-ON (the fix) vs memory
forced-OFF (the old vanilla path that hallucinated a random ball once the curtain outlasted
the 4-frame window). Reveal latent is decoded with the probe's native-order decoder +
detector so there is no channel-order ambiguity.
"""
import sys
import pathlib
import random
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT / "src" / "D_dynamics_model", ROOT / "src" / "C_multi_image_auto_encoder",
          ROOT / "src" / "probe"):
    sys.path.insert(0, str(p))

import play_dynamics_checkpoint as play
from revisit_probe import detect_ball, _decode_frame

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TOK = ROOT / "trained_autoencoder.pt"
CKPT = ROOT / "experiments" / "EXP-010" / "k3" / "ff7_k3_s0.pt"
FRAMES, ACTS = ROOT / "occluded.npy", ROOT / "occluded_actions.npy"


def build():
    payload = torch.load(CKPT, map_location=DEV, weights_only=False)
    cfg = play._config_from_checkpoint(payload["config"], play.DynamicsModelConfig)
    model = play.DynamicsModel(cfg).to(DEV)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    tok = play.load_tokenizer(TOK, DEV)
    tcfg = play._config_from_checkpoint(
        torch.load(TOK, map_location=DEV, weights_only=False)["config"], play.AutoEncoderConfig)
    ro = play.InteractiveRollout(
        raw=np.load(FRAMES), actions_raw=np.load(ACTS), tokenizer=tok, model=model, device=DEV,
        tokenizer_context=tcfg.max_temporal_length, inference_steps=cfg.inference_steps, h=64, w=64)
    return ro, tok


def reset_with_visible_ball(ro, max_tries=200):
    """Keep resetting until the seeded context's last frame has a detectable (visible) ball."""
    for _ in range(max_tries):
        ro.reset()
        last_gt = ro.raw[ro.ep, ro.frame_base + play.ROLLOUT_CTX - 1]
        found, *_rest = detect_ball(last_gt)
        if found and all(a == 0 for a in ro.actions):   # all context frames curtain-up
            gt_color = detect_ball(last_gt)[3].astype(np.float32)
            return gt_color
    raise RuntimeError("no visible-ball context found")


def run(force_memory, n_down=8, seed=0):
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    ro, tok = build()
    ro.use_memory = force_memory          # same checkpoint, both inference paths
    gt_color = reset_with_visible_ball(ro)
    for _ in range(n_down):
        ro.step(1)                         # curtain down (ball hidden)
    ro.step(0)                             # reveal
    reveal_latent = ro.latents[:, -1:]     # carried/generated latent for the revealed frame
    frame = _decode_frame(tok, reveal_latent[:, 0])     # native order, like the probe
    found, x, y, color = detect_ball(frame)
    drgb = float(np.abs(color.astype(np.float32) - gt_color).mean()) if found else float("nan")
    return {"path": "memory(FF7)" if force_memory else "vanilla(forced)",
            "uses_memory_step": force_memory, "ball_found": bool(found),
            "reveal_color_dRGB_vs_gt": round(drgb, 1), "gt_color_native": gt_color.tolist()}


if __name__ == "__main__":
    print(f"n_down=8 curtain frames (well past the 4-frame vanilla window), ff7_k3, seed 0")
    for fm in (True, False):
        print(" ", run(fm))
