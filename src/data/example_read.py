"""
Example: reading the bouncing shape dataset.

Usage:
    python read_bounce_data.py
    python read_bounce_data.py --frames bouncing.npy
"""

import argparse

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", default="bouncing.npy")
    return p.parse_args()

args = parse_args()

frames = np.load(args.frames)   # (N, T, H, W, 3)  uint8

N, T, H, W, C = frames.shape
print(f"Dataset: {N} episodes, {T} frames each, {H}×{W} px, {C} channels")
print()

# ---------------------------------------------------------------------------
# Access a single episode
# ---------------------------------------------------------------------------

ep_idx = 7                          # ← pick any episode

episode_frames = frames[ep_idx]     # (T, H, W, 3)

print(f"Episode {ep_idx}:")
print(f"  frames shape : {episode_frames.shape}")
print()

# ---------------------------------------------------------------------------
# Access individual frames within that episode
# ---------------------------------------------------------------------------

frame_idx = 42                                  # ← pick any frame

single_frame = episode_frames[frame_idx]        # (H, W, 3)  uint8

print(f"  Frame {frame_idx}:")
print(f"    image shape : {single_frame.shape}")
print()

# ---------------------------------------------------------------------------
# Consecutive frame pairs  (t → t+1)  — what you'll use for the dynamics model
# ---------------------------------------------------------------------------

obs   = episode_frames[:-1]     # (T-1, H, W, 3)  — current frame
obs_next = episode_frames[1:]   # (T-1, H, W, 3)  — next frame

print(f"  Consecutive pairs: obs={obs.shape}  obs_next={obs_next.shape}")
print()

# ---------------------------------------------------------------------------
# Normalise to float [0, 1]  or  [-1, 1]
# ---------------------------------------------------------------------------

# Option A: [0, 1]
frames_01  = episode_frames.astype(np.float32) / 255.0

# Option B: [-1, 1]  (common for VAE decoders with tanh output)
frames_11  = frames_01 * 2.0 - 1.0

print(f"  Normalised [0,1]:   min={frames_01.min():.3f}  max={frames_01.max():.3f}")
print(f"  Normalised [-1,1]:  min={frames_11.min():.3f}  max={frames_11.max():.3f}")
print()

# ---------------------------------------------------------------------------
# PyTorch tensor — shape convention (N, C, H, W)
# ---------------------------------------------------------------------------
try:
    import torch

    # Full dataset as a tensor
    tensor = torch.from_numpy(frames).float() / 255.0   # (N, T, H, W, 3)
    tensor = tensor.permute(0, 1, 4, 2, 3)              # (N, T, C, H, W)
    print(f"PyTorch tensor: {tensor.shape}  dtype={tensor.dtype}")

    # Simple TensorDataset of (obs, obs_next) pairs across ALL episodes
    obs_t  = tensor[:, :-1]    # (N, T-1, C, H, W)
    obs_tp = tensor[:, 1:]     # (N, T-1, C, H, W)

    # Flatten episodes × timesteps into one big batch dimension
    obs_t  = obs_t.reshape(-1, *obs_t.shape[2:])    # (N*(T-1), C, H, W)
    obs_tp = obs_tp.reshape(-1, *obs_tp.shape[2:])  # (N*(T-1), C, H, W)

    dataset = torch.utils.data.TensorDataset(obs_t, obs_tp)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    batch_obs, batch_next = next(iter(loader))
    print(f"DataLoader batch: obs={batch_obs.shape}  next={batch_next.shape}")
    print()

except ImportError:
    print("(PyTorch not installed — skipping tensor example)\n")

# ---------------------------------------------------------------------------
# Quick visual check: play episode frames at 10 fps (single window)
# ---------------------------------------------------------------------------

FPS = 10
delay_ms = max(1, round(1000 / FPS))
scale = max(1, 400 // H)

print(f"Playing episode {ep_idx} at {FPS} fps — press Q or Esc to stop.")

for t in range(T):
    frame = episode_frames[t]
    display = cv2.resize(frame, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    cv2.putText(
        display,
        f"Episode {ep_idx}  frame {t}/{T - 1}  @ {FPS} fps",
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
    )
    cv2.imshow("Episode playback", display)
    key = cv2.waitKey(delay_ms) & 0xFF
    if key in (ord("q"), ord("Q"), 27):  # q, Q, Esc
        break

cv2.destroyAllWindows()