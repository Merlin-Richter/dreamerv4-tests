"""
Bouncing-shape dataset — generation and playback.

Drives `BouncingEnv` (src/envs/) to write the DVD-style bouncing dataset and to preview it.
The env itself (shape drawing + physics) lives in `src/envs/bouncing.py`.

Usage:
    python -u src/datagen/generate_bouncing.py --n_episodes 1000 --out bouncing.npy
    python -u src/datagen/generate_bouncing.py --debug
    python -u src/datagen/generate_bouncing.py --debug --shape star --img_size 128
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_SRC = Path(__file__).resolve().parents[1]  # .../src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from envs.bouncing import SHAPES, BouncingEnv  # noqa: E402


# ---------------------------------------------------------------------------
# Episode generator
# ---------------------------------------------------------------------------

def generate_episode(
    n_frames: int = 200,
    img_size: int = 64,
    shape: str = "circle",
    radius: int = 8,
    seed: int = 42,
    color=None,
    change_color_on_bounce: bool = False,
):
    """Returns (frames (T,H,W,3) uint8, states (T,4) float32 [x,y,vx,vy]).

    Byte-for-byte equivalent to the previous standalone generator: `BouncingEnv.reset` draws
    color → x → y → vx → vy in the same order, so a given seed yields the same episode.
    """
    env = BouncingEnv(
        img_size=img_size, shape=shape, radius=radius,
        color=color, change_color_on_bounce=change_color_on_bounce,
    )
    env.reset(seed=seed)
    frames = np.empty((n_frames, img_size, img_size, 3), dtype=np.uint8)
    states = np.empty((n_frames, 4), dtype=np.float32)
    for t in range(n_frames):
        frames[t], states[t] = env.step()
    return frames, states


# ---------------------------------------------------------------------------
# Debug playback
# ---------------------------------------------------------------------------

def debug_play(args):
    shape = args.shape if args.shape != "random" else SHAPES[0]
    print(f"[debug] shape={shape}  size={args.img_size}  radius={args.radius}  "
          f"frames={args.n_frames}  color_on_bounce={args.color_on_bounce}")
    print("Press Q or ESC to quit, SPACE to pause.")

    frames, states = generate_episode(
        n_frames=args.n_frames,
        img_size=args.img_size,
        shape=shape,
        radius=args.radius,
        seed=args.seed,
        change_color_on_bounce=args.color_on_bounce,
    )

    # Scale up small frames so they're actually visible
    display_size = max(args.img_size, 400)
    scale = display_size // args.img_size

    paused = False
    for i, frame in enumerate(frames):
        big = cv2.resize(frame, (args.img_size * scale, args.img_size * scale),
                         interpolation=cv2.INTER_NEAREST)
        cv2.putText(big, f"{shape}  frame {i+1}/{len(frames)}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.imshow("Bouncing shape [Q/ESC=quit  SPACE=pause]", big)

        while True:
            delay = 0 if paused else max(1, int(1000 / args.fps))
            key = cv2.waitKey(delay) & 0xFF
            if key in (ord('q'), 27):       # Q or ESC
                cv2.destroyAllWindows()
                return
            if key == ord(' '):             # SPACE toggles pause
                paused = not paused
            if not paused:
                break

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(args):
    rng_meta = np.random.default_rng(args.seed)

    all_frames = []
    all_states = []

    t0 = time.time()
    for i in range(args.n_episodes):
        shape = (
            SHAPES[rng_meta.integers(len(SHAPES))]
            if args.shape == "random"
            else args.shape
        )
        frames, states = generate_episode(
            n_frames=args.n_frames,
            img_size=args.img_size,
            shape=shape,
            radius=args.radius,
            seed=int(rng_meta.integers(1 << 31)),
            change_color_on_bounce=args.color_on_bounce,
        )
        all_frames.append(frames)
        all_states.append(states)

        if (i + 1) % max(1, args.n_episodes // 10) == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{args.n_episodes} episodes  ({elapsed:.1f}s)")

    frames_arr = np.stack(all_frames)   # (N, T, H, W, 3)  uint8
    states_arr = np.stack(all_states)   # (N, T, 4)         float32

    out_frames = args.out
    out_states = args.out.replace(".npy", "_states.npy")

    np.save(out_frames, frames_arr)
    np.save(out_states, states_arr)

    print(f"\nSaved frames → {out_frames}   shape={frames_arr.shape}")
    print(f"Saved states → {out_states}   shape={states_arr.shape}")
    print(f"Disk size: frames={frames_arr.nbytes / 1e6:.1f} MB  "
          f"states={states_arr.nbytes / 1e6:.2f} MB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Bouncing shape dataset generator")
    p.add_argument("--debug", action="store_true",
                   help="Play a single episode in a window instead of saving")
    p.add_argument("--shape", default="random",
                   choices=SHAPES + ["random"],
                   help="Shape to use (default: random per episode)")
    p.add_argument("--n_episodes", type=int, default=1000,
                   help="Number of episodes to generate (save mode)")
    p.add_argument("--n_frames", type=int, default=200,
                   help="Frames per episode")
    p.add_argument("--img_size", type=int, default=64,
                   help="Square image size in pixels")
    p.add_argument("--radius", type=int, default=8,
                   help="Shape radius / half-size in pixels")
    p.add_argument("--seed", type=int, default=42,
                   help="Master random seed")
    p.add_argument("--out", default="bouncing.npy",
                   help="Output .npy path for frames (save mode)")
    p.add_argument("--fps", type=int, default=30,
                   help="Playback FPS (debug mode only)")
    p.add_argument("--color_on_bounce", action="store_true",
                   help="Change color on each wall bounce (like real DVD logo)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        debug_play(args)
    else:
        print(f"Generating {args.n_episodes} episodes × {args.n_frames} frames "
              f"@ {args.img_size}×{args.img_size}  shape={args.shape}")
        generate_dataset(args)
