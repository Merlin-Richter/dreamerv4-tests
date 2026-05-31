"""
Bouncing shape dataset generator — DVD screensaver style.

Shapes: circle, rect, donut, diamond, triangle, star, cross

Usage:
    # Generate and save dataset
    python generate_bounce_data.py --n_episodes 1000 --out bouncing.npy

    # Debug: preview a single episode in a window
    python generate_bounce_data.py --debug
    python generate_bounce_data.py --debug --shape star --img_size 128
"""

import argparse
import time

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Shape drawing
# ---------------------------------------------------------------------------

def _polygon(frame, cx, cy, n_sides, radius, angle_offset, color):
    """Draw a filled regular polygon."""
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False) + angle_offset
    pts = np.array(
        [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [pts], color)


def draw_shape(frame, shape, cx, cy, radius, color):
    """Draw `shape` centred at (cx, cy) with given radius and color."""

    if shape == "circle":
        cv2.circle(frame, (cx, cy), radius, color, -1)

    elif shape == "rect":
        cv2.rectangle(
            frame,
            (cx - radius, cy - radius),
            (cx + radius, cy + radius),
            color, -1,
        )

    elif shape == "donut":
        cv2.circle(frame, (cx, cy), radius, color, -1)
        # Punch a hole — use black, ~40 % of outer radius
        hole_r = max(1, int(radius * 0.42))
        cv2.circle(frame, (cx, cy), hole_r, (0, 0, 0), -1)

    elif shape == "diamond":
        # 4-sided polygon rotated 45°
        _polygon(frame, cx, cy, 4, radius, np.pi / 4, color)

    elif shape == "triangle":
        # Equilateral, pointing up
        _polygon(frame, cx, cy, 3, radius, -np.pi / 2, color)

    elif shape == "star":
        # 5-point star: alternate outer/inner vertices
        outer_r = radius
        inner_r = max(1, int(radius * 0.42))
        pts = []
        for i in range(10):
            r = outer_r if i % 2 == 0 else inner_r
            a = np.pi * i / 5 - np.pi / 2
            pts.append((int(cx + r * np.cos(a)), int(cy + r * np.sin(a))))
        cv2.fillPoly(frame, [np.array(pts, dtype=np.int32)], color)

    elif shape == "cross":
        arm = radius
        thickness = max(2, radius // 3)
        cv2.rectangle(frame, (cx - arm, cy - thickness),
                              (cx + arm, cy + thickness), color, -1)
        cv2.rectangle(frame, (cx - thickness, cy - arm),
                              (cx + thickness, cy + arm), color, -1)

    else:
        raise ValueError(f"Unknown shape: {shape!r}")


SHAPES = ["circle", "rect", "donut", "diamond", "triangle", "star", "cross"]


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
    """
    Returns:
        frames : np.ndarray  (T, H, W, 3)  uint8
        states : np.ndarray  (T, 4)         float32  [x, y, vx, vy]
    """
    rng = np.random.default_rng(seed)

    def random_color():
        return tuple(rng.integers(80, 255, size=3).tolist())

    c = random_color() if color is None else color

    # Start well inside so the shape is fully visible
    margin = radius + 1
    x  = rng.uniform(margin, img_size - margin)
    y  = rng.uniform(margin, img_size - margin)
    vx = rng.choice([-1, 1]) * rng.uniform(1.5, 3.0)
    vy = rng.choice([-1, 1]) * rng.uniform(1.5, 3.0)

    frames = []
    states = []

    for _ in range(n_frames):
        x += vx
        y += vy

        bounced = False
        if x - radius < 0:
            vx = abs(vx)
            x = float(radius)
            bounced = True
        elif x + radius > img_size - 1:
            vx = -abs(vx)
            x = float(img_size - 1 - radius)
            bounced = True

        if y - radius < 0:
            vy = abs(vy)
            y = float(radius)
            bounced = True
        elif y + radius > img_size - 1:
            vy = -abs(vy)
            y = float(img_size - 1 - radius)
            bounced = True

        if bounced and change_color_on_bounce:
            c = random_color()

        frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        draw_shape(frame, shape, int(round(x)), int(round(y)), radius, c)
        frames.append(frame)
        states.append([x, y, vx, vy])

    return np.stack(frames), np.array(states, dtype=np.float32)


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