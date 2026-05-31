"""
Occluded bouncing-ball dataset — a dense, action-conditioned memory environment.

Why this env:
  * **Dense frames.** Every frame has a smooth per-episode textured background, so the
    reconstruction target is never the degenerate all-black image. (A ~5%-area object on a
    black background is a pathological case for MSE: predicting all-black is a deep local
    optimum. Dense backgrounds remove that trap.)
  * **Actions.** Two *absolute* actions set the curtain state for the current frame:
        action[t] = 0  -> curtain UP   (revealed: background + ball visible)
        action[t] = 1  -> curtain DOWN (occluded: an opaque curtain hides the whole frame)
    Absolute (not toggle) actions are Markov in the action, so the curtain itself needs no
    memory — only the *hidden ball* does.
  * **Memory.** The ball keeps bouncing (with wall reflections) behind the curtain. To predict
    the frame where the curtain lifts, a model must integrate the ball's hidden position and
    velocity across the occluded stretch — information absent from every occluded frame.

Convention: ``action[t]`` describes frame ``t`` (the curtain state you observe at t), so a
per-frame action token lines up with the frame it explains.

Outputs (saved next to the other datasets, repo root by default):
    occluded.npy         (N, T, H, W, 3)  uint8    frames
    occluded_actions.npy (N, T)           uint8    0 = up/revealed, 1 = down/occluded
    occluded_states.npy  (N, T, 5)        float32  [x, y, vx, vy, curtain]

Usage:
    python occluded_bouncing.py --n_episodes 1000 --out occluded.npy
    python occluded_bouncing.py --play              # interactive: you control the curtain
    python occluded_bouncing.py --debug             # preview one scheduled episode
    python occluded_bouncing.py --play --frames occluded.npy --episode 0
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Background + curtain
# ---------------------------------------------------------------------------

# Curtain is a fixed colour across the whole dataset so "curtain down" is a single, easily
# learned appearance fully determined by the action. Mid-tone (not near-black) on purpose.
CURTAIN_COLOR = (52, 48, 60)  # BGR

WINDOW_TITLE = "Occluded bouncing [Q/ESC=quit  SPACE=pause]"


def make_background(rng: np.random.Generator, img_size: int) -> np.ndarray:
    """Smooth, dense, low-frequency texture: a small random colour grid bilinearly upsampled.

    Low frequency keeps it easy for a patch (ViT) tokenizer while still filling every pixel.
    """
    lo = rng.integers(40, 215, size=(4, 4, 3)).astype(np.uint8)
    bg = cv2.resize(lo, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    return bg


# ---------------------------------------------------------------------------
# Curtain schedule
# ---------------------------------------------------------------------------

def make_curtain_schedule(
    rng: np.random.Generator,
    n_frames: int,
    min_visible: int = 2,
    max_visible: int = 8,
    min_cover: int = 1,
    max_cover: int = 6,
    start_visible: int = 2,
) -> np.ndarray:
    """Alternating runs of revealed/occluded frames.

    Always starts with ``start_visible`` revealed frames so velocity is observable before any
    occlusion. ``max_cover`` is kept short so an occlusion and its surrounding visible frames
    comfortably fit inside the dynamics model's temporal window.
    """
    curtain = np.zeros(n_frames, dtype=np.uint8)
    t = max(0, start_visible)
    covered = True  # first scheduled run after the forced-visible prefix is a cover run
    while t < n_frames:
        if covered:
            run = int(rng.integers(min_cover, max_cover + 1))
            curtain[t : t + run] = 1
        else:
            run = int(rng.integers(min_visible, max_visible + 1))
        t += run
        covered = not covered
    return curtain[:n_frames]


# ---------------------------------------------------------------------------
# Step-by-step environment
# ---------------------------------------------------------------------------

class OccludedBouncingEnv:
    """Single-episode env: physics always runs; the action picks revealed vs occluded render."""

    def __init__(self, img_size: int = 64, radius: int = 10):
        self.img_size = img_size
        self.radius = radius
        self.rng = np.random.default_rng()
        self.bg = None
        self.color = None
        self.x = self.y = self.vx = self.vy = 0.0
        self.t = 0

    def reset(self, seed: int = 42) -> "OccludedBouncingEnv":
        self.rng = np.random.default_rng(seed)
        self.bg = make_background(self.rng, self.img_size)
        self.color = tuple(int(v) for v in self.rng.integers(80, 256, size=3))
        margin = self.radius + 1
        self.x = float(self.rng.uniform(margin, self.img_size - margin))
        self.y = float(self.rng.uniform(margin, self.img_size - margin))
        self.vx = float(self.rng.choice([-1, 1]) * self.rng.uniform(1.5, 3.0))
        self.vy = float(self.rng.choice([-1, 1]) * self.rng.uniform(1.5, 3.0))
        self.t = 0
        return self

    def _advance_physics(self) -> None:
        img_size, radius = self.img_size, self.radius
        self.x += self.vx
        self.y += self.vy
        if self.x - radius < 0:
            self.vx = abs(self.vx)
            self.x = float(radius)
        elif self.x + radius > img_size - 1:
            self.vx = -abs(self.vx)
            self.x = float(img_size - 1 - radius)
        if self.y - radius < 0:
            self.vy = abs(self.vy)
            self.y = float(radius)
        elif self.y + radius > img_size - 1:
            self.vy = -abs(self.vy)
            self.y = float(img_size - 1 - radius)

    def _render(self, action: int) -> np.ndarray:
        if action:
            return np.full((self.img_size, self.img_size, 3), CURTAIN_COLOR, dtype=np.uint8)
        frame = self.bg.copy()
        cv2.circle(
            frame,
            (int(round(self.x)), int(round(self.y))),
            self.radius,
            self.color,
            -1,
        )
        return frame

    def step(self, action: int) -> tuple[np.ndarray, np.ndarray]:
        """Advance one frame. Returns (frame uint8 HWC, state float32[5])."""
        self._advance_physics()
        frame = self._render(action)
        state = np.array(
            (self.x, self.y, self.vx, self.vy, float(action)), dtype=np.float32
        )
        self.t += 1
        return frame, state


# ---------------------------------------------------------------------------
# Episode generator
# ---------------------------------------------------------------------------

def generate_episode(
    n_frames: int = 200,
    img_size: int = 64,
    radius: int = 10,
    seed: int = 42,
):
    """Returns (frames (T,H,W,3) uint8, actions (T,) uint8, states (T,5) float32)."""
    env = OccludedBouncingEnv(img_size=img_size, radius=radius)
    env.reset(seed=seed)
    curtain = make_curtain_schedule(env.rng, n_frames)

    frames = np.empty((n_frames, img_size, img_size, 3), dtype=np.uint8)
    states = np.empty((n_frames, 5), dtype=np.float32)
    for t in range(n_frames):
        frames[t], states[t] = env.step(int(curtain[t]))
    return frames, curtain.copy(), states


# ---------------------------------------------------------------------------
# Playback helpers
# ---------------------------------------------------------------------------

def _display_scale(img_size: int) -> int:
    return max(img_size, 400) // img_size


def _annotate_frame(frame: np.ndarray, action: int, label: str, scale: int) -> np.ndarray:
    h, w = frame.shape[:2]
    big = cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    if action:
        cv2.rectangle(big, (0, 0), (big.shape[1] - 1, big.shape[0] - 1), (0, 0, 255), 3)
    cv2.putText(big, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
    return big


def _playback_loop(frames, actions, img_size: int, fps: int, header: str) -> None:
    scale = _display_scale(img_size)
    paused = False
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
    for i, frame in enumerate(frames):
        label = f"{header}  frame {i + 1}/{len(frames)}  action={actions[i]}"
        big = _annotate_frame(frame, int(actions[i]), label, scale)
        cv2.imshow(WINDOW_TITLE, big)
        while True:
            delay = 0 if paused else max(1, int(1000 / fps))
            key = cv2.waitKey(delay) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                cv2.destroyAllWindows()
                return
            if key == ord(" "):
                paused = not paused
            if not paused:
                break
    cv2.destroyAllWindows()


def debug_play(args):
    print("Scheduled episode preview.")
    print("Press Q/ESC to quit, SPACE to pause.  Red border = curtain down (occluded).")
    frames, actions, _ = generate_episode(
        n_frames=args.n_frames, img_size=args.img_size, radius=args.radius, seed=args.seed
    )
    _playback_loop(frames, actions, args.img_size, args.fps, header="scheduled")


def interactive_play(args):
    print("Interactive mode — ball physics run continuously; you choose the curtain each frame.")
    print("  U / 0  curtain up (revealed)     D / 1  curtain down (occluded)")
    print("  R      reset episode             SPACE  pause")
    print("  Q/ESC  quit")

    env = OccludedBouncingEnv(img_size=args.img_size, radius=args.radius)
    env.reset(seed=args.seed)
    action = 0
    frame_idx = 0
    frame = np.full((args.img_size, args.img_size, 3), CURTAIN_COLOR, dtype=np.uint8)
    scale = _display_scale(args.img_size)
    paused = False
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

    while True:
        if not paused and frame_idx < args.n_frames:
            frame, _ = env.step(action)
            frame_idx += 1

        label = (
            f"interactive  frame {frame_idx}/{args.n_frames}  action={action}"
            + ("  [PAUSED]" if paused else "")
        )
        cv2.imshow(WINDOW_TITLE, _annotate_frame(frame, action, label, scale))

        delay = 0 if paused else max(1, int(1000 / args.fps))
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif key in (ord("u"), ord("U"), ord("0")):
            action = 0
        elif key in (ord("d"), ord("D"), ord("1")):
            action = 1
        elif key in (ord("r"), ord("R")):
            env.reset(seed=args.seed)
            action = 0
            frame_idx = 0
            frame = np.full((args.img_size, args.img_size, 3), CURTAIN_COLOR, dtype=np.uint8)

    cv2.destroyAllWindows()


def play_saved(args):
    frames_path = Path(args.frames)
    if not frames_path.exists():
        raise FileNotFoundError(f"Frames file not found: {frames_path}")

    frames = np.load(frames_path, mmap_mode="r")
    actions_path = frames_path.with_name(frames_path.stem + "_actions.npy")
    if actions_path.exists():
        actions = np.load(actions_path, mmap_mode="r")
    else:
        actions = np.zeros(frames.shape[1], dtype=np.uint8)

    ep = args.episode
    if ep < 0 or ep >= frames.shape[0]:
        raise ValueError(f"--episode {ep} out of range [0, {frames.shape[0] - 1}]")

    episode_frames = frames[ep]
    episode_actions = actions[ep]
    _, _, h, w, _ = frames.shape
    print(f"Playing saved episode {ep} from {frames_path}  ({episode_frames.shape[0]} frames)")
    print("Press Q/ESC to quit, SPACE to pause.  Red border = curtain down (occluded).")
    _playback_loop(
        episode_frames,
        episode_actions,
        img_size=h,
        fps=args.fps,
        header=f"saved ep {ep}",
    )


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(args):
    rng_meta = np.random.default_rng(args.seed)
    frames_arr = np.empty((args.n_episodes, args.n_frames, args.img_size, args.img_size, 3),
                          dtype=np.uint8)
    actions_arr = np.empty((args.n_episodes, args.n_frames), dtype=np.uint8)
    states_arr = np.empty((args.n_episodes, args.n_frames, 5), dtype=np.float32)

    t0 = time.time()
    for i in range(args.n_episodes):
        frames, actions, states = generate_episode(
            n_frames=args.n_frames, img_size=args.img_size, radius=args.radius,
            seed=int(rng_meta.integers(1 << 31)),
        )
        frames_arr[i], actions_arr[i], states_arr[i] = frames, actions, states
        if (i + 1) % max(1, args.n_episodes // 10) == 0:
            print(f"  {i+1}/{args.n_episodes} episodes  ({time.time() - t0:.1f}s)")

    out_frames = args.out
    out_actions = args.out.replace(".npy", "_actions.npy")
    out_states = args.out.replace(".npy", "_states.npy")
    np.save(out_frames, frames_arr)
    np.save(out_actions, actions_arr)
    np.save(out_states, states_arr)

    occ = actions_arr.mean()
    print(f"\nSaved frames  -> {out_frames}   {frames_arr.shape}")
    print(f"Saved actions -> {out_actions}  {actions_arr.shape}  (occluded fraction {occ:.2f})")
    print(f"Saved states  -> {out_states}   {states_arr.shape}")
    print(f"Disk: frames {frames_arr.nbytes / 1e6:.1f} MB")


def parse_args():
    p = argparse.ArgumentParser(description="Occluded bouncing-ball memory dataset")
    p.add_argument("--play", action="store_true",
                   help="Interactive window: control the curtain (U/0=up, D/1=down)")
    p.add_argument("--debug", action="store_true",
                   help="Preview one randomly scheduled episode instead of saving")
    p.add_argument("--frames", default=None,
                   help="Play a saved .npy dataset (use with --play or --debug)")
    p.add_argument("--episode", type=int, default=0,
                   help="Episode index when playing a saved dataset")
    p.add_argument("--n_episodes", type=int, default=1000)
    p.add_argument("--n_frames", type=int, default=200)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--radius", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="occluded.npy")
    p.add_argument("--fps", type=int, default=20, help="Playback FPS")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.frames:
        play_saved(args)
    elif args.play:
        interactive_play(args)
    elif args.debug:
        debug_play(args)
    else:
        print(f"Generating {args.n_episodes} eps x {args.n_frames} frames @ "
              f"{args.img_size}x{args.img_size}  radius={args.radius}")
        generate_dataset(args)
