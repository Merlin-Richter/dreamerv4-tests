"""
GridWorld occluded-memory dataset — generation, playback, and preview.

Drives `GridWorldEnv` (src/envs/gridworld.py) to write the discrete memory dataset and to
inspect it. The env (physics + rendering) lives in `src/envs/gridworld.py`; this module is the
data-generation + viewer layer on top.

Outputs (saved under data/ by default; gitignored):
    data/gridworld.npy         (N, T, H, W, 3)  uint8    frames (BGR)
    data/gridworld_actions.npy (N, T)           uint8    0 = up/revealed, 1 = down/occluded
    data/gridworld_states.npy  (N, T, 5)        float32  [col, row, dcol, drow, curtain]
    data/gridworld_colors.npy  (N, 2)           uint8    [bg_color_idx, square_color_idx] (PALETTE order)

The categorical colors are stored per-episode (they are constant within an episode) so evals can
score 4-way color recall without re-deriving them from pixels.

Usage:
    python -u src/datagen/generate_gridworld.py --n_episodes 1000        # -> data/gridworld.npy
    python -u src/datagen/generate_gridworld.py --play              # interactive: you control the curtain
    python -u src/datagen/generate_gridworld.py --debug             # preview one scheduled episode
    python -u src/datagen/generate_gridworld.py --frames data/gridworld.npy --episode 0   # play saved
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

from envs.gridworld import COLOR_NAMES, CURTAIN_COLOR, GridWorldEnv  # noqa: E402

WINDOW_TITLE = "GridWorld [Q/ESC=quit  SPACE=pause]"


# ---------------------------------------------------------------------------
# Curtain schedule (block sampler: see generate_gridworld.md)
# ---------------------------------------------------------------------------

def make_curtain_schedule(
    rng: np.random.Generator,
    n_frames: int,
    p_single: float = 0.90,
    p_run_visible: float = 0.05,
    p_run_occluded: float = 0.05,
    run_len: int = 8,
    single_p_occlude: float = 0.5,
    start_visible: int = 2,
) -> np.ndarray:
    """Block sampler (Merlin's spec). At each decision point draw one of three blocks:
        - p_single       (0.90): emit ONE frame with a random curtain (single_p_occlude => occluded)
        - p_run_visible  (0.05): emit `run_len` (8) REVEALED frames in a row
        - p_run_occluded (0.05): emit `run_len` (8) OCCLUDED frames in a row
    So the dataset contains both the common 1-step mix AND long stretches of either scenario.

    `start_visible` forces the first frames revealed so the square's direction is observable
    before any occlusion (set 0 to disable). Probabilities should sum to 1.
    """
    curtain = np.zeros(n_frames, dtype=np.uint8)
    t = min(max(0, start_visible), n_frames)  # forced-revealed prefix stays 0
    while t < n_frames:
        u = rng.random()
        if u < p_single:
            curtain[t] = 1 if rng.random() < single_p_occlude else 0
            t += 1
        elif u < p_single + p_run_visible:
            t += run_len  # revealed run: leave zeros, just advance
        else:
            curtain[t : t + run_len] = 1
            t += run_len
    return curtain[:n_frames]


# ---------------------------------------------------------------------------
# Episode generator
# ---------------------------------------------------------------------------

def generate_episode(n_frames: int = 200, img_size: int = 64, seed: int = 42):
    """Returns (frames (T,H,W,3) uint8, actions (T,) uint8, states (T,5) float32,
    colors (2,) uint8 = [bg_idx, square_idx] in PALETTE order)."""
    env = GridWorldEnv(img_size=img_size)
    env.reset(seed=seed)
    curtain = make_curtain_schedule(env.rng, n_frames)

    frames = np.empty((n_frames, img_size, img_size, 3), dtype=np.uint8)
    states = np.empty((n_frames, 5), dtype=np.float32)
    for t in range(n_frames):
        frames[t], states[t] = env.step(int(curtain[t]))
    colors = np.array(
        [COLOR_NAMES.index(env.bg_name), COLOR_NAMES.index(env.color_name)], dtype=np.uint8
    )
    return frames, curtain.copy(), states, colors


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
    print("Scheduled episode preview.  Q/ESC=quit  SPACE=pause.  Red border = occluded.")
    frames, actions, _, _ = generate_episode(
        n_frames=args.n_frames, img_size=args.img_size, seed=args.seed
    )
    _playback_loop(frames, actions, args.img_size, args.fps, header="scheduled")


def interactive_play(args):
    print("Interactive — square steps continuously; you choose the curtain each frame.")
    print("  U/0 curtain up (revealed)   D/1 curtain down (occluded)   R reset   SPACE pause   Q/ESC quit")
    env = GridWorldEnv(img_size=args.img_size)
    env.reset(seed=args.seed)
    action, frame_idx = 0, 0
    frame = np.full((args.img_size, args.img_size, 3), CURTAIN_COLOR, dtype=np.uint8)
    scale = _display_scale(args.img_size)
    paused = False
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
    while True:
        if not paused and frame_idx < args.n_frames:
            frame, _ = env.step(action)
            frame_idx += 1
        label = f"interactive  frame {frame_idx}/{args.n_frames}  action={action}" + (
            "  [PAUSED]" if paused else ""
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
            action, frame_idx = 0, 0
            frame = np.full((args.img_size, args.img_size, 3), CURTAIN_COLOR, dtype=np.uint8)
    cv2.destroyAllWindows()


def play_saved(args):
    frames_path = Path(args.frames)
    if not frames_path.exists():
        raise FileNotFoundError(f"Frames file not found: {frames_path}")
    frames = np.load(frames_path, mmap_mode="r")
    actions_path = frames_path.with_name(frames_path.stem + "_actions.npy")
    actions = np.load(actions_path, mmap_mode="r") if actions_path.exists() else np.zeros(
        frames.shape[1], dtype=np.uint8
    )
    ep = args.episode
    if ep < 0 or ep >= frames.shape[0]:
        raise ValueError(f"--episode {ep} out of range [0, {frames.shape[0] - 1}]")
    _, _, h, _, _ = frames.shape
    print(f"Playing saved episode {ep} from {frames_path}  ({frames.shape[1]} frames)")
    print("Q/ESC=quit  SPACE=pause.  Red border = occluded.")
    _playback_loop(frames[ep], actions[ep], img_size=h, fps=args.fps, header=f"saved ep {ep}")


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(args):
    rng_meta = np.random.default_rng(args.seed)
    out_frames = args.out
    Path(out_frames).parent.mkdir(parents=True, exist_ok=True)
    out_actions = args.out.replace(".npy", "_actions.npy")
    out_states = args.out.replace(".npy", "_states.npy")
    out_colors = args.out.replace(".npy", "_colors.npy")

    # Frames can be many GB (3000x200x64x64x3 ~= 7.4 GB) -> stream straight to a disk-backed
    # memmap so we never hold the whole array in RAM. The small arrays stay in memory.
    frames_arr = np.lib.format.open_memmap(
        out_frames, mode="w+", dtype=np.uint8,
        shape=(args.n_episodes, args.n_frames, args.img_size, args.img_size, 3),
    )
    actions_arr = np.empty((args.n_episodes, args.n_frames), dtype=np.uint8)
    states_arr = np.empty((args.n_episodes, args.n_frames, 5), dtype=np.float32)
    colors_arr = np.empty((args.n_episodes, 2), dtype=np.uint8)

    t0 = time.time()
    for i in range(args.n_episodes):
        frames, actions, states, colors = generate_episode(
            n_frames=args.n_frames, img_size=args.img_size, seed=int(rng_meta.integers(1 << 31))
        )
        frames_arr[i], actions_arr[i], states_arr[i], colors_arr[i] = frames, actions, states, colors
        if (i + 1) % max(1, args.n_episodes // 10) == 0:
            print(f"  {i+1}/{args.n_episodes} episodes  ({time.time() - t0:.1f}s)")

    frames_arr.flush()
    del frames_arr  # close the memmap (already on disk via open_memmap)
    np.save(out_actions, actions_arr)
    np.save(out_states, states_arr)
    np.save(out_colors, colors_arr)
    frames_arr = np.load(out_frames, mmap_mode="r")  # for the summary stats below

    occ = actions_arr.mean()
    print(f"\nSaved frames  -> {out_frames}   {frames_arr.shape}")
    print(f"Saved actions -> {out_actions}  {actions_arr.shape}  (occluded fraction {occ:.2f})")
    print(f"Saved states  -> {out_states}   {states_arr.shape}")
    print(f"Saved colors  -> {out_colors}   {colors_arr.shape}  (PALETTE order: {COLOR_NAMES})")
    print(f"Disk: frames {frames_arr.nbytes / 1e6:.1f} MB")


def parse_args():
    p = argparse.ArgumentParser(description="GridWorld occluded-memory dataset")
    p.add_argument("--play", action="store_true", help="Interactive window (U/0=up, D/1=down)")
    p.add_argument("--debug", action="store_true", help="Preview one scheduled episode")
    p.add_argument("--frames", default=None, help="Play a saved .npy dataset")
    p.add_argument("--episode", type=int, default=0, help="Episode index for saved playback")
    p.add_argument("--n_episodes", type=int, default=1000)
    p.add_argument("--n_frames", type=int, default=200)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/gridworld.npy")
    p.add_argument("--fps", type=int, default=10, help="Playback FPS")
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
              f"{args.img_size}x{args.img_size}")
        generate_dataset(args)
