"""GridWorldV2 dataset — generation, playback, preview (spec: specs/datagen/generate_gridworldv2.md, DRAFT).

Drives `GridWorldV2Env` (src/envs/gridworldv2.py). Outputs (gitignored):
    data/gridworldv2.npy          (N, T, H, W, 3) uint8   frames (BGR)
    data/gridworldv2_actions.npy  (N, T)          uint8   actions 0..6 (n_actions=7 auto-detected)
    data/gridworldv2_states.npy   (N, T, 3)       float32 [col, row, curtain]
    data/gridworldv2_colors.npy   (N, 2)          uint8   [bg_idx, square_idx] (PALETTE order)

Usage:
    python -u src/datagen/generate_gridworldv2.py --n_episodes 5000
    python -u src/datagen/generate_gridworldv2.py --debug                 # preview one episode
    python -u src/datagen/generate_gridworldv2.py --frames data/gridworldv2.npy --episode 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]  # .../src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from datagen.generate_gridworld import _playback_loop  # noqa: E402  (shared cv2 viewer)
from envs.gridworldv2 import (A_HIDE, A_REVEAL, COLOR_NAMES, GridWorldV2Env,  # noqa: E402
                              sample_moves)


def make_action_schedule(rng: np.random.Generator, n_frames: int,
                         revealed_run: tuple[int, int] = (3, 10),
                         occluded_run: tuple[int, int] = (2, 12),
                         start_visible: int = 3, run_max: int = 4) -> np.ndarray:
    """Alternating revealed/occluded runs (see spec). An occluded run = one hide tick +
    (O-1) movement ticks; a revealed run = one reveal tick + (R-1) movement ticks. Movement
    ticks come from ONE per-episode `sample_moves` stream (matches the recall eval policy)."""
    moves = iter(sample_moves(rng, n_frames, run_max=run_max))
    actions = np.empty(n_frames, dtype=np.uint8)
    t = 0
    for _ in range(min(start_visible, n_frames)):
        actions[t] = next(moves)
        t += 1
    occluded = True  # first alternating run after the visible prefix is occluded
    while t < n_frames:
        toggle = A_HIDE if occluded else A_REVEAL
        lo, hi = occluded_run if occluded else revealed_run
        run = int(rng.integers(lo, hi + 1))
        actions[t] = toggle
        t += 1
        for _ in range(min(run - 1, n_frames - t)):
            actions[t] = next(moves)
            t += 1
        occluded = not occluded
    return actions


def generate_episode(n_frames: int = 200, img_size: int = 64, seed: int = 42):
    """Returns (frames (T,H,W,3) uint8, actions (T,) uint8, states (T,3) float32,
    colors (2,) uint8 = [bg_idx, square_idx] in PALETTE order)."""
    env = GridWorldV2Env(img_size=img_size)
    env.reset(seed=seed)
    actions = make_action_schedule(env.rng, n_frames)
    frames = np.empty((n_frames, img_size, img_size, 3), dtype=np.uint8)
    states = np.empty((n_frames, 3), dtype=np.float32)
    for t in range(n_frames):
        frames[t], states[t] = env.step(int(actions[t]))
    colors = np.array(
        [COLOR_NAMES.index(env.bg_name), COLOR_NAMES.index(env.color_name)], dtype=np.uint8
    )
    return frames, actions, states, colors


def write_dataset(n_episodes: int, n_frames: int, out: Path, seed0: int = 0) -> None:
    frames = np.empty((n_episodes, n_frames, 64, 64, 3), dtype=np.uint8)
    actions = np.empty((n_episodes, n_frames), dtype=np.uint8)
    states = np.empty((n_episodes, n_frames, 3), dtype=np.float32)
    colors = np.empty((n_episodes, 2), dtype=np.uint8)
    for i in range(n_episodes):
        frames[i], actions[i], states[i], colors[i] = generate_episode(n_frames, seed=seed0 + i)
        if (i + 1) % 500 == 0 or i + 1 == n_episodes:
            print(f"  {i + 1}/{n_episodes} episodes")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, frames)
    np.save(out.with_name(out.stem + "_actions.npy"), actions)
    np.save(out.with_name(out.stem + "_states.npy"), states)
    np.save(out.with_name(out.stem + "_colors.npy"), colors)
    occ = float((states[..., 2] == 1).mean())
    print(f"wrote {out} {frames.shape} (+_actions/_states/_colors)  occluded-frame fraction {occ:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="GridWorldV2 dataset writer / preview.")
    ap.add_argument("--n_episodes", type=int, default=1000)
    ap.add_argument("--n_frames", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("data/gridworldv2.npy"))
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--debug", action="store_true", help="cv2 preview of one scheduled episode.")
    ap.add_argument("--frames", type=Path, default=None, help="Play back a saved dataset.")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    if args.debug:
        fr, ac, st, _ = generate_episode(args.n_frames, seed=args.seed0)
        occl = (ac == A_HIDE).sum()
        print(f"preview: {len(fr)} frames, {occl} hide ticks; actions head: {ac[:24].tolist()}")
        _playback_loop(fr, (st[:, 2] > 0).astype(np.uint8), 64, args.fps, "gridworldv2 preview")
        return
    if args.frames is not None:
        fr = np.load(args.frames, mmap_mode="r")[args.episode]
        st = np.load(args.frames.with_name(args.frames.stem + "_states.npy"))[args.episode]
        _playback_loop(np.asarray(fr), (st[:, 2] > 0).astype(np.uint8), 64, args.fps,
                       f"gridworldv2 ep {args.episode}")
        return
    write_dataset(args.n_episodes, args.n_frames, args.out, seed0=args.seed0)


if __name__ == "__main__":
    main()
