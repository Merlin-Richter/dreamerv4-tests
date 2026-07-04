"""Playable Memory Maze rendered by the dynamics model (spec: specs/interactive/play_memmaze.md).

The neural twin of external/memory-maze/gui/run_gui.py: same pygame window, same keymap
(noop/UP/LEFT/RIGHT/UP+LEFT/UP+RIGHT = the dataset's actions 0..5), same fps pacing — but frames
come from the trained dynamics model's carrying rollout in the frozen tokenizer's latent space
(local GPU) instead of MuJoCo. Reset seeds the model with a full dynamics context window of REAL
episode frames (replayed on screen so you see the walk the model saw), then every tick is one
committed rollout_step conditioned on your keyboard action. You play inside the world model.

Run from repo root (repo venv — CUDA torch + pygame):
    venv/Scripts/python.exe -u src/interactive/play_memmaze.py \
        --checkpoint checkpoints/memmaze/dynamics_vanilla.pt \
        --tokenizer checkpoints/memmaze/tokenizer.pt

Headless smoke test (no window, scripted actions, prints per-step latency):
    venv/Scripts/python.exe -u src/interactive/play_memmaze.py --selftest 12 \
        --checkpoint ... --tokenizer ...
"""

import argparse
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]   # .../src (the package root)
_ROOT = _SRC.parent                          # repo root
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.gridworld.recall import _tokenizer_window            # noqa: E402
from interactive.play_dynamics import load_dynamics, load_tokenizer  # noqa: E402

PANEL_LEFT = 250
PANEL_RIGHT = 250
K_NONE = tuple()
ACTION_NAMES = {0: "noop", 1: "forward", 2: "left", 3: "right", 4: "fwd+left", 5: "fwd+right"}


def get_keymap(pygame):
    """Identical to run_gui.py: held-key combos -> the dataset's 6 discrete actions."""
    return {
        tuple(): 0,
        (pygame.K_UP,): 1,
        (pygame.K_LEFT,): 2,
        (pygame.K_RIGHT,): 3,
        (pygame.K_UP, pygame.K_LEFT): 4,
        (pygame.K_UP, pygame.K_RIGHT): 5,
    }


class RolloutGame:
    """The carried dynamics rollout behind the window: reset = encode + rollout_init a full real
    context window; step = ONE committed rollout_step + trailing-window decode. pygame-free."""

    def __init__(self, *, frames, actions, ids, tokenizer, model, device,
                 n_ctx, decode_ctx, K, window=None, episode=None, seed=None):
        self.frames, self.actions, self.ids = frames, actions, ids
        self.tokenizer, self.model, self.device = tokenizer, model, device
        self.K, self.decode_ctx, self.episode = K, decode_ctx, episode
        self.max_ctx = None if window is None else max(1, window - 1)
        self.window = window if window is not None else model.config.max_temporal_length
        self.rng = random.Random(seed)

        tok_w = _tokenizer_window(tokenizer)
        self.n_ctx = n_ctx if n_ctx is not None else model.config.max_temporal_length
        assert self.n_ctx <= tok_w, f"n_ctx={self.n_ctx} exceeds the tokenizer window {tok_w} (one-shot encode)"
        if frames.shape[1] < self.n_ctx:
            raise ValueError(f"Episodes have T={frames.shape[1]} < n_ctx={self.n_ctx} frames.")

        self.state = None            # carried rollout state (rollout_init dict)
        self.lat_buf = None          # (1, <=decode_ctx, L, D) trailing latents for the decoder
        self.replay_pos = 0          # next context frame to show; >= n_ctx -> play phase
        self.steps = 0               # generated frames so far
        self.step_ms = None          # last model step latency (ms)
        self.last_action = 0
        self.current = None          # displayed frame, uint8 (H, W, 3), stored channel order

    @property
    def in_replay(self) -> bool:
        return self.replay_pos < self.n_ctx

    def episode_label(self) -> str:
        return f"{self.ep}" + (f" (id {int(self.ids[self.ep])})" if self.ids is not None else "")

    @torch.no_grad()
    def reset(self):
        self.ep = self.episode if self.episode is not None else self.rng.randrange(len(self.frames))
        self.start = self.rng.randrange(self.frames.shape[1] - self.n_ctx + 1)

        clip = np.asarray(self.frames[self.ep, self.start:self.start + self.n_ctx])
        x = torch.from_numpy(clip.astype(np.float32) / 255.0).unsqueeze(0).to(self.device)
        ctx_lat = self.tokenizer.encoder(x)
        ctx_act = None
        if self.actions is not None and self.model.n_actions > 0:
            a = np.asarray(self.actions[self.ep, self.start:self.start + self.n_ctx]).astype(np.int64)
            ctx_act = torch.from_numpy(a).unsqueeze(0).to(self.device)
        self.state = self.model.rollout_init(ctx_lat, ctx_act, K=self.K, max_ctx=self.max_ctx)
        self.lat_buf = ctx_lat[:, -self.decode_ctx:]

        self._ctx_frames = clip                       # raw dataset frames for the replay
        self._ctx_actions = (np.asarray(self.actions[self.ep, self.start:self.start + self.n_ctx])
                             if self.actions is not None else np.zeros(self.n_ctx, np.int64))
        self.replay_pos = self.steps = 0
        self.step_ms = None
        self.advance_replay()
        print(f"reset: episode {self.episode_label()}  start t={self.start}  "
              f"ctx {self.n_ctx} frames  window {self.window}", flush=True)

    def advance_replay(self):
        """Show the next REAL context frame (already committed in rollout_init)."""
        self.current = self._ctx_frames[self.replay_pos]
        self.last_action = int(self._ctx_actions[self.replay_pos])
        self.replay_pos += 1

    @torch.no_grad()
    def step(self, action: int):
        """One committed rollout_step conditioned on the player's action, then decode the last
        frame of the trailing latent window (the temporal decoder needs past context)."""
        t0 = time.perf_counter()
        a = None
        if self.model.n_actions > 0:
            a = torch.tensor([[action]], device=self.device, dtype=torch.long)
        nxt = self.model.rollout_step(self.state, a, commit=True)
        self.lat_buf = torch.cat((self.lat_buf, nxt), dim=1)[:, -self.decode_ctx:]
        frame = self.tokenizer.decoder(self.lat_buf)[0, -1].clamp(0.0, 1.0)
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.step_ms = (time.perf_counter() - t0) * 1000.0
        self.current = (frame.cpu().float().numpy() * 255.0).round().astype(np.uint8)
        self.last_action = int(action)
        self.steps += 1


def stats_text(game, model, device):
    kvs = [("## Stats ##", "")]
    kvs.append(("", ""))
    kvs.append(("episode", game.episode_label()))
    kvs.append(("start t", game.start))
    if game.in_replay:
        kvs.append(("phase", "CTX REPLAY"))
        kvs.append(("ctx frame", f"{game.replay_pos}/{game.n_ctx}"))
    else:
        kvs.append(("phase", "ROLLOUT"))
        kvs.append(("step", game.steps))
    kvs.append(("action", ACTION_NAMES.get(game.last_action, game.last_action)))
    if game.step_ms is not None:
        kvs.append(("model ms", f"{game.step_ms:.0f}"))
        kvs.append(("model fps", f"{1000.0 / max(game.step_ms, 1e-6):.1f}"))
    kvs.append(("window", game.window))
    kvs.append(("memory", f"n_mem={model.n_memory}" if model.n_memory > 0 else "vanilla"))
    kvs.append(("device", device))
    return [f"{k:<11} {v!s:>10}" for k, v in kvs]


def keymap_text():
    kvs = [("## Commands ##", ""), ("", ""),
           ("forward", "up arrow"), ("left", "left arrow"), ("right", "right arrow"),
           ("", ""),
           ("reset", "backspace"), ("pause", "space"), ("speed up", "tab"), ("quit", "esc")]
    return [f"{k:<15} {v}" for k, v in kvs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Playable Memory Maze rendered by the dynamics model.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Dynamics checkpoint.")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Frozen tokenizer checkpoint.")
    parser.add_argument("--frames", type=Path, default=_ROOT / "data" / "memmaze9x9_val12.npy",
                        help="Frames .npy (N, T, H, W, 3) uint8 (held-out episodes).")
    parser.add_argument("--actions", type=Path, default=None,
                        help="Actions .npy (N, T). Default: '<frames>_actions.npy' if present.")
    parser.add_argument("--n-ctx", type=int, default=None,
                        help="REAL context frames committed before play (default: the model's window).")
    parser.add_argument("--decode-ctx", type=int, default=16,
                        help="Trailing latents fed to the temporal decoder each frame.")
    parser.add_argument("--K", type=int, default=4, help="Shortcut denoising steps per frame.")
    parser.add_argument("--window", type=int, default=None,
                        help="Force a shorter sliding context window (total frames); default native.")
    parser.add_argument("--episode", type=int, default=None, help="Fix the episode index.")
    parser.add_argument("--seed", type=int, default=None, help="Seed episode/offset sampling.")
    parser.add_argument("--size", type=int, nargs=2, default=(600, 600))
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--nonoop", action="store_true", help="No keys = pause instead of noop.")
    parser.add_argument("--selftest", type=int, default=None, metavar="N",
                        help="Headless smoke mode: scripted actions, quit after N generated frames.")
    args = parser.parse_args()

    if args.selftest:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame          # viewer-only dependency; after the SDL driver choice
    import pygame.freetype

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_dynamics(args.checkpoint, device)
    tokenizer = load_tokenizer(args.tokenizer, device)

    frames = np.load(args.frames, mmap_mode="r")
    if frames.ndim != 5 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames (N, T, H, W, 3), got {frames.shape}")
    actions = None
    act_path = args.actions or args.frames.with_name(args.frames.stem + "_actions.npy")
    if Path(act_path).is_file():
        actions = np.load(act_path, mmap_mode="r")
    elif model.n_actions > 0:
        raise SystemExit(f"ERROR: model is action-conditioned (n_actions={model.n_actions}) but no "
                         f"actions file at {act_path} — an unconditioned memmaze rollout is meaningless.")
    if model.n_actions == 0:
        print("!! unconditioned model — action keys have no effect (free run).", flush=True)
    ids_path = args.frames.with_name(args.frames.stem + "_ids.npy")
    ids = np.load(ids_path) if ids_path.is_file() else None

    game = RolloutGame(frames=frames, actions=actions, ids=ids, tokenizer=tokenizer, model=model,
                       device=device, n_ctx=args.n_ctx, decode_ctx=args.decode_ctx, K=args.K,
                       window=args.window, episode=args.episode, seed=args.seed)

    render_size = tuple(args.size)
    window_size = (render_size[0] + PANEL_LEFT + PANEL_RIGHT, render_size[1])
    pygame.init()
    screen = pygame.display.set_mode(window_size, pygame.FULLSCREEN if args.fullscreen else 0)
    pygame.display.set_caption("Memory Maze — dynamics rollout")
    clock = pygame.time.Clock()
    font = pygame.freetype.SysFont("Mono", 16)
    fontsmall = pygame.freetype.SysFont("Mono", 12)
    keymap = get_keymap(pygame)

    game.reset()
    running, paused, speedup = True, False, False
    latencies = []
    script = [1, 1, 1, 2, 1, 1, 3, 1]      # selftest action cycle: walk + turns

    while running:
        # Rendering ------------------------------------------------------------------
        screen.fill((64, 64, 64))
        surf = pygame.surfarray.make_surface(game.current.transpose((1, 0, 2)))
        surf = pygame.transform.scale(surf, render_size)
        screen.blit(surf, (PANEL_LEFT, 0))
        if game.in_replay:                                     # green border = real frames
            pygame.draw.rect(screen, (0, 200, 0), (PANEL_LEFT, 0, *render_size), 4)
        y = 5
        for line in stats_text(game, model, device):
            ts, _ = font.render(line, (255, 255, 255))
            screen.blit(ts, (16, y))
            y += font.size + 2
        y = 5
        for line in keymap_text():
            ts, _ = fontsmall.render(line, (255, 255, 255))
            screen.blit(ts, (render_size[0] + PANEL_LEFT + 16, y))
            y += fontsmall.size + 2
        pygame.display.flip()
        clock.tick(0 if (speedup or args.selftest) else args.fps)

        # Keyboard input (identical resolution to run_gui.py) --------------------------
        pygame.event.pump()
        keys_down = defaultdict(bool)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                keys_down[event.key] = True
        keys_hold = pygame.key.get_pressed()

        action = keymap[K_NONE]
        for keys, act in keymap.items():
            if keys and all(keys_hold[key] or keys_down[key] for key in keys):
                action = act                                   # last all-pressed entry wins

        force_reset = False
        speedup = keys_hold[pygame.K_TAB]
        if keys_down[pygame.K_ESCAPE]:
            running = False
        if keys_down[pygame.K_SPACE]:
            paused = not paused
        elif action != keymap[K_NONE]:
            paused = False                                     # unpause on action press
        if keys_down[pygame.K_BACKSPACE]:
            force_reset = True

        if paused:
            continue
        if force_reset:
            game.reset()
            continue

        # Advance --------------------------------------------------------------------
        if game.in_replay:                                     # memorization phase: real frames
            game.advance_replay()
            continue
        if args.selftest:
            if game.steps >= args.selftest:
                running = False
                continue
            action = script[game.steps % len(script)]
        elif action == keymap[K_NONE] and args.nonoop:
            continue
        game.step(action)
        latencies.append(game.step_ms)

    pygame.quit()
    if latencies:
        arr = np.asarray(latencies)
        print(f"{len(arr)} generated frames | model step ms: mean {arr.mean():.0f}  "
              f"median {np.median(arr):.0f}  max {arr.max():.0f}  "
              f"(~{1000.0 / arr.mean():.1f} fps)", flush=True)
    if args.selftest:
        print("SELFTEST OK", flush=True)


if __name__ == "__main__":
    main()
