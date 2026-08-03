#!/usr/bin/env python3
"""Play Memory Maze inside the trained community Dreamer 4 world model.

The controls and reset flow mirror ``src/interactive/play_memmaze.py``. Reset encodes a short
held-out episode prefix with the community tokenizer and replays those real frames with a green
border. Once the replay ends, every displayed frame is generated autoregressively from the
player's action; no recorded future frames are consulted.

The external checkout is deliberately not vendored. Pass ``--dreamer4`` or place the pinned
checkout below ``runs/dreamer4-community-baseline/upstream-*``.
"""
from __future__ import annotations

import argparse
import importlib
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import ModuleType

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
PANEL_LEFT = 250
PANEL_RIGHT = 250
ACTION_NAMES = {
    0: "noop",
    1: "forward",
    2: "left",
    3: "right",
    4: "fwd+left",
    5: "fwd+right",
}


def get_keymap(pygame):
    """Held-key combinations mapped to the native six-way Memory Maze actions."""
    return {
        tuple(): 0,
        (pygame.K_UP,): 1,
        (pygame.K_LEFT,): 2,
        (pygame.K_RIGHT,): 3,
        (pygame.K_UP, pygame.K_LEFT): 4,
        (pygame.K_UP, pygame.K_RIGHT): 5,
    }


def resolve_dreamer4(path: Path | None) -> Path:
    candidates: list[Path] = []
    if path is not None:
        candidates.append(path)
    if os.environ.get("D4_ROOT"):
        candidates.append(Path(os.environ["D4_ROOT"]))
    candidates.extend(sorted((ROOT / "runs" / "dreamer4-community-baseline").glob("upstream-*")))
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "dreamer4" / "model.py").is_file():
            return candidate
        if candidate.name == "dreamer4" and (candidate / "model.py").is_file():
            return candidate.parent
    searched = ", ".join(str(x) for x in candidates) or "no checkout candidates"
    raise SystemExit(
        "ERROR: community Dreamer 4 checkout not found. Pass --dreamer4 PATH to the pinned "
        f"b8abafbf checkout (searched: {searched})."
    )


def import_dreamer4_model(root: Path) -> ModuleType:
    source = root / "dreamer4"
    sys.path.insert(0, str(source))
    model = importlib.import_module("model")
    required = (
        "Encoder", "Decoder", "Tokenizer", "Dynamics", "temporal_patchify",
        "temporal_unpatchify", "pack_bottleneck_to_spatial", "unpack_spatial_to_bottleneck",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise RuntimeError(f"incompatible Dreamer 4 checkout; model.py lacks {missing}")
    return model


def _load_models(
    model_mod: ModuleType,
    *,
    tokenizer_path: Path,
    dynamics_path: Path,
    device: torch.device,
):
    tok_ckpt = torch.load(tokenizer_path, map_location="cpu", weights_only=False)
    tok_cfg = dict(tok_ckpt.get("args", {}))
    H = int(tok_cfg.get("H", 128))
    W = int(tok_cfg.get("W", 128))
    C = int(tok_cfg.get("C", 3))
    patch = int(tok_cfg.get("patch", 4))
    n_latents = int(tok_cfg.get("n_latents", 16))
    d_bottleneck = int(tok_cfg.get("d_bottleneck", 32))
    n_patches = (H // patch) * (W // patch)
    d_patch = patch * patch * C
    common = dict(
        d_model=int(tok_cfg.get("d_model", 256)),
        n_heads=int(tok_cfg.get("n_heads", 4)),
        depth=int(tok_cfg.get("depth", 8)),
        dropout=0.0,
        mlp_ratio=float(tok_cfg.get("mlp_ratio", 4.0)),
        time_every=int(tok_cfg.get("time_every", 1)),
        latents_only_time=bool(tok_cfg.get("latents_only_time", True)),
        scale_pos_embeds=bool(tok_cfg.get("scale_pos_embeds", True)),
    )
    encoder = model_mod.Encoder(
        patch_dim=d_patch,
        n_latents=n_latents,
        n_patches=n_patches,
        d_bottleneck=d_bottleneck,
        mae_p_min=0.0,
        mae_p_max=0.0,
        **common,
    )
    decoder = model_mod.Decoder(
        d_bottleneck=d_bottleneck,
        n_latents=n_latents,
        n_patches=n_patches,
        d_patch=d_patch,
        **common,
    )
    tokenizer = model_mod.Tokenizer(encoder, decoder)
    tokenizer.load_state_dict(tok_ckpt["model"], strict=True)
    tokenizer = tokenizer.to(device).eval()
    tokenizer.requires_grad_(False)

    dyn_ckpt = torch.load(dynamics_path, map_location="cpu", weights_only=False)
    dyn_cfg = dict(dyn_ckpt.get("args", {}))
    if not bool(dyn_cfg.get("use_actions", False)):
        raise AssertionError("dynamics checkpoint is not action-conditioned")
    packing = int(dyn_cfg.get("packing_factor", 2))
    if n_latents % packing:
        raise AssertionError(f"n_latents={n_latents} is not divisible by packing_factor={packing}")
    n_spatial = n_latents // packing
    dynamics = model_mod.Dynamics(
        d_model=int(dyn_cfg.get("d_model_dyn", 512)),
        d_bottleneck=d_bottleneck,
        d_spatial=d_bottleneck * packing,
        n_spatial=n_spatial,
        n_register=int(dyn_cfg.get("n_register", 4)),
        n_agent=int(dyn_cfg.get("n_agent", 1)),
        n_heads=int(dyn_cfg.get("n_heads", 4)),
        depth=int(dyn_cfg.get("dyn_depth", 8)),
        k_max=int(dyn_cfg.get("k_max", 8)),
        dropout=float(dyn_cfg.get("dropout", 0.0)),
        mlp_ratio=float(dyn_cfg.get("mlp_ratio", 4.0)),
        time_every=int(dyn_cfg.get("time_every", 1)),
        space_mode=str(dyn_cfg.get("space_mode", "wm_agent_isolated")),
        scale_pos_embeds=bool(dyn_cfg.get("scale_pos_embeds", False)),
    )
    dynamics.load_state_dict(dyn_ckpt["dynamics"], strict=True)
    dynamics = dynamics.to(device).eval()
    dynamics.requires_grad_(False)

    info = {
        "H": H,
        "W": W,
        "C": C,
        "patch": patch,
        "n_latents": n_latents,
        "d_bottleneck": d_bottleneck,
        "packing": packing,
        "n_spatial": n_spatial,
        "k_max": int(dyn_cfg.get("k_max", 8)),
        "seq_len": int(dyn_cfg.get("seq_len", 32)),
        "checkpoint_step": int(dyn_ckpt.get("step", 0)),
    }
    if (H, W, C) != (64, 64, 3):
        raise AssertionError(f"expected native 64x64 RGB Memory Maze checkpoint, got {(H, W, C)}")
    return tokenizer.encoder, tokenizer.decoder, dynamics, info


def make_schedule(k_max: int, K: int) -> dict[str, object]:
    if K <= 0 or K > k_max or K & (K - 1) or k_max % K:
        raise ValueError(f"K must be a power of two dividing k_max={k_max}, got {K}")
    return {
        "K": K,
        "e": int(round(math.log2(K))),
        "dt": 1.0 / K,
        "tau": [i / K for i in range(K)],
        "tau_idx": [i * (k_max // K) for i in range(K)],
    }


def _one_hot(action: int, device: torch.device) -> torch.Tensor:
    if action not in ACTION_NAMES:
        raise ValueError(f"invalid Memory Maze action {action}")
    out = torch.zeros(16, device=device, dtype=torch.float32)
    out[action] = 1.0
    return out


def _action_ids(actions: np.ndarray) -> np.ndarray:
    if actions.ndim == 2:
        ids = actions.astype(np.int64, copy=False)
    elif actions.ndim == 3:
        ids = actions.argmax(axis=-1).astype(np.int64, copy=False)
    else:
        raise ValueError(f"expected actions (N,T) or (N,T,A), got {actions.shape}")
    if ids.size and (ids.min() < 0 or ids.max() >= len(ACTION_NAMES)):
        raise ValueError(f"actions must be in [0,{len(ACTION_NAMES) - 1}]")
    return ids


class RolloutGame:
    """Pygame-independent state machine for context replay and imagined rollout."""

    def __init__(
        self,
        *,
        frames: np.ndarray,
        actions: np.ndarray,
        ids: np.ndarray | None,
        model_mod: ModuleType,
        encoder,
        decoder,
        dynamics,
        info: dict[str, int],
        device: torch.device,
        n_ctx: int,
        decode_ctx: int,
        context_window: int,
        K: int,
        amp: bool,
        episode: int | None,
        start: int | None,
        seed: int | None,
    ):
        self.frames = frames
        self.action_ids = _action_ids(actions)
        self.ids = ids
        self.model_mod = model_mod
        self.encoder = encoder
        self.decoder = decoder
        self.dynamics = dynamics
        self.info = info
        self.device = device
        self.n_ctx = int(n_ctx)
        self.decode_ctx = int(decode_ctx)
        self.context_window = int(context_window)
        self.schedule = make_schedule(info["k_max"], K)
        self.amp = bool(amp and device.type == "cuda")
        self.fixed_episode = episode
        self.fixed_start = start
        self.rng = random.Random(seed)

        if frames.ndim != 5 or frames.shape[-1] != 3:
            raise ValueError(f"expected frames (N,T,H,W,3), got {frames.shape}")
        if self.action_ids.shape[:2] != frames.shape[:2]:
            raise ValueError(f"frame/action shape mismatch: {frames.shape[:2]} vs {self.action_ids.shape}")
        if not 1 <= self.n_ctx <= self.context_window:
            raise ValueError(f"n_ctx must be in [1, context_window={self.context_window}]")
        if not 1 <= self.context_window < info["seq_len"]:
            raise ValueError(
                f"context_window must be in [1,{info['seq_len'] - 1}] so target + history stays "
                "within the trained sequence length"
            )
        if not 1 <= self.decode_ctx <= info["seq_len"]:
            raise ValueError(f"decode_ctx must be in [1,{info['seq_len']}]")
        if episode is not None and not 0 <= episode < len(frames):
            raise ValueError(f"episode {episode} outside [0,{len(frames) - 1}]")

        self.latents: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.current: np.ndarray | None = None
        self.replay_pos = 0
        self.steps = 0
        self.step_ms: float | None = None
        self.last_action = 0

    @property
    def in_replay(self) -> bool:
        return self.replay_pos < self.n_ctx

    def episode_label(self) -> str:
        if self.ids is None:
            return str(self.ep)
        return f"{self.ep} (id {int(self.ids[self.ep])})"

    @torch.inference_mode()
    def reset(self) -> None:
        self.ep = self.fixed_episode if self.fixed_episode is not None else self.rng.randrange(len(self.frames))
        max_start = self.frames.shape[1] - self.n_ctx
        if max_start < 0:
            raise ValueError(f"episode length {self.frames.shape[1]} is shorter than n_ctx={self.n_ctx}")
        if self.fixed_start is not None:
            if not 0 <= self.fixed_start <= max_start:
                raise ValueError(f"start {self.fixed_start} outside [0,{max_start}]")
            self.start = self.fixed_start
        else:
            self.start = self.rng.randrange(max_start + 1)

        self.ctx_frames = np.asarray(
            self.frames[self.ep, self.start:self.start + self.n_ctx], dtype=np.uint8
        )
        clip = torch.from_numpy(self.ctx_frames.copy()).permute(0, 3, 1, 2)
        clip = clip.unsqueeze(0).to(self.device, dtype=torch.float32).div_(255.0)
        patches = self.model_mod.temporal_patchify(clip, self.info["patch"])
        latent, _ = self.encoder(patches)
        packed = self.model_mod.pack_bottleneck_to_spatial(
            latent, n_spatial=self.info["n_spatial"], k=self.info["packing"]
        )[0]
        self.latents = [z.detach() for z in packed]

        ctx_ids = self.action_ids[self.ep, self.start:self.start + self.n_ctx]
        self.actions = [torch.zeros(16, device=self.device, dtype=torch.float32)]
        self.actions.extend(_one_hot(int(a), self.device) for a in ctx_ids[1:])
        self.replay_pos = 0
        self.steps = 0
        self.step_ms = None
        self.advance_replay()
        print(
            f"reset: episode {self.episode_label()} start={self.start} ctx={self.n_ctx} "
            f"history={self.context_window} K={self.schedule['K']}",
            flush=True,
        )

    def advance_replay(self) -> None:
        self.current = self.ctx_frames[self.replay_pos]
        self.last_action = int(self.action_ids[self.ep, self.start + self.replay_pos])
        self.replay_pos += 1

    @torch.inference_mode()
    def _sample_next(self, action: int) -> torch.Tensor:
        g = len(self.latents)
        s = max(0, g - self.context_window)
        past = torch.stack(self.latents[s:g], dim=0).unsqueeze(0)
        t = past.shape[1]

        actions = torch.zeros((1, t + 1, 16), device=self.device, dtype=torch.float32)
        if t > 1:
            actions[0, 1:t] = torch.stack(self.actions[s + 1:g], dim=0)
        actions[0, t] = _one_hot(action, self.device)
        mask = torch.zeros_like(actions)
        mask[:, 1:, :6] = 1.0

        K = int(self.schedule["K"])
        e = int(self.schedule["e"])
        k_max = self.info["k_max"]
        z = torch.randn(
            (1, 1, self.info["n_spatial"], past.shape[-1]),
            device=self.device,
            dtype=past.dtype,
        )
        emax = int(round(math.log2(k_max)))
        step_idx = torch.full((1, t + 1), emax, device=self.device, dtype=torch.long)
        step_idx[:, -1] = e
        signal_idx = torch.full((1, t + 1), k_max - 1, device=self.device, dtype=torch.long)

        for i in range(K):
            tau = float(self.schedule["tau"][i])
            signal_idx[:, -1] = int(self.schedule["tau_idx"][i])
            sequence = torch.cat((past, z), dim=1)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                clean, _ = self.dynamics(
                    actions, step_idx, signal_idx, sequence, act_mask=mask, agent_tokens=None
                )
            velocity = (clean[:, -1:].float() - z.float()) / max(1e-4, 1.0 - tau)
            z = (z.float() + velocity * float(self.schedule["dt"])).to(past.dtype)
        return z[0, 0]

    @torch.inference_mode()
    def _decode_current(self) -> np.ndarray:
        packed = torch.stack(self.latents[-self.decode_ctx:], dim=0).unsqueeze(0)
        latent = self.model_mod.unpack_spatial_to_bottleneck(packed, k=self.info["packing"])
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            patches = self.decoder(latent)
            frames = self.model_mod.temporal_unpatchify(
                patches,
                self.info["H"],
                self.info["W"],
                self.info["C"],
                self.info["patch"],
            )
        frame = frames[0, -1].float().clamp_(0, 1)
        if not torch.isfinite(frame).all():
            raise RuntimeError("non-finite decoded frame")
        return (frame.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)

    @torch.inference_mode()
    def step(self, action: int) -> None:
        t0 = time.perf_counter()
        nxt = self._sample_next(action)
        if not torch.isfinite(nxt).all():
            raise RuntimeError("non-finite generated latent")
        self.latents.append(nxt.detach())
        self.actions.append(_one_hot(action, self.device))
        self.current = self._decode_current()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.step_ms = (time.perf_counter() - t0) * 1000.0
        self.last_action = action
        self.steps += 1


def stats_text(game: RolloutGame, device: torch.device) -> list[str]:
    kvs: list[tuple[str, object]] = [("## Stats ##", ""), ("", "")]
    kvs.extend([("episode", game.episode_label()), ("start t", game.start)])
    if game.in_replay:
        kvs.extend([("phase", "CTX REPLAY"), ("ctx frame", f"{game.replay_pos}/{game.n_ctx}")])
    else:
        kvs.extend([("phase", "IMAGINED"), ("step", game.steps)])
    kvs.append(("action", ACTION_NAMES[game.last_action]))
    if game.step_ms is not None:
        kvs.extend([
            ("model ms", f"{game.step_ms:.0f}"),
            ("model fps", f"{1000.0 / max(game.step_ms, 1e-6):.2f}"),
        ])
    kvs.extend([
        ("history", game.context_window),
        ("decode ctx", game.decode_ctx),
        ("shortcut K", game.schedule["K"]),
        ("device", str(device)),
    ])
    return [f"{key:<11} {value!s:>10}" for key, value in kvs]


def keymap_text() -> list[str]:
    kvs = [
        ("## Commands ##", ""), ("", ""), ("forward", "up arrow"),
        ("left", "left arrow"), ("right", "right arrow"), ("", ""),
        ("reset", "backspace"), ("pause", "space"), ("speed up", "tab"),
        ("quit", "esc"),
    ]
    return [f"{key:<15} {value}" for key, value in kvs]


def main() -> None:
    ap = argparse.ArgumentParser(description="Play inside the community Dreamer 4 Memory Maze model.")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Final community dynamics checkpoint.")
    ap.add_argument("--tokenizer", type=Path, required=True, help="Approved community tokenizer checkpoint.")
    ap.add_argument("--dreamer4", type=Path, default=None, help="Pinned community repository checkout.")
    ap.add_argument(
        "--frames", type=Path, default=ROOT / "data" / "memmaze9x9_val12.npy",
        help="Held-out episodic RGB frames (N,T,H,W,3).",
    )
    ap.add_argument("--actions", type=Path, default=None, help="Held-out actions; default beside --frames.")
    ap.add_argument("--n-ctx", type=int, default=8, help="Real frames shown and encoded on reset.")
    ap.add_argument("--context-window", type=int, default=31, help="Maximum generated-history length.")
    ap.add_argument("--decode-ctx", type=int, default=8, help="Trailing latent frames given to the decoder.")
    ap.add_argument("--K", type=int, default=4, help="Shortcut integration steps per generated frame.")
    ap.add_argument("--episode", type=int, default=None, help="Fix held-out episode index.")
    ap.add_argument("--start", type=int, default=None, help="Fix context start within the episode.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="Default: cuda when available, otherwise cpu.")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--size", type=int, nargs=2, default=(600, 600))
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--nonoop", action="store_true", help="No keys pauses instead of applying noop.")
    ap.add_argument("--selftest", type=int, default=None, metavar="N", help="Headless scripted N-frame smoke test.")
    args = ap.parse_args()

    if args.selftest is not None and args.selftest <= 0:
        ap.error("--selftest must be positive")
    if args.selftest:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import pygame
    import pygame.freetype

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dreamer4 = resolve_dreamer4(args.dreamer4)
    model_mod = import_dreamer4_model(dreamer4)
    print(f"loading tokenizer: {args.tokenizer}", flush=True)
    encoder, decoder, dynamics, info = _load_models(
        model_mod,
        tokenizer_path=args.tokenizer,
        dynamics_path=args.checkpoint,
        device=device,
    )
    print(
        f"loaded dynamics step={info['checkpoint_step']} seq_len={info['seq_len']} "
        f"k_max={info['k_max']} on {device}",
        flush=True,
    )

    frames = np.load(args.frames, mmap_mode="r")
    actions_path = args.actions or args.frames.with_name(args.frames.stem + "_actions.npy")
    if not actions_path.is_file():
        raise SystemExit(f"ERROR: action-conditioned player needs {actions_path}")
    actions = np.load(actions_path, mmap_mode="r")
    ids_path = args.frames.with_name(args.frames.stem + "_ids.npy")
    ids = np.load(ids_path) if ids_path.is_file() else None
    game = RolloutGame(
        frames=frames,
        actions=actions,
        ids=ids,
        model_mod=model_mod,
        encoder=encoder,
        decoder=decoder,
        dynamics=dynamics,
        info=info,
        device=device,
        n_ctx=args.n_ctx,
        decode_ctx=args.decode_ctx,
        context_window=args.context_window,
        K=args.K,
        amp=args.amp,
        episode=args.episode,
        start=args.start,
        seed=args.seed,
    )

    render_size = tuple(args.size)
    window_size = (render_size[0] + PANEL_LEFT + PANEL_RIGHT, render_size[1])
    pygame.init()
    screen = pygame.display.set_mode(window_size, pygame.FULLSCREEN if args.fullscreen else 0)
    pygame.display.set_caption("Memory Maze — community Dreamer 4 imagined world")
    clock = pygame.time.Clock()
    font = pygame.freetype.SysFont("Mono", 16)
    font_small = pygame.freetype.SysFont("Mono", 12)
    keymap = get_keymap(pygame)

    game.reset()
    running, paused, speedup = True, False, False
    latencies: list[float] = []
    scripted_actions = [1, 1, 1, 2, 1, 1, 3, 1]

    while running:
        screen.fill((64, 64, 64))
        assert game.current is not None
        surface = pygame.surfarray.make_surface(game.current.transpose((1, 0, 2)))
        surface = pygame.transform.scale(surface, render_size)
        screen.blit(surface, (PANEL_LEFT, 0))
        if game.in_replay:
            pygame.draw.rect(screen, (0, 200, 0), (PANEL_LEFT, 0, *render_size), 4)
        y = 5
        for line in stats_text(game, device):
            text_surface, _ = font.render(line, (255, 255, 255))
            screen.blit(text_surface, (16, y))
            y += font.size + 2
        y = 5
        for line in keymap_text():
            text_surface, _ = font_small.render(line, (255, 255, 255))
            screen.blit(text_surface, (render_size[0] + PANEL_LEFT + 16, y))
            y += font_small.size + 2
        pygame.display.flip()
        clock.tick(0 if (speedup or args.selftest) else args.fps)

        pygame.event.pump()
        keys_down = defaultdict(bool)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                keys_down[event.key] = True
        keys_hold = pygame.key.get_pressed()

        action = keymap[tuple()]
        for keys, candidate in keymap.items():
            if keys and all(keys_hold[key] or keys_down[key] for key in keys):
                action = candidate
        speedup = keys_hold[pygame.K_TAB]
        if keys_down[pygame.K_ESCAPE]:
            running = False
        if keys_down[pygame.K_SPACE]:
            paused = not paused
        elif action != keymap[tuple()]:
            paused = False
        if keys_down[pygame.K_BACKSPACE]:
            game.reset()
            continue
        if paused:
            continue

        if game.in_replay:
            game.advance_replay()
            continue
        if args.selftest:
            if game.steps >= args.selftest:
                running = False
                continue
            action = scripted_actions[game.steps % len(scripted_actions)]
        elif action == keymap[tuple()] and args.nonoop:
            continue
        game.step(action)
        assert game.step_ms is not None
        latencies.append(game.step_ms)

    pygame.quit()
    if latencies:
        values = np.asarray(latencies)
        print(
            f"{len(values)} generated frames | step ms mean={values.mean():.0f} "
            f"median={np.median(values):.0f} max={values.max():.0f} "
            f"(~{1000.0 / values.mean():.2f} fps)",
            flush=True,
        )
    if args.selftest:
        print("SELFTEST OK", flush=True)


if __name__ == "__main__":
    main()
