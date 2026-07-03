"""
Train the Dreamer-4 dynamics model on the frozen tokenizer's latents.

Latents come from an ON-DISK CACHE by default: the first run for a (frames, tokenizer) combo encodes
the whole dataset once (fp16, `<frames>.latents-<tokhash>.npy` next to the frames) and every run —
including that first one — then trains from the mmapped latents with NO tokenizer on the GPU.
`--encode-online` restores the legacy per-batch pixels->latents path.

Pipeline per step (cache path):
  latents z1 (B, L, n_latents, bottleneck_dim)  [precomputed by the frozen tokenizer encoder]
    --shortcut forcing loss-->     train the dynamics transformer to denoise each frame from
                                   its causal history.

With `--ff9 K --n-memory M` the model also carries per-frame memory tokens and adds the FF9
sufficiency loss (memory must encode the full env state so it survives past the latent window);
without them it is the vanilla memory-free model. The dynamics model never sees pixels.

Run from repo root:
    python -u src/training/train_dynamics.py

Visualize a rollout from a saved checkpoint (OpenCV window; needs a display):
    python -u src/training/train_dynamics.py --test-checkpoint

To try a NEW loss / training-flow idea without editing the spec-backed model, pass an experiment-local
subclass via `--model-module experiments/EXP-NNN/model.py:ClassName` (see experiments/README.md). The
canonical src/models/dynamics_model.py stays untouched until the idea graduates.
"""
# Defer annotation evaluation so PEP-604 unions (e.g. `str | None` in resolve_model_class) work on
# the cluster's Python < 3.10 (they are only evaluated lazily, never at def time).
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Put `src` on the path (where `wlog` and the `models` / `evals` packages live).
_SRC = Path(__file__).resolve().parents[1]   # .../src
_ROOT = _SRC.parent                           # repo root
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import wlog
from models.dynamics_model import DynamicsModel, DynamicsModelConfig
from models.tokenizer import AutoEncoder, AutoEncoderConfig


class ChunkClipDataset(Dataset):
    """Fixed-length clips along time: ``[start + j*L : start + (j+1)*L]`` per episode.

    Mirrors the tokenizer's dataset so clip boundaries shift across epochs.

    ``frames`` stays memory-mapped on disk (``np.load(..., mmap_mode="r")``); each clip is
    converted to float32 only on access, so the full dataset is never materialized in RAM.
    Accepts EITHER uint8 pixel frames (N, T, H, W, 3) — scaled to [0,1] float — OR precomputed
    fp16 latents (N, T, n_latents, bottleneck_dim) from the latent cache, passed through as float32.
    ``episode_indices`` selects this split's episodes without copying (fancy-indexing a memmap
    would silently pull the whole subset into memory). ``actions`` is the full (N, T) tensor,
    indexed with the same absolute episode indices.
    """

    def __init__(self, frames: np.ndarray, episode_indices: np.ndarray, chunk_len: int,
                 start_offset: int = 0, actions: torch.Tensor = None) -> None:
        self.frames = frames  # (N, T, H, W, 3) uint8
        self.episode_indices = np.asarray(episode_indices)
        self.actions = actions  # (N, T) long, or None for unlabeled video
        self.chunk_len = int(chunk_len)
        self.start_offset = int(start_offset)
        self._rebuild_index()

    def set_start_offset(self, start_offset: int) -> None:
        self.start_offset = int(start_offset)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        o, L = self.start_offset, self.chunk_len
        T = self.frames.shape[1]
        pairs: list[tuple[int, int]] = []
        for ep in self.episode_indices:
            for j in range((T - o) // L):
                pairs.append((int(ep), o + j * L))
        self._pairs = pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int):
        ep, start = self._pairs[idx]
        clip_np = np.asarray(self.frames[ep, start:start + self.chunk_len])
        if clip_np.dtype == np.uint8:   # pixel frames -> [0,1] float32
            clip = torch.from_numpy(clip_np.astype(np.float32) / 255.0)
        else:                           # precomputed latents (fp16 cache) -> float32 as-is
            clip = torch.from_numpy(clip_np.astype(np.float32))
        if self.actions is None:
            return clip
        act = self.actions[ep, start:start + self.chunk_len].clone()
        return clip, act


def _split_batch(batch, device):
    """DataLoader yields either frames or (frames, actions); return (frames, actions_or_None)."""
    if isinstance(batch, (list, tuple)):
        frames, actions = batch
        return frames.to(device), actions.to(device)
    return batch.to(device), None


def _config_from_checkpoint(cfg_dict: dict, cls):
    """Build a dataclass config from a saved dict; ignore unknown keys for forward compat."""
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in cfg_dict.items() if k in allowed})


def resolve_model_class(spec: str | None):
    """Resolve which dynamics-model class to train/visualize.

    ``spec`` None  -> the canonical, spec-backed ``DynamicsModel``.
    ``spec`` ``"experiments/EXP-NNN/model.py:ClassName"`` -> an experiment-local subclass (NOT
    spec-backed; it lives under experiments/ and dies with the experiment). The class MUST subclass
    ``DynamicsModel`` and take a ``DynamicsModelConfig``, so checkpoints, ``generate()`` and the eval
    path keep working unchanged — a loss/training-flow experiment overrides only ``loss()``. This is
    the seam that lets variant ideas be tried WITHOUT editing src/models/dynamics_model.py (which is
    spec-backed); see experiments/README.md.
    """
    if not spec:
        return DynamicsModel
    # rpartition (not partition): a Windows absolute path carries a drive-letter colon (C:/...), so
    # split on the LAST ':' to separate the path from the class name.
    mod_path, sep, cls_name = spec.rpartition(":")
    if not sep or not cls_name.isidentifier():
        raise ValueError(f"--model-module must be 'path/to/file.py:ClassName', got {spec!r}")
    file = Path(mod_path)
    if not file.is_file():
        raise FileNotFoundError(f"--model-module file not found: {file}")
    module_spec = importlib.util.spec_from_file_location(file.stem, file)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)  # src/ is already on sys.path, so its imports resolve
    cls = getattr(module, cls_name, None)
    if cls is None:
        raise AttributeError(f"class {cls_name!r} not found in {file}")
    if not (isinstance(cls, type) and issubclass(cls, DynamicsModel)):
        raise TypeError(f"{cls_name} must subclass DynamicsModel")
    print(f"[--model-module] training experiment model {cls_name} from {file}")
    return cls


def load_tokenizer(checkpoint: Path, device: str) -> AutoEncoder:
    """Load the frozen causal tokenizer (C autoencoder)."""
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
    """frames: (B, T, H, W, 3) in [0,1] -> latents (B, T, n_latents, bottleneck_dim).

    A single frozen-tokenizer encode pass (clips are <= the model/tokenizer temporal window)."""
    return tokenizer.encoder(frames)


def _file_sha256(path: Path, n_hex: int = 12) -> str:
    """First ``n_hex`` hex chars of the file's sha256 (streamed; identifies a tokenizer ckpt)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:n_hex]


def latent_cache_paths(frames_path: Path, tokenizer_path: Path) -> "tuple[Path, Path]":
    """(npy, json-meta) cache paths for this (frames file, tokenizer ckpt) combo, next to the frames."""
    key = _file_sha256(tokenizer_path)
    npy = frames_path.with_name(f"{frames_path.stem}.latents-{key}.npy")
    meta = frames_path.with_name(f"{frames_path.stem}.latents-{key}.json")
    return npy, meta


def ensure_latent_cache(frames_path: Path, tokenizer_path: Path, device: str,
                        batch_episodes: int = 4) -> Path:
    """Return the latent-cache .npy for (frames, tokenizer), building it once if missing.

    Cache HIT (npy + meta exist, episode/time shape matches the frames): return immediately —
    the tokenizer is never loaded, freeing its VRAM + per-batch encode compute for the whole run.
    Cache MISS: load the frozen tokenizer, encode the entire dataset in non-overlapping windows of
    the tokenizer's ``max_temporal_length`` (trailing partial window encoded as-is; the encoder is
    eval-mode + deterministic, MAE dropout off), store fp16 ``(N, T, n_latents, bottleneck_dim)``
    via ``open_memmap`` and atomically rename into place (parallel jobs may both build; last rename
    wins, contents identical). fp16 is exact for bf16-computed values, ~12x smaller than uint8 pixels
    for memmaze.

    Because the causal encoder sees only its own window, a frame's latent depends on the window
    start; downstream chunking slices the cache at ARBITRARY offsets (Merlin 2026-07-03: per-frame
    reconstruction gives no pressure to use temporal context, so latents are ~window-invariant —
    measured by experiments/memmaze-dynamics/probe_window_invariance.py, and the GridWorld mem2mem
    winner already trained on boundary-crossing latents).
    """
    frames_path, tokenizer_path = Path(frames_path), Path(tokenizer_path)
    npy, meta_p = latent_cache_paths(frames_path, tokenizer_path)
    raw = np.load(frames_path, mmap_mode="r")
    if npy.is_file() and meta_p.is_file():
        lat = np.load(npy, mmap_mode="r")
        if lat.shape[:2] == raw.shape[:2]:
            print(f"[latent-cache] HIT {npy.name}  {lat.shape} {lat.dtype}")
            return npy
        print(f"[latent-cache] STALE (cache {lat.shape[:2]} != frames {raw.shape[:2]}) -> rebuild")
    payload = torch.load(tokenizer_path, map_location="cpu", weights_only=False)
    tok_T = int(payload["config"]["max_temporal_length"])
    n_lat = int(payload["config"]["n_latents"])
    bdim = int(payload["config"]["bottleneck_dim"])
    del payload
    tokenizer = load_tokenizer(tokenizer_path, device)
    n, t = raw.shape[:2]
    use_amp = device == "cuda"
    print(f"[latent-cache] MISS -> encoding {n} eps x {t} frames in windows of {tok_T} "
          f"-> {npy.name} (fp16, ({n}, {t}, {n_lat}, {bdim}))", flush=True)
    tmp = npy.with_name(npy.name + f".tmp{os.getpid()}")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, t, n_lat, bdim))
    t0 = time.perf_counter()
    with torch.no_grad():
        for e0 in range(0, n, batch_episodes):
            e1 = min(n, e0 + batch_episodes)
            clip = torch.from_numpy(
                np.asarray(raw[e0:e1]).astype(np.float32) / 255.0).to(device)
            zs = []
            for w0 in range(0, t, tok_T):
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                    z = encode_frames(tokenizer, clip[:, w0:w0 + tok_T])
                zs.append(z.float().cpu())
            out[e0:e1] = torch.cat(zs, dim=1).numpy().astype(np.float16)
            if e1 % max(batch_episodes * 25, 1) < batch_episodes or e1 == n:
                dt = time.perf_counter() - t0
                print(f"[latent-cache]   {e1}/{n} eps  ({e1 / max(dt, 1e-9):.1f} eps/s)", flush=True)
    out.flush()
    del out
    meta_p.write_text(json.dumps({
        "frames": str(frames_path.name), "frames_shape": list(raw.shape),
        "tokenizer": str(tokenizer_path), "tokenizer_sha256_12": _file_sha256(tokenizer_path),
        "window": tok_T, "dtype": "float16",
        "latents_shape": [n, t, n_lat, bdim],
    }, indent=2))
    os.replace(tmp, npy)
    del tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"[latent-cache] BUILT {npy.name} in {time.perf_counter() - t0:.0f}s", flush=True)
    return npy


def run_test_checkpoint(args: argparse.Namespace) -> None:
    import cv2

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"--test-checkpoint needs an existing --checkpoint, got: {args.checkpoint}")

    raw = np.load(args.frames, mmap_mode="r")
    if raw.ndim != 5 or raw.shape[-1] != 3:
        raise ValueError(f"Expected (N, T, H, W, 3), got {raw.shape}")
    n_eps, n_frames, h, w, _ = raw.shape

    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = _config_from_checkpoint(payload["config"], DynamicsModelConfig)
    model = resolve_model_class(args.model_module)(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    # Action-conditioned models need the per-frame action stream to roll out correctly.
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

    tokenizer = load_tokenizer(args.tokenizer, device)

    L = cfg.max_temporal_length
    n_ctx = args.context_frames
    n_gen = L - n_ctx
    if n_ctx < 1 or n_gen < 1:
        raise ValueError(f"--context-frames must be in [1, {L - 1}] for clip length {L}.")

    K = cfg.inference_steps if args.inference_steps is None else args.inference_steps
    if K < 1 or K > cfg.max_sampling_steps or (K & (K - 1)) != 0:
        raise ValueError(
            f"--inference-steps must be a power of two in [1, {cfg.max_sampling_steps}], got {K}."
        )

    win = "Dynamics rollout: GT (top) | context+prediction (bottom)  (space=new, q/Esc=quit)"
    scale = max(2, 512 // (max(h, w) * L))

    def tensor01_to_bgr(img_t: torch.Tensor) -> np.ndarray:
        x = img_t.detach().cpu().float().clamp(0.0, 1.0).numpy()
        return cv2.cvtColor((x * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)

    def pick_and_show() -> None:
        ep = random.randrange(n_eps)
        start = random.randrange(max(1, n_frames - L + 1))
        clip = raw[ep, start:start + L].astype(np.float32) / 255.0
        x = torch.from_numpy(clip).unsqueeze(0).to(device)  # (1, L, H, W, 3)

        action_idx = None
        if actions_raw is not None:
            action_idx = torch.from_numpy(
                actions_raw[ep, start:start + L].astype(np.int64)
            ).unsqueeze(0).to(device)  # (1, L), aligned with the clip

        with torch.no_grad():
            latents = encode_frames(tokenizer, x)             # (1, L, n_lat, dim)
            context = latents[:, :n_ctx]
            gen = model.generate(context, n_generate=n_gen, K=K, action_idx=action_idx)
            full = torch.concat((context, gen), dim=1)        # (1, L, n_lat, dim)
            recon = tokenizer.decoder(full)[0]                # (L, H, W, 3)

        gt_row = np.hstack([tensor01_to_bgr(x[0, t]) for t in range(L)])
        pred_row = np.hstack([tensor01_to_bgr(recon[t]) for t in range(L)])
        # Mark the boundary between given context and generated frames.
        boundary = n_ctx * w
        cv2.line(pred_row, (boundary, 0), (boundary, pred_row.shape[0]), (0, 0, 255), 1)
        pair = np.vstack([gt_row, pred_row])

        disp = cv2.resize(
            pair, (pair.shape[1] * scale, pair.shape[0] * scale), interpolation=cv2.INTER_NEAREST
        )
        cv2.putText(
            disp,
            f"ep {ep}  ctx={n_ctx}  gen={n_gen}  K={K}   SPACE new   q/Esc quit",
            (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
        )
        cv2.imshow(win, disp)

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    pick_and_show()
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord(" "):
            pick_and_show()
        if key in (ord("q"), ord("Q"), 27):
            break
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True,
                        help="Path to frames .npy (N, T, H, W, C) uint8 (required; env-specific).")
    parser.add_argument("--actions", type=Path, default=None,
                        help="Path to actions .npy (N, T) ints. Default: '<frames>_actions.npy' "
                             "if it exists, else unlabeled (no action conditioning).")
    parser.add_argument("--tokenizer", type=Path, required=True,
                        help="Frozen tokenizer checkpoint (required; env-specific, e.g. "
                             "checkpoints/gridworld/tokenizer.pt).")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker processes. Default: auto from SLURM_CPUS_PER_TASK / os.cpu_count() "
             "(capped at 8). 0 = synchronous load on the main thread (starves the GPU).",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Where to SAVE weights + config each epoch (required; env-specific, e.g. "
                             "checkpoints/gridworld/dynamics.pt). With --test-checkpoint, the checkpoint "
                             "to load and visualize.")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Load weights + config to start training FROM (optional; default: random "
                             "init). Use to continue a run or warm-start FF9 from a vanilla checkpoint.")
    parser.add_argument("--test-checkpoint", action="store_true",
                        help="Load --checkpoint and visualize an autoregressive rollout.")
    parser.add_argument("--context-frames", type=int, default=4,
                        help="Frames of ground-truth context before generation (test mode).")
    parser.add_argument("--inference-steps", type=int, default=None,
                        help="Shortcut steps K per generated frame in --test-checkpoint "
                             "(power of 2; default: checkpoint config, typically 4). "
                             "Use 128 for the finest K_max schedule.")
    parser.add_argument("--context-length", type=int, default=None,
                        help="Clip length in frames. Default: DynamicsModelConfig.max_temporal_length.")
    parser.add_argument("--embedding-dim", type=int, default=None,
                        help="Transformer width (default: DynamicsModelConfig default; env-dependent).")
    parser.add_argument("--depth", type=int, default=None,
                        help="Transformer depth; use a multiple of 3 (3x[spatial,temporal,spatial]).")
    parser.add_argument("--n-heads", type=int, default=None, help="Attention heads.")
    parser.add_argument("--n-registers", type=int, default=None, help="Scratch register tokens/frame.")
    parser.add_argument("--val-offset", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--ff9", type=int, default=0, metavar="K",
                        help="FF9 memory-sufficiency training: enable the per-frame memory tokens + "
                             "the memory-only-sufficiency loss with lookahead K (1-3 sensible; 0=off, "
                             "i.e. vanilla memory-free). Sets config.ff9_k.")
    parser.add_argument("--n-memory", type=int, default=4,
                        help="Number of per-frame MEMORY tokens when --ff9 > 0 (default 4).")
    parser.add_argument("--model-module", type=str, default=None, metavar="FILE.py:Class",
                        help="Train an experiment-local DynamicsModel SUBCLASS instead of the canonical "
                             "model, e.g. 'experiments/EXP-034/model.py:DynamicsModelEXP034'. The seam "
                             "for trying a new loss/training-flow WITHOUT editing the spec-backed "
                             "src/models/dynamics_model.py. Default: the canonical model. See "
                             "experiments/README.md.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for torch/numpy/random (model init, tau/noise sampling).")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Cap the number of TRAINING episodes (subset, for fast iteration). "
                             "Default: all. Val split is unaffected. Subset is the first "
                             "--max-episodes of the fixed seed-0 train permutation (reproducible).")
    parser.add_argument("--encode-online", action="store_true",
                        help="Bypass the latent disk cache: keep the tokenizer on the GPU and encode "
                             "pixels -> latents per batch (the legacy path; debug/AB only).")
    parser.add_argument("--build-latent-cache-only", action="store_true",
                        help="Build the latent cache for (--frames, --tokenizer) if missing, then exit "
                             "without training (for prep jobs, so parallel training jobs don't race "
                             "to build it).")
    parser.add_argument("--cache-batch", type=int, default=4,
                        help="Episodes per encode batch while BUILDING the latent cache (default 4; "
                             "raise on big GPUs for a faster one-time build).")
    wlog.add_args(parser)
    args = parser.parse_args()
    # Enable TF32 for any fp32 matmuls outside the bf16 autocast region (free on H100).
    torch.set_float32_matmul_precision("high")

    if args.test_checkpoint:
        run_test_checkpoint(args)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Memory-mapped uint8; clips are converted to float32 per-batch in the dataset, so RAM
    # stays at ~the touched pages instead of a full 4x float32 copy of the dataset.
    raw = np.load(args.frames, mmap_mode="r")
    if raw.ndim != 5 or raw.shape[-1] != 3:
        raise ValueError(f"Expected (N, T, H, W, 3), got {raw.shape}")
    n, t, h, w, c = raw.shape
    if n < 2:
        raise ValueError(f"Need at least 2 episodes to split train/val, got n={n}.")

    # Latent source: DEFAULT is the on-disk latent cache (tokenizer runs ONCE per (frames, tokenizer)
    # combo, then never occupies VRAM again; the dataset is ~12x smaller than pixels). --encode-online
    # keeps the legacy per-batch pixels->latents path.
    tokenizer = None
    if args.encode_online:
        tokenizer = load_tokenizer(args.tokenizer, device)
        n_latents = tokenizer.encoder.n_latents
        bottleneck_dim = tokenizer.encoder.bottleneck_proj.out_features
        data_arr = raw
    else:
        cache = ensure_latent_cache(args.frames, args.tokenizer, device,
                                    batch_episodes=args.cache_batch)
        if args.build_latent_cache_only:
            print("[latent-cache] --build-latent-cache-only: done.")
            return
        data_arr = np.load(cache, mmap_mode="r")   # (N, T, n_latents, bottleneck_dim) fp16
        n_latents, bottleneck_dim = int(data_arr.shape[2]), int(data_arr.shape[3])

    # Optional discrete actions, aligned per frame with `raw`.
    actions_path = args.actions
    if actions_path is None:
        cand = args.frames.with_name(args.frames.stem + "_actions.npy")
        actions_path = cand if cand.is_file() else None
    if actions_path is not None:
        actions_np = np.load(actions_path)
        if actions_np.shape != (n, t):
            raise ValueError(f"actions shape {actions_np.shape} != frames (N,T)=({n},{t}).")
        actions = torch.from_numpy(actions_np.astype(np.int64))
        n_actions = int(actions_np.max()) + 1
        print(f"Loaded actions from {actions_path}  (n_actions={n_actions})")
    else:
        actions = None
        n_actions = 0
        print("No actions found -> training unlabeled (learned action embedding only).")

    n_val = min(max(1, int(round(n * args.val_fraction))), n - 1)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    train_idx, val_idx = perm[n_val:].numpy(), perm[:n_val].numpy()
    if args.max_episodes is not None:
        train_idx = train_idx[:args.max_episodes]
        print(f"[--max-episodes] capping training to {len(train_idx)} episodes (val unchanged).")

    # Build dynamics config; tokenizer-tied dims come from the tokenizer checkpoint (online path)
    # or from the latent cache's trailing dims (cache path) — the two always agree by construction.
    base = DynamicsModelConfig()
    chunk_len = args.context_length if args.context_length is not None else base.max_temporal_length
    if t < chunk_len:
        raise ValueError(f"Episode length T={t} must be >= clip length L={chunk_len}.")

    # Model-dim CLI overrides (unset = GridWorld dataclass defaults), mirroring train_tokenizer.
    dims = {k: v for k, v in dict(embedding_dim=args.embedding_dim, depth=args.depth,
                                  n_heads=args.n_heads, n_registers=args.n_registers).items()
            if v is not None}
    cfg = DynamicsModelConfig(
        max_temporal_length=chunk_len,
        n_latents=n_latents,
        bottleneck_dim=bottleneck_dim,
        n_actions=n_actions,
        n_memory=(args.n_memory if args.ff9 > 0 else 0),
        ff9_k=args.ff9,
        **dims,
    )

    train_ds = ChunkClipDataset(data_arr, train_idx, chunk_len, start_offset=0, actions=actions)
    val_ds = ChunkClipDataset(data_arr, val_idx, chunk_len,
                              start_offset=args.val_offset % (chunk_len + 1), actions=actions)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"No clips with L={chunk_len}; try shorter clips or smaller --val-offset.")

    # DataLoader throughput (D-041): default num_workers=0 loads each batch synchronously on the main
    # thread (memmap read + uint8->float32) while the GPU idles. Background workers + pinned memory +
    # prefetch keep the GPU fed. num_workers auto-derives from the SLURM CPU allocation (request enough
    # via submit_job --cpus); override with --num-workers. __getitem__ returns CPU tensors (the .to(device)
    # is in _split_batch, in the loop), so workers are safe (no CUDA in subprocesses).
    nw = args.num_workers
    if nw is None:
        if os.name == "nt":
            nw = 0  # native-Windows local smokes: avoid spawn/memmap-pickle overhead (cluster is Linux)
        else:
            slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
            nw = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 1)
            nw = max(0, min(nw, 8))
    pin = torch.cuda.is_available()
    loader_kw = dict(num_workers=nw, pin_memory=pin)
    if nw > 0:
        # persistent_workers MUST stay False: train_ds.set_start_offset() rebuilds self._pairs with a
        # DIFFERENT length every epoch (shifting clip boundaries), so persistent workers would keep a
        # stale _pairs from epoch 0 and crash with IndexError once the new epoch's __len__ grows (the
        # same bug fixed in the tokenizer dataset @ f1e3d6c — only reproduces with num_workers>0, i.e.
        # on the cluster, not local Windows smokes where nw=0).
        loader_kw.update(persistent_workers=False, prefetch_factor=4)
    print(f"DataLoader: num_workers={nw} pin_memory={pin}"
          + (" prefetch_factor=4 persistent_workers=False" if nw > 0 else ""))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)

    model_cls = resolve_model_class(args.model_module)
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        cfg = _config_from_checkpoint(payload["config"], DynamicsModelConfig)
        if args.ff9 > 0:  # warm-starting into FF9 training: add the memory carrier
            cfg.n_memory = args.n_memory
            cfg.ff9_k = args.ff9
        model = model_cls(cfg).to(device)
        result = model.load_state_dict(payload["model_state_dict"], strict=False)
        if result.missing_keys:
            print(f"[resume] randomly-initialized (not in checkpoint): {result.missing_keys}")
        if result.unexpected_keys:
            print(f"[resume] ignored (in checkpoint, not in model): {result.unexpected_keys}")
        print(f"Loaded weights from {args.resume}")
    else:
        model = model_cls(cfg).to(device)
    print(f"DynamicsModel: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params | "
          f"dim={cfg.embedding_dim} depth={cfg.depth} heads={cfg.n_heads} L={cfg.max_temporal_length} "
          f"n_latents={cfg.n_latents} bottleneck={cfg.bottleneck_dim} n_memory={cfg.n_memory} "
          f"ff9_k={cfg.ff9_k} n_actions={cfg.n_actions}")

    # cfg is final here (resume may have replaced it from the checkpoint).
    wlog.init(args, cfg, project="transformer-D-dynamics")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # Per-step warmup -> flat -> late-cosine cooldown (mirrors train_tokenizer). A plain
    # cosine-from-step-0 (the previous CosineAnnealingLR) spends most of training at low LR; instead
    # warm up, hold at peak, and cool down only in the final 20%, so the model trains at full LR for
    # the bulk of the (now longer) run.
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = max(200, int(0.05 * total_steps))
    decay_start = int(0.8 * total_steps)
    eta_min_ratio = 1e-6 / args.lr

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if step < decay_start:
            return 1.0
        p = (step - decay_start) / max(1, total_steps - decay_start)
        return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (1.0 + np.cos(np.pi * p))

    scheduler = LambdaLR(opt, lr_lambda)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    epoch_bar = tqdm(range(args.epochs), desc="Epochs", position=0, mininterval=1.0)
    for epoch in epoch_bar:
        train_off = random.randint(0, chunk_len)
        train_ds.set_start_offset(train_off)
        if len(train_ds) == 0:
            train_ds.set_start_offset(0)
            train_off = 0

        model.train()
        train_loss = 0.0
        train_parts: dict[str, float] = {}
        n_train_samples = 0
        train_t0 = time.perf_counter()
        for batch in tqdm(train_loader, desc=f"Train {epoch + 1}/{args.epochs}",
                          leave=False, position=1, mininterval=1.0):
            batch_x, batch_a = _split_batch(batch, device)
            n_train_samples += batch_x.shape[0]
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                # cache path: batch_x IS the latents; online path: encode pixels (frozen, no grad)
                z1 = batch_x if tokenizer is None else encode_frames(tokenizer, batch_x)
                loss, parts = model.loss(z1, batch_a, return_parts=True)
            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()  # per-step schedule (warmup -> flat -> late cosine)
            train_loss += loss.item()
            train_parts["grad_norm"] = train_parts.get("grad_norm", 0.0) + float(grad_norm)
            for name, value in parts.items():
                train_parts[name] = train_parts.get(name, 0.0) + value.item()

        train_time = time.perf_counter() - train_t0
        samples_per_s = n_train_samples / train_time if train_time > 0 else 0.0

        model.eval()
        val_loss = 0.0
        val_parts: dict[str, float] = {}
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False, position=1, mininterval=1.0):
                batch_x, batch_a = _split_batch(batch, device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                    z1 = batch_x if tokenizer is None else encode_frames(tokenizer, batch_x)
                    loss, parts = model.loss(z1, batch_a, return_parts=True)
                    val_loss += loss.item()
                    for name, value in parts.items():
                        val_parts[name] = val_parts.get(name, 0.0) + value.item()

        train_l = train_loss / len(train_loader)
        val_l = val_loss / len(val_loader)
        current_lr = opt.param_groups[0]["lr"]
        epoch_bar.set_postfix(train=f"{train_l:.6f}", val=f"{val_l:.6f}",
                              tr_off=train_off, lr=f"{current_lr:.2e}")
        print(f"Epoch {epoch + 1} | train: {train_l:.6f} | val: {val_l:.6f} "
              f"| train_clip_offset={train_off} | lr: {current_lr:.2e}")
        metrics = {
            "train/loss": train_l,
            "val/loss": val_l,
            "lr": current_lr,
            "train_clip_offset": train_off,
            "perf/train_time_s": train_time,        # wall-clock of the train loop only
            "perf/samples_per_s": samples_per_s,    # training throughput (clips/sec)
        }
        metrics.update({f"train/loss_{k}": v / len(train_loader) for k, v in train_parts.items()})
        metrics.update({f"val/loss_{k}": v / len(val_loader) for k, v in val_parts.items()})
        wlog.log(metrics, step=epoch)

        torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg)}, args.checkpoint)
        tqdm.write(f"Saved checkpoint to {args.checkpoint}")

    wlog.finish()


if __name__ == "__main__":
    main()
