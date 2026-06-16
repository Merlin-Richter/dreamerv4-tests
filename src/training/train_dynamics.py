"""
Train the Dreamer 4 dynamics model on bouncing.npy, on top of the frozen causal tokenizer
from the frozen tokenizer (`models.tokenizer`).

Pipeline per step:
  frames (B, L, H, W, 3)
    --frozen tokenizer encoder-->  clean latents z1 (B, L, n_latents, bottleneck_dim)
    --shortcut forcing loss-->     train the dynamics transformer to denoise each frame from
                                   its causal history.

The bouncing dataset has no actions, so the dynamics model is trained unconditionally (only
the learned action embedding is used) -- the "unlabeled video" case from the paper.

Run from this folder:
    python train_dynamics_model.py

Or from repo root:
    python src/training/train_dynamics.py

Log metrics to Weights & Biases (opt-in; off by default):
    python train_dynamics_model.py --wandb [--wandb-entity TEAM] [--wandb-name run1]

Visualize a rollout from a saved checkpoint (OpenCV window; needs a display):
    python src/training/train_dynamics.py --test-checkpoint

Interactive single-frame rollout (4-frame dynamics context, key 0/1 actions):
    python src/interactive/play_dynamics.py
"""

import argparse
import random
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Put `src` on the path (where `wlog` and the `models` / `evals` packages live).
_SRC = Path(__file__).resolve().parents[1]   # .../src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import wlog
from models.dynamics_model import DynamicsModel, DynamicsModelConfig
from models.tokenizer import AutoEncoder, AutoEncoderConfig


class ChunkClipDataset(Dataset):
    """Fixed-length clips along time: ``[start + j*L : start + (j+1)*L]`` per episode.

    Mirrors the tokenizer's dataset so clip boundaries shift across epochs.

    ``frames`` stays memory-mapped uint8 on disk (``np.load(..., mmap_mode="r")``); each clip is
    converted to float32 only on access, so the full dataset is never materialized in RAM.
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
        clip_u8 = np.asarray(self.frames[ep, start:start + self.chunk_len])
        clip = torch.from_numpy(clip_u8.astype(np.float32) / 255.0)
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
    """frames: (B, T, H, W, 3) in [0,1] -> latents (B, T, n_latents, bottleneck_dim)."""
    return tokenizer.encoder(frames)


def run_test_checkpoint(args: argparse.Namespace) -> None:
    import cv2

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    raw = np.load(args.frames, mmap_mode="r")
    if raw.ndim != 5 or raw.shape[-1] != 3:
        raise ValueError(f"Expected (N, T, H, W, 3), got {raw.shape}")
    n_eps, n_frames, h, w, _ = raw.shape

    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = _config_from_checkpoint(payload["config"], DynamicsModelConfig)
    model = DynamicsModel(cfg).to(device)
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
    parser.add_argument("--frames", type=Path, default=_SRC.parent.parent / "bouncing.npy",
                        help="Path to frames .npy (N, T, H, W, C) uint8.")
    parser.add_argument("--actions", type=Path, default=None,
                        help="Path to actions .npy (N, T) ints. Default: '<frames>_actions.npy' "
                             "if it exists, else unlabeled (no action conditioning).")
    parser.add_argument("--tokenizer", type=Path,
                        default=_TOKENIZER_DIR / "autoencoder_bouncing.pt",
                        help="Frozen C tokenizer checkpoint.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint", type=Path, default=_SRC / "dynamics_bouncing.pt",
                        help="Where to save weights + config.")
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
    parser.add_argument("--val-offset", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore any existing --checkpoint and train from random init.")
    parser.add_argument("--ff7", type=int, default=0, metavar="K",
                        help="FF7 register-memory training (D-014): add the single-timestep-"
                             "sufficiency loss with lookahead depth K (1-3 sensible; 0=off). "
                             "Also sets use_register_memory in the saved config, so "
                             "generate() carries register state at inference.")
    parser.add_argument("--lambda-ff7", type=float, default=1.0,
                        help="Weight of the FF7 loss term in the total (default 1.0).")
    parser.add_argument("--ff9", type=int, default=0, metavar="K",
                        help="FF9 v2 memory-only-sufficiency training (D-024): adds the distinct "
                             "MEMORY-token carrier + the memory-only-sufficiency loss with lookahead "
                             "K (1-3 sensible; 0=off). Sets n_memory + use_full_state_memory in the "
                             "saved config. Registers stay pure scratch (independent of --ff7).")
    parser.add_argument("--lambda-ff9", type=float, default=1.0,
                        help="Weight of the FF9 loss term in the total (default 1.0).")
    parser.add_argument("--n-memory", type=int, default=4,
                        help="Number of distinct MEMORY tokens when --ff9 > 0 (default 4).")
    parser.add_argument("--multistep", type=int, default=0, metavar="H",
                        help="C1 multi-step motion loss (D-027): add the time-axis DAgger term with "
                             "self-rollout depth H (4 sensible; 0=off). Loss-only — does NOT change "
                             "inference or add params. Fixes autoregressive compounding (EXP-018).")
    parser.add_argument("--lambda-multistep", type=float, default=1.0,
                        help="Peak weight of the C1 multi-step term (default 1.0).")
    parser.add_argument("--multistep-warmup", type=int, default=0, metavar="EPOCHS",
                        help="Linearly ramp lambda_multistep from 0 to its peak over the first EPOCHS "
                             "epochs (mandatory mitigation for single-frame/multi-step capacity "
                             "tension, V-T017-C1). 0 = no ramp (full weight from epoch 0).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for torch/numpy/random (model init, tau/noise sampling).")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Cap the number of TRAINING episodes (subset, for fast iteration). "
                             "Default: all. Val split is unaffected. Subset is the first "
                             "--max-episodes of the fixed seed-0 train permutation (reproducible).")
    wlog.add_args(parser)
    args = parser.parse_args()

    if args.test_checkpoint:
        run_test_checkpoint(args)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = load_tokenizer(args.tokenizer, device)

    # Memory-mapped uint8; clips are converted to float32 per-batch in the dataset, so RAM
    # stays at ~the touched pages instead of a full 4x float32 copy of the dataset.
    raw = np.load(args.frames, mmap_mode="r")
    if raw.ndim != 5 or raw.shape[-1] != 3:
        raise ValueError(f"Expected (N, T, H, W, 3), got {raw.shape}")
    n, t, h, w, c = raw.shape
    if n < 2:
        raise ValueError(f"Need at least 2 episodes to split train/val, got n={n}.")

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

    # Build dynamics config; tokenizer-tied dims come from the tokenizer checkpoint.
    base = DynamicsModelConfig()
    chunk_len = args.context_length if args.context_length is not None else base.max_temporal_length
    if t < chunk_len:
        raise ValueError(f"Episode length T={t} must be >= clip length L={chunk_len}.")

    # Read tokenizer dims from its loaded modules so the two models always agree.
    n_latents = tokenizer.encoder.n_latents
    bottleneck_dim = tokenizer.encoder.bottleneck_proj.out_features
    cfg = DynamicsModelConfig(
        max_temporal_length=chunk_len,
        n_latents=n_latents,
        bottleneck_dim=bottleneck_dim,
        n_actions=n_actions,
        use_register_memory=args.ff7 > 0,
        ff7_k=args.ff7,
        n_memory=(args.n_memory if args.ff9 > 0 else 0),
        use_full_state_memory=args.ff9 > 0,
        ff9_k=args.ff9,
        multistep_h=args.multistep,
    )

    train_ds = ChunkClipDataset(raw, train_idx, chunk_len, start_offset=0, actions=actions)
    val_ds = ChunkClipDataset(raw, val_idx, chunk_len,
                              start_offset=args.val_offset % (chunk_len + 1), actions=actions)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"No clips with L={chunk_len}; try shorter clips or smaller --val-offset.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    if args.checkpoint.is_file() and not args.fresh:
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cfg = _config_from_checkpoint(payload["config"], DynamicsModelConfig)
        if args.ff7 > 0:  # resuming an older checkpoint into FF7 training: record the flags
            cfg.use_register_memory = True
            cfg.ff7_k = args.ff7
        if args.ff9 > 0:  # resuming into FF9 training: add the memory-token carrier flags
            cfg.n_memory = args.n_memory
            cfg.use_full_state_memory = True
            cfg.ff9_k = args.ff9
        if args.multistep > 0:  # resuming into C1 training: record the lookahead (loss-only)
            cfg.multistep_h = args.multistep
        model = DynamicsModel(cfg).to(device)
        result = model.load_state_dict(payload["model_state_dict"], strict=False)
        if result.missing_keys:
            print(f"[resume] randomly-initialized (not in checkpoint): {result.missing_keys}")
        if result.unexpected_keys:
            print(f"[resume] ignored (in checkpoint, not in model): {result.unexpected_keys}")
        print(f"Loaded weights from {args.checkpoint}")
    else:
        model = DynamicsModel(cfg).to(device)

    # cfg is final here (resume may have replaced it from the checkpoint).
    wlog.init(args, cfg, project="transformer-D-dynamics")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    epoch_bar = tqdm(range(args.epochs), desc="Epochs", position=0, mininterval=1.0)
    for epoch in epoch_bar:
        # C1 lambda ramp (V-T017-C1): 0 -> peak over --multistep-warmup epochs.
        lam_ms = args.lambda_multistep
        if args.multistep > 0 and args.multistep_warmup > 0:
            lam_ms = args.lambda_multistep * min(1.0, (epoch + 1) / args.multistep_warmup)
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
                z1 = encode_frames(tokenizer, batch_x)  # frozen, no grad
                loss, parts = model.loss(z1, batch_a, ff7_k=args.ff7, lambda_ff7=args.lambda_ff7,
                                         ff9_k=args.ff9, lambda_ff9=args.lambda_ff9,
                                         multistep_h=args.multistep, lambda_multistep=lam_ms,
                                         return_parts=True)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
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
                    z1 = encode_frames(tokenizer, batch_x)
                    loss, parts = model.loss(z1, batch_a, ff7_k=args.ff7, lambda_ff7=args.lambda_ff7,
                                             ff9_k=args.ff9, lambda_ff9=args.lambda_ff9,
                                             return_parts=True)
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

        scheduler.step()
        torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg)}, args.checkpoint)
        tqdm.write(f"Saved checkpoint to {args.checkpoint}")

    wlog.finish()


if __name__ == "__main__":
    main()
