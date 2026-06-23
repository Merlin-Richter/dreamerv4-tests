"""
Train the temporal VideoAutoEncoder on bouncing.npy (same layout as data_generators/load_data.py).

Episodes are sliced into fixed-length clips of ``max_temporal_length`` frames (e.g. 32): ``[o:o+L]``,
``[o+L:o+2L]``, … along time. Each training epoch picks a new random start offset ``o`` in
``[0, L]`` so boundaries shift across epochs. Validation uses a fixed offset (default 0).

Run from this folder:
    python train_autoencoder_bouncing.py

Or from repo root:
    python src/training/train_tokenizer.py

Log metrics to Weights & Biases (opt-in; off by default):
    python train_autoencoder_bouncing.py --wandb [--wandb-entity TEAM] [--wandb-name run1]

Visualize a saved checkpoint (OpenCV window; needs a display):
    python src/training/train_tokenizer.py --test-checkpoint
"""

import argparse
import os
import random
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
# NOTE: `import lpips` is intentionally deferred to the --lpips branch below so that, when the
# flag is off, the module (and its torchvision backbone imports) impose zero RAM/processing cost.

# Put `src` on path (where `wlog` and the `models` package live).
_SRC = Path(__file__).resolve().parents[1]   # .../src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import wlog
from models.tokenizer import AutoEncoder, AutoEncoderConfig


class ChunkClipDataset(Dataset):
    """Fixed-length clips along time: ``[start + j*L : start + (j+1)*L]`` per episode.

    ``frames`` stays memory-mapped uint8 on disk (``np.load(..., mmap_mode="r")``); each clip is
    converted to float32 only on access, so the full dataset is never materialized in RAM.
    ``episode_indices`` selects this split's episodes without copying (fancy-indexing a memmap
    would silently pull the whole subset into memory).
    """

    def __init__(
        self,
        frames: np.ndarray,
        episode_indices: np.ndarray,
        chunk_len: int,
        start_offset: int = 0,
    ) -> None:
        self.frames = frames  # (N, T, H, W, 3) uint8
        self.episode_indices = np.asarray(episode_indices)
        self.chunk_len = int(chunk_len)
        self.start_offset = int(start_offset)
        self._rebuild_index()

    def set_start_offset(self, start_offset: int) -> None:
        self.start_offset = int(start_offset)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        o = self.start_offset
        L = self.chunk_len
        T = self.frames.shape[1]
        pairs: list[tuple[int, int]] = []
        for ep in self.episode_indices:
            n_chunks = (T - o) // L
            for j in range(n_chunks):
                pairs.append((int(ep), o + j * L))
        self._pairs = pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ep, start = self._pairs[idx]
        clip_u8 = np.asarray(self.frames[ep, start : start + self.chunk_len])
        clip = torch.from_numpy(clip_u8.astype(np.float32) / 255.0)
        return clip, clip


def _config_from_checkpoint(cfg_dict: dict) -> AutoEncoderConfig:
    """Build AutoEncoderConfig from a saved dict; ignore unknown keys for forward compat."""
    allowed = {f.name for f in fields(AutoEncoderConfig)}
    return AutoEncoderConfig(**{k: v for k, v in cfg_dict.items() if k in allowed})


def foreground_mask(frames: torch.Tensor, thresh: float = 0.1) -> torch.Tensor:
    """Per-clip foreground (moving-object) mask from the VISIBLE input frames.

    The static scene (background colour + grid lines + border) is constant across a clip; the
    only thing that changes cell-to-cell is the ball. So the per-pixel TEMPORAL median over the
    clip's frames is the static scene (the ball, in a different cell each tick, medians out), and
    a pixel's deviation from that median isolates the ball (plus any moving overlay such as the
    occlusion curtain). This is NOT privileged information: the tokenizer reconstructs fully
    visible frames, so locating the moving object from the input pixels is just emphasising
    visible content (D-042).

    Args:
        frames: (B, T, H, W, 3) in [0, 1].
        thresh: per-pixel L1 deviation (summed over RGB) above which a pixel is foreground.
    Returns:
        (B, T, H, W) float mask in {0, 1}.
    """
    template = frames.median(dim=1, keepdim=True).values     # (B, 1, H, W, 3) static scene
    dev = (frames - template).abs().sum(dim=-1)               # (B, T, H, W)
    return (dev > thresh).float()


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Per-pixel-weighted MSE. ``weight`` is (B, T, H, W), broadcast over the RGB channel.

    Normalised by the total weight so the scale stays comparable to a plain mean MSE and the
    background is still reconstructed (it is easy / low-loss); the weighting only re-prioritises
    gradient toward the upweighted foreground pixels (D-042).
    """
    w = weight.unsqueeze(-1)                                  # (B, T, H, W, 1)
    se = (pred - target) ** 2                                 # (B, T, H, W, 3)
    return (w * se).sum() / (w.expand_as(se).sum() + 1e-8)


def region_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Recon MSE restricted to foreground vs background pixels, for the validity metric (D-042).

    Returns (fg_mse, bg_mse, fg_frac). If the ball is dropped, fg_mse stays high while bg_mse → 0
    even as the aggregate MSE looks healthy — the signal that aggregate MSE is blind to.
    """
    se_pix = ((pred - target) ** 2).mean(dim=-1)             # (B, T, H, W), mean over RGB
    fg = mask
    bg = 1.0 - mask
    fg_sum, bg_sum = fg.sum(), bg.sum()
    fg_mse = (se_pix * fg).sum() / (fg_sum + 1e-8)
    bg_mse = (se_pix * bg).sum() / (bg_sum + 1e-8)
    fg_frac = fg_sum / fg.numel()
    return fg_mse.item(), bg_mse.item(), fg_frac.item()


def run_test_checkpoint(args: argparse.Namespace) -> None:
    import cv2

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    raw = np.load(args.frames, mmap_mode="r")
    if raw.ndim != 5 or raw.shape[-1] != 3:
        raise ValueError(f"Expected (N, T, H, W, 3), got {raw.shape}")

    n_eps, n_frames, h, w, _ = raw.shape
    if n_eps < 1 or n_frames < 1:
        raise ValueError("Dataset must have at least one episode and one frame.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = _config_from_checkpoint(payload["config"])
    model = AutoEncoder(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    np.set_printoptions(precision=4, suppress=True, linewidth=120)

    win = "AutoEncoder: input | reconstruction (space=new, q/Esc=quit)"
    scale = max(2, 512 // max(h, w))

    def uint8_rgb_to_bgr(frame_rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def tensor01_to_bgr(img_t: torch.Tensor) -> np.ndarray:
        """(H, W, 3) float ~[0,1] RGB -> uint8 BGR for display."""
        x = img_t.detach().cpu().clamp(0.0, 1.0).numpy()
        rgb = (x * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    L = cfg.max_temporal_length

    def pick_and_show() -> None:
        ep = random.randrange(n_eps)
        o = random.randint(0, L)
        n_chunks = (n_frames - o) // L
        if n_chunks <= 0:
            raise ValueError(
                f"Episode length {n_frames} is too short for context length {L} "
                f"with start offset {o} (need at least o+L frames)."
            )
        j = random.randrange(n_chunks)
        start = o + j * L
        ti = random.randrange(L)
        clip_u8 = raw[ep, start : start + L]
        frame_rgb = clip_u8[ti]
        clip = clip_u8.astype(np.float32) / 255.0
        x = torch.from_numpy(clip).unsqueeze(0).to(device)

        with torch.no_grad():
            z = model.encoder(x)
            recon = model.decoder(z)

        z_np = z[0, ti].detach().cpu().numpy()
        print(
            f"\n--- random sample  episode={ep}  clip=[{start}:{start+L})  t_in_clip={ti}  "
            f"z[batch,t] shape={z_np.shape} ---"
        )
        for i, token in enumerate(z_np):
            print(f"bottleneck token {i}: {token}")

        inp_bgr = uint8_rgb_to_bgr(frame_rgb)
        rec_bgr = tensor01_to_bgr(recon[0, ti])
        pair = np.hstack([inp_bgr, rec_bgr])
        disp_h, disp_w = pair.shape[0] * scale, pair.shape[1] * scale
        display = cv2.resize(pair, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
        cv2.putText(
            display,
            f"ep {ep}  clip {start}:{start + L}  t {ti}   |   SPACE new   q/Esc quit",
            (6, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(win, display)

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    pick_and_show()
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord(" "):
            pick_and_show()
        if key in (ord("q"), ord("Q"), 27):
            break
    cv2.destroyAllWindows()


def run_save_recon(args: argparse.Namespace) -> None:
    """Headless: render N temporal input/reconstruction strips to a single PNG (no GUI).

    Layout per clip: two rows (top=ground-truth input frames, bottom=reconstruction)
    laid out left->right across the L timesteps. Clips are stacked vertically with a
    thin separator. Reproducible via --seed. For async review of tokenizer quality,
    including curtain/occlusion frames.
    """
    import cv2

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    rng = random.Random(args.seed)
    raw = np.load(args.frames, mmap_mode="r")
    if raw.ndim != 5 or raw.shape[-1] != 3:
        raise ValueError(f"Expected (N, T, H, W, 3), got {raw.shape}")
    n_eps, n_frames, h, w, _ = raw.shape

    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = _config_from_checkpoint(payload["config"])
    model = AutoEncoder(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    L = cfg.max_temporal_length
    if n_frames < L:
        raise ValueError(f"Episode length {n_frames} < context length {L}.")

    sep = 3  # px separator (mid-gray) between rows/clips, pre-scale
    gray = 128
    total_mse = 0.0
    clip_rows = []
    for _ in range(args.n_samples):
        ep = rng.randrange(n_eps)
        start = rng.randrange(0, n_frames - L + 1)
        clip_u8 = np.asarray(raw[ep, start : start + L])  # (L, H, W, 3) RGB uint8
        clip = clip_u8.astype(np.float32) / 255.0
        x = torch.from_numpy(clip).unsqueeze(0).to(device)
        with torch.no_grad():
            recon = model.decoder(model.encoder(x))
        rec01 = recon[0].clamp(0, 1).cpu().numpy()  # (L, H, W, 3) RGB
        total_mse += float(((rec01 - clip) ** 2).mean())

        def strip(frames_rgb):  # (L, H, W, 3) -> BGR row with 1px gaps
            cells = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames_rgb]
            row = []
            for k, c in enumerate(cells):
                if k:
                    row.append(np.full((h, 1, 3), gray, np.uint8))
                row.append(c)
            return np.hstack(row)

        inp_row = strip(clip_u8)
        rec_row = strip((rec01 * 255).round().astype(np.uint8))
        wbar = np.full((sep, inp_row.shape[1], 3), gray, np.uint8)
        clip_rows.append(np.vstack([inp_row, wbar, rec_row]))

    gap = np.full((sep * 3, clip_rows[0].shape[1], 3), 60, np.uint8)
    stacked = clip_rows[0]
    for cr in clip_rows[1:]:
        stacked = np.vstack([stacked, gap, cr])

    scale = max(2, 768 // stacked.shape[1])
    big = cv2.resize(
        stacked,
        (stacked.shape[1] * scale, stacked.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    args.save_recon.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.save_recon), big)
    mean_mse = total_mse / args.n_samples
    print(
        f"Saved {args.n_samples} input/recon strips (L={L}) to {args.save_recon}  "
        f"| mean per-pixel MSE over rendered clips: {mean_mse:.5f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames",
        type=Path,
        default=_SRC.parent.parent / "bouncing.npy",
        help="Path to bouncing.npy (N, T, H, W, C) uint8.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker processes. Default: auto from SLURM_CPUS_PER_TASK / os.cpu_count() "
             "(capped at 8). 0 = synchronous load on the main thread (starves the GPU).",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_SRC.parent / "checkpoints" / "bouncing" / "tokenizer.pt",
        help="Where to save weights + config (also loaded by --test-checkpoint). Env-specific: "
             "pass --checkpoint, e.g. checkpoints/gridworld/tokenizer.pt.",
    )
    parser.add_argument(
        "--test-checkpoint",
        action="store_true",
        help="Load --checkpoint and open a window: random L-frame clip vs reconstruction; SPACE = new sample; q/Esc = quit. L = max_temporal_length from checkpoint.",
    )
    parser.add_argument(
        "--save-recon",
        type=Path,
        default=None,
        help="Headless: load --checkpoint, render N temporal input/reconstruction strips to this PNG (no GUI). For async review.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=4,
        help="Number of clips to render for --save-recon.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for --save-recon clip selection (reproducible views).",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Clip length in frames (must match AutoEncoderConfig.max_temporal_length). Default: value from AutoEncoderConfig (typically 32).",
    )
    parser.add_argument(
        "--val-offset",
        type=int,
        default=0,
        help="Fixed start offset o for validation chunks (windows [o:o+L], [o+L:o+2L], …).",
    )
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Train on only the first N episodes of --frames (fast local validation on a subset). "
        "Default: use all episodes.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any existing --checkpoint and train from random init (useful after architecture changes).",
    )
    parser.add_argument(
        "--lpips",
        action="store_true",
        help="Add an LPIPS perceptual term to the reconstruction loss. Off by default; when off, "
        "the lpips module is never imported and no LPIPS net is built (zero RAM/VRAM/compute cost).",
    )
    parser.add_argument(
        "--lpips-net",
        type=str,
        default="vgg",
        choices=["vgg", "alex", "squeeze"],
        help="Backbone for the LPIPS net (only used when --lpips is set). Default: vgg.",
    )
    parser.add_argument(
        "--lpips-weight",
        type=float,
        default=0.2,
        help="Weight of the LPIPS term added to MSE (only used when --lpips is set).",
    )
    parser.add_argument(
        "--fg-weight",
        type=float,
        default=0.0,
        help="Foreground (moving-object) upweighting alpha for the recon loss (D-042). Per-pixel "
        "weight = 1 + alpha*fg, where fg = deviation from the per-clip temporal-median (static "
        "scene) > --fg-thresh, which isolates the ball. 0.0 = uniform mean MSE (old behaviour). "
        "Needed for sparse objects: the GridWorld ball is ~1%% of pixels, so uniform MSE ignores it.",
    )
    parser.add_argument(
        "--fg-thresh",
        type=float,
        default=0.1,
        help="Per-pixel deviation-from-static-scene threshold (L1 summed over RGB, [0,1] scale) "
        "for the foreground mask used by --fg-weight and the val/fg_mse validity metric.",
    )
    parser.add_argument(
        "--adam-beta2",
        type=float,
        default=0.999,
        help="AdamW beta2 (D-043). The torch default 0.999 keeps a long second-moment memory; when the "
        "recon loss gets very low (the sparse ball has been learned) the second moment shrinks and a "
        "single odd batch produces an oversized Adam step -> loss explosion (observed EXP-024 ep10: "
        "val/mse 6e-4 -> 3.8e-2, never fully recovered). Lower (e.g. 0.95) makes v adapt faster and is "
        "the mechanistic fix; recommended for the sparse GridWorld ball.",
    )
    parser.add_argument(
        "--adam-eps",
        type=float,
        default=1e-8,
        help="AdamW epsilon (D-043). Raising it (e.g. 1e-6) also caps the per-parameter step lr/(sqrt(v)+eps) "
        "when the second moment is tiny; a secondary stability knob.",
    )
    parser.add_argument(
        "--grad-spike-mult",
        type=float,
        default=0.0,
        help="Gradient-spike backstop (D-043). If >0, skip the optimizer step (zero grads, still advance "
        "the LR schedule) when the pre-clip grad norm is non-finite or exceeds this multiple of its "
        "running EMA (after the warmup grace). Catches the spike-shaped manifestation of the explosion; "
        "0 = clip-only (old behaviour). Recommended ~5.0.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Log per-step train MSE/loss + grad norm to W&B every N optimizer steps (D-043). The "
        "epoch-only curve (30 points) is too coarse to see an intra-epoch loss explosion.",
    )
    wlog.add_args(parser)
    args = parser.parse_args()
    # Enable TF32 for any fp32 matmuls that remain outside the bf16 autocast region (free on H100).
    torch.set_float32_matmul_precision("high")

    if args.test_checkpoint:
        run_test_checkpoint(args)
        return

    if args.save_recon is not None:
        run_save_recon(args)
        return

    # Memory-mapped uint8; clips are converted to float32 per-batch in the dataset, so RAM
    # stays at ~the touched pages instead of a full 4x float32 copy of the dataset.
    raw = np.load(args.frames, mmap_mode="r")
    if raw.ndim != 5 or raw.shape[-1] != 3:
        raise ValueError(f"Expected (N, T, H, W, 3), got {raw.shape}")

    n, t, h, w, c = raw.shape
    if args.max_episodes is not None and args.max_episodes < n:
        raw = raw[: args.max_episodes]   # lazy mmap slice
        n = args.max_episodes
        print(f"--max-episodes: training on first {n} episodes only")
    if n < 2:
        raise ValueError(
            f"Temporal training splits by episode; need at least 2 episodes, got n={n}."
        )

    n_val = max(1, int(round(n * args.val_fraction)))
    n_val = min(n_val, n - 1)
    n_train = n - n_val
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    val_idx = perm[:n_val].numpy()
    train_idx = perm[n_val:].numpy()

    base = AutoEncoderConfig(img_input_H=h, img_input_W=w)
    chunk_len = args.context_length if args.context_length is not None else base.max_temporal_length
    if chunk_len < 1:
        raise ValueError("--context-length must be positive.")
    if t < chunk_len:
        raise ValueError(
            f"Episode length T={t} must be >= context length L={chunk_len} "
            "to form at least one full clip."
        )
    cfg = AutoEncoderConfig(
        img_input_H=h, img_input_W=w, max_temporal_length=chunk_len
    )

    train_ds = ChunkClipDataset(raw, train_idx, chunk_len, start_offset=0)
    val_ds = ChunkClipDataset(raw, val_idx, chunk_len, start_offset=args.val_offset % (chunk_len + 1))
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            f"No clips with L={chunk_len}, val_offset={args.val_offset}. "
            "Try a smaller --val-offset or longer episodes."
        )

    # DataLoader throughput (D-041): num_workers=0 loads each batch synchronously on the main
    # thread (memmap read + uint8->float32) while the GPU idles -> the 0<->100% util sawtooth /
    # cold H100 we observed. Use background workers + pinned memory + persistent workers +
    # prefetch so batches are ready before the GPU asks. num_workers auto-derives from the SLURM
    # CPU allocation (job MUST request enough --cpus, else workers contend); override with --num-workers.
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
        loader_kw.update(persistent_workers=True, prefetch_factor=4)
    print(f"DataLoader: num_workers={nw} pin_memory={pin}"
          + (f" prefetch_factor=4 persistent_workers=True" if nw > 0 else ""))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- latent-collapse health probe -------------------------------------------------
    # MSE is a misleading health signal here: trivial curtain frames + the static background
    # dominate it, so a fully-collapsed ("gray mush") tokenizer can still show a healthy-looking
    # MSE. The real signal is whether DISTINCT frames get DISTINCT latents. We hold out a fixed
    # set of revealed (curtain-up) frames and each epoch report their pairwise latent cosine
    # (->1.0 == collapsed, <0.7 == escaped) and per-image output std (~0.01 == mean mush,
    # >0.04 == rendering real content). Auto-uses <frames>_actions.npy to pick revealed frames;
    # falls back to random distinct frames (noisier, since curtain frames are near-identical).
    def probe_frame(ep: int, ti: int) -> torch.Tensor:
        """Single frame from the memmap as float32 in [0,1]."""
        return torch.from_numpy(np.asarray(raw[ep, ti]).astype(np.float32) / 255.0)

    probe = None
    n_probe = 64
    actions_path = args.frames.with_name(args.frames.stem + "_actions.npy")
    if actions_path.is_file():
        acts_all = np.load(actions_path)  # (N, T) 0=revealed 1=curtain
        val_acts = acts_all[val_idx]
        pframes = []
        for ei in range(val_acts.shape[0]):
            rev = np.where(val_acts[ei] == 0)[0]
            if len(rev):
                pframes.append(probe_frame(int(val_idx[ei]), int(rev[0])))
            if len(pframes) >= n_probe:
                break
        if pframes:
            probe = torch.stack(pframes).unsqueeze(1).to(device)  # (P,1,H,W,3)
            print(f"[health] latent-collapse probe: {probe.shape[0]} revealed frames from {actions_path.name}")
    if probe is None:
        ne = min(n_probe, len(val_idx))
        probe = torch.stack([probe_frame(int(val_idx[i]), 0) for i in range(ne)]).unsqueeze(1).to(device)
        print(f"[health] latent-collapse probe: {probe.shape[0]} random frames (no actions file; metric noisier)")

    def latent_health():
        model.eval()
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                z = model.encoder(probe).float()
                pred = model(probe).float()
        p = z.shape[0]
        zf = z.reshape(p, -1)
        zf = zf / (zf.norm(dim=1, keepdim=True) + 1e-6)
        cos = (zf @ zf.T)[~torch.eye(p, dtype=torch.bool, device=zf.device)].mean().item()
        pstd = pred.reshape(p, -1).std(1).mean().item()
        return cos, pstd

    if args.checkpoint.is_file() and not args.fresh:
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cfg = _config_from_checkpoint(payload["config"])
        model = AutoEncoder(cfg).to(device)
        # Tolerant load so architecture changes during development warm-start instead of
        # crashing; report any mismatches so they are never silent.
        result = model.load_state_dict(payload["model_state_dict"], strict=False)
        if result.missing_keys:
            print(f"[resume] randomly-initialized (not in checkpoint): {result.missing_keys}")
        if result.unexpected_keys:
            print(f"[resume] ignored (in checkpoint, not in model): {result.unexpected_keys}")
        print(f"Loaded weights from {args.checkpoint}")
    else:
        model = AutoEncoder(cfg).to(device)

    # cfg is final here (the resume branch may have replaced it from the checkpoint), so it
    # accurately describes the architecture wandb.config will record.
    wlog.init(args, cfg, project="transformer-C-tokenizer")

    # betas[1]=--adam-beta2 (D-043): 0.95 makes the 2nd moment adapt faster so it can't go stale-tiny
    # at the loss minimum and amplify one batch into a destructive step (the EXP-024 ep10 explosion).
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, args.adam_beta2), eps=args.adam_eps
    )
    print(f"AdamW: lr={args.lr} betas=(0.9, {args.adam_beta2}) eps={args.adam_eps}"
          + (f" | grad-spike skip at {args.grad_spike_mult}x EMA" if args.grad_spike_mult > 0 else ""))
    # Per-STEP warmup -> constant -> late-cosine-cooldown. The latent-collapse escape is a
    # saddle-point plateau that only ignites after ~2k steps and needs SUSTAINED lr to complete;
    # a plain CosineAnnealingLR over few epochs decays through the escape window and freezes the
    # tokenizer in the collapsed (gray-mush) basin. Keep lr flat through the escape, cool down
    # only in the final 25% for crisp convergence. See memory: qk-norm-attention-temperature.
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = max(200, int(0.05 * total_steps))
    decay_start = int(0.75 * total_steps)
    eta_min_ratio = 1e-6 / args.lr

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if step < decay_start:
            return 1.0
        p = (step - decay_start) / max(1, total_steps - decay_start)
        return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (1.0 + np.cos(np.pi * p))

    scheduler = LambdaLR(opt, lr_lambda)
    loss_fn = nn.MSELoss()
    # Only when --lpips is set do we import the module and build the net; otherwise this stays
    # None and the train loop never touches LPIPS, so it costs nothing.
    lpips_loss_fn = None
    if args.lpips:
        import lpips  # deferred so the off-path has no import/RAM footprint

        lpips_loss_fn = lpips.LPIPS(net=args.lpips_net).to(device)
        lpips_loss_fn.eval()
        for p in lpips_loss_fn.parameters():
            p.requires_grad_(False)
        print(f"LPIPS enabled (net={args.lpips_net}, weight={args.lpips_weight})")
    # Running magnitude estimates (detached) used to put LPIPS on the MSE scale so that
    # --lpips-weight is a scale-free ratio (w=1 -> equal contribution). We rescale LPIPS onto
    # MSE rather than normalizing both terms, which would let the global loss/grad scale drift
    # as MSE shrinks and disturb the LR schedule tuned for the collapse escape (see qk-norm memory).
    LPIPS_EMA_DECAY = 0.99
    ema_mse = None
    ema_lpips = None
    use_amp = device == "cuda"

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    last_ckpt = args.checkpoint.with_name(args.checkpoint.stem + "_last" + args.checkpoint.suffix)

    # Training-stability + safety-net state (D-043).
    global_step = 0
    gn_ema = None              # running EMA of the pre-clip grad norm (spike-skip reference)
    best_fg = float("inf")     # best val/fg_mse seen -> the canonical checkpoint tracks the BEST
                               # ball-encoding model, so a loss explosion can't discard it (EXP-024).

    epoch_bar = tqdm(range(args.epochs), desc="Epochs", position=0, mininterval=1.0)
    for epoch in epoch_bar:
        train_off = random.randint(0, chunk_len)
        train_ds.set_start_offset(train_off)
        if len(train_ds) == 0:
            train_ds.set_start_offset(0)
            train_off = 0
        val_ds.set_start_offset(args.val_offset % (chunk_len + 1))
        if len(val_ds) == 0:
            raise RuntimeError(
                "Validation clip index is empty for this val_offset and episode length. "
                "Try --val-offset 0 or a smaller offset."
            )

        model.train()
        train_loss = 0.0
        train_lpips = 0.0
        n_train_samples = 0
        epoch_grad_norm = 0.0   # sum of finite pre-clip grad norms this epoch (D-043)
        epoch_skipped = 0       # steps skipped by the spike guard this epoch
        train_t0 = time.perf_counter()
        for batch_x, _ in tqdm(
            train_loader,
            desc=f"Train {epoch + 1}/{args.epochs}",
            leave=False,
            position=1,
            mininterval=1.0,
        ):
            B, T, H, W, C = batch_x.shape
            n_train_samples += B
            batch_x = batch_x.to(device)
            # Foreground mask from the (fp32) input, outside autocast (D-042). None when disabled.
            fg = foreground_mask(batch_x, args.fg_thresh) if args.fg_weight > 0 else None
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                pred = model(batch_x)
                mse_plain = loss_fn(pred, batch_x)        # plain mean MSE: logged + LPIPS scale anchor
                mse_raw = mse_plain.detach().item()
                # Foreground-weighted MSE drives the loss when --fg-weight > 0; else plain mean MSE.
                mse = weighted_mse(pred, batch_x, 1.0 + args.fg_weight * fg) if fg is not None else mse_plain
                loss = mse
                if lpips_loss_fn is not None:
                    lpips_val = lpips_loss_fn(
                        pred.reshape(B * T, H, W, 3).permute(0, 3, 1, 2),
                        batch_x.reshape(B * T, H, W, 3).permute(0, 3, 1, 2),
                        normalize=True,
                    ).mean()
                    lpips_raw = lpips_val.detach().item()
                    # Update detached running magnitudes (lazy-init to first observation).
                    if ema_mse is None:
                        ema_mse, ema_lpips = mse_raw, lpips_raw
                    else:
                        ema_mse = LPIPS_EMA_DECAY * ema_mse + (1.0 - LPIPS_EMA_DECAY) * mse_raw
                        ema_lpips = LPIPS_EMA_DECAY * ema_lpips + (1.0 - LPIPS_EMA_DECAY) * lpips_raw
                    # Map LPIPS onto the MSE scale; `scale` is a plain float so no grad flows through it.
                    scale = ema_mse / (ema_lpips + 1e-8)
                    loss = mse + args.lpips_weight * scale * lpips_val
                    train_lpips += lpips_raw
            opt.zero_grad()
            loss.backward()
            # Gradient clipping: without it, a single large-gradient batch under bf16 can land a
            # destructive update that spikes the loss and knocks the model back into the latent-
            # collapse basin (observed as lat_cos jumping ~0.36 -> ~0.99 mid-training). Standard
            # transformer safeguard; keeps the escape monotonic. clip_grad_norm_ returns the PRE-clip
            # total norm — the observability + spike-skip signal (D-043).
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0))
            # Spike guard (D-043): skip the update on a non-finite norm, or (when --grad-spike-mult>0)
            # a norm far above its running EMA past the warmup grace. Note: this catches the
            # spike-SHAPED explosion; the Adam-2nd-moment CAUSE is handled by --adam-beta2.
            spike = (not np.isfinite(gn)) or (
                args.grad_spike_mult > 0
                and gn_ema is not None
                and global_step > warmup_steps
                and gn > args.grad_spike_mult * gn_ema
            )
            if spike:
                opt.zero_grad(set_to_none=True)
                epoch_skipped += 1
            else:
                opt.step()
                gn_ema = gn if gn_ema is None else 0.98 * gn_ema + 0.02 * gn
                if np.isfinite(gn):
                    epoch_grad_norm += gn
            scheduler.step()
            global_step += 1
            train_loss += mse_raw  # log raw reconstruction MSE, not the LPIPS-augmented total
            # Per-step W&B logging so an intra-epoch loss explosion is visible (auto-step axis; carries
            # global_step/epoch for the x-axis). Cheap relative to the bf16 forward/backward (D-043).
            if global_step % args.log_every == 0:
                wlog.log({
                    "train/step_mse": mse_raw,
                    "train/step_loss": float(loss.detach().item()),
                    "train/grad_norm": gn,
                    "train/grad_norm_ema": gn_ema if gn_ema is not None else 0.0,
                    "train/skipped_cumulative": epoch_skipped,
                    "lr": opt.param_groups[0]["lr"],
                    "global_step": global_step,
                    "epoch": epoch,
                })

        train_time = time.perf_counter() - train_t0
        samples_per_s = n_train_samples / train_time if train_time > 0 else 0.0

        model.eval()
        val_loss = 0.0
        val_fg_mse = val_bg_mse = val_fg_frac = 0.0  # ball-region validity metric (D-042)
        with torch.no_grad():
            for batch_x, _ in tqdm(
                val_loader,
                desc="Val",
                leave=False,
                position=1,
                mininterval=1.0,
            ):
                batch_x = batch_x.to(device)
                fg = foreground_mask(batch_x, args.fg_thresh)
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                    pred = model(batch_x)
                    val_loss += loss_fn(pred, batch_x).item()
                fmse, bmse, ffrac = region_mse(pred.float(), batch_x, fg)
                val_fg_mse += fmse
                val_bg_mse += bmse
                val_fg_frac += ffrac

        train_mse = train_loss / len(train_loader)
        val_mse = val_loss / len(val_loader)
        n_stepped = max(1, len(train_loader) - epoch_skipped)
        grad_norm_mean = epoch_grad_norm / n_stepped   # mean pre-clip grad norm over stepped batches
        n_val = len(val_loader)
        val_fg_mse /= n_val   # recon MSE on ball/moving pixels — THE validity guard: stays high if
        val_bg_mse /= n_val   # the ball is dropped even while val/mse looks healthy (background-only)
        val_fg_frac /= n_val
        train_lpips_mean = train_lpips / len(train_loader) if lpips_loss_fn is not None else None
        lat_cos, pred_std = latent_health()
        current_lr = opt.param_groups[0]["lr"]
        epoch_bar.set_postfix(
            train=f"{train_mse:.6f}",
            val=f"{val_mse:.6f}",
            lat_cos=f"{lat_cos:.3f}",
            pstd=f"{pred_std:.4f}",
            lr=f"{current_lr:.2e}",
        )
        print(
            f"Epoch {epoch + 1} | train MSE: {train_mse:.6f} | val MSE: {val_mse:.6f} "
            f"| val fg_mse: {val_fg_mse:.6f} bg_mse: {val_bg_mse:.6f} (fg_frac {val_fg_frac:.3f}) "
            f"| latent_cos: {lat_cos:.3f} (<0.7=escaped) | pred_std: {pred_std:.4f} (>0.04=content) "
            f"| grad_norm: {grad_norm_mean:.3f} (skipped {epoch_skipped}/{len(train_loader)}) "
            f"| train_clip_offset={train_off} | lr: {current_lr:.2e}"
        )
        # Auto-step axis (D-043): no explicit step=, so this shares one monotonic W&B step with the
        # per-step logs above; global_step/epoch are carried as fields for use as the x-axis.
        wlog.log(
            {
                "train/mse": train_mse,
                "val/mse": val_mse,
                "val/fg_mse": val_fg_mse,   # recon MSE on ball/moving pixels — validity guard (D-042)
                "val/bg_mse": val_bg_mse,   # recon MSE on static background pixels
                "val/fg_frac": val_fg_frac, # fraction of pixels flagged foreground
                "val/best_fg_mse": min(best_fg, val_fg_mse),  # best ball-recon so far (canonical ckpt)
                "latent_cos": lat_cos,  # <0.7 == escaped collapse
                "pred_std": pred_std,   # >0.04 == rendering real content
                "lr": current_lr,
                "train/grad_norm_epoch": grad_norm_mean,   # mean pre-clip grad norm (stability)
                "train/skipped_epoch": epoch_skipped,      # spike-guard skips this epoch
                "global_step": global_step,
                "epoch": epoch,
                "train_clip_offset": train_off,
                "perf/train_time_s": train_time,        # wall-clock of the train loop only
                "perf/samples_per_s": samples_per_s,    # training throughput (clips/sec)
                **(
                    {
                        "train/lpips": train_lpips_mean,            # raw LPIPS, scale-independent
                        "train/lpips_scale": ema_mse / (ema_lpips + 1e-8),  # MSE/LPIPS magnitude ratio
                    }
                    if train_lpips_mean is not None
                    else {}
                ),
            },
        )

        # Safety net (D-043): the canonical --checkpoint tracks the BEST val/fg_mse (best ball recon),
        # so a later loss explosion can never discard the good model (the EXP-024 mistake). _last.pt
        # always holds the most recent epoch for resume/inspection.
        payload = {"model_state_dict": model.state_dict(), "config": asdict(cfg)}
        torch.save(payload, last_ckpt)
        if val_fg_mse < best_fg:
            best_fg = val_fg_mse
            torch.save(payload, args.checkpoint)
            tqdm.write(f"new best val/fg_mse {best_fg:.6f} -> saved {args.checkpoint}")
        else:
            tqdm.write(f"val/fg_mse {val_fg_mse:.6f} >= best {best_fg:.6f}; kept best in {args.checkpoint} (last -> {last_ckpt.name})")

    wlog.finish()


if __name__ == "__main__":
    main()
