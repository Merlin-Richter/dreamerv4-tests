"""
Train the temporal AutoEncoder (tokenizer) to reconstruct frames, then freeze it.

Episodes are sliced into fixed-length clips of ``max_temporal_length`` frames. Each epoch
picks a new random start offset so clip boundaries shift; validation uses a fixed offset.
Loss = MSE (+ optional LPIPS) in [0,1]; MAE patch-dropout is active in train mode.

Run with -u (unbuffered). From repo root:
    python -u src/training/train_tokenizer.py --frames data/gridworld.npy --lpips
Visualize a checkpoint:
    python -u src/training/train_tokenizer.py --test-checkpoint --checkpoint <ckpt.pt>
"""

import argparse
import os
import random
import sys
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
# NOTE: `import lpips` is deferred to the --lpips branch so the off-path has zero import/RAM cost.

_SRC = Path(__file__).resolve().parents[1]   # .../src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import wlog
from models.tokenizer import AutoEncoder, AutoEncoderConfig


class ChunkClipDataset(Dataset):
    """Fixed-length clips along time: ``[start + j*L : start + (j+1)*L]`` per episode.

    ``frames`` stays memory-mapped uint8 on disk; each clip is converted to float32 only on
    access, so the full dataset is never materialized in RAM. ``episode_indices`` selects this
    split's episodes without copying.
    """

    def __init__(self, frames, episode_indices, chunk_len, start_offset=0):
        self.frames = frames  # (N, T, H, W, 3) uint8
        self.episode_indices = np.asarray(episode_indices)
        self.chunk_len = int(chunk_len)
        self.start_offset = int(start_offset)
        self._rebuild_index()

    def set_start_offset(self, start_offset):
        self.start_offset = int(start_offset)
        self._rebuild_index()

    def _rebuild_index(self):
        o, L, T = self.start_offset, self.chunk_len, self.frames.shape[1]
        self._pairs = [
            (int(ep), o + j * L)
            for ep in self.episode_indices
            for j in range((T - o) // L)
        ]

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        ep, start = self._pairs[idx]
        clip_u8 = np.asarray(self.frames[ep, start : start + self.chunk_len])
        clip = torch.from_numpy(clip_u8.astype(np.float32) / 255.0)
        return clip, clip


def _config_from_checkpoint(cfg_dict):
    """Build AutoEncoderConfig from a saved dict; ignore unknown keys for forward compat."""
    allowed = {f.name for f in fields(AutoEncoderConfig)}
    return AutoEncoderConfig(**{k: v for k, v in cfg_dict.items() if k in allowed})


def foreground_mask(frames, thresh=0.1):
    """Per-clip foreground (moving-object) mask from the visible input frames.

    The static scene is the per-pixel temporal median over the clip; a pixel's deviation from it
    isolates the moving object. Not privileged info — the tokenizer reconstructs visible frames.

    Args:
        frames: (B, T, H, W, 3) in [0, 1].
        thresh: per-pixel L1 deviation (summed over RGB) above which a pixel is foreground.
    Returns:
        (B, T, H, W) float mask in {0, 1}.
    """
    template = frames.median(dim=1, keepdim=True).values     # (B, 1, H, W, 3) static scene
    dev = (frames - template).abs().sum(dim=-1)               # (B, T, H, W)
    return (dev > thresh).float()


def weighted_mse(pred, target, weight):
    """Per-pixel-weighted MSE. ``weight`` is (B, T, H, W), broadcast over RGB, normalised by
    total weight so the scale stays comparable to plain mean MSE (D-042)."""
    w = weight.unsqueeze(-1)                                  # (B, T, H, W, 1)
    se = (pred - target) ** 2
    return (w * se).sum() / (w.expand_as(se).sum() + 1e-8)


def region_mse(pred, target, mask):
    """Recon MSE on foreground vs background pixels (validity metric, D-042).

    Returns (fg_mse, bg_mse, fg_frac). If the ball is dropped, fg_mse stays high while bg_mse → 0
    even as the aggregate MSE looks healthy.
    """
    se_pix = ((pred - target) ** 2).mean(dim=-1)             # (B, T, H, W), mean over RGB
    fg, bg = mask, 1.0 - mask
    fg_sum, bg_sum = fg.sum(), bg.sum()
    fg_mse = (se_pix * fg).sum() / (fg_sum + 1e-8)
    bg_mse = (se_pix * bg).sum() / (bg_sum + 1e-8)
    return fg_mse.item(), bg_mse.item(), (fg_sum / fg.numel()).item()


def _load_model(args, device):
    """Load AutoEncoder + config from --checkpoint for the viewer paths."""
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = _config_from_checkpoint(payload["config"])
    model = AutoEncoder(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, cfg


def run_test_checkpoint(args):
    """Interactive OpenCV window: random L-frame clip input vs reconstruction strips."""
    import cv2

    raw = np.load(args.frames, mmap_mode="r")
    if raw.ndim != 5 or raw.shape[-1] != 3:
        raise ValueError(f"Expected (N, T, H, W, 3), got {raw.shape}")
    n_eps, n_frames, h, w, _ = raw.shape

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = _load_model(args, device)
    L = cfg.max_temporal_length
    if n_frames < L:
        raise ValueError(f"Episode length {n_frames} < context length {L}.")

    win = "AutoEncoder: input | reconstruction (space=new, q/Esc=quit)"
    scale = max(2, 512 // max(h, w))

    def to_bgr01(img_t):
        x = img_t.detach().cpu().clamp(0, 1).numpy()
        return cv2.cvtColor((x * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)

    def pick_and_show():
        ep = random.randrange(n_eps)
        start = random.randrange(0, n_frames - L + 1)
        ti = random.randrange(L)
        clip_u8 = raw[ep, start : start + L]
        x = torch.from_numpy(clip_u8.astype(np.float32) / 255.0).unsqueeze(0).to(device)
        with torch.no_grad():
            recon = model.decoder(model.encoder(x))
        inp_bgr = cv2.cvtColor(clip_u8[ti], cv2.COLOR_RGB2BGR)
        pair = np.hstack([inp_bgr, to_bgr01(recon[0, ti])])
        display = cv2.resize(pair, (pair.shape[1] * scale, pair.shape[0] * scale),
                             interpolation=cv2.INTER_NEAREST)
        cv2.putText(display, f"ep {ep}  clip {start}:{start + L}  t {ti}   |   SPACE new   q/Esc quit",
                    (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
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


def run_save_recon(args):
    """Headless: render N temporal input/reconstruction strips to a single PNG (no GUI).

    Per clip: top row = ground-truth frames, bottom row = reconstruction, left→right over the L
    timesteps. The recon visual is the real success check (a low MSE can hide a dropped square).
    """
    import cv2

    rng = random.Random(args.seed)
    raw = np.load(args.frames, mmap_mode="r")
    if raw.ndim != 5 or raw.shape[-1] != 3:
        raise ValueError(f"Expected (N, T, H, W, 3), got {raw.shape}")
    n_eps, n_frames, h, w, _ = raw.shape

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = _load_model(args, device)
    L = cfg.max_temporal_length
    if n_frames < L:
        raise ValueError(f"Episode length {n_frames} < context length {L}.")

    sep, gray = 3, 128
    total_mse = 0.0
    clip_rows = []
    for _ in range(args.n_samples):
        ep = rng.randrange(n_eps)
        start = rng.randrange(0, n_frames - L + 1)
        clip_u8 = np.asarray(raw[ep, start : start + L])
        clip = clip_u8.astype(np.float32) / 255.0
        x = torch.from_numpy(clip).unsqueeze(0).to(device)
        with torch.no_grad():
            recon = model.decoder(model.encoder(x))
        rec01 = recon[0].clamp(0, 1).cpu().numpy()
        total_mse += float(((rec01 - clip) ** 2).mean())

        def strip(frames_rgb):
            row = []
            for k, f in enumerate(frames_rgb):
                if k:
                    row.append(np.full((h, 1, 3), gray, np.uint8))
                row.append(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
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
    big = cv2.resize(stacked, (stacked.shape[1] * scale, stacked.shape[0] * scale),
                     interpolation=cv2.INTER_NEAREST)
    args.save_recon.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.save_recon), big)
    print(f"Saved {args.n_samples} input/recon strips (L={L}) to {args.save_recon} "
          f"| mean per-pixel MSE: {total_mse / args.n_samples:.5f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, default=_SRC.parent / "data" / "gridworld.npy",
                        help="Path to frames (N, T, H, W, C) uint8.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader workers. Default: auto from SLURM_CPUS_PER_TASK/cpu_count "
                             "(capped 8); 0 on Windows. 0 = synchronous load on the main thread.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Where to SAVE {config, model_state_dict} (BEST val/fg_mse) (required; "
                             "env-specific). Also the checkpoint loaded by the --test-checkpoint / "
                             "--save-recon viewers.")
    parser.add_argument("--test-checkpoint", action="store_true",
                        help="Load --checkpoint and open a window: clip vs reconstruction.")
    parser.add_argument("--save-recon", type=Path, default=None,
                        help="Headless: render N input/recon strips to this PNG (no GUI).")
    parser.add_argument("--n-samples", type=int, default=4, help="Clips to render for --save-recon.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for torch/numpy/random (model init, per-epoch offset/shuffle; "
                             "also --save-recon selection). The train/val split is fixed (seed 0).")
    parser.add_argument("--context-length", type=int, default=None,
                        help="Clip length L (= AutoEncoderConfig.max_temporal_length). Default: config value.")
    # Model dims (env-dependent). Default None = keep the AutoEncoderConfig dataclass default (GridWorld);
    # the Memory-Maze run overrides these to the larger LOCKED config.
    parser.add_argument("--embedding-dim", type=int, default=None,
                        help="Transformer width (AutoEncoderConfig.embedding_dim). Default: config value (256).")
    parser.add_argument("--depth", type=int, default=None,
                        help="Attention layer count (3x[spatial,temporal,spatial]; temporal at i%%3==1). "
                             "Default: config value (9).")
    parser.add_argument("--n-heads", type=int, default=None,
                        help="Attention heads (embedding_dim must be divisible by this). Default: config value (16).")
    parser.add_argument("--n-latents", type=int, default=None,
                        help="Latent tokens per frame. Default: config value (4).")
    parser.add_argument("--bottleneck-dim", type=int, default=None,
                        help="Per-latent channel dim; frame bottleneck = n_latents x bottleneck_dim. "
                             "Default: config value (64).")
    parser.add_argument("--val-offset", type=int, default=0, help="Fixed start offset for val chunks.")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Train on only the first N episodes (fast local validation).")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Load weights + config to start training FROM (optional; default: random init).")
    parser.add_argument("--lpips", action="store_true",
                        help="Add an LPIPS perceptual term (off by default; module imported only when set).")
    parser.add_argument("--lpips-net", type=str, default="vgg", choices=["vgg", "alex", "squeeze"])
    parser.add_argument("--lpips-weight", type=float, default=0.2, help="Weight of the LPIPS term.")
    parser.add_argument("--fg-weight", type=float, default=0.0,
                        help="Foreground upweighting alpha (D-042): per-pixel weight = 1 + alpha*fg. "
                             "0 = uniform mean MSE. Needed for sparse objects (~1%% of pixels).")
    parser.add_argument("--fg-thresh", type=float, default=0.1,
                        help="Per-pixel deviation-from-static-scene threshold for the foreground mask.")
    parser.add_argument("--adam-beta2", type=float, default=0.95,
                        help="AdamW beta2 (D-043). 0.95 makes the 2nd moment adapt faster so it can't go "
                             "stale-tiny at the loss minimum and amplify one batch into a destructive step "
                             "(the EXP-024 ep10 explosion). The torch default 0.999 is the failure mode.")
    parser.add_argument("--adam-eps", type=float, default=1e-8,
                        help="AdamW epsilon (D-043). Raising it (e.g. 1e-6) caps the step when v is tiny.")
    parser.add_argument("--grad-spike-mult", type=float, default=0.0,
                        help="Grad-spike backstop (D-043). If >0, skip the optimizer step (still advance "
                             "the LR schedule) when the pre-clip grad norm is non-finite or > this multiple "
                             "of its running EMA (after warmup). 0 = clip-only. Recommended ~5.0.")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Log per-step MSE/loss + grad norm every N steps (D-043) so an intra-epoch "
                             "explosion is visible.")
    wlog.add_args(parser)
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")  # TF32 for stray fp32 matmuls (free on H100)

    if args.test_checkpoint:
        run_test_checkpoint(args)
        return
    if args.save_recon is not None:
        run_save_recon(args)
        return

    # Seed model init + per-epoch offset/shuffle sampling for reproducibility (the train/val split
    # below stays pinned at manual_seed(0) so the val set is stable across --seed, matching train_dynamics).
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Memory-mapped uint8; clips become float32 per-batch in the dataset, so RAM stays small.
    raw = np.load(args.frames, mmap_mode="r")
    if raw.ndim != 5 or raw.shape[-1] != 3:
        raise ValueError(f"Expected (N, T, H, W, 3), got {raw.shape}")
    n, t, h, w, c = raw.shape
    if args.max_episodes is not None and args.max_episodes < n:
        raw = raw[: args.max_episodes]
        n = args.max_episodes
        print(f"--max-episodes: training on first {n} episodes only")
    if n < 2:
        raise ValueError(f"Splits by episode; need at least 2 episodes, got n={n}.")

    n_val = min(max(1, int(round(n * args.val_fraction))), n - 1)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    val_idx = perm[:n_val].numpy()
    train_idx = perm[n_val:].numpy()

    base = AutoEncoderConfig(img_input_H=h, img_input_W=w)
    chunk_len = args.context_length if args.context_length is not None else base.max_temporal_length
    if chunk_len < 1:
        raise ValueError("--context-length must be positive.")
    if t < chunk_len:
        raise ValueError(f"Episode length T={t} must be >= context length L={chunk_len}.")
    # Model dims are env-dependent: unset CLI flags keep the dataclass default (GridWorld); the Memory-Maze
    # run overrides via --embedding-dim/--depth/--n-heads/--n-latents/--bottleneck-dim.
    cfg_overrides = dict(img_input_H=h, img_input_W=w, max_temporal_length=chunk_len)
    for attr, val in (("embedding_dim", args.embedding_dim), ("depth", args.depth),
                      ("n_heads", args.n_heads), ("n_latents", args.n_latents),
                      ("bottleneck_dim", args.bottleneck_dim)):
        if val is not None:
            cfg_overrides[attr] = val
    cfg = AutoEncoderConfig(**cfg_overrides)
    print(f"AutoEncoderConfig: embedding_dim={cfg.embedding_dim} depth={cfg.depth} n_heads={cfg.n_heads} "
          f"n_latents={cfg.n_latents} bottleneck_dim={cfg.bottleneck_dim} L={cfg.max_temporal_length} "
          f"img={cfg.img_input_H}x{cfg.img_input_W}")

    train_ds = ChunkClipDataset(raw, train_idx, chunk_len, start_offset=0)
    val_ds = ChunkClipDataset(raw, val_idx, chunk_len, start_offset=args.val_offset % (chunk_len + 1))
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"No clips with L={chunk_len}, val_offset={args.val_offset}.")

    # DataLoader throughput (D-041): background workers + pinned memory + prefetch so batches are
    # ready before the GPU asks. persistent_workers MUST be False: each epoch mutates the train
    # dataset's start offset (clip count changes), and persistent workers cache a stale index ->
    # IndexError at the epoch boundary (EXP-025). Non-persistent workers re-pickle the fresh index.
    nw = args.num_workers
    if nw is None:
        if os.name == "nt":
            nw = 0  # native-Windows: avoid spawn/memmap-pickle overhead
        else:
            slurm = os.environ.get("SLURM_CPUS_PER_TASK")
            nw = max(0, min(int(slurm) if slurm else (os.cpu_count() or 1), 8))
    pin = torch.cuda.is_available()
    loader_kw = dict(num_workers=nw, pin_memory=pin)
    if nw > 0:
        loader_kw.update(persistent_workers=False, prefetch_factor=4)
    print(f"DataLoader: num_workers={nw} pin_memory={pin}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    # --- latent-collapse health probe -------------------------------------------------
    # MSE is a misleading health signal (trivial curtain frames + static bg dominate it). The real
    # signal: do DISTINCT frames get DISTINCT latents? Hold out fixed revealed frames; each epoch
    # report their pairwise latent cosine (->1.0 collapsed, <0.7 escaped) and output std (~0.01 mush,
    # >0.04 real content). Uses <frames>_actions.npy to pick revealed frames; else random frames.
    def probe_frame(ep, ti):
        return torch.from_numpy(np.asarray(raw[ep, ti]).astype(np.float32) / 255.0)

    probe = None
    n_probe = 64
    actions_path = args.frames.with_name(args.frames.stem + "_actions.npy")
    if actions_path.is_file():
        val_acts = np.load(actions_path)[val_idx]  # (N, T) 0=revealed 1=curtain
        pframes = []
        for ei in range(val_acts.shape[0]):
            rev = np.where(val_acts[ei] == 0)[0]
            if len(rev):
                pframes.append(probe_frame(int(val_idx[ei]), int(rev[0])))
            if len(pframes) >= n_probe:
                break
        if pframes:
            probe = torch.stack(pframes).unsqueeze(1).to(device)
            print(f"[health] latent-collapse probe: {probe.shape[0]} revealed frames from {actions_path.name}")
    if probe is None:
        ne = min(n_probe, len(val_idx))
        probe = torch.stack([probe_frame(int(val_idx[i]), 0) for i in range(ne)]).unsqueeze(1).to(device)
        print(f"[health] latent-collapse probe: {probe.shape[0]} random frames (no actions file)")

    def latent_health():
        model.eval()
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                z = model.encoder(probe).float()
                pred = model(probe).float()
        p = z.shape[0]
        zf = z.reshape(p, -1)
        zf = zf / (zf.norm(dim=1, keepdim=True) + 1e-6)
        cos = (zf @ zf.T)[~torch.eye(p, dtype=torch.bool, device=zf.device)].mean().item()
        pstd = pred.reshape(p, -1).std(1).mean().item()
        return cos, pstd

    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        cfg = _config_from_checkpoint(payload["config"])
        model = AutoEncoder(cfg).to(device)
        # Tolerant load so architecture changes warm-start instead of crashing; report mismatches.
        result = model.load_state_dict(payload["model_state_dict"], strict=False)
        if result.missing_keys:
            print(f"[resume] randomly-initialized: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"[resume] ignored: {result.unexpected_keys}")
        print(f"Loaded weights from {args.resume}")
    else:
        model = AutoEncoder(cfg).to(device)

    wlog.init(args, cfg, project="transformer-C-tokenizer")

    # betas[1]=--adam-beta2 (D-043): 0.95 keeps the 2nd moment from going stale-tiny at the minimum.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, args.adam_beta2), eps=args.adam_eps)
    print(f"AdamW: lr={args.lr} betas=(0.9, {args.adam_beta2}) eps={args.adam_eps}"
          + (f" | grad-spike skip at {args.grad_spike_mult}x EMA" if args.grad_spike_mult > 0 else ""))

    # Per-step warmup -> flat -> late-cosine cooldown. The latent-collapse escape is a saddle plateau
    # that ignites after ~2k steps and needs SUSTAINED lr; a plain cosine decays through the escape
    # window and freezes the tokenizer in the collapsed basin. Cool down only in the final 25%.
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

    # LPIPS imported/built only when --lpips is set; otherwise None and never touched.
    lpips_loss_fn = None
    if args.lpips:
        import lpips
        lpips_loss_fn = lpips.LPIPS(net=args.lpips_net).to(device)
        lpips_loss_fn.eval()
        for p in lpips_loss_fn.parameters():
            p.requires_grad_(False)
        print(f"LPIPS enabled (net={args.lpips_net}, weight={args.lpips_weight})")
    # Running magnitudes (detached) put LPIPS on the MSE scale so --lpips-weight is a scale-free ratio.
    LPIPS_EMA_DECAY = 0.99
    ema_mse = ema_lpips = None

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    last_ckpt = args.checkpoint.with_name(args.checkpoint.stem + "_last" + args.checkpoint.suffix)

    # Stability + safety-net state (D-043): the canonical --checkpoint tracks the BEST val/fg_mse, so
    # a loss explosion can't discard the good model (the EXP-024 mistake). _last.pt holds the latest.
    global_step = 0
    gn_ema = None              # running EMA of the pre-clip grad norm (spike-skip reference)
    best_fg = float("inf")

    epoch_bar = tqdm(range(args.epochs), desc="Epochs", position=0, mininterval=1.0)
    for epoch in epoch_bar:
        train_off = random.randint(0, chunk_len)
        train_ds.set_start_offset(train_off)
        if len(train_ds) == 0:
            train_ds.set_start_offset(0)
            train_off = 0
        val_ds.set_start_offset(args.val_offset % (chunk_len + 1))

        model.train()
        train_loss = 0.0
        epoch_skipped = 0
        for batch_x, _ in tqdm(train_loader, desc=f"Train {epoch + 1}/{args.epochs}",
                               leave=False, position=1, mininterval=1.0):
            B, T, H, W, C = batch_x.shape
            batch_x = batch_x.to(device)
            # Foreground mask from the fp32 input, outside autocast (D-042). None when disabled.
            fg = foreground_mask(batch_x, args.fg_thresh) if args.fg_weight > 0 else None
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                pred = model(batch_x)
                mse_plain = loss_fn(pred, batch_x)
                mse_raw = mse_plain.detach().item()
                mse = weighted_mse(pred, batch_x, 1.0 + args.fg_weight * fg) if fg is not None else mse_plain
                loss = mse
                if lpips_loss_fn is not None:
                    lpips_val = lpips_loss_fn(
                        pred.reshape(B * T, H, W, 3).permute(0, 3, 1, 2),
                        batch_x.reshape(B * T, H, W, 3).permute(0, 3, 1, 2),
                        normalize=True,
                    ).mean()
                    lpips_raw = lpips_val.detach().item()
                    if ema_mse is None:
                        ema_mse, ema_lpips = mse_raw, lpips_raw
                    else:
                        ema_mse = LPIPS_EMA_DECAY * ema_mse + (1.0 - LPIPS_EMA_DECAY) * mse_raw
                        ema_lpips = LPIPS_EMA_DECAY * ema_lpips + (1.0 - LPIPS_EMA_DECAY) * lpips_raw
                    scale = ema_mse / (ema_lpips + 1e-8)  # plain float, no grad
                    loss = mse + args.lpips_weight * scale * lpips_val
            opt.zero_grad()
            loss.backward()
            # Clip + spike guard (D-043): clip_grad_norm_ returns the PRE-clip norm. Skip the update
            # on a non-finite norm or (when --grad-spike-mult>0) a norm far above its EMA past warmup —
            # the spike-SHAPED explosion; the Adam-2nd-moment CAUSE is handled by --adam-beta2.
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0))
            spike = (not np.isfinite(gn)) or (
                args.grad_spike_mult > 0 and gn_ema is not None
                and global_step > warmup_steps and gn > args.grad_spike_mult * gn_ema
            )
            if spike:
                opt.zero_grad(set_to_none=True)
                epoch_skipped += 1
            else:
                opt.step()
                gn_ema = gn if gn_ema is None else 0.98 * gn_ema + 0.02 * gn
            scheduler.step()  # still advance the schedule on skipped steps
            global_step += 1
            train_loss += mse_raw
            # Per-step logging so an intra-epoch loss explosion is visible (D-043).
            if global_step % args.log_every == 0:
                wlog.log({
                    "train/step_mse": mse_raw,
                    "train/grad_norm": gn,
                    "train/grad_norm_ema": gn_ema if gn_ema is not None else 0.0,
                    "lr": opt.param_groups[0]["lr"],
                    "global_step": global_step,
                })

        model.eval()
        val_loss = 0.0
        val_fg_mse = val_bg_mse = val_fg_frac = 0.0
        with torch.no_grad():
            for batch_x, _ in tqdm(val_loader, desc="Val", leave=False, position=1, mininterval=1.0):
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
        nb = len(val_loader)
        val_fg_mse /= nb   # recon MSE on ball/moving pixels — THE validity guard (D-042)
        val_bg_mse /= nb
        val_fg_frac /= nb
        lat_cos, pred_std = latent_health()
        current_lr = opt.param_groups[0]["lr"]
        epoch_bar.set_postfix(train=f"{train_mse:.6f}", val=f"{val_mse:.6f}",
                              lat_cos=f"{lat_cos:.3f}", pstd=f"{pred_std:.4f}", lr=f"{current_lr:.2e}")
        print(
            f"Epoch {epoch + 1} | train MSE: {train_mse:.6f} | val MSE: {val_mse:.6f} "
            f"| val fg_mse: {val_fg_mse:.6f} bg_mse: {val_bg_mse:.6f} (fg_frac {val_fg_frac:.3f}) "
            f"| latent_cos: {lat_cos:.3f} (<0.7=escaped) | pred_std: {pred_std:.4f} (>0.04=content) "
            f"| skipped {epoch_skipped}/{len(train_loader)} | lr: {current_lr:.2e}"
        )
        wlog.log({
            "train/mse": train_mse,
            "val/mse": val_mse,
            "val/fg_mse": val_fg_mse,
            "val/bg_mse": val_bg_mse,
            "val/fg_frac": val_fg_frac,
            "latent_cos": lat_cos,
            "pred_std": pred_std,
            "lr": current_lr,
            "train/skipped_epoch": epoch_skipped,
            "global_step": global_step,
            "epoch": epoch,
        })

        # Safety net (D-043): --checkpoint = BEST val/fg_mse so an explosion can't discard the good
        # model; _last.pt always holds the latest epoch for resume/inspection.
        payload = {"model_state_dict": model.state_dict(), "config": asdict(cfg)}
        torch.save(payload, last_ckpt)
        if val_fg_mse < best_fg:
            best_fg = val_fg_mse
            torch.save(payload, args.checkpoint)
            tqdm.write(f"new best val/fg_mse {best_fg:.6f} -> saved {args.checkpoint}")
        else:
            tqdm.write(f"val/fg_mse {val_fg_mse:.6f} >= best {best_fg:.6f}; kept best (last -> {last_ckpt.name})")

    wlog.finish()


if __name__ == "__main__":
    main()
