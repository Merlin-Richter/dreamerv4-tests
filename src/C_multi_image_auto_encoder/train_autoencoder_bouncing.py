"""
Train the temporal VideoAutoEncoder on bouncing.npy (same layout as data_generators/load_data.py).

Episodes are sliced into fixed-length clips of ``max_temporal_length`` frames (e.g. 32): ``[o:o+L]``,
``[o+L:o+2L]``, … along time. Each training epoch picks a new random start offset ``o`` in
``[0, L]`` so boundaries shift across epochs. Validation uses a fixed offset (default 0).

Run from this folder:
    python train_autoencoder_bouncing.py

Or from repo root:
    python src/C_multi_image_auto_encoder/train_autoencoder_bouncing.py

Visualize a saved checkpoint (OpenCV window; needs a display):
    python src/C_multi_image_auto_encoder/train_autoencoder_bouncing.py --test-checkpoint --checkpoint src/C_multi_image_auto_encoder/autoencoder_bouncing.pt
"""

import argparse
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
import lpips

# Running from repo root: put `src` on path so imports match `train.py`
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from video_auto_encoder import AutoEncoder, AutoEncoderConfig


class ChunkClipDataset(Dataset):
    """Fixed-length clips along time: ``[start + j*L : start + (j+1)*L]`` per episode."""

    def __init__(
        self,
        episodes: torch.Tensor,
        chunk_len: int,
        start_offset: int = 0,
    ) -> None:
        self.episodes = episodes
        self.chunk_len = int(chunk_len)
        self.start_offset = int(start_offset)
        self._rebuild_index()

    def set_start_offset(self, start_offset: int) -> None:
        self.start_offset = int(start_offset)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        o = self.start_offset
        L = self.chunk_len
        n_eps, T = self.episodes.shape[0], self.episodes.shape[1]
        pairs: list[tuple[int, int]] = []
        for ep in range(n_eps):
            n_chunks = (T - o) // L
            for j in range(n_chunks):
                pairs.append((ep, o + j * L))
        self._pairs = pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ep, start = self._pairs[idx]
        clip = self.episodes[ep, start : start + self.chunk_len].clone()
        return clip, clip


def _config_from_checkpoint(cfg_dict: dict) -> AutoEncoderConfig:
    """Build AutoEncoderConfig from a saved dict; ignore unknown keys for forward compat."""
    allowed = {f.name for f in fields(AutoEncoderConfig)}
    return AutoEncoderConfig(**{k: v for k, v in cfg_dict.items() if k in allowed})


def run_test_checkpoint(args: argparse.Namespace) -> None:
    import cv2

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    raw = np.load(args.frames)
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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_SRC / "autoencoder_bouncing.pt",
        help="Where to save weights + config (also loaded by --test-checkpoint).",
    )
    parser.add_argument(
        "--test-checkpoint",
        action="store_true",
        help="Load --checkpoint and open a window: random L-frame clip vs reconstruction; SPACE = new sample; q/Esc = quit. L = max_temporal_length from checkpoint.",
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
        "--fresh",
        action="store_true",
        help="Ignore any existing --checkpoint and train from random init (useful after architecture changes).",
    )
    args = parser.parse_args()

    if args.test_checkpoint:
        run_test_checkpoint(args)
        return

    raw = np.load(args.frames)
    if raw.ndim != 5 or raw.shape[-1] != 3:
        raise ValueError(f"Expected (N, T, H, W, 3), got {raw.shape}")

    n, t, h, w, c = raw.shape
    if n < 2:
        raise ValueError(
            f"Temporal training splits by episode; need at least 2 episodes, got n={n}."
        )
    x = torch.from_numpy(raw.astype(np.float32) / 255.0)

    n_val = max(1, int(round(n * args.val_fraction)))
    n_val = min(n_val, n - 1)
    n_train = n - n_val
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_eps = x[train_idx]
    val_eps = x[val_idx]

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
        img_input_H=h, img_input_W=w, max_temporal_length=chunk_len, mae_max_mask=0
    )

    train_ds = ChunkClipDataset(train_eps, chunk_len, start_offset=0)
    val_ds = ChunkClipDataset(val_eps, chunk_len, start_offset=args.val_offset % (chunk_len + 1))
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            f"No clips with L={chunk_len}, val_offset={args.val_offset}. "
            "Try a smaller --val-offset or longer episodes."
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
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
    # lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
    use_amp = device == "cuda"

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

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
        for batch_x, _ in tqdm(
            train_loader,
            desc=f"Train {epoch + 1}/{args.epochs}",
            leave=False,
            position=1,
            mininterval=1.0,
        ):
            B, T, H, W, C = batch_x.shape
            batch_x = batch_x.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                pred = model(batch_x)
                # lpips_val = lpips_loss_fn(
                #     pred.reshape(B * T, H, W, 3).permute(0, 3, 1, 2),
                #     batch_x.reshape(B * T, H, W, 3).permute(0, 3, 1, 2),
                #     normalize=True,
                # ).mean()
                loss = loss_fn(pred, batch_x) # + 0.2 * lpips_val
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, _ in tqdm(
                val_loader,
                desc="Val",
                leave=False,
                position=1,
                mininterval=1.0,
            ):
                batch_x = batch_x.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                    val_loss += loss_fn(model(batch_x), batch_x).item()

        train_mse = train_loss / len(train_loader)
        val_mse = val_loss / len(val_loader)
        current_lr = opt.param_groups[0]["lr"]
        epoch_bar.set_postfix(
            train=f"{train_mse:.6f}",
            val=f"{val_mse:.6f}",
            tr_off=train_off,
            lr=f"{current_lr:.2e}",
        )
        print(
            f"Epoch {epoch + 1} | train MSE: {train_mse:.6f} | val MSE: {val_mse:.6f} "
            f"| train_clip_offset={train_off} | lr: {current_lr:.2e}"
        )

        torch.save(
            {"model_state_dict": model.state_dict(), "config": asdict(cfg)},
            args.checkpoint,
        )
        tqdm.write(f"Saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
