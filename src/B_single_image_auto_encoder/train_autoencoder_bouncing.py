"""
Train VideoAutoEncoder on bouncing.npy (same layout as data_generators/load_data.py).

Run from this folder:
    python train_autoencoder_bouncing.py

Or from repo root:
    python src/train_autoencoder_bouncing.py

Visualize a saved checkpoint (OpenCV window; needs a display):
    python src/train_autoencoder_bouncing.py --test-checkpoint --checkpoint src/autoencoder_bouncing.pt
"""

import argparse
import sys
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

# Running from repo root: put `src` on path so imports match `train.py`
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from video_auto_encoder import AutoEncoder, AutoEncoderConfig


def _config_from_checkpoint(cfg_dict: dict) -> AutoEncoderConfig:
    """Build AutoEncoderConfig from a saved dict; ignore unknown keys for forward compat."""
    allowed = {f.name for f in fields(AutoEncoderConfig)}
    return AutoEncoderConfig(**{k: v for k, v in cfg_dict.items() if k in allowed})


def run_test_checkpoint(args: argparse.Namespace) -> None:
    import random

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
    print("Using Device: " + device)
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

    def pick_and_show() -> None:
        ep = random.randrange(n_eps)
        ti = random.randrange(n_frames)
        frame_rgb = raw[ep, ti]
        x = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0).unsqueeze(0).to(device)

        with torch.no_grad():
            z = model.encoder(x)
            recon = model.decoder(z)

        z_np = z.squeeze(0).cpu().numpy()
        print(f"\n--- random sample  episode={ep}  frame={ti}  latent shape={z_np.shape} ---")
        for i, token in enumerate(z_np):
            print(f"bottleneck token {i}: {token}")

        inp_bgr = uint8_rgb_to_bgr(frame_rgb)
        rec_bgr = tensor01_to_bgr(recon[0])
        print(rec_bgr.shape)
        
        print(rec_bgr[:, :, 0].mean())
        print(rec_bgr[:, :, 1].mean())
        print(rec_bgr[:, :, 2].mean())
        pair = np.hstack([inp_bgr, rec_bgr])
        disp_h, disp_w = pair.shape[0] * scale, pair.shape[1] * scale
        display = cv2.resize(pair, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
        cv2.putText(
            display,
            f"ep {ep}  t {ti}   |   SPACE new   q/Esc quit",
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
        help="Load --checkpoint and open a window: input vs reconstruction; SPACE = new random frame; q/Esc = quit. Prints bottleneck latents to the console.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()

    if args.test_checkpoint:
        run_test_checkpoint(args)
        return

    raw = np.load(args.frames)
    if raw.ndim != 5 or raw.shape[-1] != 3:
        raise ValueError(f"Expected (N, T, H, W, 3), got {raw.shape}")

    n, t, h, w, c = raw.shape
    x = torch.from_numpy(raw.astype(np.float32) / 255.0).reshape(-1, h, w, c)

    n_total = x.shape[0]
    n_val = max(1, int(n_total * args.val_fraction))
    n_train = n_total - n_val
    full_ds = TensorDataset(x, x)
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using Device: " + device)

    if args.checkpoint.is_file():
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cfg = _config_from_checkpoint(payload["config"])
        model = AutoEncoder(cfg).to(device)
        model.load_state_dict(payload["model_state_dict"])
        print(f"Loaded weights from {args.checkpoint}")
    else:
        cfg = AutoEncoderConfig(img_input_H=h, img_input_W=w)
        model = AutoEncoder(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    loss_fn = nn.MSELoss()

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    epoch_bar = tqdm(range(args.epochs), desc="Epochs", position=0, mininterval=1.0)
    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        for batch_x, _ in tqdm(
            train_loader,
            desc=f"Train {epoch + 1}/{args.epochs}",
            leave=False,
            position=1,
            mininterval=1.0,
        ):
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_x)
            opt.zero_grad()
            loss.backward()
            opt.step()
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
                val_loss += loss_fn(model(batch_x), batch_x).item()

        train_mse = train_loss / len(train_loader)
        val_mse = val_loss / len(val_loader)
        current_lr = opt.param_groups[0]["lr"]
        epoch_bar.set_postfix(
            train=f"{train_mse:.6f}", val=f"{val_mse:.6f}", lr=f"{current_lr:.2e}"
        )
        print(
            f"Epoch {epoch + 1} | train MSE: {train_mse:.6f} | val MSE: {val_mse:.6f} | lr: {current_lr:.2e}"
        )

        scheduler.step()

        torch.save(
            {"model_state_dict": model.state_dict(), "config": asdict(cfg)},
            args.checkpoint,
        )
        tqdm.write(f"Saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
