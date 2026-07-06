"""ColorField tokenizer training — one-time prep of the FROZEN LAYER.

Lean adaptation of src/training/train_tokenizer.py (D-043 stability stack kept:
AdamW beta2=0.95, grad-clip 1.0 + spike-skip vs EMA, warmup->flat->late-cosine LR,
best-by-val canonical checkpoint + _last safety net, latent-collapse health probe)
for the procedural ColorField dataset: frames are rendered on the fly from
(map, start, actions) sidecars — no frames file exists.

Dropped vs src: fg-weighting (no moving foreground here), LPIPS (flat colors, MSE
is the right loss), W&B (self-contained; prints instead).

Acceptance gate (run after training, also standalone via --verify-only):
reconstructions must be READOUT-EXACT on held-out val frames — every visible cell
in the decoded frame reads back the true color through readout.read_cells. This is
what "the tokenizer is good enough for the comeback eval" MEANS here.

Usage (repo root):
  venv/Scripts/python.exe -u -m autoresearch.frozen.train_tokenizer \
    --train data/colorfield --val data/colorfield_val \
    --checkpoint checkpoints/colorfield/tokenizer.pt --epochs 20
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from .datagen import ColorFieldDataset
from .env import build_world, positions_from, render
from .readout import read_cells
from .tokenizer_model import AutoEncoder, AutoEncoderConfig


class ProceduralClipDataset(Dataset):
    """Random L-frame clips rendered on the fly. Each epoch resamples which
    (episode, start) clips are visited (fresh subset + fresh offsets), so the
    model sees new worlds/windows every epoch without materializing frames."""

    def __init__(self, ds: ColorFieldDataset, L: int, clips_per_epoch: int, seed: int):
        self.ds = ds
        self.L = L
        self.n_clips = clips_per_epoch
        self.rng = np.random.default_rng(seed)
        # positions precomputed once: (N, T, 2) int16
        self.positions = np.stack([
            positions_from(tuple(ds.starts[i]), ds.actions[i]) for i in range(len(ds))
        ]).astype(np.int16)
        self.T = ds.actions.shape[1]
        self.resample()

    def resample(self):
        self.eps = self.rng.integers(0, len(self.ds), size=self.n_clips)
        self.starts = self.rng.integers(0, self.T - self.L + 1, size=self.n_clips)

    def __len__(self):
        return self.n_clips

    def __getitem__(self, idx):
        ep, s = int(self.eps[idx]), int(self.starts[idx])
        world = build_world(self.ds.maps[ep])          # ~microseconds
        clip = np.stack([render(world, tuple(self.positions[ep, s + j]))
                         for j in range(self.L)])
        x = torch.from_numpy(clip.astype(np.float32) / 255.0)
        return x, x


def verify_readout_exact(model, val: ColorFieldDataset, device, n_frames=512, seed=0,
                         L=16):
    """Fraction of (frame, visible cell) reads where decode(encode(frame)) reads
    back the true color. Returns (cell_acc, frame_exact_frac)."""
    rng = np.random.default_rng(seed)
    model.eval()
    n_cells = n_cells_ok = n_frames_exact = 0
    n_done = 0
    while n_done < n_frames:
        ep = int(rng.integers(0, len(val)))
        s = int(rng.integers(0, val.actions.shape[1] - L + 1))
        world = build_world(val.maps[ep])
        pos_all = positions_from(tuple(val.starts[ep]), val.actions[ep])
        clip = np.stack([render(world, tuple(pos_all[s + j])) for j in range(L)])
        x = torch.from_numpy(clip.astype(np.float32) / 255.0).unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16,
                                             enabled=device == "cuda"):
            rec = model(x)
        rec_u8 = (rec.float().clamp(0, 1) * 255.0).round().to(torch.uint8)[0].cpu().numpy()
        for j in range(L):
            pos = tuple(int(v) for v in pos_all[s + j])
            true_reads = read_cells(clip[j], pos)
            rec_reads = read_cells(rec_u8[j], pos)
            ok = sum(rec_reads[k].color == r.color for k, r in true_reads.items())
            n_cells += len(true_reads)
            n_cells_ok += ok
            n_frames_exact += ok == len(true_reads)
            n_done += 1
            if n_done >= n_frames:
                break
    return n_cells_ok / n_cells, n_frames_exact / n_done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="data/colorfield")
    ap.add_argument("--val", default="data/colorfield_val")
    ap.add_argument("--checkpoint", default="checkpoints/colorfield/tokenizer.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--clips-per-epoch", type=int, default=25000)
    ap.add_argument("--val-clips", type=int, default=1500)
    ap.add_argument("--context-length", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--adam-beta2", type=float, default=0.95)
    ap.add_argument("--grad-spike-mult", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify-only", action="store_true",
                    help="Load --checkpoint and run the readout-exactness gate only.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    val_ds_raw = ColorFieldDataset(args.val)

    if args.verify_only:
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cfg = AutoEncoderConfig(**{k: v for k, v in payload["config"].items()
                                   if k in AutoEncoderConfig.__dataclass_fields__ and k != "dtype"})
        model = AutoEncoder(cfg).to(device)
        model.load_state_dict(payload["model_state_dict"])
        acc, fexact = verify_readout_exact(model, val_ds_raw, device)
        print(f"[verify] cell readout acc {acc:.6f} | frame-exact {fexact:.4f}")
        return

    train_ds_raw = ColorFieldDataset(args.train)
    L = args.context_length
    cfg = AutoEncoderConfig(img_input_H=64, img_input_W=64, max_temporal_length=L)
    print(f"AutoEncoderConfig: emb={cfg.embedding_dim} depth={cfg.depth} heads={cfg.n_heads} "
          f"n_latents={cfg.n_latents} bottleneck={cfg.bottleneck_dim} L={L}")
    model = AutoEncoder(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.1f}M | device {device}")

    train_ds = ProceduralClipDataset(train_ds_raw, L, args.clips_per_epoch, seed=args.seed + 1)
    val_ds = ProceduralClipDataset(val_ds_raw, L, args.val_clips, seed=12345)
    nw = 0 if os.name == "nt" else 4
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=nw, pin_memory=use_amp)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=nw, pin_memory=use_amp)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, args.adam_beta2))
    total_steps = max(1, (args.clips_per_epoch // args.batch_size) * args.epochs)
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

    # latent-collapse health probe: 64 fixed val frames
    prng = np.random.default_rng(7)
    probe_frames = []
    for _ in range(64):
        ep = int(prng.integers(0, len(val_ds_raw)))
        t = int(prng.integers(0, val_ds_raw.actions.shape[1]))
        world = build_world(val_ds_raw.maps[ep])
        pos = positions_from(tuple(val_ds_raw.starts[ep]), val_ds_raw.actions[ep])[t]
        probe_frames.append(render(world, tuple(pos)))
    probe = (torch.from_numpy(np.stack(probe_frames).astype(np.float32) / 255.0)
             .unsqueeze(1).to(device))

    def latent_health():
        model.eval()
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16,
                                             enabled=use_amp):
            z = model.encoder(probe).float()
            pred = model(probe).float()
        p = z.shape[0]
        zf = z.reshape(p, -1)
        zf = zf / (zf.norm(dim=1, keepdim=True) + 1e-6)
        cos = (zf @ zf.T)[~torch.eye(p, dtype=torch.bool, device=zf.device)].mean().item()
        return cos, pred.reshape(p, -1).std(1).mean().item()

    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    last_ckpt = args.checkpoint.replace(".pt", "_last.pt")
    best_val = float("inf")
    gn_ema = None
    global_step = 0

    for epoch in range(args.epochs):
        train_ds.resample()
        model.train()
        tl, nb, skipped = 0.0, 0, 0
        t0 = time.time()
        for x, _ in train_loader:
            x = x.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                loss = loss_fn(model(x), x)
            opt.zero_grad()
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0))
            spike = (not np.isfinite(gn)) or (
                args.grad_spike_mult > 0 and gn_ema is not None
                and global_step > warmup_steps and gn > args.grad_spike_mult * gn_ema)
            if spike:
                opt.zero_grad(set_to_none=True)
                skipped += 1
            else:
                opt.step()
                gn_ema = gn if gn_ema is None else 0.98 * gn_ema + 0.02 * gn
            scheduler.step()
            global_step += 1
            tl += loss.item()
            nb += 1

        model.eval()
        vl, vb = 0.0, 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                    vl += loss_fn(model(x), x).item()
                vb += 1
        vl /= max(1, vb)
        cos, pstd = latent_health()
        dt = time.time() - t0
        print(f"epoch {epoch + 1}/{args.epochs} | train {tl / max(1, nb):.6f} | val {vl:.6f} "
              f"| latent_cos {cos:.3f} pred_std {pstd:.3f} | skipped {skipped} "
              f"| lr {opt.param_groups[0]['lr']:.2e} | {dt:.0f}s", flush=True)

        payload = {"model_state_dict": model.state_dict(),
                   "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__ if k != "dtype"},
                   "epoch": epoch, "val_mse": vl}
        torch.save(payload, last_ckpt)
        if vl < best_val:
            best_val = vl
            torch.save(payload, args.checkpoint)

    # acceptance gate
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    acc, fexact = verify_readout_exact(model, val_ds_raw, device)
    print(f"[verify] BEST ckpt (val {payload['val_mse']:.6f}): cell readout acc {acc:.6f} "
          f"| frame-exact {fexact:.4f}", flush=True)


if __name__ == "__main__":
    main()
