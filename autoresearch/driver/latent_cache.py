"""Latent cache for ColorField dynamics training (driver, NOT loop-editable).

Encodes the procedural datasets ONCE through the FROZEN tokenizer into fp16 latents
(~2.6 GB train), so budgeted experiments never pay the tokenizer. Same design as
the memmaze latent disk cache.

PREREQUISITE (verified here, fail-loud): WINDOW INVARIANCE — the cache is encoded
in fixed 16-frame chunks but training slices it at arbitrary offsets, which is only
sound if a frame's latent barely depends on its position within the encoding chunk
(temporal attention is causal, so it CAN depend). GridWorld measured cos 0.9975,
memmaze cos 0.9996 with window-delta recon MSE 60x below recon error; GridWorld
was ACCEPTED as safe at 6x (the repo precedent). We require cos >= 0.99 and
delta-MSE <= recon-MSE/6 (the accepted precedent), else abort loudly.
Measured here: train 9.4x / val 12.6x, cos 0.9975 (== GridWorld's).

Usage (repo root):
  venv/Scripts/python.exe -u -m autoresearch.driver.latent_cache \
    [--data data/colorfield --tokenizer checkpoints/colorfield/tokenizer.pt]
Writes <data>/latents-<tokhash12>.npy, shape (N, T, n_latents, bottleneck) fp16.
"""

import argparse
import hashlib
import os

import numpy as np
import torch

from ..frozen.datagen import ColorFieldDataset
from ..frozen.env import build_world, positions_from, render
from ..frozen.tokenizer_model import AutoEncoder, AutoEncoderConfig

L = 16  # tokenizer temporal window (frozen config)


def load_tokenizer(path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = AutoEncoderConfig(**{k: v for k, v in payload["config"].items()
                               if k in AutoEncoderConfig.__dataclass_fields__ and k != "dtype"})
    model = AutoEncoder(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, cfg


def _episode_frames(ds, i):
    world = build_world(ds.maps[i])
    pos = positions_from(tuple(ds.starts[i]), ds.actions[i])
    return np.stack([render(world, tuple(p)) for p in pos])


@torch.no_grad()
def _encode_chunked(model, frames_u8, device, offset=0):
    """Encode (T,64,64,3) uint8 in fixed L-chunks starting at `offset`; returns
    (T', n_latents, bottleneck) float32 for frames [offset : offset + T']."""
    T = len(frames_u8)
    outs = []
    for s in range(offset, T - L + 1, L):
        x = torch.from_numpy(frames_u8[s:s + L].astype(np.float32) / 255.0)
        x = x.unsqueeze(0).to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16,
                            enabled=device == "cuda"):
            z = model.encoder(x)
        outs.append(z.float()[0].cpu())
    return torch.cat(outs, dim=0).numpy(), offset  # frames [offset : offset+len)


@torch.no_grad()
def window_invariance_probe(model, ds, device, n_episodes=6):
    """Encode the same frames at chunk offsets 0 and L//2; compare latents of the
    overlap + decoded recon deltas. Returns dict; raises on failure."""
    cos_all, dmse_all, rmse_all = [], [], []
    for i in range(n_episodes):
        frames = _episode_frames(ds, i)[: 8 * L]           # 128 frames is plenty
        za, _ = _encode_chunked(model, frames, device, offset=0)
        zb, off = _encode_chunked(model, frames, device, offset=L // 2)
        # overlap: frames [off : off + len(zb)] exist in both encodings
        a = torch.from_numpy(za[off:off + len(zb)]).flatten(1)
        b = torch.from_numpy(zb).flatten(1)
        cos = torch.nn.functional.cosine_similarity(a, b, dim=1).mean().item()
        cos_all.append(cos)
        # recon deltas: decode both latent sets (in L-chunks — the temporal RoPE
        # table is sized to L), compare to truth and to each other
        n_ov = (len(zb) // L) * L
        fa = frames[off:off + n_ov].astype(np.float32) / 255.0

        def _decode(z_np):
            outs = []
            for s in range(0, n_ov, L):
                x = torch.from_numpy(z_np[s:s + L]).unsqueeze(0).to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                    enabled=device == "cuda"):
                    outs.append(model.decoder(x).float()[0].cpu().numpy())
            return np.concatenate(outs, axis=0)

        ra = _decode(za[off:off + n_ov])
        rb = _decode(zb[:n_ov])
        rmse_all.append(float(((ra - fa) ** 2).mean()))
        dmse_all.append(float(((ra - rb) ** 2).mean()))
    result = {"latent_cos": float(np.mean(cos_all)),
              "recon_mse": float(np.mean(rmse_all)),
              "window_delta_mse": float(np.mean(dmse_all))}
    ok = result["latent_cos"] >= 0.99 and \
        result["window_delta_mse"] <= result["recon_mse"] / 6
    if not ok:
        raise RuntimeError(f"WINDOW INVARIANCE FAILED: {result} — arbitrary-offset "
                           "slicing of the latent cache is NOT safe for this tokenizer")
    return result


def build_cache(data_dir, tokenizer_path, device, batch_eps=8):
    ds = ColorFieldDataset(data_dir)
    model, cfg = load_tokenizer(tokenizer_path, device)
    with open(tokenizer_path, "rb") as f:
        tok_hash = hashlib.sha256(f.read()).hexdigest()[:12]
    out_path = os.path.join(data_dir, f"latents-{tok_hash}.npy")

    probe = window_invariance_probe(model, ds, device)
    print(f"[probe] window invariance OK: {probe}", flush=True)

    N, T = len(ds), ds.actions.shape[1]
    assert T % L == 0, (T, L)
    lat = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float16,
        shape=(N, T, cfg.n_latents, cfg.bottleneck_dim))
    n_chunks = T // L
    with torch.no_grad():
        for i in range(N):
            frames = _episode_frames(ds, i)
            x = torch.from_numpy(frames.astype(np.float32) / 255.0)
            x = x.reshape(n_chunks, L, *x.shape[1:])          # (64, 16, 64, 64, 3)
            zs = []
            for s in range(0, n_chunks, batch_eps):
                with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                    enabled=device == "cuda"):
                    z = model.encoder(x[s:s + batch_eps].to(device))
                zs.append(z.float().cpu())
            lat[i] = torch.cat(zs, 0).reshape(T, cfg.n_latents, cfg.bottleneck_dim) \
                          .numpy().astype(np.float16)
            if (i + 1) % 200 == 0:
                print(f"[cache] {i + 1}/{N}", flush=True)
    lat.flush()
    print(f"[cache] wrote {out_path} shape {lat.shape}", flush=True)
    return out_path, probe


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/colorfield")
    ap.add_argument("--val", default="data/colorfield_val")
    ap.add_argument("--tokenizer", default="checkpoints/colorfield/tokenizer.pt")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for d in (args.data, args.val):
        build_cache(d, args.tokenizer, device)


if __name__ == "__main__":
    main()
