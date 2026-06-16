"""
Minimal web UI to explore tokenizer bottleneck latents.

Sliders (-1 … 1) feed the frozen C autoencoder decoder; the image updates live.

Run from repo root:
    python src/interactive/latent_explorer/run.py

Then open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from dataclasses import dataclass, fields
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parents[1]                       # .../src (the `models` package)
_REPO = _HERE.parents[2]                      # repo root
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.tokenizer import AutoEncoder, AutoEncoderConfig  # noqa: E402

HTML_PATH = _HERE / "index.html"
DEFAULT_TOKENIZER = _REPO / "trained_autoencoder.pt"
DEFAULT_FRAMES = _REPO / "bouncing.npy"


def _config_from_checkpoint(cfg_dict: dict, cls):
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in cfg_dict.items() if k in allowed})


def load_tokenizer(checkpoint: Path, device: str) -> AutoEncoder:
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Tokenizer not found: {checkpoint}\n"
            "Train C first or pass --tokenizer PATH"
        )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = _config_from_checkpoint(payload["config"], AutoEncoderConfig)
    model = AutoEncoder(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@dataclass
class AppState:
    tokenizer: AutoEncoder
    device: str
    frames_path: Path
    n_latents: int
    bottleneck_dim: int
    img_h: int
    img_w: int


def make_state(tokenizer: AutoEncoder, device: str, frames_path: Path) -> AppState:
    enc = tokenizer.encoder
    dec = tokenizer.decoder
    return AppState(
        tokenizer=tokenizer,
        device=device,
        frames_path=frames_path,
        n_latents=enc.n_latents,
        bottleneck_dim=enc.bottleneck_proj.out_features,
        img_h=dec.h_patches * dec.patch_size,
        img_w=dec.w_patches * dec.patch_size,
    )


@torch.no_grad()
def decode_latents(state: AppState, latents: list) -> bytes:
    z = torch.tensor(latents, dtype=torch.float32, device=state.device)
    if z.shape != (state.n_latents, state.bottleneck_dim):
        raise ValueError(
            f"Expected latents ({state.n_latents}, {state.bottleneck_dim}), got {tuple(z.shape)}"
        )
    z = z.clamp(-1.0, 1.0).unsqueeze(0).unsqueeze(0)  # (1, 1, L, D)
    rgb = state.tokenizer.decoder(z)[0, 0].detach().cpu().float().clamp(0.0, 1.0).numpy()
    u8 = (rgb * 255.0).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(u8).save(buf, format="PNG")
    return buf.getvalue()


@torch.no_grad()
def encode_random_frame(state: AppState) -> list:
    if not state.frames_path.is_file():
        raise FileNotFoundError(
            f"No frames file at {state.frames_path} (pass --frames or generate bouncing.npy)"
        )
    raw = np.load(state.frames_path, mmap_mode="r")
    if raw.ndim != 5:
        raise ValueError(f"Expected (N,T,H,W,3), got {raw.shape}")
    ep = random.randrange(raw.shape[0])
    t = random.randrange(raw.shape[1])
    frame = raw[ep, t].astype(np.float32) / 255.0
    x = torch.from_numpy(frame).to(state.device).unsqueeze(0).unsqueeze(0)
    z = state.tokenizer.encoder(x)[0, 0].cpu().tolist()
    return z


def make_handler(state: AppState):
    html = HTML_PATH.read_text(encoding="utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if args and str(args[0]).startswith("GET /decode"):
                return
            super().log_message(fmt, *args)

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n)) if n else {}

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
                return
            if self.path == "/api/info":
                payload = json.dumps({
                    "n_latents": state.n_latents,
                    "bottleneck_dim": state.bottleneck_dim,
                    "img_h": state.img_h,
                    "img_w": state.img_w,
                }).encode("utf-8")
                self._send(200, payload, "application/json")
                return
            self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            try:
                if self.path == "/decode":
                    data = self._read_json()
                    png = decode_latents(state, data["latents"])
                    self._send(200, png, "image/png")
                    return
                if self.path == "/encode_sample":
                    latents = encode_random_frame(state)
                    self._send(200, json.dumps(latents).encode("utf-8"), "application/json")
                    return
                self._send(404, b"not found", "text/plain")
            except Exception as e:
                self._send(400, str(e).encode("utf-8"), "text/plain")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore bottleneck latents in the browser.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES,
                        help="Optional .npy for 'Load random frame' (default: repo/bouncing.npy).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading tokenizer from {args.tokenizer} ({device})…", flush=True)
    tokenizer = load_tokenizer(args.tokenizer, device)
    state = make_state(tokenizer, device, args.frames)

    server = HTTPServer((args.host, args.port), make_handler(state))
    url = f"http://{args.host}:{args.port}"
    print(f"Open {url}", flush=True)
    print("Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
