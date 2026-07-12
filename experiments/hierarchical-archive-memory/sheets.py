"""Memory Maze rollout sheets for an archive checkpoint, with same-checkpoint gate ablation."""
from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evals.gridworld.sheets import _load, save_sheet  # noqa: E402
from evals.memmaze.sheets import rollout_sheet, val_episodes  # noqa: E402
from models.tokenizer import AutoEncoder, AutoEncoderConfig  # noqa: E402
from model import ArchiveDynamicsConfig, DynamicsModelArchive  # noqa: E402


def load_archive(path: Path, device: str):
    payload = torch.load(path, map_location=device, weights_only=False)
    allowed = {f.name for f in fields(ArchiveDynamicsConfig)}
    cfg = ArchiveDynamicsConfig(**{k: v for k, v in payload["config"].items() if k in allowed})
    model = DynamicsModelArchive(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--actions", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=_ROOT / "outputs" / "sheets")
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--n-pre", type=int, default=64)
    p.add_argument("--n-ctx", type=int, default=8)
    p.add_argument("--n-gen", type=int, default=56)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, nargs="*", default=None)
    p.add_argument("--zero-archive", action="store_true",
                   help="Same checkpoint, but force every archive reader gate to zero.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = _load(args.tokenizer, AutoEncoder, AutoEncoderConfig, device)
    tokenizer.requires_grad_(False)
    model = load_archive(args.checkpoint, device)
    if args.zero_archive:
        for gate in model.archive_gates.values():
            gate.data.zero_()

    frames = np.load(args.frames, mmap_mode="r")
    action_path = args.actions or args.frames.with_name(args.frames.stem + "_actions.npy")
    actions = np.load(action_path, mmap_mode="r") if Path(action_path).is_file() else None
    if actions is None and model.n_actions > 0:
        raise SystemExit(f"action-conditioned model but no actions at {action_path}")
    episodes = args.episodes or val_episodes(len(frames))[:args.n_samples]
    sheet = rollout_sheet(
        model, tokenizer, frames, actions, episodes=episodes, n_samples=args.n_samples,
        n_pre=args.n_pre, n_ctx=args.n_ctx, n_gen=args.n_gen, K=args.K,
        device=device, scale=args.scale, seed=args.seed, window=None)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_archive_zero" if args.zero_archive else "_archive_on"
    save_sheet(args.out_dir / f"sheet_memmaze{suffix}.png", sheet)


if __name__ == "__main__":
    main()
