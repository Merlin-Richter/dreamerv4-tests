"""Qualitative Memory-Maze rollout sheets (spec: specs/evals/memmaze/sheets.md).

The memmaze counterpart of `evals/gridworld/sheets.py`. No controlled env / curtain here (occlusion in
Memory Maze is natural — the agent looks away and later looks back), so there is ONE sheet kind:

  * rollout_sheet — held-out dataset episodes. TOP = ground truth, BOTTOM = model rollout (first
    `n_ctx` cols are context reconstructions, then free-run predictions conditioned on the TRUE action
    sequence). `n_gen` may exceed the model's window (the carrying rollout slides) — the qualitative
    long-horizon / maze-consistency check.

Episodes default to the trainer's held-out val split (reproduced here: randperm seed 0 — MUST stay in
sync with train_dynamics.py). Drawing/decode/checkpoint-loading reused from evals.gridworld.sheets
(one source of truth). cv2 only, BGR end-to-end, no channel swap. Run with -u.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/
from evals.gridworld.sheets import (  # noqa: E402  — shared drawing/decode/loading layer
    _assemble, _decode_seq, _load, _sample_block, save_sheet,
)

SCALE = 2  # default upscale per cell (memmaze blocks are wide: 32+ columns)


def val_episodes(n_episodes: int, val_fraction: float = 0.05) -> np.ndarray:
    """The trainer's held-out episode ids — reproduces train_dynamics.py's split exactly
    (torch.randperm with a fresh seed-0 generator; n_val = min(max(1, round(n*frac)), n-1))."""
    n_val = min(max(1, int(round(n_episodes * val_fraction))), n_episodes - 1)
    g = torch.Generator().manual_seed(0)
    return torch.randperm(n_episodes, generator=g)[:n_val].numpy()


@torch.no_grad()
def _action_rollout(model, tokenizer, frames_ep, actions_ep, start, n_ctx, n_gen, K, device,
                    max_ctx=None) -> list[np.ndarray]:
    """Encode `n_ctx` true frames, generate `n_gen` more conditioned on the true action sequence,
    decode the whole thing. Returns n_ctx+n_gen uint8 BGR frames."""
    wf = np.asarray(frames_ep[start:start + n_ctx]).astype(np.float32) / 255.0
    ctx = tokenizer.encoder(torch.from_numpy(wf).unsqueeze(0).to(device))
    act = None
    if actions_ep is not None:
        a = np.asarray(actions_ep[start:start + n_ctx + n_gen]).astype(np.int64)
        act = torch.from_numpy(a).unsqueeze(0).to(device)
    gen = model.generate(ctx, n_gen, K=K, action_idx=act, max_ctx=max_ctx)
    return _decode_seq(tokenizer, torch.cat((ctx, gen), dim=1))


@torch.no_grad()
def rollout_sheet(model, tokenizer, frames, actions, *, episodes=None, n_samples=4, n_ctx=8,
                  n_gen=None, K=4, device="cpu", scale=SCALE, seed=0, window=None) -> np.ndarray:
    """Held-out GT-vs-rollout sheet (uint8 BGR), one block per episode. TOP = ground truth, BOTTOM =
    context recon then action-conditioned free-run. `actions` may be None only for an unlabeled model.

    ``episodes`` overrides the default val-split selection; the start offset inside each episode is
    drawn from a `seed`-seeded generator. ``window`` (total frames) forces a shorter sliding window
    than the model trained with; None = native."""
    n_gen = (model.config.max_temporal_length - n_ctx) if n_gen is None else n_gen
    win = n_ctx + n_gen
    max_ctx = None if window is None else max(1, window - 1)
    if episodes is None:
        episodes = val_episodes(len(frames))[:n_samples]
    win_lbl = "" if window is None else f"window={window}  "

    g = torch.Generator().manual_seed(seed)
    blocks = []
    for ep in episodes:
        f = np.asarray(frames[ep])
        if f.shape[0] < win:
            print(f"!! ep{ep}: T={f.shape[0]} < {win} frames needed — skipped")
            continue
        s = int(torch.randint(0, f.shape[0] - win + 1, (1,), generator=g).item())
        a_ep = actions[ep] if actions is not None else None
        pred = _action_rollout(model, tokenizer, f, a_ep, s, n_ctx, n_gen, K, device, max_ctx=max_ctx)
        act_lbl = ""
        if a_ep is not None:
            digits = "".join(str(int(a)) for a in np.asarray(a_ep[s:s + win]))
            act_lbl = f"  acts {digits[:n_ctx]}|{digits[n_ctx:]}"
        label = (f"MEMMAZE ep{ep} t{s}  {n_ctx} ctx | {n_gen} action-conditioned rollout  {win_lbl}"
                 f"TOP=ground truth  BOTTOM=model rollout{act_lbl}")
        blocks.append(_sample_block(list(f[s:s + win]), pred, label, boundary=n_ctx, scale=scale))
        if len(blocks) >= n_samples:
            break
    return _assemble(blocks)


# --------------------------------------------------------------------------- CLI
def main() -> None:
    from models.dynamics_model import DynamicsModel, DynamicsModelConfig
    from models.tokenizer import AutoEncoder, AutoEncoderConfig

    root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description="Render qualitative Memory-Maze rollout sheets.")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Dynamics checkpoint.")
    ap.add_argument("--tokenizer", type=Path, required=True, help="Frozen tokenizer checkpoint.")
    ap.add_argument("--frames", type=Path, default=root / "data" / "memmaze9x9.npy",
                    help="Frames .npy (N,T,H,W,3) uint8.")
    ap.add_argument("--actions", type=Path, default=None,
                    help="Actions .npy (N,T). Default: '<frames>_actions.npy' if present.")
    ap.add_argument("--out-dir", type=Path, default=root / "outputs" / "sheets",
                    help="Where the PNG lands (default: outputs/sheets/, gitignored).")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--n-ctx", type=int, default=8)
    ap.add_argument("--n-gen", type=int, default=None,
                    help="Rollout length; default = model window - n_ctx. May exceed the window.")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--scale", type=int, default=SCALE)
    ap.add_argument("--seed", type=int, default=0, help="Seeds the per-episode start offsets.")
    ap.add_argument("--val-fraction", type=float, default=0.05,
                    help="Held-out split fraction (must match training); 0 = sample from ALL episodes.")
    ap.add_argument("--episodes", type=int, nargs="*", default=None,
                    help="Explicit episode ids (overrides the val-split selection).")
    ap.add_argument("--window", type=int, default=None,
                    help="Force the sliding context window (total frames); default = native.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = _load(args.tokenizer, AutoEncoder, AutoEncoderConfig, device)
    for p in tok.parameters():
        p.requires_grad_(False)
    model = _load(args.checkpoint, DynamicsModel, DynamicsModelConfig, device)

    frames = np.load(args.frames, mmap_mode="r")
    actions = None
    ap_path = args.actions or args.frames.with_name(args.frames.stem + "_actions.npy")
    if Path(ap_path).is_file():
        actions = np.load(ap_path, mmap_mode="r")
    elif model.n_actions > 0:
        raise SystemExit(f"ERROR: model is action-conditioned (n_actions={model.n_actions}) but no "
                         f"actions file at {ap_path} — an unconditioned memmaze rollout is meaningless.")
    else:
        print("!! unlabeled model and no actions file — rollout is a free run.")

    episodes = args.episodes
    if episodes is None:
        pool = (val_episodes(len(frames), args.val_fraction) if args.val_fraction > 0
                else np.arange(len(frames)))
        episodes = pool[:args.n_samples]

    sheet = rollout_sheet(model, tok, frames, actions, episodes=episodes, n_samples=args.n_samples,
                          n_ctx=args.n_ctx, n_gen=args.n_gen, K=args.K, device=device,
                          scale=args.scale, seed=args.seed, window=args.window)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_sheet(args.out_dir / "sheet_memmaze_rollout.png", sheet)


if __name__ == "__main__":
    main()
