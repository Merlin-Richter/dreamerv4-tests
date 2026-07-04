"""Qualitative Memory-Maze rollout sheets (spec: specs/evals/memmaze/sheets.md).

The memmaze counterpart of `evals/gridworld/sheets.py`. No controlled env / curtain here (occlusion in
Memory Maze is natural — the agent looks away and later looks back), so there is ONE sheet kind:

  * rollout_sheet — held-out dataset episodes. The model PREFILLS `n_pre` (default 64) true frames
    through the sliding window before generation (with 8 frames the maze is mostly unobserved and the
    env is impossible by construction; long prefill is the intended usage — memory models absorb the
    pre-window part via the relay). The sheet displays only the last `n_ctx` context columns. TOP =
    ground truth, BOTTOM = model rollout (context reconstructions, then free-run predictions
    conditioned on the TRUE action sequence). `n_gen` may exceed the model's window (the carrying
    rollout slides) — the qualitative long-horizon / maze-consistency check.

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
from evals.gridworld.recall import _tokenizer_window  # noqa: E402
from evals.gridworld.sheets import (  # noqa: E402  — shared drawing/loading layer
    _assemble, _load, _sample_block, save_sheet,
)

SCALE = 2   # default upscale per cell (memmaze blocks are wide: 32+ columns)
N_PRE = 64  # default context frames the model PREFILLS (2x the W=32 dynamics window; also the
            # tokenizer window = the one-shot encode limit). The sheet displays only the tail.


def val_episodes(n_episodes: int, val_fraction: float = 0.05) -> np.ndarray:
    """The trainer's held-out episode ids — reproduces train_dynamics.py's split exactly
    (torch.randperm with a fresh seed-0 generator; n_val = min(max(1, round(n*frac)), n-1))."""
    n_val = min(max(1, int(round(n_episodes * val_fraction))), n_episodes - 1)
    g = torch.Generator().manual_seed(0)
    return torch.randperm(n_episodes, generator=g)[:n_val].numpy()


@torch.no_grad()
def _action_rollout(model, tokenizer, frames_ep, actions_ep, start, n_pre, n_ctx, n_gen, K, device,
                    max_ctx=None) -> list[np.ndarray]:
    """Encode `n_pre` true frames and let the model PREFILL them all (rollout_init's long-context
    teacher-forced sliding commits — memory relays the pre-window part), then generate `n_gen` frames
    conditioned on the true action sequence. Returns only the DISPLAYED tail as uint8 BGR: the last
    `n_ctx` context frames (reconstructions) followed by the `n_gen` rollout frames."""
    tok_w = _tokenizer_window(tokenizer)
    assert n_pre <= tok_w, f"n_pre={n_pre} exceeds the tokenizer window {tok_w} (one-shot encode)"
    wf = np.asarray(frames_ep[start:start + n_pre]).astype(np.float32) / 255.0
    ctx = tokenizer.encoder(torch.from_numpy(wf).unsqueeze(0).to(device))
    act = None
    if actions_ep is not None:
        a = np.asarray(actions_ep[start:start + n_pre + n_gen]).astype(np.int64)
        act = torch.from_numpy(a).unsqueeze(0).to(device)
    gen = model.generate(ctx, n_gen, K=K, action_idx=act, max_ctx=max_ctx)
    seq = torch.cat((ctx, gen), dim=1)
    disp = n_ctx + n_gen
    assert disp <= tok_w, f"displayed strip {disp} exceeds the tokenizer window {tok_w}"
    # One-shot decode of the trailing tokenizer window, then slice the displayed tail (safe by the
    # window-invariance probe: window-delta recon error is far below the recon error itself).
    dec = tokenizer.decoder(seq[:, -min(seq.shape[1], tok_w):])[0, -disp:].clamp(0, 1)
    return [(f * 255.0).round().astype(np.uint8) for f in dec.cpu().float().numpy()]


@torch.no_grad()
def rollout_sheet(model, tokenizer, frames, actions, *, episodes=None, n_samples=4, n_pre=N_PRE,
                  n_ctx=8, n_gen=None, K=4, device="cpu", scale=SCALE, seed=0,
                  window=None) -> np.ndarray:
    """Held-out GT-vs-rollout sheet (uint8 BGR), one block per episode. The model PREFILLS `n_pre`
    true frames (long context through the sliding window — memory absorbs the pre-window part) but
    the sheet displays only the last `n_ctx` of them, then the `n_gen` rollout columns. TOP = ground
    truth, BOTTOM = context recon then action-conditioned free-run. `actions` may be None only for an
    unlabeled model.

    ``episodes`` overrides the default val-split selection; the start offset inside each episode is
    drawn from a `seed`-seeded generator. ``window`` (total frames) forces a shorter sliding window
    than the model trained with; None = native."""
    n_gen = (model.config.max_temporal_length - n_ctx) if n_gen is None else n_gen
    assert n_ctx <= n_pre, f"n_ctx={n_ctx} must be <= n_pre={n_pre} (shown ctx is a suffix)"
    need = n_pre + n_gen                       # frames consumed from the episode
    disp = n_ctx + n_gen                       # columns on the sheet
    max_ctx = None if window is None else max(1, window - 1)
    if episodes is None:
        episodes = val_episodes(len(frames))[:n_samples]
    win_lbl = "" if window is None else f"window={window}  "

    g = torch.Generator().manual_seed(seed)
    blocks = []
    for ep in episodes:
        f = np.asarray(frames[ep])
        if f.shape[0] < need:
            print(f"!! ep{ep}: T={f.shape[0]} < {need} frames needed — skipped")
            continue
        s = int(torch.randint(0, f.shape[0] - need + 1, (1,), generator=g).item())
        a_ep = actions[ep] if actions is not None else None
        pred = _action_rollout(model, tokenizer, f, a_ep, s, n_pre, n_ctx, n_gen, K, device,
                               max_ctx=max_ctx)
        d0 = s + n_pre - n_ctx                 # first DISPLAYED frame
        act_lbl = ""
        if a_ep is not None:
            digits = "".join(str(int(a)) for a in np.asarray(a_ep[d0:d0 + disp]))
            act_lbl = f"  acts {digits[:n_ctx]}|{digits[n_ctx:]}"
        label = (f"MEMMAZE ep{ep} t{s}  {n_pre} ctx ({n_ctx} shown) | {n_gen} action-conditioned "
                 f"rollout  {win_lbl}TOP=ground truth  BOTTOM=model rollout{act_lbl}")
        blocks.append(_sample_block(list(f[d0:d0 + disp]), pred, label, boundary=n_ctx, scale=scale))
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
    ap.add_argument("--n-pre", type=int, default=N_PRE,
                    help="Context frames the model PREFILLS before generating (only the last n-ctx "
                         "are displayed). Must be <= the tokenizer window (one-shot encode).")
    ap.add_argument("--n-ctx", type=int, default=8, help="Displayed context columns (suffix of n-pre).")
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
                          n_pre=args.n_pre, n_ctx=args.n_ctx, n_gen=args.n_gen, K=args.K,
                          device=device, scale=args.scale, seed=args.seed, window=args.window)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_sheet(args.out_dir / "sheet_memmaze_rollout.png", sheet)


if __name__ == "__main__":
    main()
