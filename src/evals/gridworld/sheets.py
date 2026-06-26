"""Qualitative GridWorld rollout sheets (spec: specs/evals/gridworld/sheets.md).

The visual companion to `recall.py`: `recall` gives the retention *number*, `sheets` gives the *picture*.
Each sheet is a vertical stack of per-sample blocks; a block is two filmstrip rows (TOP/BOTTOM), columns =
timesteps, with a header label and a thick yellow bar where context ends and the rollout begins.

  * occlusion_sheet — controlled GridWorldEnv: `n_ctx` revealed frames then `n_occ` OCCLUDED rollout steps.
    TOP = true underlying square (what is really behind the curtain), BOTTOM = the model's belief (a
    read-only curtain-up peek decoded every step). Same branching rollout `recall` scores, so strip and
    curve agree; a memory model holds the belief on the true square past the window, a vanilla model decays.
  * normal_sheet    — held-out fully-revealed dataset clips. TOP = ground truth, BOTTOM = model rollout
    (first `n_ctx` cols are context reconstructions, then free-run predictions). In-the-clear sanity check.

Library functions take an already-loaded model + frozen tokenizer (like `recall`) and return a uint8 BGR
image; `save_sheet` and the `__main__` CLI are the local-run convenience layer. cv2 only (BGR, no
matplotlib — runs in the cluster venv and locally), NO RGB/BGR swap (tokenizer is BGR in/out). Run with -u.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/
from envs.gridworld import GridWorldEnv, make_grid_background, stamp_square  # noqa: E402
from evals.gridworld.recall import _tokenizer_window  # noqa: E402

SCALE = 4        # default nearest-neighbour upscale per cell
SEP = 2          # gray separator between cells (px, post-scale)
BND = 7          # thick context|rollout boundary line width (px, post-scale)


# --------------------------------------------------------------------------- rollouts
def _decode_seq(tokenizer, lat: torch.Tensor) -> list[np.ndarray]:
    """Decode a (1, T, L, D) latent sequence to T uint8 BGR frames, sliding the tokenizer's temporal
    window so any T is handled (each frame decoded as the last of its <=window slice)."""
    tok_w = _tokenizer_window(tokenizer)
    T = lat.shape[1]
    if T <= tok_w:
        dec = tokenizer.decoder(lat)[0].clamp(0, 1).cpu().float().numpy()
        return [(f * 255.0).round().astype(np.uint8) for f in dec]
    out = []
    for t in range(T):
        win = lat[:, max(0, t - tok_w + 1):t + 1]
        f = tokenizer.decoder(win)[0, -1].clamp(0, 1).cpu().float().numpy()
        out.append((f * 255.0).round().astype(np.uint8))
    return out


def _controlled_episode(seed: int, n_ctx: int, n_occ: int):
    """Directly from the env: `n_ctx` revealed frames then `n_occ` occluded frames. Returns
    (ctx_frames[n_ctx], truth_frames[n_ctx+n_occ]) — truth = true underlying square rendered every step."""
    env = GridWorldEnv().reset(seed)
    base = make_grid_background(env.bg_color)
    ctx, truth = [], []
    for i in range(n_ctx + n_occ):
        f, s = env.step(0 if i < n_ctx else 1)
        if i < n_ctx:
            ctx.append(f)
        t = base.copy()
        stamp_square(t, int(s[0]), int(s[1]), env.color)
        truth.append(t)
    return np.array(ctx), np.array(truth)


@torch.no_grad()
def _occlusion_belief(model, tokenizer, ctx_frames_u8, n_occ, K, device, window=None) -> list[np.ndarray]:
    """Controlled occlusion rollout via the carried cache. Encode the revealed context, then for each
    occluded step: a READ-ONLY reveal peek (the belief) followed by the occluded commit. Memory is carried
    automatically by `rollout_step`. Returns n_ctx+n_occ uint8 BGR beliefs (context recons then peeks).

    ``window`` (total frames) forces a shorter sliding window than the model trained with; None = native."""
    use_act = model.n_actions > 0
    tok_w = _tokenizer_window(tokenizer)
    max_ctx = None if window is None else max(1, window - 1)
    n_ctx = len(ctx_frames_u8)
    cfx = torch.from_numpy(ctx_frames_u8.astype(np.float32) / 255.0).unsqueeze(0).to(device)
    ctx_lat = tokenizer.encoder(cfx)                                       # (1, n_ctx, L, D)
    ctx_act = torch.zeros((1, n_ctx), dtype=torch.long, device=device) if use_act else None
    state = model.rollout_init(ctx_lat, ctx_act, K, max_ctx=max_ctx)
    lat_buf = ctx_lat[:, -(tok_w - 1):]                                    # rolling decode window

    belief = _decode_seq(tokenizer, ctx_lat)[-n_ctx:]                      # context reconstructions
    a0 = torch.zeros((1,), dtype=torch.long, device=device) if use_act else None  # reveal
    a1 = torch.ones((1,), dtype=torch.long, device=device) if use_act else None   # occlude
    for _ in range(n_occ):
        z_rev = model.rollout_step(state, a0, commit=False)                # read-only belief peek
        win = torch.cat((lat_buf, z_rev), dim=1)[:, -tok_w:]
        d = tokenizer.decoder(win)[0, -1].clamp(0, 1).cpu().float().numpy()
        belief.append((d * 255.0).round().astype(np.uint8))
        z_occ = model.rollout_step(state, a1, commit=True)                 # commit the occluded tick
        lat_buf = torch.cat((lat_buf, z_occ), dim=1)[:, -(tok_w - 1):]
    return belief


@torch.no_grad()
def _free_rollout(model, tokenizer, frames_ep, start, n_ctx, n_gen, K, device) -> list[np.ndarray]:
    """Curtain-up free rollout for the normal sheet: encode `n_ctx` true frames, generate `n_gen` more
    with the curtain up, decode the whole window. Returns n_ctx+n_gen uint8 BGR frames."""
    use_act = model.n_actions > 0
    wf = frames_ep[start:start + n_ctx].astype(np.float32) / 255.0
    ctx = tokenizer.encoder(torch.from_numpy(wf).unsqueeze(0).to(device))
    act = (torch.zeros((1, n_ctx + n_gen), dtype=torch.long, device=device) if use_act else None)
    gen = model.generate(ctx, n_gen, K=K, action_idx=act)
    return _decode_seq(tokenizer, torch.cat((ctx, gen), dim=1))


# --------------------------------------------------------------------------- drawing
def _row(frames, scale):
    cells = []
    for i, f in enumerate(frames):
        c = cv2.resize(f, (f.shape[1] * scale, f.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        cells.append(c)
        if i + 1 < len(frames):
            cells.append(np.full((c.shape[0], SEP, 3), 90, np.uint8))
    return np.hstack(cells)


def _sample_block(top, bot, label, boundary, scale):
    """One sample: TOP over BOTTOM filmstrip, a header label, and a yellow ctx|rollout boundary bar."""
    tr, br = _row(top, scale), _row(bot, scale)
    bar = np.full((SEP, tr.shape[1], 3), 90, np.uint8)
    block = np.vstack([tr, bar, br])
    x = boundary * (top[0].shape[1] * scale + SEP) - SEP // 2
    cv2.rectangle(block, (x - BND // 2, 0), (x + BND // 2, block.shape[0]), (0, 230, 255), -1)
    head = np.full((18, block.shape[1], 3), 30, np.uint8)
    cv2.putText(head, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
    return np.vstack([head, block])


def _assemble(blocks) -> np.ndarray:
    """Pad blocks to equal width and vstack with a black gap. Empty -> a 0-sized array."""
    if not blocks:
        return np.zeros((0, 0, 3), np.uint8)
    w = max(b.shape[1] for b in blocks)
    blocks = [np.pad(b, ((0, 0), (0, w - b.shape[1]), (0, 0)), constant_values=30) for b in blocks]
    gap = np.zeros((8, w, 3), np.uint8)
    sheet = blocks[0]
    for b in blocks[1:]:
        sheet = np.vstack([sheet, gap, b])
    return sheet


def save_sheet(path, sheet: np.ndarray) -> None:
    """cv2.imwrite wrapper; warns and skips an empty sheet."""
    path = Path(path)
    if sheet.size == 0:
        print(f"!! no samples for {path.name} — nothing written")
        return
    cv2.imwrite(str(path), sheet)
    print(f"wrote {path}  ({sheet.shape[1]}x{sheet.shape[0]})")


# --------------------------------------------------------------------------- public sheets
@torch.no_grad()
def occlusion_sheet(model, tokenizer, *, seeds, n_ctx=4, n_occ=16, K=4, device="cpu",
                    scale=SCALE, window=None) -> np.ndarray:
    """Controlled-env occlusion sheet (uint8 BGR), one block per seed. TOP = true underlying square,
    BOTTOM = the model's belief (read-only reveal peek each occluded step).

    ``window`` (total frames) forces a shorter sliding window than the model trained with; None = native."""
    blocks = []
    win_lbl = "" if window is None else f"window={window}  "
    for seed in seeds:
        ctx_frames, truth = _controlled_episode(seed, n_ctx, n_occ)
        belief = _occlusion_belief(model, tokenizer, ctx_frames, n_occ, K, device, window=window)
        label = (f"OCCLUSION seed{seed}  {n_ctx} revealed | {n_occ} occluded rollout  {win_lbl} "
                 f"TOP=true underlying  BOTTOM=model belief (curtain-up peek)")
        blocks.append(_sample_block(list(truth), belief, label, boundary=n_ctx, scale=scale))
    return _assemble(blocks)


@torch.no_grad()
def normal_sheet(model, tokenizer, frames, actions=None, *, n_samples=5, n_ctx=4, n_gen=None,
                 K=4, device="cpu", scale=SCALE, seed=0) -> np.ndarray:
    """Free-run-in-the-clear sheet (uint8 BGR) over fully-revealed dataset clips. TOP = ground truth,
    BOTTOM = model rollout. With `actions`, only windows whose curtain is up throughout are used."""
    maxT = model.config.max_temporal_length
    n_gen = (maxT - n_ctx) if n_gen is None else n_gen
    win = n_ctx + n_gen
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(frames), generator=g).numpy()

    blocks = []
    for ep in order:
        f = np.asarray(frames[ep])
        if f.shape[0] < win:
            continue
        if actions is not None:
            cur = np.asarray(actions[ep])
            s = next((t for t in range(len(cur) - win + 1) if not np.any(cur[t:t + win])), None)
            if s is None:
                continue
        else:
            s = 0
        pred = _free_rollout(model, tokenizer, f, s, n_ctx, n_gen, K, device)
        label = (f"NORMAL ep{ep} t{s}  {n_ctx} ctx | {n_gen} free-run (curtain up)   "
                 f"TOP=ground truth  BOTTOM=model rollout")
        blocks.append(_sample_block(list(f[s:s + win]), pred, label, boundary=n_ctx, scale=scale))
        if len(blocks) >= n_samples:
            break
    return _assemble(blocks)


# --------------------------------------------------------------------------- checkpoint loading (CLI)
def _config_from_checkpoint(cfg_dict: dict, cls):
    allowed = {fld.name for fld in fields(cls)}
    return cls(**{k: v for k, v in cfg_dict.items() if k in allowed})


def _load(checkpoint, cls_model, cls_cfg, device):
    """Load a {config, model_state_dict} payload (same convention as interactive/play_dynamics.py)."""
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = _config_from_checkpoint(payload["config"], cls_cfg)
    model = cls_model(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


# --------------------------------------------------------------------------- CLI
def main() -> None:
    from models.dynamics_model import DynamicsModel, DynamicsModelConfig
    from models.tokenizer import AutoEncoder, AutoEncoderConfig

    root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description="Render qualitative GridWorld rollout sheets.")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Dynamics checkpoint.")
    ap.add_argument("--tokenizer", type=Path, required=True, help="Frozen tokenizer checkpoint.")
    ap.add_argument("--frames", type=Path, default=root / "data" / "gridworld.npy",
                    help="Frames .npy (N,T,H,W,3); only needed for the normal sheet.")
    ap.add_argument("--actions", type=Path, default=None,
                    help="Actions .npy (N,T). Default: '<frames>_actions.npy' if present.")
    ap.add_argument("--out-dir", type=Path, default=root / "outputs" / "sheets",
                    help="Where the PNGs land (default: outputs/sheets/, gitignored).")
    ap.add_argument("--kind", choices=["occlusion", "normal", "both"], default="both")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--n-ctx", type=int, default=4)
    ap.add_argument("--n-occ", type=int, default=16)
    ap.add_argument("--occ-seed0", type=int, default=100, help="First env seed for occlusion samples.")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--window", type=int, default=None,
                    help="Force the sliding context window (frames) for the occlusion sheet; "
                         "default = the model's native max_temporal_length.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = _load(args.tokenizer, AutoEncoder, AutoEncoderConfig, device)
    for p in tok.parameters():
        p.requires_grad_(False)
    model = _load(args.checkpoint, DynamicsModel, DynamicsModelConfig, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.kind in ("occlusion", "both"):
        if model.n_actions == 0:
            print("!! model is action-unlabeled (n_actions=0): the curtain has no meaning, so the "
                  "occlusion sheet just shows a free rollout.")
        seeds = range(args.occ_seed0, args.occ_seed0 + args.n_samples)
        sheet = occlusion_sheet(model, tok, seeds=seeds, n_ctx=args.n_ctx, n_occ=args.n_occ,
                                K=args.K, device=device, window=args.window)
        save_sheet(args.out_dir / "sheet_occlusion.png", sheet)

    if args.kind in ("normal", "both"):
        frames = np.load(args.frames, mmap_mode="r")
        actions = None
        if model.n_actions > 0:
            ap_path = args.actions or args.frames.with_name(args.frames.stem + "_actions.npy")
            if Path(ap_path).is_file():
                actions = np.load(ap_path, mmap_mode="r")
            else:
                print(f"!! no actions file at {ap_path}; normal sheet may include occluded windows.")
        sheet = normal_sheet(model, tok, frames, actions, n_samples=args.n_samples,
                             n_ctx=args.n_ctx, K=args.K, device=device)
        save_sheet(args.out_dir / "sheet_normal.png", sheet)


if __name__ == "__main__":
    main()
