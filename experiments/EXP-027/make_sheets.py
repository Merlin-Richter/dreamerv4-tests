"""EXP-027 qualitative rollout sheets (vanilla GridWorld dynamics).

Two sheets, each = several samples stacked; per sample two rows, columns = timesteps:

  * sheet_normal.png    — free-running rollout, curtain UP the whole time. TOP = ground-truth frames,
    BOTTOM = model rollout (first n_ctx cols are context reconstructions, then predictions). Motion
    tracking in the clear. (held-out val episodes that happen to be fully revealed.)

  * sheet_occlusion.png — CONTROLLED env scenario built directly from GridWorldEnv (no dataset): the
    model sees n_ctx=4 revealed frames, then the curtain stays DOWN for n_occ=16 frames. TOP = the TRUE
    underlying square (rendered from env state — what is really behind the curtain), BOTTOM = the model's
    BELIEF (a curtain-UP "peek" decoded every step → where it thinks the square is). The model is held
    to a sliding context window of W=8 frames, so once the last revealed frame slides out (~7 steps in)
    it is fully blind and the belief collapses — the memory cutoff, visible early. A thick yellow line
    marks where the true input ends and the rollout begins.

cv2 only (BGR, matches the data) — no matplotlib, so it runs in the cluster venv. Run with -u.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from envs.gridworld import GridWorldEnv, make_grid_background, stamp_square  # noqa: E402
from evals.gridworld.adapter import load_dynamics, load_tokenizer  # noqa: E402

SCALE = 4
SEP = 2          # gray separator between cells
BND = 7          # thick boundary line width (px, post-scale)


# --------------------------------------------------------------------------- rollouts
@torch.no_grad()
def free_rollout(model, tok, frames_u8, curtain, start, n_ctx, n_gen, device):
    """Curtain-UP free rollout for the NORMAL sheet: encode n_ctx true frames, generate n_gen with
    curtain up, decode the whole window. Returns (n_ctx+n_gen, H, W, 3) uint8 BGR."""
    wf = frames_u8[start:start + n_ctx].astype(np.float32) / 255.0
    ctx = tok.encoder(torch.from_numpy(wf).unsqueeze(0).to(device))
    act = torch.zeros((1, n_ctx + n_gen), dtype=torch.long, device=device)  # curtain up throughout
    gen = model.generate_cached(ctx, n_gen, action_idx=act)
    full = torch.cat((ctx, gen), dim=1)
    dec = tok.decoder(full)[0].clamp(0, 1).cpu().float().numpy()
    return (dec * 255.0).round().astype(np.uint8)


@torch.no_grad()
def occlusion_belief(model, tok, ctx_frames_u8, n_occ, device, W):
    """Controlled occlusion rollout with a forced sliding window W. Encode the revealed context, then
    for each of n_occ occluded steps: COMMIT the curtain-down prediction (what the model actually sees),
    and PEEK with curtain-up to render the believed square. The model only ever attends to the last W
    frames. Returns (n_ctx + n_occ, H, W, 3) uint8 BGR beliefs (context recons then per-step beliefs)."""
    n_ctx = len(ctx_frames_u8)
    wf = ctx_frames_u8.astype(np.float32) / 255.0
    seq = tok.encoder(torch.from_numpy(wf).unsqueeze(0).to(device))     # (1, n_ctx, ...)
    acts = [0] * n_ctx
    belief = [(f * 255).round().astype(np.uint8)
              for f in tok.decoder(seq)[0].clamp(0, 1).cpu().float().numpy()]
    for _ in range(n_occ):
        ctx_in = seq[:, -(W - 1):]                                      # sliding window (<= W-1 context)
        a_in = acts[-(W - 1):]
        peek = model.generate_cached(ctx_in, 1,
                                     action_idx=torch.tensor([a_in + [0]], dtype=torch.long, device=device))
        d = tok.decoder(torch.cat((ctx_in, peek), 1))[0, -1].clamp(0, 1).cpu().float().numpy()
        belief.append((d * 255).round().astype(np.uint8))
        nxt = model.generate_cached(ctx_in, 1,
                                    action_idx=torch.tensor([a_in + [1]], dtype=torch.long, device=device))
        seq = torch.cat((seq, nxt), 1)
        acts.append(1)
    return belief


def gen_controlled_episode(seed, n_ctx, n_occ):
    """Directly from the env: n_ctx revealed frames then n_occ occluded frames. Returns
    (ctx_frames[n_ctx], truth_frames[n_ctx+n_occ]) — truth = true underlying square every step."""
    env = GridWorldEnv().reset(seed)
    base = make_grid_background(env.bg_color)
    ctx, truth = [], []
    for i in range(n_ctx + n_occ):
        f, s = env.step(0 if i < n_ctx else 1)
        if i < n_ctx:
            ctx.append(f)
        t = base.copy(); stamp_square(t, int(s[0]), int(s[1]), env.color); truth.append(t)
    return np.array(ctx), np.array(truth)


# --------------------------------------------------------------------------- drawing
def _row(frames):
    cells = []
    for i, f in enumerate(frames):
        c = cv2.resize(f, (f.shape[1] * SCALE, f.shape[0] * SCALE), interpolation=cv2.INTER_NEAREST)
        cells.append(c)
        if i + 1 < len(frames):
            cells.append(np.full((c.shape[0], SEP, 3), 90, np.uint8))
    return np.hstack(cells)


def _sample_block(top, bot, label, boundary):
    tr, br = _row(top), _row(bot)
    bar = np.full((SEP, tr.shape[1], 3), 90, np.uint8)
    block = np.vstack([tr, bar, br])
    x = boundary * (top[0].shape[1] * SCALE + SEP) - SEP // 2          # between ctx and rollout
    cv2.rectangle(block, (x - BND // 2, 0), (x + BND // 2, block.shape[0]), (0, 230, 255), -1)
    head = np.full((18, block.shape[1], 3), 30, np.uint8)
    cv2.putText(head, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
    return np.vstack([head, block])


def _save(path, blocks):
    if not blocks:
        print(f"!! no samples for {path.name}"); return
    w = max(b.shape[1] for b in blocks)
    blocks = [np.pad(b, ((0, 0), (0, w - b.shape[1]), (0, 0)), constant_values=30) for b in blocks]
    gap = np.full((8, w, 3), 0, np.uint8)
    sheet = blocks[0]
    for b in blocks[1:]:
        sheet = np.vstack([sheet, gap, b])
    cv2.imwrite(str(path), sheet)
    print(f"wrote {path}  ({sheet.shape[1]}x{sheet.shape[0]})")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=str(ROOT / "gridworld.npy"))
    ap.add_argument("--tokenizer", default=str(ROOT / "checkpoints/gridworld/tokenizer.pt"))
    ap.add_argument("--dynamics", default=str(ROOT / "checkpoints/gridworld/dynamics_vanilla.pt"))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--n-ctx", type=int, default=4)
    ap.add_argument("--n-occ", type=int, default=16)
    ap.add_argument("--window", type=int, default=8, help="forced sliding context window for occlusion")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, _ = load_tokenizer(args.tokenizer, device)
    model, cfg = load_dynamics(args.dynamics, device)
    maxT = cfg.max_temporal_length
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- OCCLUSION: controlled env scenario, sliding window, thick boundary ----
    blocks = []
    for seed in range(100, 100 + args.n_samples):
        ctx_frames, truth = gen_controlled_episode(seed, args.n_ctx, args.n_occ)
        belief = occlusion_belief(model, tok, ctx_frames, args.n_occ, device, args.window)
        blocks.append(_sample_block(
            list(truth), belief,
            f"OCCLUSION seed{seed}  {args.n_ctx} revealed | {args.n_occ} occluded rollout  "
            f"window={args.window}   TOP=true underlying  BOTTOM=model belief (curtain-up peek)",
            boundary=args.n_ctx))
    _save(out_dir / "sheet_occlusion.png", blocks)

    # ---- NORMAL: free-run in the clear (held-out val episodes, fully revealed window) ----
    frames = np.load(args.frames, mmap_mode="r")
    actions = np.load((args.frames[:-4] if args.frames.endswith(".npy") else args.frames) + "_actions.npy",
                      mmap_mode="r")
    g = torch.Generator().manual_seed(0)
    val_idx = torch.randperm(len(frames), generator=g).numpy()[:150]
    n_gen = maxT - args.n_ctx
    blocks, used = [], 0
    for ep in val_idx:
        cur = np.asarray(actions[ep])
        s = next((t for t in range(len(cur) - maxT) if not np.any(cur[t:t + maxT])), None)
        if s is None:
            continue
        f = np.asarray(frames[ep])
        pred = free_rollout(model, tok, f, cur, s, args.n_ctx, n_gen, device)
        blocks.append(_sample_block(list(f[s:s + maxT]), list(pred),
                      f"NORMAL ep{ep} t{s}  {args.n_ctx} ctx | {n_gen} free-run (curtain up)  "
                      f"TOP=ground truth  BOTTOM=model rollout", boundary=args.n_ctx))
        used += 1
        if used >= args.n_samples:
            break
    _save(out_dir / "sheet_normal.png", blocks)


if __name__ == "__main__":
    main()
