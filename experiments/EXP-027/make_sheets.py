"""EXP-027 qualitative rollout sheets (vanilla GridWorld dynamics).

Two sheets, each = several samples stacked; per sample two rows, columns = timesteps:
  * sheet_normal.png    — free-running rollout with the curtain UP the whole time. TOP = ground-truth
    frames, BOTTOM = model rollout (first n_ctx are context reconstructions, then predictions). Shows
    motion-tracking quality in the clear.
  * sheet_occlusion.png — rollout through a curtain-DOWN run then reveal. TOP = TRUE UNDERLYING frames
    (square always rendered = oracle, i.e. what is really happening behind the curtain), BOTTOM = model
    rollout (its imagination behind the curtain + the reveal prediction). Occluded columns are tinted;
    a vertical line marks the context/generation boundary. This is the memory view: does the bottom
    square track the top square while hidden?

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
from evals.gridworld.adapter import load_dynamics, load_tokenizer  # noqa: E402
from evals.gridworld.recall import find_reveal_events, oracle_frames  # noqa: E402

SCALE = 3
SEP = 2  # gray separator px (pre-scale handled after upscale)


@torch.no_grad()
def rollout_window(model, tok, frames_u8, curtain, start, n_ctx, n_gen, device, force_curtain=None):
    """Decode a (n_ctx+n_gen)-frame window: encode n_ctx true frames, generate n_gen feeding the true
    (or forced) curtain actions, decode the whole window. Returns pred_frames (T,H,W,3) uint8 BGR."""
    T = n_ctx + n_gen
    wf = frames_u8[start:start + n_ctx].astype(np.float32) / 255.0
    ctx = tok.encoder(torch.from_numpy(wf).unsqueeze(0).to(device))      # (1, n_ctx, n_lat, dim)
    act = np.asarray(curtain[start:start + T]).astype(np.int64).copy()
    if force_curtain is not None:
        act[n_ctx:] = force_curtain
    act_t = torch.from_numpy(act).unsqueeze(0).to(device)
    gen = model.generate_cached(ctx, n_gen, action_idx=act_t)            # (1, n_gen, n_lat, dim)
    full = torch.cat((ctx, gen), dim=1)                                  # (1, T<=max_T, ...)
    dec = tok.decoder(full)[0].clamp(0, 1).cpu().float().numpy()         # (T, H, W, 3)
    return (dec * 255.0).round().astype(np.uint8)


@torch.no_grad()
def belief_rollout(model, tok, frames_u8, curtain, start, n_ctx, n_gen, device, maxT):
    """Per-timestep BELIEF rollout for the occlusion view. Roll the latent forward with the TRUE curtain
    (so the model is genuinely blind during the occluded run), but at every step also decode a
    'curtain-UP peek' — the frame the model WOULD predict if the curtain lifted now — to render where it
    currently thinks the square is. Returns n_ctx+n_gen believed frames (uint8 BGR)."""
    wf = frames_u8[start:start + n_ctx].astype(np.float32) / 255.0
    seq = tok.encoder(torch.from_numpy(wf).unsqueeze(0).to(device))      # (1, n_ctx, ...)
    acts_true = [int(curtain[start + j]) for j in range(n_ctx + n_gen)]
    belief = [(f * 255).round().astype(np.uint8)
              for f in tok.decoder(seq)[0].clamp(0, 1).cpu().float().numpy()]  # context reconstructions
    cur_acts = acts_true[:n_ctx]
    for i in range(n_gen):
        peek_act = torch.tensor([cur_acts + [0]], dtype=torch.long, device=device)   # curtain UP peek
        peek = model.generate_cached(seq, 1, action_idx=peek_act)
        win = torch.cat((seq, peek), 1)[:, -maxT:]
        d = tok.decoder(win)[0, -1].clamp(0, 1).cpu().float().numpy()
        belief.append((d * 255).round().astype(np.uint8))
        commit_act = torch.tensor([cur_acts + [acts_true[n_ctx + i]]], dtype=torch.long, device=device)
        seq = torch.cat((seq, model.generate_cached(seq, 1, action_idx=commit_act)), 1)
        cur_acts.append(acts_true[n_ctx + i])
    return belief


def _row(frames, occluded_mask=None, boundary=None):
    """Upscale + concat a list of frames into one row; tint occluded columns; mark boundary."""
    cells = []
    for i, f in enumerate(frames):
        c = cv2.resize(f, (f.shape[1] * SCALE, f.shape[0] * SCALE), interpolation=cv2.INTER_NEAREST)
        if occluded_mask is not None and occluded_mask[i]:
            c[:3, :] = (0, 140, 255); c[-3:, :] = (0, 140, 255)  # orange top/bottom border = occluded
        cells.append(c)
        if i + 1 < len(frames):
            cells.append(np.full((c.shape[0], SEP, 3), 90, np.uint8))
    row = np.hstack(cells)
    if boundary is not None:
        x = boundary * (frames[0].shape[1] * SCALE + SEP)
        cv2.line(row, (x, 0), (x, row.shape[0]), (0, 0, 255), 1)
    return row


def _sample_block(top, bot, label, occ_mask=None, boundary=None):
    tr = _row(top, None, boundary)
    br = _row(bot, occ_mask, boundary)
    bar = np.full((SEP, tr.shape[1], 3), 90, np.uint8)
    block = np.vstack([tr, bar, br])
    head = np.full((16, block.shape[1], 3), 30, np.uint8)
    cv2.putText(head, label, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)
    return np.vstack([head, block])


def find_normal_windows(curtain, T, n_needed):
    """Windows of length T fully curtain-UP (no occlusion)."""
    out = []
    for s in range(0, len(curtain) - T):
        if not np.any(np.asarray(curtain[s:s + T])):
            out.append(s)
            if len(out) >= n_needed:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=str(ROOT / "gridworld.npy"))
    ap.add_argument("--tokenizer", default=str(ROOT / "checkpoints/gridworld/tokenizer.pt"))
    ap.add_argument("--dynamics", default=str(ROOT / "checkpoints/gridworld/dynamics_vanilla.pt"))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--n-ctx", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stem = args.frames[:-4] if args.frames.endswith(".npy") else args.frames
    frames = np.load(args.frames, mmap_mode="r")
    states = np.load(stem + "_states.npy", mmap_mode="r")
    colors = np.load(stem + "_colors.npy", mmap_mode="r")
    actions = np.load(stem + "_actions.npy", mmap_mode="r")  # == curtain
    tok, _ = load_tokenizer(args.tokenizer, device)
    model, cfg = load_dynamics(args.dynamics, device)
    maxT = cfg.max_temporal_length
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # deterministic seed-0 val split (same as training/eval) — sheets on held-out episodes
    g = torch.Generator().manual_seed(0)
    val_idx = torch.randperm(len(frames), generator=g).numpy()[:150]

    # ---- NORMAL: free-run, curtain UP, full maxT window ----
    n_gen = maxT - args.n_ctx
    blocks, used = [], 0
    for ep in val_idx:
        cur = np.asarray(actions[ep])
        ws = find_normal_windows(cur, maxT, 1)
        if not ws:
            continue
        s = ws[0]
        f = np.asarray(frames[ep])
        pred = rollout_window(model, tok, f, cur, s, args.n_ctx, n_gen, device, force_curtain=0)
        gt = f[s:s + maxT]
        blocks.append(_sample_block(list(gt), list(pred),
                      f"NORMAL ep{ep} t{s}..{s+maxT-1}  (ctx={args.n_ctx}, free-run curtain-up)",
                      boundary=args.n_ctx))
        used += 1
        if used >= args.n_samples:
            break
    _save(out_dir / "sheet_normal.png", blocks)

    # ---- OCCLUSION: context (visible) -> through an occluded run -> reveal ----
    blocks, used = [], 0
    for ep in val_idx:
        cur = np.asarray(actions[ep]); f = np.asarray(frames[ep])
        oracle = oracle_frames(np.asarray(states[ep]), np.asarray(colors[ep]), cur)  # true underlying
        for ev in find_reveal_events(cur):
            lv, k, t = ev["last_visible_t"], ev["k"], ev["reveal_t"]
            if not (4 <= k <= maxT - 3):
                continue
            s = max(0, lv - (args.n_ctx - 1))
            T = min(maxT, t - s + 2)          # context + occlusion + reveal (+1 post if room), capped
            if T < 6 or s + T > len(cur):
                continue
            n_ctx = lv - s + 1
            n_gen = T - n_ctx
            belief = belief_rollout(model, tok, f, cur, s, n_ctx, n_gen, device, maxT)
            top = list(oracle[s:s + T])       # true underlying square (what's behind the curtain)
            occ = [bool(cur[s + i]) for i in range(T)]
            blocks.append(_sample_block(top, belief,
                          f"OCC ep{ep} t{s}..{s+T-1}  k={k} reveal@{t}  top=TRUE underlying  bot=model BELIEF (curtain-up peek; orange=blind)",
                          occ_mask=occ, boundary=n_ctx))
            used += 1
            break
        if used >= args.n_samples:
            break
    _save(out_dir / "sheet_occlusion.png", blocks)


def _save(path, blocks):
    if not blocks:
        print(f"!! no samples for {path.name}")
        return
    w = max(b.shape[1] for b in blocks)
    blocks = [np.pad(b, ((0, 0), (0, w - b.shape[1]), (0, 0)), constant_values=30) for b in blocks]
    gap = np.full((6, w, 3), 0, np.uint8)
    sheet = blocks[0]
    for b in blocks[1:]:
        sheet = np.vstack([sheet, gap, b])
    cv2.imwrite(str(path), sheet)
    print(f"wrote {path}  ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
