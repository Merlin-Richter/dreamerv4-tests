"""Shared probe driver for the D_sparse_n8 k=4-6 dip investigation.

Differences from evals/gridworldv2/recall.py's driver (alignment kept IDENTICAL):
  * branches at EVERY k in 1..max_k (read-only branches don't perturb the rollout);
  * records the predicted (col,row,color) AND the full true trajectory p_0..p_max_k per
    rollout, so beliefs can be classified offline (match vs p_{k-1}/p_k/p_{k+1}/p_0/p_write);
  * optional no-hide mode (fully revealed rollout — driver/action-alignment validation);
  * optional teacher-forced commits (true latents committed instead of generated ones);
  * optional sparse-mask ablation applied ONLY to the read-only branch pass:
      - "no_mem_read": memory-slot queries see NO write keys (only info-free scratch keys);
      - "mem_only":    non-memory slots see only their own frame in TIME (memory keeps writes).
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "src", ROOT / "experiments" / "sparse-write-slots"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import model as spmod  # noqa: E402  (experiments/sparse-write-slots/model.py)
from model import DynamicsModelSparseWS  # noqa: E402
from envs.gridworld import PALETTE  # noqa: E402
from envs.gridworldv2 import A_HIDE, A_REVEAL, GridWorldV2Env, sample_moves  # noqa: E402
from evals.gridworld.readout import read_square  # noqa: E402
from evals.gridworld.recall import _load_checkpoint, _tokenizer_window  # noqa: E402
from models.dynamics_model import DynamicsModel, DynamicsModelConfig  # noqa: E402
from models.tokenizer import AutoEncoder, AutoEncoderConfig  # noqa: E402

COLOR_NAMES = list(PALETTE.keys())
CKPT = {
    "A": ROOT / "checkpoints/gridworldv2/dynamics_vanilla_tau0.pt",
    "D": ROOT / "checkpoints/gridworldv2/dynamics_sparse_n8.pt",
}
MODEL_CLS = {"A": DynamicsModel, "D": DynamicsModelSparseWS}
TOKENIZER = ROOT / "checkpoints/gridworld/tokenizer.pt"


def load(arm: str, device):
    tok, _ = _load_checkpoint(TOKENIZER, AutoEncoder, AutoEncoderConfig, device)
    for p in tok.parameters():
        p.requires_grad_(False)
    model, cfg = _load_checkpoint(CKPT[arm], MODEL_CLS[arm], DynamicsModelConfig, device)
    return model, cfg, tok


@contextmanager
def sparse_mask_mode(mode: str):
    """Patch the sparse mask builder for the duration of a (branch) forward. "normal" = no-op."""
    if mode == "normal":
        yield
        return
    orig = spmod.sparse_write_mask

    def patched(pos_q, pos_all, n_slots, mem_start, mem_end, n_sparse):
        causal = pos_all[None, :] > pos_q[:, None]
        full = causal.unsqueeze(0).expand(n_slots, -1, -1).clone()
        is_write = (pos_all % n_sparse) == 0
        if mode == "no_mem_read":
            # memory queries: causal AND key must NOT be a write slot -> belief flow severed
            full[mem_start:mem_end] = causal | is_write[None, :]
        elif mode == "mem_only":
            # memory keeps its normal write-read; every OTHER slot sees only its own frame
            full[mem_start:mem_end] = causal | (~is_write[None, :])
            own = pos_all[None, :] != pos_q[:, None]  # (T_q, T_all) True unless same frame
            for s in list(range(0, mem_start)) + list(range(mem_end, n_slots)):
                full[s] = causal | own
        else:
            raise ValueError(mode)
        return full

    spmod.sparse_write_mask = patched
    try:
        yield
    finally:
        spmod.sparse_write_mask = orig


def _encode_full(tokenizer, frames_u8: np.ndarray, device, tok_w: int) -> torch.Tensor:
    """Encode (B,T,H,W,3) uint8 -> (B,T,L,D) latents in non-overlapping tok_w windows."""
    fx = torch.from_numpy(frames_u8.astype(np.float32) / 255.0).to(device)
    outs = [tokenizer.encoder(fx[:, s:s + tok_w]) for s in range(0, fx.shape[1], tok_w)]
    return torch.cat(outs, dim=1)


@torch.no_grad()
def probe_rollout_batch(model, tokenizer, seeds, *, n_ctx: int, max_k: int, K: int, device,
                        window: int | None, hide: bool = True, teacher_forced: bool = False,
                        branch_mask_mode: str = "normal", spoof: str | None = None) -> list[dict]:
    """One batched rollout per seed; branch read-only at EVERY k. Returns one record per seed:
    {seed, traj: [(col,row) x (max_k+1)] (p_0 = post-context/post-hide), preds: {k: (col,row,color_idx)},
     oracle_ok: {k: 0/1}, sq_idx, bg_idx}.
    """
    tok_w = _tokenizer_window(tokenizer)
    max_ctx = None if window is None else max(1, window - 1)
    B = len(seeds)
    envs = [GridWorldV2Env().reset(s) for s in seeds]
    streams = [sample_moves(env.rng, n_ctx + max_k) for env in envs]

    cframes, cacts = [[] for _ in range(B)], [[] for _ in range(B)]
    for b, env in enumerate(envs):
        for t in range(n_ctx):
            a = streams[b][t]
            f, _ = env.step(a)
            cframes[b].append(f)
            cacts[b].append(a)
    colors = [(COLOR_NAMES.index(e.bg_name), COLOR_NAMES.index(e.color_name)) for e in envs]

    cfx = torch.from_numpy(np.stack([np.stack(c) for c in cframes]).astype(np.float32) / 255.0).to(device)
    ctx_lat = tokenizer.encoder(cfx)
    ctx_act = torch.tensor(cacts, dtype=torch.long, device=device)
    state = model.rollout_init(ctx_lat, ctx_act, K, max_ctx=max_ctx)
    lat_buf = ctx_lat[:, -(tok_w - 1):]

    a_rev = torch.full((B,), A_REVEAL, dtype=torch.long, device=device)

    if hide:
        a_hide = torch.full((B,), A_HIDE, dtype=torch.long, device=device)
        for env in envs:
            env.step(A_HIDE)
        z = model.rollout_step(state, a_hide, commit=True)
        lat_buf = torch.cat((lat_buf, z), dim=1)[:, -(tok_w - 1):]

    traj = [[(int(e.col), int(e.row))] for e in envs]  # p_0 = post-context (post-hide if hiding)
    preds: list[dict] = [dict() for _ in range(B)]
    oracle_ok: list[dict] = [dict() for _ in range(B)]

    for k in range(1, max_k + 1):
        moves = [streams[b][n_ctx + k - 1] for b in range(B)]
        step_frames = []
        for b, env in enumerate(envs):
            f, s = env.step(moves[b])
            traj[b].append((int(s[0]), int(s[1])))
            step_frames.append(f)
        mv = torch.tensor(moves, dtype=torch.long, device=device)
        # Action-token spoof (causal test): env applies the TRUE move, but the MODEL is fed
        # A_STAY at the commit of this position. "at_write": spoof positions %8==0 (the write
        # phase); "after_write": spoof positions %8==1 (control). If readers ignore the
        # write-phase action token, "at_write" leaves branch beliefs unchanged.
        if spoof is not None:
            phase = state["next_pos"] % 8
            if (spoof == "at_write" and phase == 0) or (spoof == "after_write" and phase == 1):
                mv = torch.full_like(mv, 6)  # A_STAY
        if teacher_forced:
            # commit the TRUE latent of this tick (revealed frame in no-hide mode; curtain if hiding)
            tf = _encode_full(tokenizer, np.stack(step_frames)[:, None], device, tok_w)
            model._commit_context_frame(state, tf, mv)
            z_step = tf
        else:
            z_step = model.rollout_step(state, mv, commit=True)
        lat_buf = torch.cat((lat_buf, z_step), dim=1)[:, -(tok_w - 1):]

        with sparse_mask_mode(branch_mask_mode):
            z_rev = model.rollout_step(state, a_rev, commit=False)
        win = torch.cat((lat_buf, z_rev), dim=1)[:, -tok_w:]
        dec = tokenizer.decoder(win)[:, -1].clamp(0, 1).cpu().float().numpy()
        pred = (dec * 255.0).round().astype(np.uint8)
        for b in range(B):
            rd = read_square(pred[b])
            preds[b][k] = (int(rd["col"]), int(rd["row"]), int(rd["color_idx"]))
            ro = read_square(envs[b].render_revealed())
            oracle_ok[b][k] = int((ro["col"], ro["row"]) == traj[b][-1])

    return [{"seed": int(seeds[b]), "bg_idx": colors[b][0], "sq_idx": colors[b][1],
             "traj": traj[b], "preds": preds[b], "oracle_ok": oracle_ok[b]}
            for b in range(B)]


def run_probe(arm: str, *, n_rollouts: int, n_ctx: int, max_k: int, window: int | None,
              K: int = 4, batch_size: int = 64, device=None, hide=True, teacher_forced=False,
              branch_mask_mode="normal", seed0: int = 0, spoof: str | None = None) -> list[dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, tok = run_probe._cache.get(arm, (None, None, None))
    if model is None:
        model, cfg, tok = load(arm, device)
        run_probe._cache[arm] = (model, cfg, tok)
    recs = []
    for i in range(0, n_rollouts, batch_size):
        seeds = list(range(seed0 + i, seed0 + min(i + batch_size, n_rollouts)))
        recs += probe_rollout_batch(model, tok, seeds, n_ctx=n_ctx, max_k=max_k, K=K,
                                    device=device, window=window, hide=hide,
                                    teacher_forced=teacher_forced,
                                    branch_mask_mode=branch_mask_mode, spoof=spoof)
    return recs


run_probe._cache = {}


def summarize(recs: list[dict], max_k: int, n_ctx: int, hide: bool = True) -> dict:
    """Per-k stats: acc vs p_k, p_{k-1}, p_{k+1}, p_0 (copy-last), p_write (true position at the
    newest COMMITTED write slot), Chebyshev distance mean, color acc, oracle.

    Hide mode positions: ctx 0..n_ctx-1, hide tick at n_ctx, move j at n_ctx+j, branch at
    n_ctx+k+1. No-hide: move j at n_ctx+j-1, branch at n_ctx+k."""
    out = {}
    for k in range(1, max_k + 1):
        n = same = prev = nxt = p0m = pwm = col_ok = orac = 0
        dsum = 0.0
        for r in recs:
            if k not in r["preds"]:
                continue
            pc = r["preds"][k][:2]
            tr = r["traj"]
            n += 1
            same += pc == tuple(tr[k])
            prev += pc == tuple(tr[k - 1])
            nxt += (k + 1 < len(tr)) and pc == tuple(tr[k + 1])
            p0m += pc == tuple(tr[0])
            last_committed = (n_ctx + k) if hide else (n_ctx + k - 1)
            wpos = (last_committed // 8) * 8  # newest committed write slot
            j = wpos - n_ctx if hide else wpos - n_ctx + 1  # its occluded-move index
            j = min(max(j, 0), k)
            pwm += pc == tuple(tr[j])
            dsum += max(abs(pc[0] - tr[k][0]), abs(pc[1] - tr[k][1]))
            col_ok += r["preds"][k][2] == r["sq_idx"]
            orac += r["oracle_ok"][k]
        if n:
            out[k] = {"n": n, "acc": same / n, "acc_km1": prev / n, "acc_kp1": nxt / n,
                      "acc_p0": p0m / n, "acc_pwrite": pwm / n, "cheb": dsum / n,
                      "color": col_ok / n, "oracle": orac / n}
    return out


def print_table(title: str, summ: dict):
    print(f"\n### {title}")
    print("  k :   acc   k-1   k+1    p0  p_wr  cheb  colr  orac")
    for k, s in sorted(summ.items()):
        print(f" {k:3d}: {s['acc']:5.3f} {s['acc_km1']:5.3f} {s['acc_kp1']:5.3f} "
              f"{s['acc_p0']:5.3f} {s['acc_pwrite']:5.3f} {s['cheb']:5.2f} "
              f"{s['color']:5.3f} {s['oracle']:5.3f}")
