"""GridWorldV2 action-conditioned memory recall (spec: specs/evals/gridworldv2/recall.md, DRAFT).

Same question as v1's recall but the hidden square is ACTION-driven: behind the curtain its
position is a clamped (nonlinear) function of the whole movement-action stream, so the belief
must integrate actions, not extrapolate ballistics.

ALIGNMENT (see spec): context (revealed, moving) -> one committed hide tick (no move, unscored)
-> per k one committed occluded movement tick; at checked k a READ-ONLY branch with action 0
("reveal now, no move") is scored against the true position after exactly k movements.

Reuses v1's scorer/loader/k-grid so the two evals cannot drift: score_reveal, chance_levels,
_check_ks, _tokenizer_window, _load_checkpoint.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/
from envs.gridworld import PALETTE  # noqa: E402
from envs.gridworldv2 import A_HIDE, A_REVEAL, GridWorldV2Env, sample_moves  # noqa: E402
from evals.gridworld.recall import (_check_ks, _load_checkpoint, _tokenizer_window,  # noqa: E402
                                    chance_levels, position_credit, score_reveal)

COLOR_NAMES = list(PALETTE.keys())


@torch.no_grad()
def roll_and_score_batch(model, tokenizer, seeds, n_ctx: int, max_k: int, K: int,
                         device, window: int = None) -> dict:
    """Batched v2 driver (B=len(seeds)). Each record is (k, {pos_correct, pos_score,
    color_correct}); B records per checked k. The reveal branch is read-only."""
    tok_w = _tokenizer_window(tokenizer)
    max_ctx = None if window is None else max(1, window - 1)
    B = len(seeds)
    envs = [GridWorldV2Env().reset(s) for s in seeds]
    # Per-env movement stream (seeded by the env's own rng -> reproducible, datagen statistics).
    streams = [sample_moves(env.rng, n_ctx + max_k) for env in envs]

    # Context: n_ctx REVEALED movement frames per env.
    cframes, cacts, last = [[] for _ in range(B)], [[] for _ in range(B)], [None] * B
    for b, env in enumerate(envs):
        s = None
        for t in range(n_ctx):
            a = streams[b][t]
            f, s = env.step(a)
            cframes[b].append(f)
            cacts[b].append(a)
        last[b] = (int(s[0]), int(s[1]))                  # last OBSERVED cell (copy_last belief)
    colors = [(COLOR_NAMES.index(e.bg_name), COLOR_NAMES.index(e.color_name)) for e in envs]

    cfx = torch.from_numpy(np.stack([np.stack(c) for c in cframes]).astype(np.float32) / 255.0).to(device)
    ctx_lat = tokenizer.encoder(cfx)                      # (B, n_ctx, L, D)
    ctx_act = torch.tensor(cacts, dtype=torch.long, device=device)
    state = model.rollout_init(ctx_lat, ctx_act, K, max_ctx=max_ctx)
    lat_buf = ctx_lat[:, -(tok_w - 1):]

    a_rev = torch.full((B,), A_REVEAL, dtype=torch.long, device=device)
    a_hide = torch.full((B,), A_HIDE, dtype=torch.long, device=device)

    # One committed HIDE tick (curtain latches down; square does not move; not scored).
    for env in envs:
        env.step(A_HIDE)
    z = model.rollout_step(state, a_hide, commit=True)
    lat_buf = torch.cat((lat_buf, z), dim=1)[:, -(tok_w - 1):]

    check = set(_check_ks(max_k))
    recs = {"model": [], "oracle": [], "copy_last": []}
    for k in range(1, max_k + 1):
        moves = [streams[b][n_ctx + k - 1] for b in range(B)]
        true_cells = []
        for b, env in enumerate(envs):
            _, s = env.step(moves[b])                     # occluded movement tick (truth p_k)
            true_cells.append((int(s[0]), int(s[1])))
        z_occ = model.rollout_step(state, torch.tensor(moves, dtype=torch.long, device=device),
                                   commit=True)
        lat_buf = torch.cat((lat_buf, z_occ), dim=1)[:, -(tok_w - 1):]
        if k in check:
            z_rev = model.rollout_step(state, a_rev, commit=False)  # read-only "reveal now"
            win = torch.cat((lat_buf, z_rev), dim=1)[:, -tok_w:]
            dec = tokenizer.decoder(win)[:, -1].clamp(0, 1).cpu().float().numpy()
            pred = (dec * 255.0).round().astype(np.uint8)
            for b in range(B):
                tcell = true_cells[b]
                recs["model"].append((k, score_reveal(pred[b], tcell, colors[b])))
                recs["oracle"].append((k, score_reveal(envs[b].render_revealed(), tcell, colors[b])))
                cl_dist = max(abs(last[b][0] - tcell[0]), abs(last[b][1] - tcell[1]))
                recs["copy_last"].append((k, {
                    "pos_correct": int(cl_dist == 0),
                    "pos_score": position_credit(cl_dist),
                    "color_correct": 1,                   # colour is static
                }))
    return recs


@torch.no_grad()
def recall(model, tokenizer, *, n_ctx: int = 4, max_k: int, n_rollouts: int = 64,
           K: int = 4, device="cpu", window: int = None, batch_size: int = 64) -> dict:
    """Run n_rollouts v2 occluded rollouts (batched) -> per-k recall curves (v1 result schema)."""
    metric_name = {"pos_correct": "position_acc", "pos_score": "position_score",
                   "color_correct": "color_acc"}
    acc = {src: defaultdict(lambda: defaultdict(list)) for src in ("model", "oracle", "copy_last")}
    for i in range(0, n_rollouts, max(1, batch_size)):
        seeds = list(range(i, min(i + batch_size, n_rollouts)))
        recs = roll_and_score_batch(model, tokenizer, seeds, n_ctx, max_k, K, device, window=window)
        for src, events in recs.items():
            for k, rec in events:
                for raw, name in metric_name.items():
                    acc[src][name][k].append(rec[raw])
    out = {}
    for src in ("model", "copy_last", "oracle"):
        out[src] = {name: {k: float(np.mean(v)) for k, v in sorted(acc[src][name].items())}
                    for name in metric_name.values()}
    out["chance"] = chance_levels()
    return out


def main() -> None:
    """Load a checkpoint, run v2 recall, write JSON curves (plot_recall.py-compatible)."""
    import argparse
    import json
    from models.dynamics_model import DynamicsModel, DynamicsModelConfig
    from models.tokenizer import AutoEncoder, AutoEncoderConfig

    root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description="GridWorldV2 recall on a checkpoint -> JSON curves.")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output JSON (default: outputs/recall/recallv2_<checkpoint-stem>.json).")
    ap.add_argument("--n-ctx", type=int, default=4)
    ap.add_argument("--max-k", type=int, required=True)
    ap.add_argument("--n-rollouts", type=int, default=64)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, _ = _load_checkpoint(args.tokenizer, AutoEncoder, AutoEncoderConfig, device)
    for p in tok.parameters():
        p.requires_grad_(False)
    model, cfg = _load_checkpoint(args.checkpoint, DynamicsModel, DynamicsModelConfig, device)
    window = args.window or cfg.max_temporal_length

    print(f"recall(v2): n_ctx={args.n_ctx} max_k={args.max_k} n_rollouts={args.n_rollouts} "
          f"K={args.K} window={window} n_memory={cfg.n_memory} device={device}")
    res = recall(model, tok, n_ctx=args.n_ctx, max_k=args.max_k, n_rollouts=args.n_rollouts,
                 K=args.K, device=device, window=window, batch_size=args.batch_size)
    res["meta"] = {
        "env": "gridworldv2",
        "checkpoint": str(args.checkpoint), "tokenizer": str(args.tokenizer),
        "n_ctx": args.n_ctx, "max_k": args.max_k, "n_rollouts": args.n_rollouts, "K": args.K,
        "n_memory": cfg.n_memory, "window": window, "native_window": cfg.max_temporal_length,
    }
    out = args.out or (root / "outputs" / "recall" / f"recallv2_{args.checkpoint.stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
    print("  k :  model  copy_last  oracle   (position_acc)")
    for k in sorted(res["model"]["position_acc"], key=int):
        m, c, o = (res[s]["position_acc"][k] for s in ("model", "copy_last", "oracle"))
        print(f"  {int(k):>3}: {m:6.3f}   {c:6.3f}   {o:6.3f}")


if __name__ == "__main__":
    main()
