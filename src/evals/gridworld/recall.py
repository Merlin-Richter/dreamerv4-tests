"""GridWorld memory recall eval (spec: specs/evals/gridworld/recall.md).

The question: how well and how long can the model model the environment when it can't see the state?

We drive a GridWorldEnv: show the model ``n_ctx`` REVEALED ground-truth frames as context, then roll a
long OCCLUDED rollout (action=1 every step). Periodically we branch a READ-ONLY reveal (action=0) off
the carried state, decode it, and read the square out of the predicted frame — the model's *belief* of
the hidden square at that point — and score it against the env's independently-advancing true state.
The reveal branch is discarded and the occluded rollout continues, so ONE rollout scores every k.

`n_ctx`: length of the observed context window.   `K`: shortcut diffusion steps.   `k`: occlusion length.

ALIGNMENT (a result-defining choice — flagged to Merlin). In a single branching rollout the reveal at
occlusion-length ``k`` is the read-only SIBLING of the k-th occluded tick: it is predicted at the same
absolute rollout position as that tick (branched *before* the occluded frame is committed) and scored
against the env's true square at that tick. So the belief at k reflects memory carried through the
context plus the first k-1 committed occluded frames — i.e. "the square has been hidden for k
consecutive ticks; reveal the k-th and check." This is the only self-consistent single-rollout reading
(one env.step per k, branch strictly read-only). It is applied identically to the model and the
baselines, so the curves are comparable; the absolute k-axis carries this off-by-one convention.

Baselines / ceiling, same readout:
  * oracle    — read the square out of the TRUE revealed frame (ceiling; position_acc must be ~1.0,
                an instrument self-test of the readout).
  * copy_last — the NO-MEMORY reference: freeze the square at its last observed cell; it decays as the
                true square moves away, so beating it is the operational definition of "has memory".
  * chance    — analytic floors (1/36 position, 1/4 colour, graded position-credit chance).

All colours BGR (env channel-order contract). Position readout is channel-order-independent.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/
from envs.gridworld import GRID_N, PALETTE, GridWorldEnv  # noqa: E402
from evals.gridworld.readout import read_square  # noqa: E402

COLOR_NAMES = list(PALETTE.keys())

# Graded position credit (D-040): exact = full credit, falling to 0 by Chebyshev cell-distance 3.
POSITION_CREDIT = {0: 1.0, 1: 0.25, 2: 0.0625}


def position_credit(d: int) -> float:
    """Graded position score for a Chebyshev cell-distance ``d`` (1.0 exact ... 0.0 at d>=3)."""
    return POSITION_CREDIT.get(int(d), 0.0)


def _check_ks(max_k: int) -> list[int]:
    """The occlusion lengths scored: 2,4,..,14 then multiples of 8 up to max_k (incl. max_k)."""
    ks = set(range(2, 15, 2)) | {8 * i for i in range(2, max_k // 8 + 1)} | {max_k}
    return sorted(k for k in ks if 1 <= k <= max_k)


def _tokenizer_window(tokenizer) -> int:
    """The frozen tokenizer's temporal window (its RoPE table length) — the most frames it can
    encode/decode at once. Cached on the object."""
    w = getattr(tokenizer, "_tok_window", None)
    if w is None:
        w = 1 << 30
        for m in tokenizer.modules():
            cos = getattr(m, "cos", None)
            if isinstance(cos, torch.Tensor) and cos.dim() >= 1:
                w = min(w, cos.shape[0])
        tokenizer._tok_window = w
    return w


def score_reveal(pred_frame: np.ndarray, true_cell: tuple[int, int],
                 colors: tuple[int, int]) -> dict:
    """Score one predicted reveal frame against the env's true (col,row) + square colour index.

    pred_frame: (H,W,3) uint8 BGR.   true_cell: (col,row).   colors: (bg_idx, sq_idx) PALETTE order.
    Returns {pos_correct (exact 6x6), pos_score (graded), color_correct (4-way)}.
    """
    _, sq_idx = colors
    rd = read_square(pred_frame)
    dist = max(abs(rd["col"] - true_cell[0]), abs(rd["row"] - true_cell[1]))  # Chebyshev cells
    return {
        "pos_correct": int(dist == 0),
        "pos_score": position_credit(dist),
        "color_correct": int(rd["color_idx"] == sq_idx),
    }


@torch.no_grad()
def roll_and_score_batch(model, tokenizer, seeds, n_ctx: int, max_k: int, K: int,
                         device, window: int = None) -> dict:
    """Batched: run ``len(seeds)`` occluded rollouts in PARALLEL (B=len(seeds)) and return per-event
    records for model/oracle/copy_last. Identical scoring/alignment to the single-seed path, just
    vectorised over the batch so the model forwards + tokenizer decode run at B>1 (the GPU-friendly
    path). Only the cheap per-env work (numpy env.step, closed-form readout) stays elementwise.

    Each record is ``(k, {pos_correct, pos_score, color_correct})``; there are B records per checked k.
    The reveal branch is read-only and never corrupts the main occluded rollout's carried state.
    ``window`` (total frames) FORCES a shorter sliding window than training; None = native.
    """
    tok_w = _tokenizer_window(tokenizer)
    max_ctx = None if window is None else max(1, window - 1)
    B = len(seeds)
    envs = [GridWorldEnv().reset(s) for s in seeds]

    # Context: n_ctx REVEALED frames per env (action 0 = revealed).
    cframes, last = [[] for _ in range(B)], [None] * B
    for b, env in enumerate(envs):
        s = None
        for _ in range(n_ctx):
            f, s = env.step(0)
            cframes[b].append(f)
        last[b] = (int(s[0]), int(s[1]))                  # last OBSERVED cell (copy_last belief)
    colors = [(COLOR_NAMES.index(e.bg_name), COLOR_NAMES.index(e.color_name)) for e in envs]

    cfx = torch.from_numpy(np.stack([np.stack(c) for c in cframes]).astype(np.float32) / 255.0).to(device)
    ctx_lat = tokenizer.encoder(cfx)                      # (B, n_ctx, L, D)
    ctx_act = torch.zeros((B, n_ctx), dtype=torch.long, device=device)
    state = model.rollout_init(ctx_lat, ctx_act, K, max_ctx=max_ctx)
    lat_buf = ctx_lat[:, -(tok_w - 1):]                  # rolling latents for the decode window

    check = set(_check_ks(max_k))
    a0 = torch.zeros((B,), dtype=torch.long, device=device)  # reveal action
    a1 = torch.ones((B,), dtype=torch.long, device=device)   # occlude action
    recs = {"model": [], "oracle": [], "copy_last": []}

    for k in range(1, max_k + 1):
        true_cells, f_true = [], []
        for env in envs:
            f, s = env.step(0)                            # advance physics; revealed render = oracle truth
            true_cells.append((int(s[0]), int(s[1])))
            f_true.append(f)
        if k in check:
            z_rev = model.rollout_step(state, a0, commit=False)   # read-only reveal belief at this tick
            win = torch.cat((lat_buf, z_rev), dim=1)[:, -tok_w:]
            dec = tokenizer.decoder(win)[:, -1].clamp(0, 1).cpu().float().numpy()  # (B,H,W,3)
            pred = (dec * 255.0).round().astype(np.uint8)
            for b in range(B):
                tcol, trow = true_cells[b]
                recs["model"].append((k, score_reveal(pred[b], (tcol, trow), colors[b])))
                recs["oracle"].append((k, score_reveal(f_true[b], (tcol, trow), colors[b])))
                cl_dist = max(abs(last[b][0] - tcol), abs(last[b][1] - trow))
                recs["copy_last"].append((k, {
                    "pos_correct": int(cl_dist == 0),
                    "pos_score": position_credit(cl_dist),
                    "color_correct": 1,                   # colour is static -> a memoryless guess knows it
                }))
        z_occ = model.rollout_step(state, a1, commit=True)        # commit the occluded tick
        lat_buf = torch.cat((lat_buf, z_occ), dim=1)[:, -(tok_w - 1):]
    return recs


@torch.no_grad()
def roll_and_score(model, tokenizer, seed: int, n_ctx: int, max_k: int, K: int,
                   device, window: int = None) -> dict:
    """Single-seed convenience wrapper over ``roll_and_score_batch`` (B=1)."""
    return roll_and_score_batch(model, tokenizer, [seed], n_ctx, max_k, K, device, window=window)


def chance_levels() -> dict:
    """Analytic floors. position_score is the graded credit a uniform-random cell earns, averaged
    over all true cells."""
    cells = [(c, r) for r in range(GRID_N) for c in range(GRID_N)]
    tot = sum(position_credit(max(abs(pc - tc), abs(pr - tr)))
              for tc, tr in cells for pc, pr in cells)
    return {
        "position_acc": 1.0 / (GRID_N * GRID_N),
        "position_score": tot / (len(cells) ** 2),
        "color_acc": 1.0 / len(COLOR_NAMES),
    }


@torch.no_grad()
def recall(model, tokenizer, *, n_ctx: int = 4, max_k: int, n_rollouts: int = 64,
           K: int = 4, device="cpu", window: int = None, batch_size: int = 64) -> dict:
    """Run ``n_rollouts`` occluded rollouts and return per-k recall curves.

    Rollouts run BATCHED: seeds are processed in chunks of ``batch_size`` as one B-batch through the
    model/decoder (the GPU-friendly path). Lower ``batch_size`` if GPU memory is tight.
    ``window`` (total frames) forces a shorter sliding context window than the model trained with;
    None = the model's native ``max_temporal_length``.
    Returns ``{"model": {position_acc, position_score, color_acc each {k: v}}, "copy_last": …,
    "oracle": …, "chance": {position_acc, position_score, color_acc}}``.
    """
    metric_name = {"pos_correct": "position_acc", "pos_score": "position_score",
                   "color_correct": "color_acc"}
    # acc[src][metric][k] -> list of per-rollout values
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


# --------------------------------------------------------------------------- CLI (run + dump JSON)
def _load_checkpoint(path, cls_model, cls_cfg, device):
    """Load a {config, model_state_dict} payload (same convention as interactive/play_dynamics.py).

    NOTE: this small loader is duplicated in sheets.py and play_dynamics.py — see the follow-up to
    extract a shared evals/gridworld loader module.
    """
    from dataclasses import fields
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = cls_cfg(**{k: v for k, v in payload["config"].items()
                     if k in {f.name for f in fields(cls_cfg)}})
    model = cls_model(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, cfg


def main() -> None:
    """Run the recall eval on a checkpoint and write a JSON of the per-k curves (consumed by
    plot_recall.py). Keeps recall() pure: this is just the load + run + serialize shell."""
    import argparse
    import json
    from models.dynamics_model import DynamicsModel, DynamicsModelConfig
    from models.tokenizer import AutoEncoder, AutoEncoderConfig

    root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description="Run GridWorld recall on a checkpoint -> JSON curves.")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Dynamics checkpoint.")
    ap.add_argument("--tokenizer", type=Path, required=True, help="Frozen tokenizer checkpoint.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output JSON (default: outputs/recall/recall_<checkpoint-stem>.json).")
    ap.add_argument("--n-ctx", type=int, default=4)
    ap.add_argument("--max-k", type=int, required=True)
    ap.add_argument("--n-rollouts", type=int, default=64)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--window", type=int, default=None,
                    help="Force the sliding context window to this many frames (default: the model's "
                         "max_temporal_length). E.g. 8 to probe memory at a shorter window than 16.")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Rollouts run batched in chunks of this size (B). Lower if GPU memory is tight.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, _ = _load_checkpoint(args.tokenizer, AutoEncoder, AutoEncoderConfig, device)
    for p in tok.parameters():
        p.requires_grad_(False)
    model, cfg = _load_checkpoint(args.checkpoint, DynamicsModel, DynamicsModelConfig, device)
    window = args.window or cfg.max_temporal_length            # total frames in the sliding window

    print(f"recall: n_ctx={args.n_ctx} max_k={args.max_k} n_rollouts={args.n_rollouts} "
          f"K={args.K} window={window} batch_size={args.batch_size} n_memory={cfg.n_memory} "
          f"device={device}")
    res = recall(model, tok, n_ctx=args.n_ctx, max_k=args.max_k, n_rollouts=args.n_rollouts,
                 K=args.K, device=device, window=window, batch_size=args.batch_size)
    res["meta"] = {
        "checkpoint": str(args.checkpoint), "tokenizer": str(args.tokenizer),
        "n_ctx": args.n_ctx, "max_k": args.max_k, "n_rollouts": args.n_rollouts, "K": args.K,
        "n_memory": cfg.n_memory, "window": window, "native_window": cfg.max_temporal_length,
    }
    out = args.out or (root / "outputs" / "recall" / f"recall_{args.checkpoint.stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
    # quick console digest (model vs copy_last on exact position)
    print("  k :  model  copy_last  oracle   (position_acc)")
    for k in sorted(res["model"]["position_acc"], key=int):
        m, c, o = (res[s]["position_acc"][k] for s in ("model", "copy_last", "oracle"))
        print(f"  {int(k):>3}: {m:6.3f}   {c:6.3f}   {o:6.3f}")


if __name__ == "__main__":
    main()
