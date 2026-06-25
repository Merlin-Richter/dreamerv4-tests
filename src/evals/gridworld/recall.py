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
def roll_and_score(model, tokenizer, seed: int, n_ctx: int, max_k: int, K: int,
                   device) -> dict:
    """One long occluded rollout for one env seed; returns per-event records for model/oracle/copy_last.

    Each record is ``(k, {pos_correct, pos_score, color_correct})``. The reveal branch is read-only and
    never corrupts the main occluded rollout's carried memory / latent window.
    """
    tok_w = _tokenizer_window(tokenizer)
    env = GridWorldEnv().reset(seed)

    # Context: n_ctx REVEALED frames the model observes (action 0 = revealed).
    cframes, s = [], None
    for _ in range(n_ctx):
        f, s = env.step(0)
        cframes.append(f)
    last_col, last_row = int(s[0]), int(s[1])             # last OBSERVED cell (copy_last belief)
    colors = (COLOR_NAMES.index(env.bg_name), COLOR_NAMES.index(env.color_name))

    cfx = torch.from_numpy(np.stack(cframes).astype(np.float32) / 255.0).unsqueeze(0).to(device)
    ctx_lat = tokenizer.encoder(cfx)                      # (1, n_ctx, L, D)
    ctx_act = torch.zeros((1, n_ctx), dtype=torch.long, device=device)
    state = model.rollout_init(ctx_lat, ctx_act, K)
    lat_buf = ctx_lat[:, -(tok_w - 1):]                  # rolling latents for the decode window

    check = set(_check_ks(max_k))
    a0 = torch.zeros((1,), dtype=torch.long, device=device)  # reveal action
    a1 = torch.ones((1,), dtype=torch.long, device=device)   # occlude action
    recs = {"model": [], "oracle": [], "copy_last": []}

    for k in range(1, max_k + 1):
        f_true, s_true = env.step(0)                      # advance physics; revealed render = oracle truth
        tcol, trow = int(s_true[0]), int(s_true[1])
        if k in check:
            z_rev = model.rollout_step(state, a0, commit=False)   # read-only reveal belief at this tick
            win = torch.cat((lat_buf, z_rev), dim=1)[:, -tok_w:]
            dec = tokenizer.decoder(win)[0, -1].clamp(0, 1).cpu().float().numpy()
            pred = (dec * 255.0).round().astype(np.uint8)
            recs["model"].append((k, score_reveal(pred, (tcol, trow), colors)))
            recs["oracle"].append((k, score_reveal(f_true, (tcol, trow), colors)))
            cl_dist = max(abs(last_col - tcol), abs(last_row - trow))
            recs["copy_last"].append((k, {
                "pos_correct": int(cl_dist == 0),
                "pos_score": position_credit(cl_dist),
                "color_correct": 1,                       # colour is static -> a memoryless guess knows it
            }))
        z_occ = model.rollout_step(state, a1, commit=True)        # commit the occluded tick
        lat_buf = torch.cat((lat_buf, z_occ), dim=1)[:, -(tok_w - 1):]
    return recs


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
           K: int = 4, device="cpu") -> dict:
    """Run ``n_rollouts`` occluded rollouts and return per-k recall curves.

    Returns ``{"model": {position_acc, position_score, color_acc each {k: v}}, "copy_last": …,
    "oracle": …, "chance": {position_acc, position_score, color_acc}}``.
    """
    metric_name = {"pos_correct": "position_acc", "pos_score": "position_score",
                   "color_correct": "color_acc"}
    # acc[src][metric][k] -> list of per-rollout values
    acc = {src: defaultdict(lambda: defaultdict(list)) for src in ("model", "oracle", "copy_last")}
    for seed in range(n_rollouts):
        recs = roll_and_score(model, tokenizer, seed, n_ctx, max_k, K, device)
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
