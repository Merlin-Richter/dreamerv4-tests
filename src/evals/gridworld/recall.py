"""GridWorld recall metrics — what to score, and how, on the discrete memory env.

A "reveal event" is a curtain-up frame immediately preceded by an occluded run of length k
(the moment the curtain lifts after hiding the square for k steps). To predict that frame
correctly the model must have RETAINED the hidden square through the occlusion. Because the env
is discrete, every quantity is scored exactly:

  PRIMARY (the headline curve, vs occlusion length k):
    * position_acc[k]  = P(predicted square cell == true cell)          -- exact 6x6 recall
    * position_dist[k] = mean Chebyshev cell-distance error             -- graded partial credit
    * color_acc[k]     = P(predicted square color == true color)        -- 4-way identity recall

  SECONDARY / CONFOUND CHECK:
    * bg_acc[k]        = P(predicted background color == true bg)       -- easy static memory

  DIAGNOSTIC:
    * reflection split of position_acc: bounced-during-occlusion vs not -- did it learn the walls,
      or just extrapolate ballistically?
    * margin[k]        = readout confidence (top1-top2 distance-from-bg) -- flags smeared/halluc. preds

The metric core is PREDICTOR-AGNOSTIC: it scores any source of reveal-frame pixels. Baselines and
the ceiling are therefore just alternative frame sources fed to the SAME scorer:
    * oracle (true frames)        -> ceiling, must be position_acc==1.0  (instrument self-test)
    * copy-last (square frozen at its last observed cell) -> the NO-MEMORY reference; it decays as
      the true square moves away, so beating it is the operational definition of "has memory."
    * chance                      -> 1/36 position, 1/4 color.
The matched-horizon open-rollout control (model run curtain-UP for the same horizon) separates
"can't track motion even in the clear" from "memory lost past the window" — it needs the model, so
it lives in the Eval adapter, not this pure core.

All colors BGR. `states[t] = [col,row,dcol,drow,curtain]` (see GridWorldEnv).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/
from envs.gridworld import GRID_N, PALETTE, GridWorldEnv, make_grid_background, stamp_square  # noqa: E402
from evals.gridworld.readout import read_square  # noqa: E402

_PAL = list(PALETTE.values())

# ---------------------------------------------------------------------------
# Graded position credit (Merlin, D-040): exact = full score, partial credit MINIMAL
# and gone by distance 3. Keyed on Chebyshev cell-distance d between predicted and
# true cell. Each step of distance quarters the credit, then it's cut to 0 at d>=3.
# Super simple to read + change; the strict exact-match accuracy is reported alongside.
# ---------------------------------------------------------------------------
POSITION_CREDIT = {0: 1.0, 1: 0.25, 2: 0.0625}  # any distance not listed (>=3) -> 0.0


def position_credit(d: int) -> float:
    """Graded position score for a Chebyshev cell-distance `d` (1.0 exact ... 0.0 at d>=3)."""
    return POSITION_CREDIT.get(int(d), 0.0)


# ---------------------------------------------------------------------------
# Reveal events
# ---------------------------------------------------------------------------

def find_reveal_events(curtain: np.ndarray) -> list[dict]:
    """Each event: {reveal_t, k, last_visible_t}. A reveal is curtain 1->0; k = preceding run of
    1s; last_visible_t = the curtain-up frame just before that run (must exist, so motion is
    observed before occlusion)."""
    curtain = np.asarray(curtain).astype(int)
    events = []
    T = len(curtain)
    for t in range(1, T):
        if curtain[t] == 0 and curtain[t - 1] == 1:
            k = 0
            while t - 1 - k >= 0 and curtain[t - 1 - k] == 1:
                k += 1
            last_vis = t - 1 - k
            if last_vis >= 0:
                events.append({"reveal_t": t, "k": k, "last_visible_t": last_vis})
    return events


# ---------------------------------------------------------------------------
# Per-episode scoring
# ---------------------------------------------------------------------------

def score_episode(pred_frames: np.ndarray, states: np.ndarray, colors: np.ndarray,
                  curtain: np.ndarray) -> list[dict]:
    """Score every reveal event in one episode. `pred_frames` are the model's (or a baseline's)
    frames for this episode; only the reveal-frame indices are read. Returns per-event records."""
    sq_color_idx = int(colors[1])
    bg_color_idx = int(colors[0])
    recs = []
    for ev in find_reveal_events(curtain):
        t, k, lv = ev["reveal_t"], ev["k"], ev["last_visible_t"]
        rd = read_square(pred_frames[t])
        true_col, true_row = int(states[t, 0]), int(states[t, 1])
        dist = max(abs(rd["col"] - true_col), abs(rd["row"] - true_row))  # Chebyshev cell-distance
        # bounced during occlusion? direction at last-visible vs at reveal.
        dir_lv = (states[lv, 2], states[lv, 3])
        dir_rv = (states[t, 2], states[t, 3])
        bounced = dir_lv != dir_rv
        recs.append({
            "k": k,
            "pos_correct": int(dist == 0),               # strict exact-match (chance 1/36)
            "pos_score": position_credit(dist),          # graded headline (exact=1, falls to 0 by d3)
            "pos_dist": dist,                            # raw Chebyshev (diagnostic)
            "color_correct": int(rd["color_idx"] == sq_color_idx),  # ball 4-way
            "bg_correct": int(rd["bg_idx"] == bg_color_idx),        # background 4-way
            "margin": rd["margin"],
            "bounced": bool(bounced),
        })
    return recs


# ---------------------------------------------------------------------------
# Aggregation -> curves vs k
# ---------------------------------------------------------------------------

def aggregate(records: list[dict]) -> dict:
    """Bucket per-event records by k into the headline curves + standard errors + a reflection split.

    `*_se` are standard errors of the mean (std/sqrt(n)) per k — so a k-bin with few events shows a
    large SE and is visibly untrustworthy. With enough reveal events each k's score is a real
    "chance of keeping position", not luck on a handful of rollouts (Merlin, D-040)."""
    by_k = defaultdict(list)
    for r in records:
        by_k[r["k"]].append(r)

    def curve(metric, subset=None):
        out = {}
        for k, rs in sorted(by_k.items()):
            vals = [r[metric] for r in rs if (subset is None or subset(r))]
            if vals:
                out[k] = float(np.mean(vals))
        return out

    def curve_se(metric):
        out = {}
        for k, rs in sorted(by_k.items()):
            vals = np.array([r[metric] for r in rs], dtype=float)
            if len(vals):
                out[k] = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan")
        return out

    return {
        "n_by_k": {k: len(rs) for k, rs in sorted(by_k.items())},
        # PRIMARY
        "position_score": curve("pos_score"),       # graded headline (D-040)
        "position_score_se": curve_se("pos_score"),
        "position_acc": curve("pos_correct"),       # strict exact-match (chance 1/36)
        "position_acc_se": curve_se("pos_correct"),
        "color_acc": curve("color_correct"),        # ball 4-way
        "color_acc_se": curve_se("color_correct"),
        "bg_acc": curve("bg_correct"),              # background 4-way
        # DIAGNOSTIC
        "position_dist": curve("pos_dist"),
        "margin": curve("margin"),
        "position_score_bounced": curve("pos_score", lambda r: r["bounced"]),
        "position_score_straight": curve("pos_score", lambda r: not r["bounced"]),
        "n_events": len(records),
    }


# ---------------------------------------------------------------------------
# Baseline / ceiling frame sources (no model needed)
# ---------------------------------------------------------------------------

def oracle_frames(states: np.ndarray, colors: np.ndarray, curtain: np.ndarray) -> np.ndarray:
    """Render the TRUE revealed frame at every step (ceiling: position_acc must be 1.0)."""
    bg = _PAL[int(colors[0])]
    sq = _PAL[int(colors[1])]
    T = len(states)
    frames = np.empty((T, GridWorldEnv.img_size, GridWorldEnv.img_size, 3), dtype=np.uint8)
    base = make_grid_background(bg)
    for t in range(T):
        f = base.copy()
        stamp_square(f, int(states[t, 0]), int(states[t, 1]), sq)
        frames[t] = f
    return frames


def copylast_frames(states: np.ndarray, colors: np.ndarray, curtain: np.ndarray) -> np.ndarray:
    """No-memory baseline: during/after each occluded run, freeze the square at its LAST OBSERVED
    cell (the cell it was in at the last curtain-up frame). Reveals are scored against this frozen
    guess -> the accuracy a memoryless model would get, decaying as the true square moves away."""
    bg = _PAL[int(colors[0])]
    sq = _PAL[int(colors[1])]
    T = len(states)
    base = make_grid_background(bg)
    frames = np.empty((T, GridWorldEnv.img_size, GridWorldEnv.img_size, 3), dtype=np.uint8)
    last_col, last_row = int(states[0, 0]), int(states[0, 1])
    for t in range(T):
        # Render with the belief CARRIED IN (what a memoryless model knows before seeing frame t)...
        f = base.copy()
        stamp_square(f, last_col, last_row, sq)
        frames[t] = f
        # ...THEN, if this frame is observed, update the belief for subsequent steps. (Update after
        # stamping so the reveal frame is scored against the pre-occlusion belief, not the answer.)
        if curtain[t] == 0:
            last_col, last_row = int(states[t, 0]), int(states[t, 1])
    return frames


def chance_levels() -> dict:
    """Analytic floors for reference lines. `position_score` is the graded credit a UNIFORM-random
    predicted cell earns, averaged over all true cells (edge cells have fewer close neighbours)."""
    cells = [(c, r) for r in range(GRID_N) for c in range(GRID_N)]
    tot = 0.0
    for tc, tr in cells:
        for pc, pr in cells:
            tot += position_credit(max(abs(pc - tc), abs(pr - tr)))
    pos_score_chance = tot / (len(cells) ** 2)
    return {
        "position_score": pos_score_chance,
        "position_acc": 1.0 / (GRID_N * GRID_N),
        "color_acc": 1.0 / len(_PAL),
        "bg_acc": 1.0 / len(_PAL),
    }


def flatten_for_wandb(agg: dict, prefix: str = "gw") -> dict:
    """Flatten the per-k headline curves into scalar keys for a periodic W&B eval (D-040). E.g.
    {"gw/position_score/k=4": 0.31, ...}. Ready for a during-training hook (wiring TBD with Merlin)."""
    out = {}
    for metric in ("position_score", "position_acc", "color_acc", "bg_acc"):
        for k, v in agg.get(metric, {}).items():
            out[f"{prefix}/{metric}/k={k}"] = v
    out[f"{prefix}/n_events"] = agg.get("n_events", 0)
    return out


# ---------------------------------------------------------------------------
# Dataset-level convenience
# ---------------------------------------------------------------------------

def evaluate_dataset(frame_source, frames, states, actions, colors, n_episodes=None) -> dict:
    """Run `frame_source(states_i, colors_i, curtain_i) -> pred_frames_i` over the dataset and
    aggregate. `frame_source` is a baseline generator here; the model adapter passes the decoded
    rollout. `frames` is unused by baselines but kept in the signature for the model adapter."""
    n = n_episodes or len(states)
    all_recs = []
    for i in range(n):
        pred = frame_source(states[i], colors[i], actions[i])
        all_recs += score_episode(pred, states[i], colors[i], actions[i])
    return aggregate(all_recs)


def _print_curve(name: str, agg: dict, metric: str, se_key: str | None = None) -> None:
    ch = chance_levels().get(metric)
    line = f"  {name:<14}" + ("  chance=%.3f" % ch if ch is not None else "")
    print(line)
    ks = sorted(agg[metric])
    cells = "  ".join(
        f"k{k}:{agg[metric][k]:.2f}" + (f"+/-{agg[se_key][k]:.02f}" if se_key and k in agg.get(se_key, {}) else "")
        + f"(n{agg['n_by_k'][k]})"
        for k in ks
    )
    print("    " + cells)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run GridWorld recall baselines on a dataset.")
    ap.add_argument("--frames", default="gridworld.npy", help="dataset .npy (uses _states/_colors/_actions)")
    ap.add_argument("--n_episodes", type=int, default=None, help="limit episodes (default: all)")
    args = ap.parse_args()
    stem = args.frames[:-4] if args.frames.endswith(".npy") else args.frames
    states = np.load(stem + "_states.npy")
    colors = np.load(stem + "_colors.npy")
    actions = np.load(stem + "_actions.npy")
    n = args.n_episodes or len(states)
    print(f"GridWorld recall baselines on {args.frames}  ({n} episodes)")
    print(f"chance: {chance_levels()}\n")
    for name, src in (("oracle", oracle_frames), ("copy-last", copylast_frames)):
        agg = evaluate_dataset(src, None, states, actions, colors, n_episodes=n)
        print(f"== {name}  ({agg['n_events']} reveal events) ==")
        _print_curve(name, agg, "position_score", "position_score_se")
        _print_curve(name, agg, "position_acc", "position_acc_se")
        _print_curve(name, agg, "color_acc", "color_acc_se")
        _print_curve(name, agg, "bg_acc")
        print()
