"""Compare GridWorld recall curves across runs (spec: specs/evals/gridworld/plot_recall.md).

Reads one or more recall-result JSONs (written by `recall.py`'s CLI) and renders a 2x2 comparison
figure: per-k model curves for every series overlaid, with copy_last / oracle / chance references from
the FIRST series, and a vertical marker at the latent-window edge. The picture EXP-027/028/030 made by
hand, now first-class — eval each checkpoint once to JSON, then plot/compare freely without re-evaling.

This is a LOCAL post-hoc analysis tool: it uses matplotlib (unlike sheets.py, which is cv2-only for the
cluster venv). Plotting is not part of training; run it locally on the JSONs you pulled back.

  python -u src/evals/gridworld/plot_recall.py \
    --series "vanilla|outputs/recall/recall_dynamics_vanilla.json|tab:red" \
    --series "FF9 (carry)|outputs/recall/recall_dynamics_ff9.json|tab:green" \
    --out outputs/recall/compare.png

Each --series is "label|path|color" ('|' separated so matplotlib colours like 'tab:red' work; color is
optional). Paths are resolved relative to the current working directory. Metrics are auto-detected from
the first series, so this works with the current 3-metric recall and any future additions (e.g. bg_acc).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Metric key -> panel title, in display order. Only those present in the first series are drawn.
METRIC_TITLES = {
    "position_acc": "exact position (chance 1/36)",
    "position_score": "graded position",
    "color_acc": "ball colour (1/4)",
    "bg_acc": "bg colour (1/4)",
}


def _curve(metric_dict: dict):
    """(xs, ys) for a {k: value} dict with integer-string keys, sorted by k. Empty -> ([], [])."""
    ks = sorted((k for k in metric_dict if str(k).lstrip("-").isdigit()), key=int)
    return [int(k) for k in ks], [metric_dict[k] for k in ks]


def _parse_series(spec: str):
    """'label|path|color' -> (label, dict, color|None). color optional."""
    parts = spec.split("|")
    if len(parts) == 2:
        (label, path), color = parts, None
    elif len(parts) == 3:
        label, path, color = parts
    else:
        raise SystemExit(f"--series must be 'label|path' or 'label|path|color', got: {spec!r}")
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"series JSON not found: {p} (paths are relative to the current directory)")
    return label, json.loads(p.read_text()), (color or None)


def plot_recall(series, out_path, *, window=None, title=None):
    """Render the 2x2 comparison figure. series: list of (label, result_dict, color|None)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = series[0][1]
    chance = ref.get("chance", {})
    if window is None:
        window = ref.get("meta", {}).get("window")   # total frames in the sliding window
    metrics = [(m, t) for m, t in METRIC_TITLES.items() if m in ref.get("model", {})]
    if not metrics:
        raise SystemExit("no known metrics found in the first series' 'model' block.")

    ncols = 1 if len(metrics) == 1 else 2
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), squeeze=False)
    flat = axes.ravel()

    for ax, (m, mtitle) in zip(flat, metrics):
        for label, j, color in series:
            xs, ys = _curve(j.get("model", {}).get(m, {}))
            if xs:
                ax.plot(xs, ys, label=label, marker="o", ms=3, c=color)
        # references from the FIRST series only (shared baselines/ceiling)
        for nm, key, stl in [("copy-last", "copy_last", dict(c="gray", ls=":", marker="x", ms=3)),
                             ("oracle", "oracle", dict(c="k", ls="--", lw=1))]:
            xs, ys = _curve(ref.get(key, {}).get(m, {}))
            if xs:
                ax.plot(xs, ys, label=nm, **stl)
        if m in chance:
            ax.axhline(chance[m], c="lightgray", lw=1, label="chance")
        if window is not None:
            edge = window - 1                         # k where the model has rolled a full window past ctx
            ax.axvline(edge, c="purple", lw=0.8, alpha=0.5)
            ax.text(edge + 0.2, 0.02, f"window={window}", color="purple", fontsize=6)
        ax.set_title(mtitle)
        ax.set_xlabel("occlusion length k")
        ax.set_ylabel(m)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in flat[len(metrics):]:                       # hide unused panels (e.g. 3 metrics in a 2x2)
        ax.axis("off")

    if title is None:
        meta = ref.get("meta", {})
        title = (f"GridWorld recall: {len(series)} run(s) "
                 f"(n_rollouts={meta.get('n_rollouts', '?')}, n_ctx={meta.get('n_ctx', '?')})")
    fig.suptitle(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    print(f"wrote {out_path}  ({len(metrics)} panels, {len(series)} series)")


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description="Overlay GridWorld recall curves from result JSONs.")
    ap.add_argument("--series", action="append", required=True,
                    help="'label|path|color' (repeatable; color optional, e.g. tab:green).")
    ap.add_argument("--out", type=Path, default=root / "outputs" / "recall" / "compare.png")
    ap.add_argument("--window", type=int, default=None,
                    help="Sliding-window size in frames; marker drawn at k=window-1 "
                         "(default: first series' meta.window).")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    series = [_parse_series(s) for s in args.series]
    plot_recall(series, args.out, window=args.window, title=args.title)


if __name__ == "__main__":
    main()
