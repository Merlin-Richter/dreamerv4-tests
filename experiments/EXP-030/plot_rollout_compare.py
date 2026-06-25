"""Overlay FF9 rollout-training recall curves on the EXP-027/028 vanilla + FF9 baselines.

Usage:
  python experiments/EXP-030/plot_rollout_compare.py \
     --series "vanilla:../EXP-028/recall_env_vanilla.json:tab:red" \
     --series "FF9 (no rollout):../EXP-028/recall_env_ff9.json:tab:green" \
     --series "FF9+rollout h24 (relay):recall_env_ff9roll_m24_relay.json:tab:blue" \
     --series "FF9+rollout h44 (relay):recall_env_ff9roll_d44_relay.json:tab:purple" \
     --out compare_rollout.png

Each --series is "label:path:color" (path relative to this dir). copy-last/oracle/chance come from
the FIRST series' json. Plots position_acc / position_score / color_acc / bg_acc vs k. The window
edge (k=15 for the 16-frame models; the vanilla-w32 control would be 31) is marked.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--series", action="append", required=True, help="label:path:color")
ap.add_argument("--out", default="compare_rollout.png")
ap.add_argument("--window", type=int, default=15, help="window-edge marker (k); 15 for window-16.")
args = ap.parse_args()
D = Path(__file__).resolve().parent

series = []
for s in args.series:
    label, path, color = s.rsplit(":", 2)
    j = json.loads((D / path).read_text())
    series.append((label, j, color))
ref = series[0][1]
ch = ref["chance"]

metrics = [("position_acc", "exact position (chance 1/36)"), ("position_score", "graded position"),
           ("color_acc", "ball colour (1/4)"), ("bg_acc", "bg colour (1/4)")]
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, (m, title) in zip(axes.ravel(), metrics):
    for label, j, color in series:
        d = j["model"].get(m, {})
        ks = sorted(d, key=int)
        if ks:
            ax.plot([int(k) for k in ks], [d[k] for k in ks], label=label, marker="o", ms=3, c=color)
    # references from the first series
    for nm, key, stl in [("copy-last", "copy_last", dict(c="gray", ls=":", marker="x", ms=3)),
                         ("oracle", "oracle", dict(c="k", ls="--", lw=1))]:
        d = ref.get(key, {})
        ks = sorted(d, key=int)
        if ks:
            ax.plot([int(k) for k in ks], [d[k] for k in ks], label=nm, **stl)
    if m in ch:
        ax.axhline(ch[m], c="lightgray", lw=1)
    ax.axvline(args.window, c="purple", lw=0.8, alpha=0.5)
    ax.text(args.window + 0.2, 0.02, "window", color="purple", fontsize=6)
    ax.set_title(title); ax.set_xlabel("occlusion length k"); ax.set_ylabel(m)
    ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.suptitle(f"GridWorld recall (env-direct): FF9 rollout-training vs baselines "
             f"(N/k={ref['n_per_k']}, n_ctx={ref['n_ctx']})")
fig.tight_layout()
fig.savefig(D / args.out, dpi=110)
print(f"wrote {D / args.out}")
