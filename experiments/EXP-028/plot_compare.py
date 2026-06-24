"""Overlay vanilla vs FF9 env-direct recall curves (reads recall_env_vanilla.json + recall_env_ff9.json
from --dir; writes compare.png). Local (matplotlib)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=str(Path(__file__).resolve().parent))
args = ap.parse_args()
D = Path(args.dir)
van = json.loads((D / "recall_env_vanilla.json").read_text())
ff9w = json.loads((D / "recall_env_ff9_windowed.json").read_text())
ff9m = json.loads((D / "recall_env_ff9_memory.json").read_text())
ch = van["chance"]

metrics = [("position_acc", "exact position (1/36)"), ("position_score", "graded position"),
           ("color_acc", "ball colour (1/4)"), ("bg_acc", "bg colour (1/4)")]
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, (m, title) in zip(axes.ravel(), metrics):
    series = [("vanilla (windowed)", van["model"], dict(c="tab:red", marker="o")),
              ("FF9 base (windowed)", ff9w["model"], dict(c="tab:green", marker="s")),
              ("FF9 memory", ff9m["model"], dict(c="tab:blue", marker="o")),
              ("copy-last", van["copy_last"], dict(c="gray", ls=":", marker="x")),
              ("oracle", van["oracle"], dict(c="k", ls="--"))]
    for label, src, stl in series:
        d = src.get(m, {})
        ks = sorted(d, key=int)
        if ks:
            ax.plot([int(k) for k in ks], [d[k] for k in ks], label=label, **stl)
    if m in ch:
        ax.axhline(ch[m], c="lightgray", lw=1)
    ax.axvline(15, c="purple", lw=0.8, alpha=0.5); ax.text(15.2, 0.02, "window", color="purple", fontsize=6)
    ax.set_title(title); ax.set_xlabel("occlusion length k"); ax.set_ylabel(m)
    ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle(f"GridWorld recall (env-direct): vanilla vs FF9  (N/k={van['n_per_k']}, n_ctx={van['n_ctx']})")
fig.tight_layout()
fig.savefig(D / "compare.png", dpi=110)
print(f"wrote {D / 'compare.png'}")
