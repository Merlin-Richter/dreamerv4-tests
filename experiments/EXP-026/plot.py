"""EXP-026 headline view: recall vs occlusion length k — tokenizer-roundtrip ceiling vs oracle vs
copy-last, with chance line. Reads results.json, writes headline.png."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = Path(__file__).resolve().parent
res = json.loads((D / "results.json").read_text())
ch = res["chance"]

metrics = [("position_score", "graded position"), ("position_acc", "exact position (1/36)"),
           ("color_acc", "ball colour (1/4)"), ("bg_acc", "bg colour (1/4)")]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (m, title) in zip(axes.ravel(), metrics):
    for src, style in (("oracle", dict(c="k", ls="--", marker="")),
                       ("tokenizer_roundtrip", dict(c="tab:green", marker="o")),
                       ("copy_last", dict(c="tab:red", marker="x"))):
        d = res[src][m]
        ks = sorted(d, key=int)
        ax.plot([int(k) for k in ks], [d[k] for k in ks], label=src, **style)
    ax.axhline(ch[m], c="gray", ls=":", lw=1, label="chance")
    ax.set_title(title)
    ax.set_xlabel("occlusion length k")
    ax.set_ylabel(m)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle(f"EXP-026 GridWorld tokenizer-roundtrip recall CEILING  (n={res['n_episodes']} eps)")
fig.tight_layout()
out = D / "headline.png"
fig.savefig(out, dpi=110)
print(f"wrote {out}")
