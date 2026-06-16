"""EXP-022 view: open-loop pos_err vs horizon at several context_signal levels, one panel per model.
Shows the inference-trust lever: lower context_signal flattens ff7's compounding; C1 is already best at 0.9.
Run: python -u experiments/EXP-022/make_view.py
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
data = json.loads((HERE / "sweep.json").read_text(encoding="utf-8"))
models = data["models"]
show_s = ["0.50", "0.70", "0.90", "0.99"]
cmap = {"0.50": "#d62728", "0.70": "#ff7f0e", "0.90": "#1f77b4", "0.99": "#2ca02c"}

fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.8), sharey=True)
if len(models) == 1:
    axes = [axes]
for ax, (name, per_s) in zip(axes, models.items()):
    for s in show_s:
        if s not in per_s:
            continue
        curve = per_s[s]["curve"]
        xs = sorted(int(k) for k in curve)
        ys = [curve[str(x)] if str(x) in curve else curve[x] for x in xs]
        lw = 2.4 if s == "0.90" else 1.6
        ax.plot(xs, ys, "-o", ms=3, lw=lw, color=cmap[s],
                label=f"s={s}" + (" (default)" if s == "0.90" else ""))
    ax.axhline(18.0, ls=":", color="gray", lw=1, alpha=0.7, label="chance ~18")
    ax.set_title(name)
    ax.set_xlabel("horizon h")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
axes[0].set_ylabel("open-loop ball pos err (px)")
fig.suptitle("EXP-022: open-loop compounding vs inference context_signal (lower s = less trust in self-gen context)",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = HERE / "sweep.png"
fig.savefig(out, dpi=110)
print(f"wrote {out}")
