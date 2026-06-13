"""EXP-014 headline chart: 1-step teacher-forced pos_err, vanilla vs relay path, per model.
Run: venv/Scripts/python.exe experiments/EXP-014/make_png.py"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
d = json.loads((HERE / "results.json").read_text())
models = list(d["models"].keys())

van = [d["models"][m]["vanilla"]["model_1step_pos_err_mean"] for m in models]
rel = [d["models"][m]["relay"]["model_1step_pos_err_mean"] for m in models]
gt_step = d["models"][models[0]]["vanilla"]["gt_1step_displacement_mean"]
copy_last = gt_step  # copy-last 1-step error == mean GT displacement

x = np.arange(len(models))
w = 0.38
fig, ax = plt.subplots(figsize=(8.2, 5.0))
b1 = ax.bar(x - w / 2, van, w, label="vanilla path (windowed, NO relay)", color="#3b6fb0")
b2 = ax.bar(x + w / 2, rel, w, label="relay path (window-1 + carried register)", color="#c0613a")

ax.axhline(copy_last, ls="--", lw=1.2, color="gray",
           label=f"copy-last (freeze ball) = {copy_last:.2f}px")
van_s0 = d["models"]["vanilla_s0"]["vanilla"]["model_1step_pos_err_mean"]
ax.axhline(van_s0, ls=":", lw=1.2, color="#3b6fb0",
           label=f"vanilla_s0 windowed baseline = {van_s0:.2f}px")

for bars in (b1, b2):
    for b in bars:
        ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("1-step teacher-forced position error (px)  [lower = better]")
ax.set_title("EXP-014 (D-019): FF7's base-dynamics gain is the LOSS, not the relay\n"
             "FF7 weights through the plain windowed path (no relay) already ~4.5x better than vanilla_s0",
             fontsize=10.5)
ax.legend(fontsize=8.5, loc="upper right")
ax.set_ylim(0, max(van + rel) * 1.25)
fig.tight_layout()
out = HERE / "headline.png"
fig.savefig(out, dpi=130)
print("wrote", out)
