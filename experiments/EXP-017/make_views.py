"""EXP-017 low-friction views: (1) frozen-probe color recall vs n_occ for the 3 models, the
headline; (2) within-window memory-sufficiency L(mem) vs L(no-mem). Reads frozen_color.json +
primary.json. Run: venv/Scripts/python.exe -u experiments/EXP-017/make_views.py"""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "experiments" / "EXP-017"
fc = json.loads((D / "frozen_color.json").read_text())
pr = json.loads((D / "primary.json").read_text())

# ---------- view 1: headline color recall vs n_occ ----------
models = fc["models"]
grid = fc["meta"]["grid"]
colors = {"ff9v2_s0": "#1f77b4", "ff7_k3": "#ff7f0e", "vanilla_s0": "#d62728"}
labels = {"ff9v2_s0": "FF9 v2 (full-state memory, A1+B1)", "ff7_k3": "FF7 k3 (register relay)",
          "vanilla_s0": "vanilla (sliding window)"}

fig, ax = plt.subplots(figsize=(9, 5.5))
for name in ["vanilla_s0", "ff7_k3", "ff9v2_s0"]:
    if name not in models:
        continue
    y = [models[name]["color_dRGB_by_occ"][str(k)] for k in grid]
    ax.plot(grid, y, "-o", color=colors[name], label=labels[name], lw=2, ms=5)
# reference lines from FF9's controls (ceiling/chance ~ model-agnostic on this probe)
ceil = models.get("ff9v2_s0", models[list(models)[0]])["ceiling"]["color_dRGB"]
chance = models.get("ff9v2_s0", models[list(models)[0]])["chance"]["color_dRGB"]
ax.axhline(ceil, ls=":", color="green", lw=1.2, label=f"ceiling ~{ceil:.0f}")
ax.axhline(chance, ls=":", color="black", lw=1.2, label=f"chance ~{chance:.0f}")
ax.axhline(63, ls="--", color="purple", lw=1.2, label="T-004 H3 bar (63)")
ax.axvspan(0, 7, color="grey", alpha=0.08)
ax.text(3.5, chance * 0.92, "inside\nN=8 window", ha="center", va="top", fontsize=8, color="grey")
ax.set_xlabel("n_occ (occluded frames after the prefix scrolls out)")
ax.set_ylabel("hidden-color recall  ΔRGB  (lower = better)")
ax.set_title("EXP-017 — beyond-window hidden-COLOR recall (frozen probe 5503e75, 64 eps/pt)")
ax.legend(fontsize=8, loc="center right")
ax.set_ylim(0, max(chance * 1.1, 120))
ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(D / "headline_color.png", dpi=130); plt.close(fig)
print("wrote headline_color.png")

# ---------- view 2: memory sufficiency within window ----------
ms = pr["part2_memory_sufficiency"]
chance_v = ms["chance"]
js = sorted(int(j) for j in ms["by_j"])
tau0 = [ms["by_j"][str(j)]["tau_term"]["0.00"] for j in js]
mem = [r["mem"] for r in tau0]
nomem = [r["no_mem"] for r in tau0]
copyl = [ms["by_j"][str(j)]["copy_last"] for j in js]

fig, ax = plt.subplots(figsize=(7.5, 5))
x = np.arange(len(js)); wd = 0.25
ax.bar(x - wd, mem, wd, label="L(memory)", color="#1f77b4")
ax.bar(x, nomem, wd, label="L(no memory)", color="#aaaaaa")
ax.bar(x + wd, copyl, wd, label="copy-last (freeze)", color="#ff7f0e")
ax.axhline(chance_v, ls=":", color="black", label=f"chance (var)={chance_v:.2f}")
ax.set_xticks(x); ax.set_xticklabels([f"j={j}" for j in js])
ax.set_ylabel("latent flow MSE @ tau_term=0 (memory is the ONLY carrier)")
ax.set_title("EXP-017 — within-window memory sufficiency (PRIMARY)\nmemory alone predicts t+j far below chance / copy-last")
ax.legend(fontsize=8)
ax.grid(alpha=0.25, axis="y")
fig.tight_layout(); fig.savefig(D / "memory_sufficiency.png", dpi=130); plt.close(fig)
print("wrote memory_sufficiency.png")
