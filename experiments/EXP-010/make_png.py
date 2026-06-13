"""EXP-010 headline PNG: hidden-color recall vs occlusion, FF7 arms vs baseline.
Phone-friendly single image. Run from repo root: python experiments/EXP-010/make_png.py
"""
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
BAR = 63.0
TEST = (12, 16, 24)
SERIES = [
    ("baseline (EXP-009)", HERE.parent / "EXP-009" / "results.json", "#888888"),
    ("FF7 k=1", HERE / "k1" / "results.json", "#d62728"),
    ("FF7 k=3", HERE / "k3" / "results.json", "#1f77b4"),
]


def load(p):
    return json.loads(p.read_text()) if p.is_file() else None


data = [(lab, load(p), c) for lab, p, c in SERIES]
data = [(lab, d, c) for lab, d, c in data if d is not None]
base = data[0][1]
grid = base["occ_grid"]
ceil_c = base["controls"]["ceiling"]["color_dRGB"]
chance_c = base["controls"]["chance"]["color_dRGB"]

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=130)
ax.axhline(ceil_c, ls="--", lw=1, color="#2ca02c", label=f"ceiling {ceil_c:.0f}")
ax.axhline(chance_c, ls="--", lw=1, color="#999999", label=f"chance {chance_c:.0f}")
ax.axhline(BAR, ls=":", lw=1.6, color="#e377c2", label=f"T-004 bar {BAR:.0f}")
ax.axvspan(8.5, 24, color="#fff3f3", zorder=0)  # beyond-window region (n_occ>=N=8 cliff)

for lab, d, col in data:
    occ = [d["color_dRGB_by_occ"][str(n)] for n in grid]
    drift = [d["matched_horizon_drift"]["color_dRGB"][str(n)] for n in grid]
    ax.plot(grid, occ, "-o", color=col, lw=2.2, ms=5, label=f"{lab}  (occluded)")
    ax.plot(grid, drift, "--", color=col, lw=1, alpha=0.55)

ax.set_xlabel("n_occ  (frames the ball is hidden; window N=8, prefix P=3)")
ax.set_ylabel("color ΔRGB at reveal  (lower = better recall)")
ax.set_title("EXP-010 — FF7 hidden-color recall vs sliding-window baseline\n"
             "dashed thin = matched-horizon drift control (same series colour)")
ax.set_ylim(0, max(chance_c * 1.18, 125))
ax.set_xticks(grid)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=8, loc="upper left", ncol=2)
for n in TEST:
    ax.axvline(n, color="#eeeeee", lw=6, zorder=0)
fig.tight_layout()
out = HERE / "headline.png"
fig.savefig(out)
print("wrote", out)
