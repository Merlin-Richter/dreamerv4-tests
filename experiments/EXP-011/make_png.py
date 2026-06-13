"""EXP-011 headline PNG: open-loop ball-position error vs horizon, model vs copy-last vs chance,
with teacher-forced 1-step and latent-probe annotations. Run: python experiments/EXP-011/make_png.py
"""
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
H = R["meta"]["horizon"]
hor = list(range(1, H + 1))
chance = R["chance_pos_err_px"]
COL = {"my_dynamics": "#000000", "ff7_k1": "#d62728", "ff7_k3": "#1f77b4"}

fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=130)
ax.axhline(chance, ls="--", lw=1.2, color="#999", label=f"chance {chance:.0f}px")

# copy-last baseline (same for all models — GT displacement vs horizon); take from my_dynamics
base = R["models"].get("my_dynamics") or next(iter(R["models"].values()))
cl = [base["open_loop"]["copy_last_pos_err"][str(h)] for h in hor]
ax.plot(hor, cl, ":", lw=2, color="#ff7f0e", label="copy-last (freeze ball)")

for name, m in R["models"].items():
    y = [m["open_loop"]["model_pos_err"].get(str(h), float("nan")) for h in hor]
    ax.plot(hor, y, "-o", ms=3.5, lw=2, color=COL.get(name, None), label=f"{name} (open-loop)")

ax.set_xlabel("rollout horizon (steps past last observed frame)")
ax.set_ylabel("ball position error (px)")
k = R["gt_kinematics"]["speed_px_per_step"]["mean"]
lp = R["latent_position_probe"]
tf = base["teacher_forced_1step"]
ax.set_title("EXP-011 — open-loop ball-position error vs horizon\n"
             f"GT ball speed {k:.1f}px/step | teacher-forced 1-step (my_dynamics) "
             f"{tf['model_1step_pos_err_mean']:.1f}px vs GT step {tf['gt_1step_displacement_mean']:.1f}px | "
             f"latent→xy probe median {lp['median_pos_err_px']:.1f}px (R²={lp['r2']:.2f})")
ax.set_ylim(0, max(chance * 1.15, max(cl) * 1.1))
ax.set_xticks(hor)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
out = HERE / "headline.png"
fig.savefig(out)
print("wrote", out)
