"""Headline PNG: open-loop ball-position error vs horizon, models vs copy-last vs chance, with a
1-step teacher-forced bar inset and latent-probe annotation. Reusable across the EXP-011
diagnostic and the EXP-012 rerun (any model set).

  python experiments/EXP-011/make_png.py                              # EXP-011 results.json
  python experiments/EXP-011/make_png.py --in experiments/EXP-012/diagnostic.json \
         --out experiments/EXP-012/headline.png --title "EXP-012"
"""
import argparse
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
PALETTE = ["#000000", "#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b"]
# stable colors for known names; others fall back to the palette by order
KNOWN = {"my_dynamics": "#000000", "ff7_k1": "#d62728", "ff7_k3": "#1f77b4",
         "vanilla_s0 (EXP-012)": "#2ca02c"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=pathlib.Path, default=HERE / "results.json")
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "headline.png")
    ap.add_argument("--title", default="EXP-011")
    args = ap.parse_args()

    R = json.loads(args.inp.read_text())
    H = R["meta"]["horizon"]
    hor = list(range(1, H + 1))
    chance = R["chance_pos_err_px"]
    models = R["models"]
    base = models.get("my_dynamics") or next(iter(models.values()))

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(13, 5.6), dpi=130,
                                  gridspec_kw={"width_ratios": [3, 1]})
    ax.axhline(chance, ls="--", lw=1.2, color="#999", label=f"chance {chance:.0f}px")
    cl = [base["open_loop"]["copy_last_pos_err"][str(h)] for h in hor]
    ax.plot(hor, cl, ":", lw=2, color="#ff7f0e", label="copy-last (freeze ball)")
    for i, (name, m) in enumerate(models.items()):
        y = [m["open_loop"]["model_pos_err"].get(str(h), float("nan")) for h in hor]
        ax.plot(hor, y, "-o", ms=3.5, lw=2, color=KNOWN.get(name, PALETTE[i % len(PALETTE)]),
                label=f"{name} (open-loop)")
    ax.set_xlabel("rollout horizon (steps past last observed frame)")
    ax.set_ylabel("ball position error (px)")
    ax.set_ylim(0, max(chance * 1.15, max(cl) * 1.1))
    ax.set_xticks(hor)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")

    # 1-step teacher-forced bar (THE attribution number) vs the GT per-step displacement.
    names = list(models.keys())
    tf = [models[n]["teacher_forced_1step"]["model_1step_pos_err_mean"] for n in names]
    gt_step = base["teacher_forced_1step"]["gt_1step_displacement_mean"]
    axb.bar(range(len(names)), tf,
            color=[KNOWN.get(n, PALETTE[i % len(PALETTE)]) for i, n in enumerate(names)])
    axb.axhline(gt_step, ls=":", color="#ff7f0e", lw=2, label=f"copy-last {gt_step:.1f}px")
    axb.set_xticks(range(len(names)))
    axb.set_xticklabels([n.split()[0] for n in names], rotation=30, ha="right", fontsize=8)
    axb.set_ylabel("teacher-forced 1-step pos err (px)")
    axb.set_title("1-step dynamics (lower=better)", fontsize=9)
    axb.legend(fontsize=8)
    axb.grid(True, axis="y", alpha=0.25)

    k = R["gt_kinematics"]["speed_px_per_step"]["mean"]
    lp = R["latent_position_probe"]
    fig.suptitle(f"{args.title} — open-loop position error vs horizon  |  GT speed {k:.1f}px/step  |  "
                 f"latent→xy probe median {lp['median_pos_err_px']:.1f}px (R²={lp['r2']:.2f})",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
