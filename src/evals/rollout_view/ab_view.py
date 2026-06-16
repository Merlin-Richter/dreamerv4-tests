"""Render a low-friction headline view from an A/B ab.json (the format ab_eval.py writes).

Two panels: open-loop pos_err vs horizon (the compounding curve) and teacher-forced pos_err vs
horizon (the per-step map). Each model is a line; copy-last + a chance line give reference. The whole
A/B read should be obvious at a glance.

Usage:  python -u src/evals/rollout_view/ab_view.py --in experiments/EXP-020/ab.json --out experiments/EXP-020/headline.png
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _curve(d):
    """dict with str/int horizon keys -> (sorted xs, ys)."""
    xs = sorted(int(k) for k in d)
    ys = [d[str(x)] if str(x) in d else d[x] for x in xs]
    return xs, ys


def plot_ab(ab_path, out_path, chance=18.0):
    data = json.loads(pathlib.Path(ab_path).read_text(encoding="utf-8"))
    models = data["models"]
    meta = data.get("meta", {})
    fig, (axo, axt) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    colors = {"control": "#888888", "c1": "#1f77b4"}

    any_copy = False
    for name, m in models.items():
        c = colors.get(name, None)
        xo, yo = _curve(m["open_loop"]["model"])
        axo.plot(xo, yo, "-o", color=c, label=name, lw=2, ms=4)
        xt, yt = _curve(m["teacher_forced"]["model"])
        axt.plot(xt, yt, "-o", color=c, label=name, lw=2, ms=4)
        cc = m.get("cross_chance_h")
        d = m.get("displacement", {})
        if cc is not None:
            axo.annotate(f"{name}: crossChance h={cc}, predDisp "
                         f"{d.get('pred_disp_mean', float('nan')):.1f}(gt{d.get('gt_disp_mean', float('nan')):.1f})",
                         xy=(0.02, 0.97 - 0.06 * list(models).index(name)),
                         xycoords="axes fraction", fontsize=8, color=c)
        if "copy_last" in m["open_loop"] and not any_copy:
            xc, yc = _curve(m["open_loop"]["copy_last"])
            axo.plot(xc, yc, "--", color="black", lw=1, alpha=0.6, label="copy-last")
            any_copy = True

    for ax, title in ((axo, "OPEN-LOOP (autoregressive)"), (axt, "TEACHER-FORCED (per-step map)")):
        ax.axhline(chance, ls=":", color="red", lw=1, alpha=0.7, label=f"chance ~{chance:.0f}")
        ax.set_xlabel("horizon h (frames after the visible prefix)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
    axo.set_ylabel("ball position error (px)")
    regime = meta.get("regime", "")
    fig.suptitle(f"A/B motion: control vs C1   ({meta.get('episodes','?')} ep, H={meta.get('horizon','?')}, {regime})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=110)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--chance", type=float, default=18.0)
    main_args = ap.parse_args()
    plot_ab(main_args.inp, main_args.out, main_args.chance)


if __name__ == "__main__":
    main()
