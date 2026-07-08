"""Pixel-tier steps-vs-quality curve figure (experiments/colorfield-pixcurve/).

Two stacked panels, shared log-x training-steps axis:
  TOP    train mem2mem flow loss — H100 416906 vs the local 4070 calcurve
         (same seed/config; the local run is the cross-backend anchor, killed at 3750).
  BOTTOM revisit-sheet on-screen cell acc per H100 snapshot (driver/sheets.py,
         map seeds 5+6; imagination revisit ages ~60-190 >> W=16; chance = 0.2).

Reads loss_h100.csv / loss_local.csv (steps,elapsed,val,mem2mem) + sheets_sweep.log.
"""
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
BLUE, ORANGE, INK, MUTED = "#1f77b4", "#ff7f0e", "#333333", "#888888"


def read_loss(name):
    steps, flow = [], []
    with open(HERE / name) as f:
        for row in csv.reader(f):
            steps.append(int(row[0])); flow.append(float(row[3]))
    return steps, flow


def read_sheets(log=HERE / "sheets_sweep.log"):
    """-> {step: [acc_seed5, acc_seed6]} from '[sheet] mean on-screen cell acc ...' lines."""
    out, step = {}, None
    for line in open(log):
        m = re.match(r"=== STEP (\d+) ===", line)
        if m:
            step = int(m.group(1)); out[step] = []
        m = re.search(r"mean on-screen cell acc over \d+ sampled imag frames: ([0-9.]+)", line)
        if m and step is not None:
            out[step].append(float(m.group(1)))
    return {k: v for k, v in out.items() if v}


def main():
    h_steps, h_flow = read_loss("loss_h100.csv")
    l_steps, l_flow = read_loss("loss_local.csv")
    acc = read_sheets()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True,
                                   gridspec_kw={"hspace": 0.12})
    for ax in (ax1, ax2):
        ax.grid(True, which="both", alpha=0.18, linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED)

    # TOP — flow loss, log-log
    ax1.plot(h_steps, h_flow, color=BLUE, lw=2, label="H100 416906 (this run)")
    ax1.plot(l_steps, l_flow, color=ORANGE, lw=2, label="local 4070 calcurve (anchor)")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_ylabel("train mem2mem flow loss", color=INK)
    ax1.text(h_steps[-1], h_flow[-1] * 1.15, f"H100  {h_flow[-1]:.4f}",
             color=BLUE, fontsize=9, ha="right", va="bottom")
    ax1.text(l_steps[-1] * 1.1, l_flow[-1], f"4070 (killed @3750)  {l_flow[-1]:.4f}",
             color=ORANGE, fontsize=9, va="center")
    ax1.legend(frameon=False, fontsize=9, loc="upper right")
    ax1.set_title("ColorField pixel tier: steps vs loss and revisit accuracy "
                  "(1.32M, bs128, rollout-only mem2mem, seed 0)",
                  fontsize=11, color=INK)

    # BOTTOM — revisit acc per snapshot
    if acc:
        steps_sorted = sorted(acc)
        means = [sum(acc[s]) / len(acc[s]) for s in steps_sorted]
        for s in steps_sorted:                       # individual seeds, small marks
            ax2.plot([s] * len(acc[s]), acc[s], "o", color=BLUE, ms=3.5, alpha=0.35)
        ax2.plot(steps_sorted, means, "-o", color=BLUE, lw=2, ms=6,
                 label="revisit on-screen cell acc (mean of map seeds 5,6)")
        ax2.text(steps_sorted[-1], means[-1] + 0.03, f"{means[-1]:.2f}",
                 color=BLUE, fontsize=9, ha="center")
    ax2.axhline(0.2, color=MUTED, lw=1.4, ls="--")
    ax2.text(100000, 0.165, "chance (0.2)", color=MUTED, fontsize=9, ha="right")
    ax2.set_ylim(0, 1.0)
    ax2.set_xlabel("training steps (log)", color=INK)
    ax2.set_ylabel("cell accuracy", color=INK)
    ax2.legend(frameon=False, fontsize=9, loc="upper left")

    out = HERE / "curve_pixcurve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved -> {out}")
    if acc:
        print("step -> acc(mean):", {s: round(sum(v) / len(v), 3) for s, v in sorted(acc.items())})


if __name__ == "__main__":
    main()
