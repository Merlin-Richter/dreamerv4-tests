"""EXP-018 views: per-horizon teacher-forced vs open-loop motion error (P1) + tau-context
sweep (P2), from diagnosis.json. Reusable for later A/B (pass --json + --out)."""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=pathlib.Path, default=HERE / "diagnosis.json")
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "diagnosis.png")
    args = ap.parse_args()
    d = json.loads(args.json.read_text(encoding="utf-8"))
    models = d["models"]
    n = len(models)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 4.2), squeeze=False)
    axes = axes[0]

    for ax, (name, m) in zip(axes, models.items()):
        ol, tf = m["P1_open_loop"], m["P1_teacher_forced"]
        hs = sorted(int(h) for h in ol["model"])
        ax.plot(hs, [ol["model"][str(h)] for h in hs], "o-", label="open-loop (self ctx)", color="crimson")
        ax.plot(hs, [tf["model"][str(h)] for h in hs], "s-", label="teacher-forced (GT ctx)", color="navy")
        ax.plot(hs, [ol["copy_last"][str(h)] for h in hs], "--", label="copy-last", color="gray")
        ax.axhline(20, ls=":", color="black", lw=0.8, label="~chance")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("horizon h (steps)")
        ax.set_ylabel("position error (px)")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 26)

    # P2 tau-context sweep, all models on the last axis
    axp = axes[-1]
    for name, m in models.items():
        p2 = m["P2_tau_sweep"]
        xs = sorted(float(k) for k in p2)
        axp.plot(xs, [p2[f"{x:.2f}"] for x in xs], "o-", label=name, lw=1.2)
    axp.axvline(0.9, ls=":", color="black", lw=0.8, label="rollout ctx=0.9")
    axp.set_title("P2: TF 1-step err vs context signal", fontsize=9)
    axp.set_xlabel("context signal fed")
    axp.set_ylabel("1-step pos err (px)")
    axp.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
