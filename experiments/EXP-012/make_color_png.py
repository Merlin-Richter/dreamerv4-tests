"""EXP-012 H3 color-recall headline: hidden-color ΔRGB vs occlusion length for the
budget-matched vanilla baseline (EXP-012) against the FF7 arms (EXP-010), on the identical
frozen probe (5503e75). Shows the architectural cliff (vanilla → chance the instant the
color-carrying prefix leaves the N=8 window) vs FF7's gentle sub-bar decay. No torch needed.

  python experiments/EXP-012/make_color_png.py
"""
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = {
    "vanilla_s0 (EXP-012, budget-matched)": (ROOT / "experiments/EXP-012/results.json", "#2ca02c"),
    "ff7_k1 (EXP-010)": (ROOT / "experiments/EXP-010/k1/results.json", "#d62728"),
    "ff7_k3 (EXP-010)": (ROOT / "experiments/EXP-010/k3/results.json", "#1f77b4"),
}
BAR = 63.0  # T-004 H3 success bar (halfway ceiling→chance)


def main():
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=130)
    ceiling = chance = None
    for name, (path, color) in SRC.items():
        R = json.loads(path.read_text())
        c = R["color_dRGB_by_occ"]
        occ = sorted(int(k) for k in c)
        ax.plot(occ, [c[str(o)] for o in occ], "-o", ms=4, lw=2, color=color, label=name)
        if ceiling is None:
            ceiling = R["controls"]["ceiling"]["color_dRGB"]
            chance = R["controls"]["chance"]["color_dRGB"]

    ax.axhline(ceiling, ls="--", lw=1, color="#888", label=f"ceiling {ceiling:.0f} (perfect recall)")
    ax.axhline(chance, ls="--", lw=1, color="#bbb", label=f"chance {chance:.0f} (no memory)")
    ax.axhline(BAR, ls=":", lw=2, color="#ff7f0e", label=f"T-004 H3 bar {BAR:.0f}")
    ax.axvline(6.5, ls="-", lw=1, color="#000", alpha=0.3)
    ax.text(6.6, chance * 0.78, "N=8 window\nboundary", fontsize=8, alpha=0.6)

    ax.set_xlabel("occlusion length n_occ (frames the ball stays hidden before reveal)")
    ax.set_ylabel("hidden-color recall error  ΔRGB  (lower = better memory)")
    ax.set_title("EXP-012 — H3 color memory: budget-matched baseline cliffs to chance; FF7 holds\n"
                 "(identical frozen probe 5503e75) — confirms EXP-010's color win is the FF7 relay, not training budget",
                 fontsize=10)
    ax.set_ylim(0, chance * 1.12)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="center right")
    out = ROOT / "experiments/EXP-012/headline_color.png"
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
