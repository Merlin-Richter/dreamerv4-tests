"""EXP-021 eval (D-029) — confound-free open-loop COMPOUNDING comparison at matched competence.

Compares the full-data C1 (this run) against the competent reference set (vanilla_s0, ff7_k3, ff9v2,
all full-data) on the frozen probe. Per EXP-022, context_signal is a real lever, so for EACH model we
SWEEP context_signal and report the BEST-s open-loop curve (not just the default 0.9). The question:
does C1's TRAINED robustness beat ff7_k3 + its best inference-tuned context_signal, at matched TF map?

Reports per model: teacher-forced curve (per-step map, ~s-independent), open-loop curve at default
s=0.9 and at best-s, and which s is best. Renders a 2-panel view (best-s open-loop overlay + TF overlay).

Run from repo root (after EXP-021 has a checkpoint):
  python -u experiments/EXP-021/eval.py --episodes 48 --horizon 24
  python -u experiments/EXP-021/eval.py --c1-full experiments/EXP-021/c1_full_s0.pt
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT / "src/probe", _ROOT / "src/C_multi_image_auto_encoder",
           _ROOT / "src/D_dynamics_model"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from eval.motion import open_loop_curve, teacher_forced_curve, cross_chance_h, N  # noqa: E402
from revisit_probe import load_models  # noqa: E402
from probe_env import make_probe_episode  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
TOK = _ROOT / "trained_autoencoder.pt"
SIGNALS = [0.5, 0.7, 0.8, 0.9, 0.95]
P = 3


def _auc(curve):
    """Mean over horizons — a single 'how much compounding' summary (lower = better)."""
    return sum(curve.values()) / len(curve)


def eval_model(name, path, episodes, device, H):
    tok, dyn, dcfg, _ = load_models(TOK, path, N, device)
    K = dcfg.inference_steps
    tf = teacher_forced_curve(tok, dyn, episodes, device, K, H)["model"]
    orig = dyn.config.context_signal
    by_s = {}
    try:
        for s in SIGNALS:
            dyn.config.context_signal = float(s)
            by_s[f"{s:.2f}"] = open_loop_curve(tok, dyn, episodes, device, K, H)["model"]
    finally:
        dyn.config.context_signal = orig
    best_s = min(by_s, key=lambda s: _auc(by_s[s]))
    return {"K": K, "tf": tf, "ol_by_s": by_s, "best_s": best_s,
            "ol_default": by_s["0.90"], "ol_best": by_s[best_s],
            "tf_mean": _auc(tf), "ol_best_auc": _auc(by_s[best_s]),
            "cross_chance_best": cross_chance_h(by_s[best_s])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1-full", type=pathlib.Path, default=_ROOT / "experiments/EXP-021/c1_full_s0.pt")
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "eval.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H = args.horizon
    eps = [make_probe_episode(seed=20000 + i, P=P, k=0, R=H) for i in range(args.episodes)]

    models = {
        "c1_full (EXP-021)": args.c1_full,
        "vanilla_s0": _ROOT / "experiments/EXP-012/vanilla_s0.pt",
        "ff7_k3": _ROOT / "experiments/EXP-010/k3/ff7_k3_s0.pt",
        "ff9v2": _ROOT / "experiments/EXP-017/ff9v2_s0.pt",
    }
    out = {"meta": {"episodes": args.episodes, "horizon": H, "signals": SIGNALS}, "models": {}}
    sel = [h for h in (1, 4, 8, 12, 16, 24) if h <= H]
    print(f"{'model':22s} {'bestS':>5s} {'TFmean':>7s} {'OLbest_h':>40s}   crossChance")
    for name, path in models.items():
        if not path.is_file():
            print(f"  !! missing {name}: {path}")
            continue
        r = eval_model(name, path, eps, device, H)
        out["models"][name] = r
        olb = "  ".join(f"{r['ol_best'][h]:4.1f}" for h in sel)
        print(f"{name:22s} {r['best_s']:>5s} {r['tf_mean']:7.2f}   {olb}   h={r['cross_chance_best']}")

    # view: best-s open-loop overlay + TF overlay
    fig, (axo, axt) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for name, r in out["models"].items():
        xo = sorted(r["ol_best"]); axo.plot(xo, [r["ol_best"][h] for h in xo], "-o", ms=3, lw=2,
                                             label=f"{name} (s={r['best_s']})")
        xt = sorted(r["tf"]); axt.plot(xt, [r["tf"][h] for h in xt], "-o", ms=3, lw=2, label=name)
    for ax, t in ((axo, "OPEN-LOOP at BEST context_signal"), (axt, "TEACHER-FORCED (per-step map)")):
        ax.axhline(18, ls=":", color="red", lw=1, alpha=0.7, label="chance ~18")
        ax.set_xlabel("horizon h"); ax.set_title(t); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    axo.set_ylabel("ball pos err (px)")
    fig.suptitle("EXP-021: does full-data C1 beat the competent set (each at its best inference trust)?",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(HERE / "headline.png", dpi=110)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out} + headline.png")


if __name__ == "__main__":
    main()
