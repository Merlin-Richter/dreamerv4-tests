"""EXP-022 (D-029) — inference-only context_signal sweep on OPEN-LOOP motion. NO TRAINING.

Tests the IDEAS "uncertainty-aware rollout" lever: the rollout feeds ALL context frames (real prefix +
self-generated) at one global signal level context_signal=0.9. Does varying that level change open-loop
compounding? If lowering it (telling the model "context less reliable") reduces the open-loop pos_err
rise, a trained per-frame confidence channel is worth building; if the curve is flat in context_signal,
the lever is inert.

Reuses the frozen probe + src/eval/motion.open_loop_curve (just overrides dyn.config.context_signal).
Run from repo root:  python -u experiments/EXP-022/context_signal_sweep.py [--episodes 32 --horizon 16]
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

import torch  # noqa: E402
from eval.motion import open_loop_curve, cross_chance_h, N  # noqa: E402
from revisit_probe import load_models  # noqa: E402
from probe_env import make_probe_episode  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
TOK = _ROOT / "trained_autoencoder.pt"
MODELS = {
    "vanilla_s0": _ROOT / "experiments/EXP-012/vanilla_s0.pt",
    "ff7_k3": _ROOT / "experiments/EXP-010/k3/ff7_k3_s0.pt",
    "c1_h4_s0": _ROOT / "experiments/EXP-020/c1_h4_s0.pt",
}
SIGNALS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
P = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "sweep.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H = args.horizon
    eps = [make_probe_episode(seed=20000 + i, P=P, k=0, R=H) for i in range(args.episodes)]
    sel = [h for h in (1, 4, 8, 12, H) if h <= H]

    out = {"meta": {"episodes": args.episodes, "horizon": H, "signals": SIGNALS, "device": device,
                    "default_context_signal": 0.9}, "models": {}}
    print(f"OPEN-LOOP pos_err vs context_signal  ({args.episodes} ep, H={H}, curtain-up)")
    for name, path in MODELS.items():
        if not path.is_file():
            print(f"  !! missing {name}: {path}")
            continue
        tok, dyn, dcfg, _ = load_models(TOK, path, N, device)
        K = dcfg.inference_steps
        orig = dyn.config.context_signal
        per_s = {}
        print(f"\n[{name}] K={K}   " + "  ".join(f"h{h}" for h in sel) + "   crossChance")
        try:
            for s in SIGNALS:
                dyn.config.context_signal = float(s)
                ol = open_loop_curve(tok, dyn, eps, device, K, H)["model"]
                per_s[f"{s:.2f}"] = {"curve": ol, "cross_chance_h": cross_chance_h(ol)}
                print(f"  s={s:.2f}      " + "  ".join(f"{ol[h]:4.1f}" for h in sel)
                      + f"      h={cross_chance_h(ol)}")
        finally:
            dyn.config.context_signal = orig
        out["models"][name] = per_s

    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
