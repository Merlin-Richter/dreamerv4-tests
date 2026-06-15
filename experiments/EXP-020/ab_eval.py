"""EXP-019/020 A/B eval: vanilla CONTROL vs C1 (multistep) on curtain-up (no-occlusion) motion.

Reuses the EXP-018 probe functions (open-loop + teacher-forced per-horizon pos_err on the frozen
probe env) and adds a collapse monitor: the model's predicted inter-frame DISPLACEMENT in open loop
(must track sim ~3.2px, not 0 — the copy-last degenerate mode from V-T017-C1 C-B(a)).

Run from repo root:
  python experiments/EXP-020/ab_eval.py --episodes 48 --horizon 24
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT / "src/probe", _ROOT / "src/C_multi_image_auto_encoder",
           _ROOT / "src/D_dynamics_model"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Shared motion-eval toolbox (src/eval/motion.py); was inline + probe_multistep pre-D-028 refactor.
from eval.motion import (  # noqa: E402
    open_loop_curve, teacher_forced_curve, open_loop_displacement, cross_chance_h, N, P,
)
from revisit_probe import load_models  # noqa: E402
from probe_env import make_probe_episode  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
TOK = _ROOT / "trained_autoencoder.pt"


def eval_ckpt(name, path, episodes, device, H):
    tok, dyn, dcfg, _ = load_models(TOK, path, N, device)
    K = dcfg.inference_steps
    ol = open_loop_curve(tok, dyn, episodes, device, K, H)
    tf = teacher_forced_curve(tok, dyn, episodes, device, K, H)
    disp = open_loop_displacement(tok, dyn, episodes, device, K, H)
    return {"open_loop": ol, "teacher_forced": tf, "displacement": disp,
            "cross_chance_h": cross_chance_h(ol["model"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", type=pathlib.Path, default=_ROOT / "experiments/EXP-019/vanilla_ctrl_s0.pt")
    ap.add_argument("--c1", type=pathlib.Path, default=_ROOT / "experiments/EXP-020/c1_h4_s0.pt")
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "ab.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H = args.horizon
    eps = [make_probe_episode(seed=20000 + i, P=P, k=0, R=H) for i in range(args.episodes)]

    out = {"meta": {"episodes": args.episodes, "horizon": H, "regime": "curtain-up (k=0)"}, "models": {}}
    for name, path in [("control", args.control), ("c1", args.c1)]:
        if not path.is_file():
            print(f"  !! missing {name}: {path}")
            continue
        out["models"][name] = eval_ckpt(name, path, eps, device, H)

    sel = [h for h in (1, 2, 4, 8, 12, 16, 24) if h <= H]
    print("\n            " + "  ".join(f"h{h:>2d}" for h in sel) + "   crossChance  predDisp(gt)")
    for name, m in out["models"].items():
        ol = m["open_loop"]["model"]; tf = m["teacher_forced"]["model"]; d = m["displacement"]
        print(f"  {name:7s} OL " + "  ".join(f"{ol[h]:4.1f}" for h in sel)
              + f"     h={m['cross_chance_h']:<3d}   {d['pred_disp_mean']:.2f}({d['gt_disp_mean']:.2f})")
        print(f"  {name:7s} TF " + "  ".join(f"{tf[h]:4.1f}" for h in sel))
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
