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
for _p in (_ROOT / "src/probe", _ROOT / "src/C_multi_image_auto_encoder",
           _ROOT / "src/D_dynamics_model", _ROOT / "experiments/EXP-018"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from revisit_probe import load_models, _encode_window, _decode_frame, detect_ball  # noqa: E402
from probe_env import make_probe_episode  # noqa: E402
import probe_multistep as pm  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
TOK = _ROOT / "trained_autoencoder.pt"
N, P = 8, 3


@torch.no_grad()
def open_loop_displacement(tok, dyn, episodes, device, K, H):
    """Mean predicted ball inter-frame displacement in open loop (collapse monitor) vs the GT
    displacement (~3.2px). Near 0 => copy-last collapse; near GT => real motion."""
    pred_disp = {h: [] for h in range(2, H + 1)}
    gt_disp = {h: [] for h in range(2, H + 1)}
    for ep in episodes:
        ctx = _encode_window(tok, ep.frames, 0, P, device)
        act = torch.from_numpy(ep.actions.astype(np.int64)).unsqueeze(0).to(device)
        full = torch.cat((ctx, dyn.generate(ctx, n_generate=H, K=K, action_idx=act)), dim=1)
        prev = None
        for h in range(1, H + 1):
            t = P - 1 + h
            found, x, y, _ = detect_ball(_decode_frame(tok, full[:, t]))
            cur = np.array([x, y]) if found else None
            if h >= 2 and prev is not None and cur is not None:
                pred_disp[h].append(float(np.hypot(*(cur - prev))))
                g = ep.states[t, :2] - ep.states[t - 1, :2]
                gt_disp[h].append(float(np.hypot(*g)))
            prev = cur
    return {
        "pred_disp_mean": float(np.mean([v for vs in pred_disp.values() for v in vs])),
        "gt_disp_mean": float(np.mean([v for vs in gt_disp.values() for v in vs])),
    }


def cross_chance_h(curve: dict, chance=18.0):
    """First horizon where open-loop error reaches `chance` (higher = better tracking)."""
    for h in sorted(int(k) for k in curve):
        if curve[str(h)] >= chance:
            return h
    return max(int(k) for k in curve) + 1


def eval_ckpt(name, path, episodes, device, H):
    tok, dyn, dcfg, _ = load_models(TOK, path, N, device)
    K = dcfg.inference_steps
    ol = pm.open_loop_curve(tok, dyn, episodes, device, K, H)
    tf = pm.teacher_forced_curve(tok, dyn, episodes, device, K, H)
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
        print(f"  {name:7s} OL " + "  ".join(f"{ol[str(h)]:4.1f}" for h in sel)
              + f"     h={m['cross_chance_h']:<3d}   {d['pred_disp_mean']:.2f}({d['gt_disp_mean']:.2f})")
        print(f"  {name:7s} TF " + "  ".join(f"{tf[str(h)]:4.1f}" for h in sel))
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
