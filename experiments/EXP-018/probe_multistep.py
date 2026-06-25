"""EXP-018 — motion-prediction diagnosis (D-026 / T-016 probes P1+P2). NO TRAINING.

Confirms the method-architect diagnosis BEFORE building anything. Reuses the EXP-011 frozen
probe (env + detector + load_models) so numbers are comparable. Curtain-UP only (k=0) = the
no-occlusion motion regime Merlin asked about.

P1 (decisive: link-3 "1-step-only fit" vs link-4 "compounding"): per-horizon position error,
TEACHER-FORCED (predict each frame from the GROUND-TRUTH context window) vs OPEN-LOOP (predict
from the model's own generated context). If TF stays flat & low while open-loop climbs ->
the per-step map is fine, the failure is autoregressive compounding (link 4). If TF ALSO
climbs -> the per-step map is a 1-step-only fit (link 3).

P2 (link-4b: tau context-distribution mismatch): teacher-forced 1-step error while sweeping
the signal level the CONTEXT frames are fed at (the rollout pins this at context_signal=0.9,
but training saw random per-frame tau). A spike near 0.9 vs the trained-tau region = mismatch.

Run from repo root:  python experiments/EXP-018/probe_multistep.py [--episodes 48 --horizon 24]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
for _p in (_SRC, _SRC / "probe", _SRC / "C_multi_image_auto_encoder", _SRC / "D_dynamics_model"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Curve logic now lives in the shared eval toolbox (src/eval/motion.py); imported here so the
# EXP-018 diagnosis entrypoint stays reproducible. (Was defined inline pre-D-028 refactor.)
from eval.motion import (  # noqa: E402
    open_loop_curve, teacher_forced_curve, tau_context_sweep, N, P,
)
from revisit_probe import load_models  # noqa: E402
from probe_env import make_probe_episode  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
TOKENIZER = _ROOT / "trained_autoencoder.pt"
MODELS = {
    "vanilla_s0 (EXP-012)": _ROOT / "experiments" / "EXP-012" / "vanilla_s0.pt",
    "ff7_k3 (EXP-010)": _ROOT / "experiments" / "EXP-010" / "k3" / "ff7_k3_s0.pt",
    "ff9v2_s0 (EXP-017)": _ROOT / "experiments" / "EXP-017" / "ff9v2_s0.pt",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "diagnosis.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H = args.horizon

    episodes = [make_probe_episode(seed=20000 + i, P=P, k=0, R=H) for i in range(args.episodes)]
    out = {"meta": {"N": N, "P": P, "horizon": H, "episodes": args.episodes, "device": device,
                    "regime": "curtain-up (k=0), no occlusion"}, "models": {}}

    for name, path in MODELS.items():
        if not path.is_file():
            print(f"  !! missing {name}: {path}")
            continue
        tok, dyn, dcfg, _ = load_models(TOKENIZER, path, N, device)
        K = dcfg.inference_steps
        print(f"\n[model] {name}  K={K}")
        ol = open_loop_curve(tok, dyn, episodes, device, K, H)
        tf = teacher_forced_curve(tok, dyn, episodes, device, K, H)
        p2 = tau_context_sweep(tok, dyn, episodes, device, K, H,
                               signals=[0.7, 0.8, 0.9, 0.95])
        out["models"][name] = {"P1_open_loop": ol, "P1_teacher_forced": tf, "P2_tau_sweep": p2}
        sel = [h for h in (1, 2, 4, 8, 12, 16, H) if h <= H]
        print("    P1  h:    " + "  ".join(f"{h:>5d}" for h in sel))
        print("    TF  err:  " + "  ".join(f"{tf['model'][h]:5.1f}" for h in sel))
        print("    OL  err:  " + "  ".join(f"{ol['model'][h]:5.1f}" for h in sel))
        print("    copylast: " + "  ".join(f"{ol['copy_last'][h]:5.1f}" for h in sel))
        print("    P2 tau-context-sweep (TF 1-step err): "
              + "  ".join(f"{k}:{v:.1f}" for k, v in p2.items()))

    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
