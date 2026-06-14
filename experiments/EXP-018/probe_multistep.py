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
for _p in (_SRC / "probe", _SRC / "C_multi_image_auto_encoder", _SRC / "D_dynamics_model"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from revisit_probe import load_models, _encode_window, _decode_frame, detect_ball  # noqa: E402
from probe_env import make_probe_episode  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
TOKENIZER = _ROOT / "trained_autoencoder.pt"
MODELS = {
    "vanilla_s0 (EXP-012)": _ROOT / "experiments" / "EXP-012" / "vanilla_s0.pt",
    "ff7_k3 (EXP-010)": _ROOT / "experiments" / "EXP-010" / "k3" / "ff7_k3_s0.pt",
    "ff9v2_s0 (EXP-017)": _ROOT / "experiments" / "EXP-017" / "ff9v2_s0.pt",
}
N, P = 8, 3  # inference window, visible prefix (match EXP-009/010/011)


def _xy(states, t):
    return states[t, :2].astype(np.float64)


@torch.no_grad()
def open_loop_curve(tok, dyn, episodes, device, K, H):
    err = {h: [] for h in range(1, H + 1)}
    copy_last = {h: [] for h in range(1, H + 1)}
    lost = {h: 0 for h in range(1, H + 1)}
    for ep in episodes:
        ctx = _encode_window(tok, ep.frames, 0, P, device)
        act = torch.from_numpy(ep.actions.astype(np.int64)).unsqueeze(0).to(device)
        gen = dyn.generate(ctx, n_generate=H, K=K, action_idx=act)
        full = torch.cat((ctx, gen), dim=1)
        last_obs = _xy(ep.states, P - 1)
        for h in range(1, H + 1):
            t = P - 1 + h
            gt = _xy(ep.states, t)
            copy_last[h].append(float(np.hypot(*(gt - last_obs))))
            found, x, y, _ = detect_ball(_decode_frame(tok, full[:, t]))
            if not found:
                lost[h] += 1
                continue
            err[h].append(float(np.hypot(x - gt[0], y - gt[1])))
    n = len(episodes)
    return {
        "model": {h: (float(np.mean(v)) if v else float("nan")) for h, v in err.items()},
        "copy_last": {h: float(np.mean(copy_last[h])) for h in copy_last},
        "lost_rate": {h: lost[h] / n for h in lost},
    }


@torch.no_grad()
def teacher_forced_curve(tok, dyn, episodes, device, K, H):
    """Per-horizon TF error: predict frame at absolute index t=P-1+h from the GT window ending
    at t-1 (model never sees its own outputs). Should be ~flat in h if the per-step map is real."""
    maxctx = N - 1
    err = {h: [] for h in range(1, H + 1)}
    lost = {h: 0 for h in range(1, H + 1)}
    for ep in episodes:
        for h in range(1, H + 1):
            t = P - 1 + h                      # frame to predict
            w = min(t, maxctx)
            lo = t - w
            ctx = _encode_window(tok, ep.frames, lo, t, device)          # GT latents [lo, t-1]
            act = torch.from_numpy(ep.actions[lo:t + 1].astype(np.int64)).unsqueeze(0).to(device)
            gen1 = dyn.generate(ctx, n_generate=1, K=K, action_idx=act)
            gt = _xy(ep.states, t)
            found, x, y, _ = detect_ball(_decode_frame(tok, gen1[:, 0]))
            if not found:
                lost[h] += 1
                continue
            err[h].append(float(np.hypot(x - gt[0], y - gt[1])))
    n = len(episodes)
    return {
        "model": {h: (float(np.mean(v)) if v else float("nan")) for h, v in err.items()},
        "lost_rate": {h: lost[h] / n for h in lost},
    }


@torch.no_grad()
def tau_context_sweep(tok, dyn, episodes, device, K, H, signals):
    """P2: TF 1-step error vs the signal level context frames are fed at. Temporarily overrides
    dyn.config.context_signal (restored after). Curtain-up; aggregates over t in [P-1, P-1+H)."""
    maxctx = N - 1
    orig = dyn.config.context_signal
    out = {}
    try:
        for s in signals:
            dyn.config.context_signal = float(s)
            errs = []
            for ep in episodes:
                for t in range(P - 1, P - 1 + H, 4):   # subsample horizons (P2 is a 1-step measure)
                    w = min(t + 1, maxctx)
                    lo = t + 1 - w
                    ctx = _encode_window(tok, ep.frames, lo, t + 1, device)
                    act = torch.from_numpy(ep.actions[lo:t + 2].astype(np.int64)).unsqueeze(0).to(device)
                    gen1 = dyn.generate(ctx, n_generate=1, K=K, action_idx=act)
                    gt = _xy(ep.states, t + 1)
                    found, x, y, _ = detect_ball(_decode_frame(tok, gen1[:, 0]))
                    if found:
                        errs.append(float(np.hypot(x - gt[0], y - gt[1])))
            out[f"{s:.2f}"] = float(np.mean(errs)) if errs else float("nan")
    finally:
        dyn.config.context_signal = orig
    return out


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
