"""Reusable motion-evaluation curves on the frozen probe env (curtain-up / open-loop / TF).

Extracted verbatim from experiments/EXP-018/probe_multistep.py + the A/B helpers from
experiments/EXP-020/ab_eval.py so motion evals are shared, not re-pasted per experiment.
Numerics are byte-identical to those originals (same logic, same N/P=8/3 geometry).

Public API:
  open_loop_curve(tok, dyn, episodes, device, K, H)      -> per-horizon open-loop pos_err (+copy_last)
  teacher_forced_curve(tok, dyn, episodes, device, K, H) -> per-horizon TF pos_err (GT context)
  tau_context_sweep(tok, dyn, episodes, device, K, H, signals) -> TF 1-step err vs context_signal
  open_loop_displacement(tok, dyn, episodes, device, K, H)     -> mean predicted inter-frame disp
  cross_chance_h(curve, chance=18.0)                     -> first horizon reaching `chance`

`episodes` are probe episodes from `probe_env.make_probe_episode`. `load_models`,
`_encode_window`, `_decode_frame`, `detect_ball` live in the FROZEN probe (src/probe/revisit_probe).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

# Bootstrap the frozen-probe + model import paths so callers only need `src` on sys.path.
_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
for _p in (_SRC / "probe", _SRC / "C_multi_image_auto_encoder", _SRC / "D_dynamics_model"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from revisit_probe import _encode_window, _decode_frame, detect_ball  # noqa: E402,F401
from revisit_probe import load_models  # noqa: E402,F401  (re-exported convenience)

# Inference window / visible prefix — fixed by the frozen probe geometry (EXP-009/010/011).
N, P = 8, 3


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
    """TF 1-step error vs the signal level context frames are fed at. Temporarily overrides
    dyn.config.context_signal (restored after). Curtain-up; aggregates over t in [P-1, P-1+H)."""
    maxctx = N - 1
    orig = dyn.config.context_signal
    out = {}
    try:
        for s in signals:
            dyn.config.context_signal = float(s)
            errs = []
            for ep in episodes:
                for t in range(P - 1, P - 1 + H, 4):   # subsample horizons (1-step measure)
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
    """First horizon where open-loop error reaches `chance` (higher = better tracking).
    Accepts int- or str-keyed curves."""
    keys = sorted(int(k) for k in curve)
    for h in keys:
        v = curve[h] if h in curve else curve[str(h)]
        if v >= chance:
            return h
    return keys[-1] + 1
