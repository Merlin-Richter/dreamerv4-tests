"""EXP-011 — position-deficit diagnostic (D-015). NO TRAINING.

Question: the world model's ball-position error hits chance even in an OPEN rollout with the
curtain UP (matched-horizon drift control, EXP-009/010). Did the model (a) never learn motion,
or (b) learn it but desync from the specific GT trajectory in open loop (chaotic bounces)?
And does the deficit live in the tokenizer C (position not encoded) or the dynamics D
(encoded but not propagated)?

Five components, all on existing checkpoints + frozen tokenizer (reuses the FROZEN probe's
env + detector so numbers are comparable to EXP-009/010):
  1. GT ball kinematics — speed/displacement read directly from states[:, vx,vy] (zero inference).
  2. Open-loop model pos_err vs horizon, against copy-last-position and chance baselines.
  3. Teacher-forced (GT-context) 1-step pos_err vs the per-step ground-truth displacement.
  4. Linear probe of frozen tokenizer latents -> (x,y): is position even encoded? (C vs D)
  5. (qualitative sheet is the existing probe drift rollout; not re-rendered here)

Run from repo root:  python experiments/EXP-011/diagnose.py [--episodes 32 --horizon 24]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]   # experiments/EXP-011/ -> repo root
_SRC = _ROOT / "src"
for _p in (_SRC / "probe", _SRC / "C_multi_image_auto_encoder", _SRC / "D_dynamics_model"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from revisit_probe import (  # noqa: E402
    load_models, _encode_window, _decode_frame, _target_latent, detect_ball,
)
from probe_env import make_probe_episode  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
TOKENIZER = _ROOT / "trained_autoencoder.pt"
# Order matters for the comparison: budget-matched vanilla baseline (EXP-012) first when present,
# then the old my_dynamics, then the FF7 arms. Missing checkpoints are skipped at runtime.
MODELS = {
    "vanilla_s0 (EXP-012)": _ROOT / "experiments" / "EXP-012" / "vanilla_s0.pt",
    "my_dynamics": _ROOT / "my_dynamics.pt",
    "ff7_k1": _ROOT / "experiments" / "EXP-010" / "k1" / "ff7_k1_s0.pt",
    "ff7_k3": _ROOT / "experiments" / "EXP-010" / "k3" / "ff7_k3_s0.pt",
}
N, P = 8, 3  # inference window, visible prefix (match EXP-009/010)


def _xy(states, t):
    return states[t, :2].astype(np.float64)


# --------------------------------------------------------------------- 1. GT kinematics
def gt_kinematics(episodes):
    speeds, steps = [], []
    for ep in episodes:
        v = ep.states[:, 2:4].astype(np.float64)
        speeds.extend(np.hypot(v[:, 0], v[:, 1]).tolist())
        xy = ep.states[:, :2].astype(np.float64)
        steps.extend(np.hypot(*(xy[1:] - xy[:-1]).T).tolist())
    speeds, steps = np.array(speeds), np.array(steps)
    return {
        "speed_px_per_step": {"mean": float(speeds.mean()), "median": float(np.median(speeds)),
                              "p90": float(np.percentile(speeds, 90)), "max": float(speeds.max())},
        "frame_to_frame_displacement_px": {"mean": float(steps.mean()),
                                           "median": float(np.median(steps)),
                                           "p90": float(np.percentile(steps, 90))},
    }


def chance_scale(episodes):
    """Expected position error of predicting a random plausible ball location."""
    pts = np.array([ep.states[t, :2] for ep in episodes for t in range(ep.states.shape[0])],
                   dtype=np.float64)
    rng = np.random.default_rng(0)
    a = pts[rng.integers(0, len(pts), 20000)]
    b = pts[rng.integers(0, len(pts), 20000)]
    return float(np.hypot(*(a - b).T).mean())


# --------------------------------------------------------------------- 2. open-loop rollout
@torch.no_grad()
def open_loop_pos_err(tok, dyn, episodes, device, K, H):
    """Model open-loop pos_err vs horizon, plus the copy-last-position baseline."""
    model_err = {h: [] for h in range(1, H + 1)}
    lost = {h: 0 for h in range(1, H + 1)}
    copy_last = {h: [] for h in range(1, H + 1)}
    for ep in episodes:
        ctx = _encode_window(tok, ep.frames, 0, P, device)
        act = torch.from_numpy(ep.actions.astype(np.int64)).unsqueeze(0).to(device)
        gen = dyn.generate(ctx, n_generate=H, K=K, action_idx=act)
        full = torch.cat((ctx, gen), dim=1)  # (1, P+H, L, d)
        last_obs = _xy(ep.states, P - 1)
        for h in range(1, H + 1):
            t = P - 1 + h  # absolute frame index of this horizon
            gt = _xy(ep.states, t)
            copy_last[h].append(float(np.hypot(*(gt - last_obs))))
            found, x, y, _ = detect_ball(_decode_frame(tok, full[:, t]))
            if not found:
                lost[h] += 1
                continue
            model_err[h].append(float(np.hypot(x - gt[0], y - gt[1])))
    n = len(episodes)
    return {
        "horizons": list(range(1, H + 1)),
        "model_pos_err": {h: (float(np.mean(v)) if v else float("nan")) for h, v in model_err.items()},
        "copy_last_pos_err": {h: float(np.mean(copy_last[h])) for h in copy_last},
        "ball_lost_rate": {h: lost[h] / n for h in lost},
    }


# --------------------------------------------------------------------- 3. teacher-forced 1-step
@torch.no_grad()
def teacher_forced_1step(tok, dyn, episodes, device, K, H, tok_win):
    """Feed GT latents as context (window <= N-1), predict the next frame, compare to GT.
    Isolates 1-step dynamics quality from open-loop compounding. Reports model vs the
    per-step GT displacement (= copy-last 1-step error)."""
    errs, disp, lost = [], [], 0
    maxctx = N - 1
    for ep in episodes:
        for t in range(P - 1, P - 1 + H):           # predict frame t+1 from GT window ending at t
            w = min(t + 1, maxctx)
            lo = t + 1 - w
            ctx = _encode_window(tok, ep.frames, lo, t + 1, device)   # (1, w, L, d) GT latents
            act = torch.from_numpy(ep.actions[lo:t + 2].astype(np.int64)).unsqueeze(0).to(device)
            gen1 = dyn.generate(ctx, n_generate=1, K=K, action_idx=act)
            gt = _xy(ep.states, t + 1)
            disp.append(float(np.hypot(*(gt - _xy(ep.states, t)))))
            found, x, y, _ = detect_ball(_decode_frame(tok, gen1[:, 0]))
            if not found:
                lost += 1
                continue
            errs.append(float(np.hypot(x - gt[0], y - gt[1])))
    n = len(errs) + lost
    return {
        "model_1step_pos_err_mean": float(np.mean(errs)) if errs else float("nan"),
        "model_1step_pos_err_median": float(np.median(errs)) if errs else float("nan"),
        "gt_1step_displacement_mean": float(np.mean(disp)),
        "ball_lost_rate": lost / max(n, 1),
        "n": n,
    }


# --------------------------------------------------------------------- 4. linear probe of latents
@torch.no_grad()
def collect_latent_xy(tok, episodes, device, tok_win):
    X, Y = [], []
    for ep in episodes:
        for t in range(ep.states.shape[0]):
            if ep.actions[t] != 0:   # curtain up only (ball visible) — here always up
                continue
            lat = _target_latent(tok, ep.frames, t, tok_win, device)  # (1,L,d)
            X.append(lat.reshape(-1).float().cpu().numpy())
            Y.append(ep.states[t, :2].astype(np.float64))
    return np.array(X), np.array(Y)


def linear_probe(Xtr, Ytr, Xte, Yte):
    A = np.concatenate([Xtr, np.ones((len(Xtr), 1))], axis=1)
    W, *_ = np.linalg.lstsq(A, Ytr, rcond=None)
    Ate = np.concatenate([Xte, np.ones((len(Xte), 1))], axis=1)
    pred = Ate @ W
    err = np.hypot(*(pred - Yte).T)
    ss_res = ((pred - Yte) ** 2).sum()
    ss_tot = ((Yte - Yte.mean(axis=0)) ** 2).sum()
    return {
        "median_pos_err_px": float(np.median(err)),
        "mean_pos_err_px": float(err.mean()),
        "r2": float(1 - ss_res / ss_tot),
        "n_train": len(Xtr), "n_test": len(Xte),
    }


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "results.json",
                    help="results path; the EXP-012 rerun passes experiments/EXP-012/diagnostic.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H = args.horizon

    episodes = [make_probe_episode(seed=20000 + i, P=P, k=0, R=H) for i in range(args.episodes)]
    probe_eps = [make_probe_episode(seed=30000 + i, P=P, k=0, R=H) for i in range(64)]  # for latent probe

    out = {"meta": {"N": N, "P": P, "horizon": H, "episodes": args.episodes, "device": device,
                    "tokenizer": TOKENIZER.name},
           "gt_kinematics": gt_kinematics(episodes),
           "chance_pos_err_px": chance_scale(episodes + probe_eps),
           "models": {}}
    print("[1] GT kinematics:", json.dumps(out["gt_kinematics"], indent=0))
    print("    chance pos_err ~ %.1f px" % out["chance_pos_err_px"])

    # tokenizer loaded once via the first model load; reuse for the latent probe
    for name, path in MODELS.items():
        if not path.is_file():
            print(f"  !! missing {name}: {path}")
            continue
        tok, dyn, dcfg, tok_win = load_models(TOKENIZER, path, N, device)
        K = dcfg.inference_steps
        print(f"\n[model] {name}  K={K} use_register_memory={getattr(dcfg,'use_register_memory',False)}")
        ol = open_loop_pos_err(tok, dyn, episodes, device, K, H)
        tf = teacher_forced_1step(tok, dyn, episodes, device, K, H, tok_win)
        out["models"][name] = {"open_loop": ol, "teacher_forced_1step": tf}
        sel = [h for h in (1, 2, 4, 8, 16, H) if h <= H]
        print("    open-loop pos_err vs horizon (model | copy-last):")
        for h in sel:
            print(f"      h={h:2d}: {ol['model_pos_err'][h]:5.1f} | {ol['copy_last_pos_err'][h]:5.1f}"
                  f"   lost={ol['ball_lost_rate'][h]:.2f}")
        print(f"    teacher-forced 1-step: model {tf['model_1step_pos_err_mean']:.2f}px "
              f"(median {tf['model_1step_pos_err_median']:.2f}) vs GT step "
              f"{tf['gt_1step_displacement_mean']:.2f}px  lost={tf['ball_lost_rate']:.2f}")

    # latent linear probe (tokenizer only, uses last-loaded tok)
    X, Y = collect_latent_xy(tok, probe_eps, device, tok_win)
    ntr = int(len(X) * 0.8)
    out["latent_position_probe"] = linear_probe(X[:ntr], Y[:ntr], X[ntr:], Y[ntr:])
    print("\n[4] linear probe frozen tokenizer latents -> (x,y):",
          json.dumps(out["latent_position_probe"], indent=0))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
