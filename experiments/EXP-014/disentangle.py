"""EXP-014 (D-019) — Disentangle FF7's base-dynamics gain: LOSS vs RELAY-INFERENCE. NO TRAINING.

ORIENT worry #4 + the EXP-012 "bonus finding": FF7 sharpens 1-step teacher-forced pos_err ~4.6x
(vanilla_s0 4.66px >> ff7 ~1.0px). But that ~1px was produced through the register-RELAY inference
path (generate() dispatches use_register_memory=True ckpts to generate_memory(), dynamics_model.py:528),
which is a WINDOW-1 relay (last latent + carried register), not the <=N-1=7-frame windowed attention
the vanilla_s0 number used. So the ~1px conflates THREE things:
  (i)   better weights from the FF7 loss,
  (ii)  the register relay,
  (iii) window size (1 vs 7).

This experiment evaluates 1-step teacher-forced pos_err for each model through BOTH inference paths on
the IDENTICAL GT window, to separate (i) from (ii)+(iii):
  - "vanilla" path: force use_register_memory=False -> generate() uses windowed attention with
    learned-init scratch registers (exactly the forward FF7's own main diffusion loss uses).
  - "relay"   path: force use_register_memory=True  -> generate_memory() window-1 + carried register.

Comparison logic (see D-019 "Would change my mind"):
  * FF7-vanilla ~ 1px  ~ FF7-relay  -> the FF7 LOSS gives better windowed dynamics (regularizer).
  * FF7-vanilla ~ 4.5px, only FF7-relay ~ 1px -> the RELAY inference carries the win, not the weights.
  * vanilla_s0-relay also ~1px -> the relay (not the loss) carries even non-FF7 weights.

Sanity anchor: FF7 relay-path must reproduce EXP-012's ~0.96-1.02px (else the harness diverges).

Run from repo root (USE THE VENV for CUDA, see HOWTO/gpu_venv.md):
  venv/Scripts/python.exe experiments/EXP-014/disentangle.py [--episodes 32 --horizon 24]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import sys

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]   # experiments/EXP-014/ -> repo root
_SRC = _ROOT / "src"
for _p in (_SRC / "probe", _SRC / "C_multi_image_auto_encoder", _SRC / "D_dynamics_model"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from revisit_probe import (  # noqa: E402
    load_models, _encode_window, _decode_frame, detect_ball,
)
from probe_env import make_probe_episode  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
TOKENIZER = _ROOT / "trained_autoencoder.pt"
MODELS = {
    "vanilla_s0": _ROOT / "experiments" / "EXP-012" / "vanilla_s0.pt",
    "ff7_k1": _ROOT / "experiments" / "EXP-010" / "k1" / "ff7_k1_s0.pt",
    "ff7_k3": _ROOT / "experiments" / "EXP-010" / "k3" / "ff7_k3_s0.pt",
}
N, P = 8, 3  # inference window, visible prefix (match EXP-009/010/011/012)


def _xy(states, t):
    return states[t, :2].astype(np.float64)


@contextlib.contextmanager
def force_memory(dyn, value: bool):
    """Temporarily force the use_register_memory dispatch flag so generate() picks the
    windowed (False) or relay (True) path, regardless of how the checkpoint was trained."""
    old = getattr(dyn.config, "use_register_memory", False)
    dyn.config.use_register_memory = value
    try:
        yield
    finally:
        dyn.config.use_register_memory = old


@torch.no_grad()
def teacher_forced_1step(tok, dyn, episodes, device, K, H, path: str):
    """1-step teacher-forced pos_err: GT window ending at t, predict frame t+1, compare to GT.
    `path` in {"vanilla","relay"} selects the inference dispatch via force_memory. The GT context
    window is IDENTICAL across paths (<= N-1 frames); only the inference path differs."""
    use_mem = (path == "relay")
    errs, disp, lost = [], [], 0
    maxctx = N - 1
    for ep in episodes:
        for t in range(P - 1, P - 1 + H):           # predict frame t+1 from GT window ending at t
            w = min(t + 1, maxctx)
            lo = t + 1 - w
            ctx = _encode_window(tok, ep.frames, lo, t + 1, device)            # (1, w, L, d) GT latents
            act = torch.from_numpy(ep.actions[lo:t + 2].astype(np.int64)).unsqueeze(0).to(device)
            with force_memory(dyn, use_mem):
                gen1 = dyn.generate(ctx, n_generate=1, K=K, action_idx=act)
            gt = _xy(ep.states, t + 1)
            disp.append(float(np.hypot(*(gt - _xy(ep.states, t)))))
            found, x, y, _ = detect_ball(_decode_frame(tok, gen1[:, 0]))
            if not found:
                lost += 1
                continue
            errs.append(float(np.hypot(x - gt[0], y - gt[1])))
    n = len(errs) + lost
    errs = np.array(errs)
    return {
        "path": path,
        "model_1step_pos_err_mean": float(errs.mean()) if len(errs) else float("nan"),
        "model_1step_pos_err_median": float(np.median(errs)) if len(errs) else float("nan"),
        "gt_1step_displacement_mean": float(np.mean(disp)),
        "ball_lost_rate": lost / max(n, 1),
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "results.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H = args.horizon

    # Same episode seeds as EXP-011/012 teacher-forced eval (20000+i) so numbers line up.
    episodes = [make_probe_episode(seed=20000 + i, P=P, k=0, R=H) for i in range(args.episodes)]

    out = {"meta": {"N": N, "P": P, "horizon": H, "episodes": args.episodes, "device": device,
                    "tokenizer": TOKENIZER.name,
                    "note": "1-step teacher-forced pos_err through vanilla(windowed) vs relay(window-1) "
                            "inference paths on identical GT windows. D-019 / worry #4."},
           "models": {}}

    for name, mpath in MODELS.items():
        if not mpath.is_file():
            print(f"  !! missing {name}: {mpath}")
            continue
        tok, dyn, dcfg, tok_win = load_models(TOKENIZER, mpath, N, device)
        K = dcfg.inference_steps
        trained_mem = getattr(dcfg, "use_register_memory", False)
        print(f"\n[model] {name}  K={K} trained_use_register_memory={trained_mem}")
        res = {"trained_use_register_memory": trained_mem, "K": K}
        for path in ("vanilla", "relay"):
            tf = teacher_forced_1step(tok, dyn, episodes, device, K, H, path)
            res[path] = tf
            print(f"    {path:7s}: 1-step pos_err {tf['model_1step_pos_err_mean']:5.2f}px "
                  f"(median {tf['model_1step_pos_err_median']:5.2f}) "
                  f"lost={tf['ball_lost_rate']:.2f}  (GT step {tf['gt_1step_displacement_mean']:.2f}px)")
        out["models"][name] = res

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
