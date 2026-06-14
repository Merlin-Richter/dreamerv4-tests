"""EXP-017 frozen-probe COLOR comparison: FF9 v2 vs vanilla_s0 vs ff7_k3.

Reuses the FROZEN probe (5503e75) functions UNCHANGED (load_models, run_condition,
validate_detector_on_gt) and the frozen env builders — only the n_occ grid is extended to
{...,32,48} (the frozen run_probe hardcodes a max of 24). No edit to the instrument (§8).
Seed conventions mirror run_probe exactly so numbers are comparable to EXP-009/010/012.

Each model's generate() auto-dispatches by its own config: vanilla -> sliding window, ff7 ->
register relay (generate_memory), ff9v2 -> generate_full_state_memory (A1+B1, V-T013-eval).

Run: venv/Scripts/python.exe -u experiments/EXP-017/frozen_color.py
"""
import sys, json, pathlib
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "probe"))
sys.path.insert(0, str(ROOT / "src" / "D_dynamics_model"))
sys.path.insert(0, str(ROOT / "src" / "C_multi_image_auto_encoder"))

from revisit_probe import load_models, run_condition, validate_detector_on_gt  # frozen
from probe_env import make_probe_batch, make_probe_episode, ACTION_DOWN        # frozen

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOK = ROOT / "trained_autoencoder.pt"
GRID = [2, 6, 8, 12, 16, 24, 32, 48]
N_EPS = 64
P, R, WINDOW_N = 3, 1, 8

MODELS = {
    "ff9v2_s0":   ROOT / "experiments" / "EXP-017" / "ff9v2_s0.pt",
    "vanilla_s0": ROOT / "experiments" / "EXP-012" / "vanilla_s0.pt",
    "ff7_k3":     ROOT / "experiments" / "EXP-010" / "k3" / "ff7_k3_s0.pt",
}
OUT = ROOT / "experiments" / "EXP-017" / "frozen_color.json"


def sweep_model(name, ckpt):
    print(f"\n=== {name}  ({ckpt.relative_to(ROOT)}) ===")
    tok, dyn, dcfg, tok_win = load_models(TOK, ckpt, WINDOW_N, DEVICE)
    K = dcfg.inference_steps
    disp = ("full_state_memory" if getattr(dcfg, "use_full_state_memory", False)
            else "register_memory" if getattr(dcfg, "use_register_memory", False)
            else "sliding_window")
    print(f"  dispatch={disp}  K={K}  n_memory={getattr(dcfg,'n_memory',0)}")
    res = {"dispatch": disp, "color_dRGB_by_occ": {}, "drift_color_dRGB_by_occ": {},
           "latent_mse_by_occ": {}, "pos_err_px_by_occ": {}, "ball_lost_rate_by_occ": {}}
    for n_occ in GRID:
        eps = make_probe_batch(k=n_occ, n_seeds=N_EPS, P=P, R=R, seed0=1000 + n_occ)
        m = run_condition(tok, dyn, eps, DEVICE, K, tok_win)
        drift_eps = [make_probe_episode(seed=7000 + n_occ * 100 + i, P=P, k=0, R=n_occ + R)
                     for i in range(N_EPS)]
        d = run_condition(tok, dyn, drift_eps, DEVICE, K, tok_win, measure_index=P + n_occ)
        res["color_dRGB_by_occ"][str(n_occ)] = m["color_dRGB"]
        res["drift_color_dRGB_by_occ"][str(n_occ)] = d["color_dRGB"]
        res["latent_mse_by_occ"][str(n_occ)] = m["latent_mse"]
        res["pos_err_px_by_occ"][str(n_occ)] = m["pos_err_px"]
        res["ball_lost_rate_by_occ"][str(n_occ)] = m["ball_lost_rate"]
        print(f"  n_occ={n_occ:2d}  dRGB={m['color_dRGB']:6.1f} (drift {d['color_dRGB']:5.1f})"
              f"  latMSE={m['latent_mse']:.3f}  posErr={m['pos_err_px']:5.1f}px  lost={m['ball_lost_rate']:.2f}")
    # controls
    ceil_eps = make_probe_batch(k=0, n_seeds=N_EPS, P=P, R=R, seed0=5000)
    chance_eps = [make_probe_episode(seed=6000 + i, P=P, k=0, R=R, prefix_action=ACTION_DOWN)
                  for i in range(N_EPS)]
    res["ceiling"] = run_condition(tok, dyn, ceil_eps, DEVICE, K, tok_win)
    res["chance"] = run_condition(tok, dyn, chance_eps, DEVICE, K, tok_win)
    print(f"  ceiling dRGB={res['ceiling']['color_dRGB']:.1f}  chance dRGB={res['chance']['color_dRGB']:.1f}")
    return res


def main():
    # Detector gate once (model-independent: GT frames). Reuse the longest grid point.
    gate = validate_detector_on_gt(make_probe_batch(k=max(GRID), n_seeds=N_EPS, P=P, R=R))
    print(f"[detector gate] pass={gate['pass']} pos_p99={gate['pos_err_px_p99']:.2f} "
          f"color_p99={gate['color_dRGB_p99']:.2f} miss={gate['miss_rate']:.3f}")
    out = {"meta": {"grid": GRID, "episodes_per_occ": N_EPS, "P": P, "R": R, "window_N": WINDOW_N,
                    "probe_frozen": "5503e75", "device": DEVICE},
           "detector_gate": gate, "models": {}}
    for name, ckpt in MODELS.items():
        if not ckpt.is_file():
            print(f"  !! missing {ckpt}"); continue
        out["models"][name] = sweep_model(name, ckpt)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
