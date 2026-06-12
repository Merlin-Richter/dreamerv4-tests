"""Revisit-consistency probe (T-002 / D-011) — the H2/H3 spine.

Measures how well the world model recalls hidden ball color + position after the
revealing frames have left its context window, as a function of occlusion length
``n_occ``, at a fixed sliding window ``N`` and visible prefix ``P``.

Protocol (per trial, PURE GENERATION):
  1. Build a probe episode: [P up | n_occ down | R up]; encode the full GT clip with the
     frozen tokenizer to get target latents.
  2. Context = the encoded *visible prefix* (P frames). Roll out n_occ+R frames with the
     real action stream; the sliding window inside generate() drops the prefix once the
     sequence exceeds N, so for n_occ >= N-1 the model must have CARRIED the info.
  3. Predicted reveal latent = last generated frame. Compare to the GT reveal latent
     (latent-MSE, primary) and decode it to read ball position/color (decomposition).

Metrics vs n_occ: latent_mse, pos_err_px, color_dRGB, ball_lost_rate.
Controls: ceiling (n_occ=0), chance (curtain-only prefix), drift (all-visible rollout).

Run:  python -m src.probe.revisit_probe --dry-run        # fast, tiny grid
      python -m src.probe.revisit_probe                  # full sweep

Window N is set by overriding the loaded model's config.max_temporal_length (RoPE is
relative; running M<N needs no retrain — D-011). Models are cast to float32 for
inference stability (checkpoints are bf16).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import fields

import numpy as np
import torch

_SRC = pathlib.Path(__file__).resolve().parents[1]            # .../src
_ROOT = _SRC.parent                                           # repo root
for _p in (_SRC / "probe", _SRC / "C_multi_image_auto_encoder", _SRC / "D_dynamics_model"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from video_auto_encoder import AutoEncoder, AutoEncoderConfig   # noqa: E402
from dynamics_model import DynamicsModel, DynamicsModelConfig   # noqa: E402

from probe_env import (  # noqa: E402
    make_probe_episode, make_probe_batch, ProbeEpisode, ACTION_DOWN,
)

DEFAULT_TOKENIZER = _ROOT / "trained_autoencoder.pt"
DEFAULT_DYNAMICS = _ROOT / "my_dynamics.pt"

# Ball detector: ball is high-VALUE (>=225) on a dark bg (<=95) / mid curtain (~60),
# so a fixed value threshold cleanly isolates it. Value = max over channels.
BALL_VALUE_THRESH = 150
BALL_MIN_AREA = 5  # px; fewer bright px than this => "ball lost"


# --------------------------------------------------------------------------- loading
def _config_from_checkpoint(cfg_dict: dict, cls):
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in cfg_dict.items() if k in allowed})


def load_models(tokenizer_path: pathlib.Path, dynamics_path: pathlib.Path,
                window_N: int, device: str):
    tok_payload = torch.load(tokenizer_path, map_location=device, weights_only=False)
    tcfg = _config_from_checkpoint(tok_payload["config"], AutoEncoderConfig)
    tok = AutoEncoder(tcfg)
    tok.load_state_dict(tok_payload["model_state_dict"])
    tok = tok.to(device).float().eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    tok_win = tcfg.max_temporal_length

    dyn_payload = torch.load(dynamics_path, map_location=device, weights_only=False)
    dcfg = _config_from_checkpoint(dyn_payload["config"], DynamicsModelConfig)
    dyn = DynamicsModel(dcfg)                       # RoPE table built at trained length (16)
    dyn.load_state_dict(dyn_payload["model_state_dict"])
    dyn = dyn.to(device).float().eval()
    for p in dyn.parameters():
        p.requires_grad_(False)
    # Set the inference window: generate() uses max_ctx = max_temporal_length - 1.
    dyn.config.max_temporal_length = window_N
    return tok, dyn, dcfg, tok_win


# --------------------------------------------------------------------------- core ops
@torch.no_grad()
def _encode_window(tok: AutoEncoder, frames_u8: np.ndarray, lo: int, hi: int,
                   device: str) -> torch.Tensor:
    """Encode frames[lo:hi] -> (1, hi-lo, n_lat, dim). hi-lo must be <= tokenizer window."""
    x = torch.from_numpy(frames_u8[lo:hi].astype(np.float32) / 255.0).unsqueeze(0).to(device)
    return tok.encoder(x)


@torch.no_grad()
def _prefix_latents(tok, frames_u8, P, device) -> torch.Tensor:
    """Context latents for the visible prefix. Causal encoder => independent of later frames."""
    return _encode_window(tok, frames_u8, 0, P, device)          # (1,P,L,d)


@torch.no_grad()
def _target_latent(tok, frames_u8, index, tok_win, device) -> torch.Tensor:
    """GT latent of frame `index`, encoded with its full causal context (a window ending
    at `index`, length <= tok_win) — matches how training targets are built."""
    lo = max(0, index - (tok_win - 1))
    lat = _encode_window(tok, frames_u8, lo, index + 1, device)  # (1, <=tok_win, L, d)
    return lat[:, -1]                                            # (1,L,d)


@torch.no_grad()
def _trial(tok, dyn, ep: ProbeEpisode, measure_index: int, device: str, K: int,
           tok_win: int):
    """One pure-generation rollout. Returns (latent_se, found, pos_err_px, color_dRGB).

    Context = encoded visible prefix; the model generates all post-prefix frames; we
    score the latent at `measure_index` against the GT tokenizer latent there, and decode
    it to read ball position/color. pos/color are NaN when no ball is detected.
    """
    context = _prefix_latents(tok, ep.frames, ep.P, device)
    action_idx = torch.from_numpy(ep.actions.astype(np.int64)).unsqueeze(0).to(device)
    gen = dyn.generate(context, n_generate=ep.frames.shape[0] - ep.P, K=K, action_idx=action_idx)
    full = torch.cat((context, gen), dim=1)                      # (1, T, L, d) predicted latents
    pred = full[:, measure_index]
    gt = _target_latent(tok, ep.frames, measure_index, tok_win, device)
    latent_se = float(((pred - gt) ** 2).mean().item())
    found, x, y, color = detect_ball(_decode_frame(tok, pred))
    if not found:
        return latent_se, False, float("nan"), float("nan")
    gx, gy = ep.states[measure_index, :2]
    pos = float(np.hypot(x - gx, y - gy))
    cdr = float(np.abs(color.astype(np.float32) - ep.ball_color.astype(np.float32)).mean())
    return latent_se, True, pos, cdr


def _aggregate(trials: list) -> dict:
    """Mean over trials; pos/color averaged only over trials where a ball was found."""
    n = len(trials)
    lat = [t[0] for t in trials]
    pos = [t[2] for t in trials if t[1]]
    col = [t[3] for t in trials if t[1]]
    lost = sum(1 for t in trials if not t[1])
    return {
        "n": n,
        "latent_mse": float(np.mean(lat)),
        "latent_mse_std": float(np.std(lat)),
        "pos_err_px": float(np.mean(pos)) if pos else float("nan"),
        "color_dRGB": float(np.mean(col)) if col else float("nan"),
        "ball_lost_rate": lost / n,
    }


@torch.no_grad()
def _decode_frame(tok: AutoEncoder, latent_1Ld: torch.Tensor) -> np.ndarray:
    """(1,L,d) -> (H,W,3) uint8 native order."""
    img = tok.decoder(latent_1Ld.unsqueeze(1))[0, 0]             # (H,W,3) in ~[0,1]
    return (img.clamp(0, 1).float().cpu().numpy() * 255.0).round().astype(np.uint8)


def detect_ball(frame_u8: np.ndarray):
    """Find the bright ball. Returns (found: bool, x: float, y: float, color: (3,) native).

    x = column, y = row (matches env / cv2.circle convention).
    """
    value = frame_u8.max(axis=2)                                 # (H,W)
    mask = value >= BALL_VALUE_THRESH
    area = int(mask.sum())
    if area < BALL_MIN_AREA:
        return False, float("nan"), float("nan"), np.array([np.nan] * 3)
    ys, xs = np.nonzero(mask)
    x = float(xs.mean())
    y = float(ys.mean())
    color = frame_u8[ys, xs].mean(axis=0)                        # mean over ball pixels
    return True, x, y, color


# --------------------------------------------------------------------------- detector gate
def validate_detector_on_gt(episodes: list[ProbeEpisode]) -> dict:
    """Acceptance #3: the detector must recover GT color/position from GT reveal frames.

    If this fails, a bad color/position number is the instrument's fault, not the model's.
    """
    pos_errs, color_errs, misses = [], [], 0
    for ep in episodes:
        gt_frame = ep.frames[ep.reveal_index]
        found, x, y, color = detect_ball(gt_frame)
        if not found:
            misses += 1
            continue
        gx, gy = ep.gt_xy
        pos_errs.append(float(np.hypot(x - gx, y - gy)))
        color_errs.append(float(np.abs(color.astype(np.float32) - ep.ball_color.astype(np.float32)).mean()))
    n = len(episodes)
    return {
        "n": n,
        "miss_rate": misses / n,
        "pos_err_px_p99": float(np.percentile(pos_errs, 99)) if pos_errs else float("nan"),
        "pos_err_px_mean": float(np.mean(pos_errs)) if pos_errs else float("nan"),
        "color_dRGB_p99": float(np.percentile(color_errs, 99)) if color_errs else float("nan"),
        "color_dRGB_mean": float(np.mean(color_errs)) if color_errs else float("nan"),
        "pass": bool(pos_errs and np.percentile(pos_errs, 99) < 1.5
                     and np.percentile(color_errs, 99) < 10.0 and misses / n <= 0.01),
    }


# --------------------------------------------------------------------------- sweep
def run_condition(tok, dyn, episodes: list[ProbeEpisode], device: str, K: int,
                  tok_win: int, measure_index=None) -> dict:
    """Aggregate metrics over episodes, scoring each at `measure_index` (default: the
    episode's reveal frame)."""
    trials = [
        _trial(tok, dyn, ep,
               ep.reveal_index if measure_index is None else measure_index,
               device, K, tok_win)
        for ep in episodes
    ]
    return _aggregate(trials)


def run_probe(args) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, dyn, dcfg, tok_win = load_models(args.tokenizer, args.dynamics, args.window_N, device)
    K = dcfg.inference_steps if args.K is None else args.K
    P, R = args.prefix_P, 1

    occ_grid = [2, 8, 16] if args.dry_run else [2, 4, 6, 7, 8, 9, 12, 16, 24]
    n_eps = 4 if args.dry_run else args.episodes

    # Detector gate FIRST (acceptance #3) — on GT frames across the grid.
    gate_eps = make_probe_batch(k=max(occ_grid), n_seeds=n_eps, P=P, R=R)
    detector = validate_detector_on_gt(gate_eps)

    results = {
        "meta": {
            "window_N": args.window_N, "prefix_P": P, "R": R, "K": K,
            "context_signal": dyn.config.context_signal, "device": device,
            "dry_run": args.dry_run, "episodes_per_occ": n_eps,
            "tokenizer": str(args.tokenizer.name), "dynamics": str(args.dynamics.name),
        },
        "detector_gate": detector,
        "occ_grid": occ_grid,
        "latent_mse_by_occ": {}, "pos_err_px_by_occ": {},
        "color_dRGB_by_occ": {}, "ball_lost_rate_by_occ": {},
        # Drift control as a matched-horizon CURVE: same rollout length / measure index
        # as each n_occ but curtain UP throughout. Differencing occluded - drift isolates
        # memory loss from ordinary autoregressive drift.
        "drift_by_occ": {"latent_mse": {}, "pos_err_px": {}, "color_dRGB": {}},
        "controls": {},
    }

    for n_occ in occ_grid:
        eps = make_probe_batch(k=n_occ, n_seeds=n_eps, P=P, R=R, seed0=1000 + n_occ)
        m = run_condition(tok, dyn, eps, device, K, tok_win)
        # Matched-horizon drift: all-visible episode of the same length, same measure idx.
        drift_eps = [make_probe_episode(seed=7000 + n_occ * 100 + i, P=P, k=0, R=n_occ + R)
                     for i in range(n_eps)]
        d = run_condition(tok, dyn, drift_eps, device, K, tok_win, measure_index=P + n_occ)
        results["latent_mse_by_occ"][str(n_occ)] = m["latent_mse"]
        results["pos_err_px_by_occ"][str(n_occ)] = m["pos_err_px"]
        results["color_dRGB_by_occ"][str(n_occ)] = m["color_dRGB"]
        results["ball_lost_rate_by_occ"][str(n_occ)] = m["ball_lost_rate"]
        results["drift_by_occ"]["latent_mse"][str(n_occ)] = d["latent_mse"]
        results["drift_by_occ"]["pos_err_px"][str(n_occ)] = d["pos_err_px"]
        results["drift_by_occ"]["color_dRGB"][str(n_occ)] = d["color_dRGB"]
        print(f"  n_occ={n_occ:2d}  latentMSE={m['latent_mse']:.4f} (drift {d['latent_mse']:.4f})"
              f"  posErr={m['pos_err_px']:.1f}px (drift {d['pos_err_px']:.1f})"
              f"  dRGB={m['color_dRGB']:.1f} (drift {d['color_dRGB']:.1f})  lost={m['ball_lost_rate']:.2f}")

    # --- Calibration controls (not the H2 test itself) ---
    # Ceiling: fully visible context (n_occ=0), best achievable.
    ceil_eps = make_probe_batch(k=0, n_seeds=n_eps, P=P, R=R, seed0=5000)
    results["controls"]["ceiling"] = run_condition(tok, dyn, ceil_eps, device, K, tok_win)

    # Chance floor: curtain-only context (prefix_action=DOWN) -> model never sees the
    # ball, so the reveal prediction is its prior. Reuses the standard path (k=0).
    chance_eps = [make_probe_episode(seed=6000 + i, P=P, k=0, R=R, prefix_action=ACTION_DOWN)
                  for i in range(n_eps)]
    results["controls"]["chance"] = run_condition(tok, dyn, chance_eps, device, K, tok_win)

    # Metric validation: does latent-MSE-vs-n_occ track the color/position errors?
    results["metric_validation"] = _metric_validation(results)
    return results


def _metric_validation(results: dict) -> dict:
    """Pearson r between latent-MSE and color/position errors across the n_occ grid."""
    occ = [str(k) for k in results["occ_grid"]]
    lat = np.array([results["latent_mse_by_occ"][k] for k in occ])

    def _r(other_key):
        y = np.array([results[other_key][k] for k in occ])
        ok = np.isfinite(lat) & np.isfinite(y)
        if ok.sum() < 3 or np.std(lat[ok]) == 0 or np.std(y[ok]) == 0:
            return float("nan")
        return float(np.corrcoef(lat[ok], y[ok])[0, 1])

    return {
        "pearson_latentMSE_vs_colorDRGB": _r("color_dRGB_by_occ"),
        "pearson_latentMSE_vs_posErr": _r("pos_err_px_by_occ"),
        "note": "latent-MSE is trustworthy as the headline iff it tracks the "
                "decomposition; position is expected to be drift-confounded (see drift_by_occ).",
    }


def parse_args():
    p = argparse.ArgumentParser(description="Revisit-consistency probe (T-002)")
    p.add_argument("--dry-run", action="store_true", help="tiny grid + few episodes")
    p.add_argument("--episodes", type=int, default=64, help="episodes per n_occ (full run)")
    p.add_argument("--window-N", type=int, default=8, help="sliding context window")
    p.add_argument("--prefix-P", type=int, default=3, help="visible prefix frames")
    p.add_argument("--K", type=int, default=None, help="diffusion/shortcut steps (default: cfg)")
    p.add_argument("--tokenizer", type=pathlib.Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--dynamics", type=pathlib.Path, default=DEFAULT_DYNAMICS)
    p.add_argument("--out", type=pathlib.Path,
                   default=_SRC / "probe" / "last_results.json",
                   help="results.json path; real experiments pass their EXP dir")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[probe] window_N={args.window_N} prefix_P={args.prefix_P} "
          f"dry_run={args.dry_run}")
    results = run_probe(args)
    g = results["detector_gate"]
    print(f"[detector gate] pass={g['pass']} pos_p99={g['pos_err_px_p99']:.2f}px "
          f"dRGB_p99={g['color_dRGB_p99']:.1f} miss={g['miss_rate']:.3f}")
    if not g["pass"]:
        print("  !! detector gate FAILED — color/position numbers are NOT trustworthy "
              "(instrument problem, not the model). Fix before reading recall.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
