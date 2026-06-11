"""EXP-008 — Context-noise rollout diagnostic (D-010).

Headless, inference-only sweep of the dynamics rollout's context-noise level
(`DynamicsModel.config.context_noise`, = tau_ctx) on the EXISTING EXP-007
checkpoint. Tests whether the EXP-007 rollout failure (ball color/position
randomized from the first generated frame, even with fully-visible context) is
caused by the context-noising convention: tau is the SIGNAL level in this codebase
(loss: z_tilde = (1-tau)*noise + tau*z1), so tau_ctx=0.1 means the context frames
are 90% noise -> the model can't read the ball from context and falls back to its
prior.

No training. No cluster. No GUI. Reuses the load/decode path from
`train_dynamics_model.run_test_checkpoint` but writes PNGs instead of imshow.

Run (full sweep):
    python experiments/EXP-008/diagnose_context_noise.py
Smoke (2 episodes):
    python experiments/EXP-008/diagnose_context_noise.py --n-episodes 2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
_DYN_DIR = _REPO / "src" / "D_dynamics_model"
_TOK_DIR = _REPO / "src" / "C_multi_image_auto_encoder"
for p in (_DYN_DIR, _TOK_DIR, _REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cv2  # noqa: E402
from dynamics_model import DynamicsModel, DynamicsModelConfig  # noqa: E402
from video_auto_encoder import AutoEncoder, AutoEncoderConfig  # noqa: E402
from train_dynamics_model import (  # noqa: E402
    _config_from_checkpoint,
    load_tokenizer,
    encode_frames,
)


def tensor01_to_bgr(img_t: torch.Tensor) -> np.ndarray:
    x = img_t.detach().cpu().float().clamp(0.0, 1.0).numpy()
    return cv2.cvtColor((x * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, default=_REPO / "occluded.npy")
    ap.add_argument("--actions", type=Path, default=_REPO / "occluded_actions.npy")
    ap.add_argument("--tokenizer", type=Path, default=_REPO / "trained_autoencoder.pt")
    ap.add_argument("--checkpoint", type=Path, default=_REPO / "my_dynamics.pt")
    ap.add_argument("--n-episodes", type=int, default=6)
    ap.add_argument("--context-frames", type=int, default=4)
    ap.add_argument("--inference-steps", type=int, default=None)
    ap.add_argument("--tau-ctx-list", type=str, default="0.1,0.5,0.9,0.99")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tau_list = [float(x) for x in args.tau_ctx_list.split(",")]

    # --- load models ---
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = _config_from_checkpoint(payload["config"], DynamicsModelConfig)
    model = DynamicsModel(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    tokenizer = load_tokenizer(args.tokenizer, device)

    L = cfg.max_temporal_length
    n_ctx = args.context_frames
    n_gen = L - n_ctx
    assert 1 <= n_ctx < L, f"context-frames must be in [1,{L-1}]"
    K = args.inference_steps or cfg.inference_steps

    raw = np.load(args.frames, mmap_mode="r")
    n_eps, n_frames, h, w, _ = raw.shape
    actions_raw = np.load(args.actions) if cfg.n_actions > 0 else None

    # Fixed (seeded) episode + start selection, shared across all tau settings.
    picks = []
    for _ in range(args.n_episodes):
        ep = int(rng.integers(0, n_eps))
        start = int(rng.integers(0, max(1, n_frames - L + 1)))
        picks.append((ep, start))

    img_dir = args.outdir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "checkpoint": str(args.checkpoint.name),
        "tokenizer": str(args.tokenizer.name),
        "n_episodes": args.n_episodes,
        "context_frames": n_ctx,
        "n_generate": n_gen,
        "K": K,
        "seed": args.seed,
        "picks": picks,
        "by_tau": {},
    }

    for tau in tau_list:
        model.config.context_noise = tau
        per_ep_frame_mse = []  # (n_episodes, n_gen)
        per_tau_sheet_rows = []
        for (ep, start) in picks:
            clip = raw[ep, start:start + L].astype(np.float32) / 255.0
            x = torch.from_numpy(clip).unsqueeze(0).to(device)  # (1,L,H,W,3)
            action_idx = None
            if actions_raw is not None:
                action_idx = torch.from_numpy(
                    actions_raw[ep, start:start + L].astype(np.int64)
                ).unsqueeze(0).to(device)

            with torch.no_grad():
                latents = encode_frames(tokenizer, x)        # (1,L,n_lat,dim)
                context = latents[:, :n_ctx]
                gen = model.generate(context, n_generate=n_gen, K=K, action_idx=action_idx)
                full = torch.concat((context, gen), dim=1)
                recon = tokenizer.decoder(full)[0]           # (L,H,W,3)

            # pixel MSE on GENERATED frames only (the part the rollout invents)
            gt_gen = x[0, n_ctx:].detach().cpu().float().clamp(0, 1)
            pr_gen = recon[n_ctx:].detach().cpu().float().clamp(0, 1)
            frame_mse = ((gt_gen - pr_gen) ** 2).mean(dim=(1, 2, 3)).numpy()  # (n_gen,)
            per_ep_frame_mse.append(frame_mse)

            gt_row = np.hstack([tensor01_to_bgr(x[0, t]) for t in range(L)])
            pr_row = np.hstack([tensor01_to_bgr(recon[t]) for t in range(L)])
            boundary = n_ctx * w
            cv2.line(pr_row, (boundary, 0), (boundary, pr_row.shape[0]), (0, 0, 255), 1)
            pair = np.vstack([gt_row, pr_row])
            scale = max(2, 512 // (max(h, w) * L))
            disp = cv2.resize(pair, (pair.shape[1] * scale, pair.shape[0] * scale),
                              interpolation=cv2.INTER_NEAREST)
            tau_tag = f"{tau:.2f}".replace(".", "p")
            cv2.imwrite(str(img_dir / f"ep{ep}_s{start}_tau{tau_tag}.png"), disp)
            per_tau_sheet_rows.append(pair)

        # contact sheet for this tau (episodes stacked)
        max_w = max(r.shape[1] for r in per_tau_sheet_rows)
        padded = [np.pad(r, ((2, 2), (0, max_w - r.shape[1]), (0, 0)),
                         constant_values=64) for r in per_tau_sheet_rows]
        sheet = np.vstack(padded)
        s = max(2, 512 // (max(h, w) * L))
        sheet = cv2.resize(sheet, (sheet.shape[1] * s, sheet.shape[0] * s),
                           interpolation=cv2.INTER_NEAREST)
        tau_tag = f"{tau:.2f}".replace(".", "p")
        cv2.imwrite(str(img_dir / f"_sheet_tau{tau_tag}.png"), sheet)

        arr = np.stack(per_ep_frame_mse)  # (n_eps, n_gen)
        results["by_tau"][f"{tau}"] = {
            "gen_mse_per_frame": arr.mean(axis=0).round(6).tolist(),
            "gen_mse_mean": float(arr.mean().round(6)),
        }
        print(f"tau_ctx={tau:<5} gen_mse_mean={arr.mean():.6f}")

    with open(args.outdir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.outdir / 'results.json'} and {img_dir}/")


if __name__ == "__main__":
    main()
