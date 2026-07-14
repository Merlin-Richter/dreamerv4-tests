"""Quantitative Memory-Maze rollout-error evaluation (spec: specs/evals/memmaze/rollout_error.md).

The quantitative companion to `evals/memmaze/sheets.py`: how fast ordinary visual prediction error
grows over a short autoregressive rollout, measured identically for every dynamics model so vanilla,
memory-token, and archive variants are directly comparable on the same held-out data.

Protocol (per scored sample = one (episode, start) pair):
  1. STREAMED PREFILL — take ``n_prefill`` (default 128) sequential ground-truth frames + their
     actions and commit them through the model's NORMAL sliding window. The dynamics window is only
     ``max_temporal_length`` (32) frames; ``rollout_init`` commits the first window in one pass then
     teacher-forces the rest one frame at a time, evicting as it slides — so this is a 128-frame
     *streamed* prefill, NOT a 128-frame context window. As frames leave the window, memory / archive
     mechanisms are free to carry their information forward; a vanilla model simply forgets them.
  2. SCORED ROLLOUT — generate the next ``n_gen`` (default 32) frames autoregressively on the TRUE
     action sequence. GENERATED frames (not ground truth) become the visual history during scoring.
  3. SCORE — decode every generated latent and compare to the real ground-truth frame with
     pixel-space MSE (frames in [0,1], mean over H·W·C). Per horizon 1..n_gen, average across samples.

Reference baselines (model-independent, computed once per eval, saved alongside the model curve):
  * ``tokenizer_floor`` — decode the TRUE latents for the scored frames. The reconstruction ceiling:
    the smallest MSE any model could reach through this frozen tokenizer. Model curves sit above it.
  * ``copy_last`` — hold the last prefill ground-truth frame constant for every horizon. The naive
    no-dynamics reference (error a static prediction accrues purely from the scene moving).

Everything is BATCHED across samples (the batch axis = episodes/starts): the 128-frame prefill and
32-frame rollout run in parallel for a whole batch at once, per the throughput requirement. Latents
are encoded in non-overlapping tokenizer-window blocks, exactly as the training latent cache does
(train_dynamics.py) — latents are ~window-invariant, so this matches the training distribution.

Frame/action alignment is the established one (train_dynamics / sheets / generate): action[t] pairs
with frame[t], so the generated frame at absolute position p is conditioned on action[p].

This is a SHORT-HORIZON visual-error instrument, not a full measure of world-model correctness:
autoregressive butterfly divergence can give a high pixel error to a rollout that stays visually
plausible. Capping the scored rollout at 32 frames keeps that manageable; claims must keep the caveat.

BGR end-to-end, no channel swap. Results are saved to reusable JSON (outputs/rollout_error/) so the
companion plot (evals/memmaze/plot_rollout_error.py) overlays runs without re-running inference. Run -u.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/
from evals.gridworld.recall import _tokenizer_window  # noqa: E402
from evals.gridworld.sheets import _load  # noqa: E402  — shared checkpoint loader

N_PREFILL = 128   # ground-truth frames streamed through the sliding window before scoring
N_GEN = 32        # scored autoregressive rollout length (the horizon axis)


# --------------------------------------------------------------------------- encode / decode
@torch.no_grad()
def _encode_windowed(tokenizer, frames01: torch.Tensor, tok_w: int) -> torch.Tensor:
    """Encode (B, T, H, W, 3) in [0,1] to latents (B, T, n_latents, bottleneck_dim) in NON-OVERLAPPING
    windows of the tokenizer temporal window — identical to the training latent cache. The causal
    encoder sees only its own window; latents are ~window-invariant so this matches training."""
    T = frames01.shape[1]
    outs = [tokenizer.encoder(frames01[:, w0:w0 + tok_w]) for w0 in range(0, T, tok_w)]
    return torch.cat(outs, dim=1)


@torch.no_grad()
def _decode_tail(tokenizer, ctx_lat: torch.Tensor, gen_lat: torch.Tensor, tok_w: int) -> torch.Tensor:
    """Decode the ``n_gen`` frames of ``gen_lat`` to (B, n_gen, H, W, 3) in [0,1], giving the decoder a
    real temporal prefix (the tail of ``ctx_lat``) then slicing the generated tail — the same trailing-
    window decode as the qualitative sheet, so sheet pixels and scored pixels agree."""
    n_gen = gen_lat.shape[1]
    prefix = ctx_lat[:, -(tok_w - n_gen):] if tok_w > n_gen else ctx_lat[:, :0]
    seq = torch.cat((prefix, gen_lat), dim=1)[:, -tok_w:]
    return tokenizer.decoder(seq)[:, -n_gen:].clamp(0, 1)


def _mse_per_horizon(pred01: torch.Tensor, gt01: torch.Tensor) -> torch.Tensor:
    """(B, n_gen, H, W, 3) pred & gt in [0,1] -> (B, n_gen) mean pixel MSE over H·W·C."""
    return ((pred01 - gt01) ** 2).flatten(2).mean(dim=2)


# --------------------------------------------------------------------------- sample selection
def val_episodes(n_episodes: int, val_fraction: float = 0.05) -> np.ndarray:
    """The trainer's held-out episode ids (train_dynamics split: randperm seed-0). For the val12 file
    every episode is already held out, so callers usually pass val_fraction=0 (use all episodes)."""
    n_val = min(max(1, int(round(n_episodes * val_fraction))), n_episodes - 1)
    g = torch.Generator().manual_seed(0)
    return torch.randperm(n_episodes, generator=g)[:n_val].numpy()


def build_samples(frames, episodes, n_samples: int, need: int, seed: int) -> list[tuple[int, int]]:
    """Deterministic (episode, start) sample list — a pure function of (episode set, n_samples, need,
    seed), so every model evaluated with the same arguments scores the IDENTICAL samples. Starts are
    drawn seeded per episode; episodes are cycled round-robin until n_samples pairs are collected."""
    g = torch.Generator().manual_seed(seed)
    usable = [ep for ep in episodes if int(np.asarray(frames[ep]).shape[0]) >= need]
    if not usable:
        raise SystemExit(f"No episode has >= {need} frames (n_prefill+n_gen).")
    pairs = []
    i = 0
    while len(pairs) < n_samples:
        ep = int(usable[i % len(usable)])
        T = int(np.asarray(frames[ep]).shape[0])
        s = int(torch.randint(0, T - need + 1, (1,), generator=g).item())
        pairs.append((ep, s))
        i += 1
    return pairs


# --------------------------------------------------------------------------- core eval
@torch.no_grad()
def rollout_error(model, tokenizer, frames, actions, *, samples, n_prefill=N_PREFILL, n_gen=N_GEN,
                  K=4, device="cpu", window=None, batch_size=16):
    """Batched streamed-prefill rollout-error over ``samples`` = list of (episode, start).

    Returns per-horizon means as numpy arrays: ``mse`` (model), ``tokenizer_floor``, ``copy_last``,
    plus ``mse_std`` and ``n_finite`` (finite-sample count per horizon). ``window`` (total frames)
    forces a shorter sliding window than training; None = native (max_temporal_length-1)."""
    tok_w = _tokenizer_window(tokenizer)
    assert n_gen <= tok_w, f"n_gen={n_gen} exceeds tokenizer window {tok_w} (one-shot decode)"
    max_ctx = None if window is None else max(1, window - 1)
    need = n_prefill + n_gen
    if actions is None and model.n_actions > 0:
        raise SystemExit(f"model is action-conditioned (n_actions={model.n_actions}) but no actions "
                         "file — an unconditioned memmaze rollout is meaningless.")

    per_sample_mse, per_sample_floor, per_sample_copy = [], [], []  # each -> (n, n_gen)
    for b0 in range(0, len(samples), batch_size):
        chunk = samples[b0:b0 + batch_size]
        fr = np.stack([np.asarray(frames[ep][s:s + need]) for ep, s in chunk]).astype(np.float32) / 255.0
        f01 = torch.from_numpy(fr).to(device)                          # (Bc, need, H, W, 3)
        act = None
        if actions is not None:
            aa = np.stack([np.asarray(actions[ep][s:s + need]) for ep, s in chunk]).astype(np.int64)
            act = torch.from_numpy(aa).to(device)                      # (Bc, need)

        all_lat = _encode_windowed(tokenizer, f01, tok_w)              # (Bc, need, L, D)
        ctx_lat = all_lat[:, :n_prefill]
        true_gen_lat = all_lat[:, n_prefill:need]
        gt_gen = f01[:, n_prefill:need]                                # (Bc, n_gen, H, W, 3)

        # --- model rollout: streamed prefill then autoregressive generation on TRUE actions ---
        ctx_act = act[:, :n_prefill] if act is not None else None
        state = model.rollout_init(ctx_lat, ctx_act, K, max_ctx=max_ctx)
        gen = []
        for i in range(n_gen):
            a = act[:, n_prefill + i:n_prefill + i + 1] if act is not None else None
            gen.append(model.rollout_step(state, a, commit=True))
        gen_lat = torch.cat(gen, dim=1)                                # (Bc, n_gen, L, D)

        pred = _decode_tail(tokenizer, ctx_lat, gen_lat, tok_w)        # (Bc, n_gen, H, W, 3)
        floor = _decode_tail(tokenizer, ctx_lat, true_gen_lat, tok_w)
        last = f01[:, n_prefill - 1:n_prefill].expand(-1, n_gen, -1, -1, -1)  # frozen last prefill frame

        per_sample_mse.append(_mse_per_horizon(pred, gt_gen).cpu().numpy())
        per_sample_floor.append(_mse_per_horizon(floor, gt_gen).cpu().numpy())
        per_sample_copy.append(_mse_per_horizon(last, gt_gen).cpu().numpy())

    mse = np.concatenate(per_sample_mse, axis=0)                       # (n, n_gen)
    floor = np.concatenate(per_sample_floor, axis=0)
    copy = np.concatenate(per_sample_copy, axis=0)
    finite = np.isfinite(mse)
    return {
        "mse": np.where(finite.any(0), np.nanmean(np.where(finite, mse, np.nan), axis=0), np.nan),
        "mse_std": np.nanstd(np.where(finite, mse, np.nan), axis=0),
        "tokenizer_floor": floor.mean(axis=0),
        "copy_last": copy.mean(axis=0),
        "n_finite": finite.sum(axis=0),
    }


# --------------------------------------------------------------------------- provenance / IO
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _model_config(model) -> dict:
    c = model.config
    keys = ["max_temporal_length", "n_memory", "ff9_k", "n_actions", "n_registers", "gqa_groups",
            "embedding_dim", "depth", "inference_steps", "context_signal"]
    return {k: getattr(c, k) for k in keys if hasattr(c, k)}


def main() -> None:
    from models.dynamics_model import DynamicsModel, DynamicsModelConfig
    from models.tokenizer import AutoEncoder, AutoEncoderConfig

    root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description="Quantitative Memory-Maze rollout-error evaluation.")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Dynamics checkpoint.")
    ap.add_argument("--tokenizer", type=Path, required=True, help="Frozen tokenizer checkpoint.")
    ap.add_argument("--frames", type=Path, default=root / "data" / "memmaze9x9_val12.npy",
                    help="Held-out frames .npy (N,T,H,W,3) uint8.")
    ap.add_argument("--actions", type=Path, default=None,
                    help="Actions .npy (N,T). Default: '<frames>_actions.npy' if present.")
    ap.add_argument("--out-dir", type=Path, default=root / "outputs" / "rollout_error",
                    help="Where the result JSON lands (default: outputs/rollout_error/, gitignored).")
    ap.add_argument("--name", type=str, default=None,
                    help="Result label / filename stem (default: checkpoint stem).")
    ap.add_argument("--n-prefill", type=int, default=N_PREFILL,
                    help="Ground-truth frames STREAMED through the sliding window before scoring.")
    ap.add_argument("--n-gen", type=int, default=N_GEN, help="Scored rollout horizon (frames).")
    ap.add_argument("--n-samples", type=int, default=24, help="Total (episode, start) scored samples.")
    ap.add_argument("--batch-size", type=int, default=16, help="Samples evaluated in parallel per batch.")
    ap.add_argument("--K", type=int, default=4, help="Shortcut inference steps.")
    ap.add_argument("--window", type=int, default=None,
                    help="Force the sliding context window (total frames); default = native.")
    ap.add_argument("--seed", type=int, default=0, help="Seeds the per-episode start offsets.")
    ap.add_argument("--val-fraction", type=float, default=0.0,
                    help="Held-out split fraction; 0 (default) = use ALL episodes (val file is held out).")
    ap.add_argument("--episodes", type=int, nargs="*", default=None,
                    help="Explicit episode ids (overrides the val-split selection).")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = _load(args.tokenizer, AutoEncoder, AutoEncoderConfig, device)
    for p in tok.parameters():
        p.requires_grad_(False)
    model = _load(args.checkpoint, DynamicsModel, DynamicsModelConfig, device)

    frames = np.load(args.frames, mmap_mode="r")
    actions = None
    ap_path = args.actions or args.frames.with_name(args.frames.stem + "_actions.npy")
    if Path(ap_path).is_file():
        actions = np.load(ap_path, mmap_mode="r")
    elif model.n_actions > 0:
        raise SystemExit(f"ERROR: action-conditioned model (n_actions={model.n_actions}) but no "
                         f"actions file at {ap_path}.")

    need = args.n_prefill + args.n_gen
    if args.episodes is not None:
        episodes = np.asarray(args.episodes)
    else:
        episodes = (val_episodes(len(frames), args.val_fraction) if args.val_fraction > 0
                    else np.arange(len(frames)))
    samples = build_samples(frames, episodes, args.n_samples, need, args.seed)

    t0 = time.time()
    res = rollout_error(model, tok, frames, actions, samples=samples, n_prefill=args.n_prefill,
                        n_gen=args.n_gen, K=args.K, device=device, window=args.window,
                        batch_size=args.batch_size)
    elapsed = time.time() - t0

    name = args.name or args.checkpoint.stem
    win = model.config.max_temporal_length if args.window is None else args.window
    out = {
        "name": name,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": _sha256(args.checkpoint),
        "tokenizer": str(args.tokenizer), "tokenizer_sha256": _sha256(args.tokenizer),
        "frames": str(args.frames), "frames_shape": list(np.asarray(frames).shape),
        "device": device, "elapsed_sec": round(elapsed, 1),
        "protocol": {
            "n_prefill": args.n_prefill, "n_gen": args.n_gen, "K": args.K, "window": int(win),
            "encode_window": _tokenizer_window(tok), "context_signal": model.config.context_signal,
            "metric": "pixel_mse_01",  # frames in [0,1], mean over H*W*C, vs RAW ground truth
        },
        "samples": {"n_samples": len(samples), "seed": args.seed,
                    "episodes": [int(e) for e, _ in samples], "starts": [int(s) for _, s in samples]},
        "model_config": _model_config(model),
        "horizons": list(range(1, args.n_gen + 1)),
        "mse": [float(x) for x in res["mse"]],
        "mse_std": [float(x) for x in res["mse_std"]],
        "tokenizer_floor": [float(x) for x in res["tokenizer_floor"]],
        "copy_last": [float(x) for x in res["copy_last"]],
        "n_finite": [int(x) for x in res["n_finite"]],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"rollout_error_{name}.json"
    out_path.write_text(json.dumps(out, indent=2))

    allfin = int(np.sum(res["n_finite"] == len(samples)))
    print(f"[rollout-error] {name}: {len(samples)} samples, {allfin}/{args.n_gen} horizons all-finite "
          f"| mse h1={out['mse'][0]:.5f} h{args.n_gen}={out['mse'][-1]:.5f} "
          f"| floor h1={out['tokenizer_floor'][0]:.5f} h{args.n_gen}={out['tokenizer_floor'][-1]:.5f} "
          f"| copy_last h{args.n_gen}={out['copy_last'][-1]:.5f} | {elapsed:.1f}s")
    print(f"[rollout-error] wrote {out_path}")


if __name__ == "__main__":
    main()
