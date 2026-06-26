"""Recall A/B: does the FF9 memory model retain hidden GridWorld state past the latent window
better than the vanilla (no-memory) model? Runs the result-defining recall eval on both retrained
dynamics models against the shared baselines (oracle ceiling, copy_last no-memory reference, chance).

PROVISIONAL on the recall k<->tick alignment (off-by-one absolute-k convention, documented atop
recall.py) — that convention is applied identically to model and baselines, so the model-vs-baseline
COMPARISON here is convention-robust; only the absolute k-axis labeling awaits Merlin's sign-off.

Run (CUDA via repo venv):  venv/Scripts/python.exe -u experiments/recall-ab/run.py
"""
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
from models.tokenizer import AutoEncoder, AutoEncoderConfig          # noqa: E402
from models.dynamics_model import DynamicsModel, DynamicsModelConfig  # noqa: E402
from evals.gridworld.recall import recall                            # noqa: E402

CKPT = _ROOT / "checkpoints" / "gridworld"
N_CTX, MAX_K, N_ROLLOUTS, K = 4, 20, 64, 4


def _cfg(cfg_dict, cls):
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in cfg_dict.items() if k in allowed})


def _load(path, model_cls, cfg_cls, device):
    p = torch.load(path, map_location=device, weights_only=False)
    m = model_cls(_cfg(p["config"], cfg_cls)).to(device)
    m.load_state_dict(p["model_state_dict"])
    return m.eval()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  n_ctx={N_CTX} max_k={MAX_K} n_rollouts={N_ROLLOUTS} K={K}")
    tok = _load(CKPT / "tokenizer.pt", AutoEncoder, AutoEncoderConfig, device)

    results = {}
    for tag, fname in [("vanilla", "dynamics_vanilla.pt"), ("ff9", "dynamics_ff9.pt")]:
        torch.manual_seed(0)
        dyn = _load(CKPT / fname, DynamicsModel, DynamicsModelConfig, device)
        results[tag] = recall(dyn, tok, n_ctx=N_CTX, max_k=MAX_K, n_rollouts=N_ROLLOUTS, K=K,
                              device=device)
        print(f"[done] recall for {tag}")

    van, ff9 = results["vanilla"], results["ff9"]
    ks = sorted(int(k) for k in van["model"]["position_acc"])

    def row(label, vals):
        return label.ljust(22) + " ".join(f"{vals.get(k, float('nan')):.3f}" for k in ks)

    print("\n=== position_acc (exact 6x6 cell; chance 1/36=0.028) — occlusion length k ===")
    print("k".ljust(22) + " ".join(f"{k:5d}" for k in ks))
    print(row("oracle (ceiling)", van["oracle"]["position_acc"]))
    print(row("FF9 memory", ff9["model"]["position_acc"]))
    print(row("vanilla", van["model"]["position_acc"]))
    print(row("copy_last (no-mem)", van["copy_last"]["position_acc"]))

    print("\n=== position_score (graded distance credit) ===")
    print(row("FF9 memory", ff9["model"]["position_score"]))
    print(row("vanilla", van["model"]["position_score"]))
    print(row("copy_last (no-mem)", van["copy_last"]["position_score"]))

    print("\n=== color_acc (4-way; chance 0.25) ===")
    print(row("FF9 memory", ff9["model"]["color_acc"]))
    print(row("vanilla", van["model"]["color_acc"]))
    print(row("copy_last (no-mem)", van["copy_last"]["color_acc"]))

    # Headline: mean over k of (model - copy_last) position_score — convention-robust signal of "memory".
    def edge(m):
        return float(np.mean([m["model"]["position_score"][k] - m["copy_last"]["position_score"][k]
                              for k in ks]))
    print(f"\nmean position_score edge over copy_last:  FF9={edge(ff9):+.3f}  vanilla={edge(van):+.3f}")
    print(f"oracle position_acc mean = {np.mean(list(van['oracle']['position_acc'].values())):.3f} "
          f"(instrument self-test; should be ~1.0)")
    print(f"chance: {van['chance']}")

    out = _ROOT / "experiments" / "recall-ab" / "results.json"
    out.write_text(json.dumps({"params": dict(n_ctx=N_CTX, max_k=MAX_K, n_rollouts=N_ROLLOUTS, K=K),
                               "results": results}, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
