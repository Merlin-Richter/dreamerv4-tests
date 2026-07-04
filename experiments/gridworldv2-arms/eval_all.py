"""GridWorldV2 4-arm recall eval: A vanilla-tau0 / B m2m+FF9 / C m2m no-FF9 / D sparse n=8.

Loads each checkpoint with its correct class (D needs DynamicsModelSparseWS), runs recallv2 at
the native window (16, max_k 64) AND the stress window (8, max_k 64), writes per-arm JSONs
(plot_recall.py-compatible) + a summary table + the k-vs-staleness decomposition for D
(age of the last write at the branch position = (n_ctx + 1 + k) % 8).

Usage:  venv/Scripts/python.exe -u experiments/gridworldv2-arms/eval_all.py [--max-k 64]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "sparse-write-slots"))
from model import DynamicsModelSparseWS  # noqa: E402
from evals.gridworld.recall import _load_checkpoint  # noqa: E402
from evals.gridworldv2.recall import recall  # noqa: E402
from models.dynamics_model import DynamicsModel, DynamicsModelConfig  # noqa: E402
from models.tokenizer import AutoEncoder, AutoEncoderConfig  # noqa: E402

ARMS = {
    "A_vanilla_tau0": ("checkpoints/gridworldv2/dynamics_vanilla_tau0.pt", DynamicsModel),
    "B_m2m_ff9": ("checkpoints/gridworldv2/dynamics_m2m_ff9.pt", DynamicsModel),
    "C_m2m_noff9": ("checkpoints/gridworldv2/dynamics_m2m_noff9.pt", DynamicsModel),
    "D_sparse_n8": ("checkpoints/gridworldv2/dynamics_sparse_n8.pt", DynamicsModelSparseWS),
}
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-k", type=int, default=64)
    ap.add_argument("--n-rollouts", type=int, default=64)
    ap.add_argument("--n-ctx", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(0)
    tok, _ = _load_checkpoint(ROOT / "checkpoints/gridworld/tokenizer.pt",
                              AutoEncoder, AutoEncoderConfig, DEV)
    for p in tok.parameters():
        p.requires_grad_(False)

    outdir = Path(__file__).parent
    summary = {}
    for name, (path, cls) in ARMS.items():
        if not (ROOT / path).exists():
            print(f"== {name}: MISSING {path} — skipped")
            continue
        model, cfg = _load_checkpoint(ROOT / path, cls, DynamicsModelConfig, DEV)
        summary[name] = {}
        for window in (16, 8):
            res = recall(model, tok, n_ctx=args.n_ctx, max_k=args.max_k,
                         n_rollouts=args.n_rollouts, K=4, device=DEV, window=window)
            res["meta"] = {"env": "gridworldv2", "checkpoint": path, "window": window,
                           "max_k": args.max_k, "n_ctx": args.n_ctx, "K": 4,
                           "n_rollouts": args.n_rollouts, "n_memory": cfg.n_memory}
            out = outdir / f"recallv2_{name}_w{window}.json"
            out.write_text(json.dumps(res, indent=2))
            pa = res["model"]["position_acc"]
            summary[name][f"w{window}"] = {int(k): round(v, 3) for k, v in pa.items()}
            print(f"== {name} w{window}: pos_acc " +
                  " ".join(f"k{int(k)}={v:.2f}" for k, v in sorted(pa.items(), key=lambda x: int(x[0]))))
        del model
        if DEV == "cuda":
            torch.cuda.empty_cache()

    # staleness decomposition for D (age of last write at the branch position)
    if "D_sparse_n8" in summary:
        print("\nD staleness view (age = (n_ctx+1+k) % 8):")
        for wtag in ("w16", "w8"):
            rows = [(k, (args.n_ctx + 1 + k) % 8, v)
                    for k, v in sorted(summary["D_sparse_n8"][wtag].items())]
            print(f"  {wtag}: " + "  ".join(f"k{k}(age{a})={v:.2f}" for k, a, v in rows))

    (outdir / "summary_recall.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {outdir / 'summary_recall.json'}")


if __name__ == "__main__":
    main()
