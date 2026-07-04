"""GQA vs MHA-tau0 head-to-head (GQA checkpoints need the DynamicsModelGQA class to load).

Runs the teacher-forced/free-run probe (same seeds as vanilla-honest-baseline/results_probe.json)
and the recall eval (w8, max_k 32) for the GQA checkpoint, and re-measures the rollout cache bytes
for both models at the real checkpoint configs.

Usage:  venv/Scripts/python.exe -u experiments/gqa-dynamics/eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "gqa-dynamics"))
sys.path.insert(0, str(ROOT / "experiments" / "vanilla-inwindow-diagnosis"))
import probe_next_pos as P  # noqa: E402
from model import DynamicsModelGQA  # noqa: E402
from evals.gridworld.recall import _load_checkpoint, recall  # noqa: E402
from models.dynamics_model import DynamicsModel, DynamicsModelConfig  # noqa: E402
from models.tokenizer import AutoEncoder, AutoEncoderConfig  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GQA_CKPT = "checkpoints/gridworld/dynamics_gqa_tau0.pt"
MHA_CKPT = "checkpoints/gridworld/dynamics_vanilla_tau0.pt"


def cache_bytes(state):
    return sum(t.numel() * t.element_size()
               for lc in state["cache"] if lc is not None
               for t in (lc["k"], lc["v"]) if t is not None)


def main():
    torch.manual_seed(0)
    tok, _ = _load_checkpoint(ROOT / "checkpoints/gridworld/tokenizer.pt",
                              AutoEncoder, AutoEncoderConfig, DEV)
    for p in tok.parameters():
        p.requires_grad_(False)
    tok_w = P._tokenizer_window(tok)

    # Probe: monkeypatch the probe module's loader so run_model builds the GQA class.
    orig_loader = P._load_checkpoint

    def gqa_loader(path, cls_model, cls_cfg, device):
        cls = DynamicsModelGQA if "gqa" in str(path).lower() else cls_model
        return orig_loader(path, cls, cls_cfg, device)

    P._load_checkpoint = gqa_loader
    results = {"probe": P.run_model("gqa_tau0", GQA_CKPT, tok, tok_w)}
    P._load_checkpoint = orig_loader

    # Recall w8 max_k32 (same params as vanilla-honest-baseline/recall_tau0_w8.json).
    model, cfg = gqa_loader(ROOT / GQA_CKPT, DynamicsModel, DynamicsModelConfig, DEV)
    res = recall(model, tok, n_ctx=4, max_k=32, n_rollouts=64, K=4, device=DEV, window=8)
    results["recall_w8"] = res
    print("\nrecall w8 (position_acc):",
          {k: round(v, 3) for k, v in res["model"]["position_acc"].items()})

    # Cache footprint at the real checkpoint configs (B=1, full window context).
    mha, _ = orig_loader(ROOT / MHA_CKPT, DynamicsModel, DynamicsModelConfig, DEV)
    with torch.no_grad():
        ctx = torch.randn(1, cfg.max_temporal_length, cfg.n_latents, cfg.bottleneck_dim,
                          device=DEV)
        act = torch.zeros((1, cfg.max_temporal_length), dtype=torch.long, device=DEV)
        bg = cache_bytes(model.rollout_init(ctx, act, 4))
        bm = cache_bytes(mha.rollout_init(ctx, act, 4))
    print(f"\ncache footprint (full-window ctx, B=1): GQA {bg / 1e3:.1f} KB "
          f"vs MHA {bm / 1e3:.1f} KB -> {bm / bg:.2f}x")
    results["cache_bytes"] = {"gqa": bg, "mha": bm, "ratio": bm / bg}

    out = Path(__file__).parent / "results_eval.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
