"""Primary eval for the honest-baseline A/B: teacher-forced 1-step + free-run position probes
(reuses experiments/vanilla-inwindow-diagnosis/probe_next_pos.py machinery; same seeds, so rows
are directly comparable to results_next_pos.json there).

Usage:  venv/Scripts/python.exe -u experiments/vanilla-honest-baseline/eval_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "vanilla-inwindow-diagnosis"))
from probe_next_pos import run_model, _load_checkpoint, _tokenizer_window  # noqa: E402
from models.tokenizer import AutoEncoder, AutoEncoderConfig  # noqa: E402

CKPTS = {
    "vanilla_ref": "checkpoints/gridworld/dynamics_vanilla.pt",          # the broken baseline
    "armC_dcurr": "checkpoints/gridworld/dynamics_vanilla_dcurr.pt",     # 415190
    "armD_tau0": "checkpoints/gridworld/dynamics_vanilla_tau0.pt",       # 415191
}

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    torch.manual_seed(0)
    tok, _ = _load_checkpoint(ROOT / "checkpoints/gridworld/tokenizer.pt",
                              AutoEncoder, AutoEncoderConfig, DEV)
    for p in tok.parameters():
        p.requires_grad_(False)
    tok_w = _tokenizer_window(tok)
    results = {}
    for name, path in CKPTS.items():
        results[name] = run_model(name, path, tok, tok_w)
    out = Path(__file__).parent / "results_probe.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
