"""Controlled archive-required overfit and same-checkpoint archive ablation.

Each batch element has a distinct static latent scene.  After the initial archive is written, both
old-half fast memory and all rollout latents are hidden.  Actions are identical.  The archive is the
only sample-identity path, so a learned archive-on model must beat the same weights with gates zeroed.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import ArchiveDynamicsConfig, DynamicsModelArchive  # noqa: E402
from rollout import archive_rollout_backward                   # noqa: E402


def run_loss(model, z, actions, seed):
    model.zero_grad(set_to_none=True)
    return archive_rollout_backward(
        model, z, actions, device=z.device,
        gen=torch.Generator(device=z.device).manual_seed(seed),
        dense_tbptt_frames=8, max_frames=z.shape[1], bootstrap=False, n_d_unlocked=1,
        force_mode="clean", force_fast_hide=True, force_hide_latents=True)["loss"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-3)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(7)
    cfg = ArchiveDynamicsConfig(
        embedding_dim=64, n_heads=4, gqa_groups=2, mlp_ratio=2.0, depth=6,
        n_latents=2, bottleneck_dim=8, n_registers=2, n_memory=2, ff9_k=0,
        max_temporal_length=8, max_sampling_steps=4, inference_steps=2,
        n_actions=2, drop_rate=0.0, att_drop_rate=0.0,
        archive_interval=4, archive_per_memory=1,
        archive_compressor_depth=1, archive_compressor_mlp_ratio=1.0,
        archive_gate_init=1.0,
    )
    model = DynamicsModelArchive(cfg).to(device).train()
    B, T = args.batch_size, 24
    scene = 0.75 * torch.randn(B, 1, cfg.n_latents, cfg.bottleneck_dim, device=device)
    z = scene.expand(-1, T, -1, -1).contiguous()
    actions = torch.zeros(B, T, dtype=torch.long, device=device)
    # Freeze the unconditional dynamics path: the only trainable way to improve is to encode/read
    # sample identity through archive modules.  Gradients still propagate through the frozen backbone.
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("archive_"))
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    initial = None
    for step in range(args.steps):
        stats_loss = run_loss(model, z, actions, 10_000 + step)
        if initial is None:
            initial = stats_loss
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if (step + 1) % 50 == 0:
            print(f"step {step+1:4d}: loss={stats_loss:.5f}")

    # Identical model weights and noise; only archive gates differ.
    learned = copy.deepcopy(model).eval()
    ablated = copy.deepcopy(model).eval()
    for gate in ablated.archive_gates.values():
        gate.data.zero_()
    on_loss = run_loss(learned, z, actions, 777)
    off_loss = run_loss(ablated, z, actions, 777)
    print(f"initial={initial:.5f} archive_on={on_loss:.5f} archive_zero={off_loss:.5f}")
    assert on_loss < initial * 0.8, "archive-required controlled batch did not overfit"
    assert off_loss > on_loss * 1.15, "same-checkpoint archive ablation had no material effect"
    print("CONTROLLED ARCHIVE-REQUIRED OVERFIT + ABLATION PASSED")


if __name__ == "__main__":
    main()
