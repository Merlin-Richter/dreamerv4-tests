"""Cheap invariants for the ColorField pixel-v3 three-arm training recipe."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = ROOT / "experiments" / "hierarchical-archive-memory"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ARCHIVE_DIR))
sys.path.insert(0, str(ROOT))

from autoresearch.editable.model import DynamicsModel, DynamicsModelConfig  # noqa: E402
from autoresearch.editable.rollout import mem2mem_rollout_loss             # noqa: E402
from model import ArchiveDynamicsConfig, DynamicsModelArchive              # noqa: E402


def cfg(**kw):
    base = dict(
        bottleneck_dim=8, n_latents=2, embedding_dim=32,
        max_temporal_length=16, n_heads=4, depth=3,
        n_actions=5, n_registers=2, n_memory=2, ff9_k=0,
        drop_rate=0.0, att_drop_rate=0.0,
    )
    base.update(kw)
    return DynamicsModelConfig(**base)


def test_tau0_train_only():
    m = DynamicsModel(cfg(n_memory=0, tau0_anchor=0.5))
    torch.manual_seed(0); m.train()
    tau, d = m.sample_tau_d(256, 16, "cpu")
    anchored = (tau == 0) & (d == m.n_d - 1)
    assert 0.48 < float(anchored.float().mean()) < 0.60
    torch.manual_seed(0); m.eval()
    tau_eval, d_eval = m.sample_tau_d(256, 16, "cpu")
    eval_mass = float(((tau_eval == 0) & (d_eval == m.n_d - 1)).float().mean())
    assert eval_mass < 0.05, eval_mass
    print("tau0 anchor is train-only: PASS")


def test_fixed_rollout_context():
    m = DynamicsModel(cfg()).train()
    z = torch.randn(1, 32, 2, 8)
    a = torch.randint(0, 5, (1, 32))
    loss, parts = mem2mem_rollout_loss(
        m, z, a, n_ctx=16, device="cpu", bootstrap=False,
        n_d_unlocked=1, use_ff9=False, max_frames=32)
    assert torch.isfinite(loss) and parts["n_ctx"] == 16.0
    print("rollout context is exactly W=16: PASS")


def test_blockwise_backward_matches_full():
    torch.manual_seed(3)
    full = DynamicsModel(cfg()).train()
    blocked = copy.deepcopy(full).train()
    z = torch.randn(1, 80, 2, 8)
    a = torch.randint(0, 5, (1, 80))

    torch.manual_seed(9)
    g1 = torch.Generator().manual_seed(11)
    loss, _ = mem2mem_rollout_loss(
        full, z, a, n_ctx=16, device="cpu", gen=g1, bootstrap=False,
        n_d_unlocked=1, use_ff9=False, max_frames=80, tbptt_frames=32)
    loss.backward()

    torch.manual_seed(9)
    g2 = torch.Generator().manual_seed(11)
    detached_loss, parts = mem2mem_rollout_loss(
        blocked, z, a, n_ctx=16, device="cpu", gen=g2, bootstrap=False,
        n_d_unlocked=1, use_ff9=False, max_frames=80, tbptt_frames=32,
        blockwise_backward=True)
    assert not detached_loss.requires_grad and parts["blockwise_backward"] == 1.0
    worst = 0.0
    for p, q in zip(full.parameters(), blocked.parameters()):
        if p.grad is not None:
            worst = max(worst, float((p.grad - q.grad).abs().max()))
    assert worst < 2e-6, worst
    print(f"blockwise TBPTT gradient matches full accumulation (max {worst:.2g}): PASS")


def test_archive_warm_start_w16():
    base = DynamicsModel(cfg())
    acfg = ArchiveDynamicsConfig(**{
        **{k: v for k, v in base.config.__dict__.items()
           if k in ArchiveDynamicsConfig.__dataclass_fields__ and k != "dtype"},
        "archive_interval": 16, "archive_per_memory": 1,
        "archive_compressor_depth": 1, "archive_compressor_mlp_ratio": 1.0,
    })
    archive = DynamicsModelArchive(acfg)
    inc = archive.load_state_dict(base.state_dict(), strict=False)
    allowed = ("archive_compressor.", "archive_readers.", "archive_norms.", "archive_gates.")
    assert not inc.unexpected_keys
    assert inc.missing_keys and all(k.startswith(allowed) for k in inc.missing_keys)
    assert archive.config.max_temporal_length == 16
    print("archive warm-start + W=16 pin: PASS")


if __name__ == "__main__":
    test_tau0_train_only()
    test_fixed_rollout_context()
    test_blockwise_backward_matches_full()
    test_archive_warm_start_w16()
    print("ALL COLORFIELD DYNAMICS V3 SMOKES PASSED")
