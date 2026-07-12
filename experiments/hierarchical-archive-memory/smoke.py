"""Deterministic CPU correctness gates for hierarchical archive memory.

Run before any GPU calibration:

    python -u experiments/hierarchical-archive-memory/smoke.py
"""
from __future__ import annotations

import copy
import sys
from dataclasses import fields
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.dynamics_model import DynamicsModel, DynamicsModelConfig  # noqa: E402
from model import ArchiveDynamicsConfig, DynamicsModelArchive         # noqa: E402
from rollout import archive_rollout_backward                          # noqa: E402


CFG = dict(
    embedding_dim=32, n_heads=4, gqa_groups=2, mlp_ratio=2.0, depth=6,
    n_latents=2, bottleneck_dim=8, n_registers=2, n_memory=2, ff9_k=0,
    max_temporal_length=8, max_sampling_steps=4, inference_steps=2,
    n_actions=3, drop_rate=0.0, att_drop_rate=0.0,
    archive_interval=4, archive_per_memory=1,
    archive_compressor_depth=1, archive_compressor_mlp_ratio=1.0,
    archive_gate_init=1.0,
)


def make(**overrides):
    cfg = dict(CFG); cfg.update(overrides)
    return DynamicsModelArchive(ArchiveDynamicsConfig(**cfg)).eval()


def test_compressor_shape_and_lane_isolation():
    torch.manual_seed(0)
    m = make()
    src = torch.randn(2, 4, 2, 32)
    out = m.archive_compressor(src)
    assert out.shape == (2, 2, 1, 32)
    changed = src.clone(); changed[:, :, 0] += torch.randn_like(changed[:, :, 0])
    out2 = m.archive_compressor(changed)
    assert (out2[:, 0] - out[:, 0]).abs().max() > 1e-5
    assert torch.equal(out2[:, 1], out[:, 1]), "slot-0 source leaked into archive group 1"
    print("1. compressor shape + slot isolation: PASS")


def test_reader_grouping_eligibility_and_cache_equivalence():
    torch.manual_seed(1)
    m = make()
    reader = m.archive_readers["1"]
    memory = torch.randn(2, 2, 2, 32)
    archive = torch.randn(2, 1, 2, 1, 32)
    apos = torch.tensor([3])

    # min age = W-N+1 = 5: query 7 is too early; query 8 is first eligible.
    early = reader(memory[:, :1], torch.tensor([7]), archive=archive, archive_positions=apos)
    assert torch.equal(early, torch.zeros_like(early))
    raw = reader(memory, torch.tensor([8, 9]), archive=archive, archive_positions=apos)
    k, v = reader.project_archive(archive, apos)
    cached = reader(memory, torch.tensor([8, 9]), archive_positions=apos,
                    archive_cache={"k": k, "v": v})
    assert torch.allclose(raw, cached, atol=1e-6, rtol=1e-6)

    altered = archive.clone(); altered[:, :, 0] += 2.0
    raw2 = reader(memory, torch.tensor([8, 9]), archive=altered, archive_positions=apos)
    assert (raw2[:, :, 0] - raw[:, :, 0]).abs().max() > 1e-6
    assert torch.equal(raw2[:, :, 1], raw[:, :, 1]), "archive group 0 leaked into query group 1"
    print("2. grouped reader + age mask + raw/cache equivalence (GQA): PASS")


def test_gate_zero_matches_base_model():
    torch.manual_seed(2)
    acfg = ArchiveDynamicsConfig(**CFG)
    allowed = {f.name for f in fields(DynamicsModelConfig)}
    bcfg = DynamicsModelConfig(**{k: v for k, v in CFG.items() if k in allowed})
    base = DynamicsModel(bcfg).eval()
    arch = DynamicsModelArchive(acfg).eval()
    arch.load_state_dict(base.state_dict(), strict=False)
    for gate in arch.archive_gates.values():
        gate.data.zero_()

    B, T = 2, 8
    z = torch.randn(B, T, 2, 8)
    tau = torch.randint(0, 4, (B, T))
    d = torch.randint(0, 3, (B, T))
    acts = base.action_features(torch.randint(0, 3, (B, T)))
    pos = torch.arange(T)
    bank = torch.randn(B, 2, 2, 1, 32)
    apos = torch.tensor([-9, -5])  # both eligible, branch still exactly gate-zero
    with torch.no_grad():
        y0 = base(z, tau, d, acts, positions=pos)
        y1 = arch(z, tau, d, acts, positions=pos,
                  archive_bank=bank, archive_positions=apos)
    assert torch.equal(y0, y1), f"gate-zero changed base output: maxdiff {(y0-y1).abs().max():.3e}"
    with torch.no_grad():
        y0_table = base(z, tau, d, acts, positions=None)
        y1_table = arch(z, tau, d, acts, positions=None,
                        archive_bank=bank, archive_positions=apos)
    assert torch.equal(y0_table, y1_table), "gate-zero changed base fixed-table forward"
    print("3. archive gate zero == base model exactly: PASS")


def _snapshot(state):
    return {
        "next": state["next_pos"],
        "apos": state["archive_positions"].clone(),
        "segment": state["segment_memory"].clone(),
        "local": [(None if x is None or x.get("k") is None else x["k"].clone())
                  for x in state["cache"]],
        "archive": [(None if x is None or x.get("k") is None else x["k"].clone())
                    for x in state["archive_cache"]],
    }


def _assert_snapshot(state, snap):
    assert state["next_pos"] == snap["next"]
    assert torch.equal(state["archive_positions"], snap["apos"])
    assert torch.equal(state["segment_memory"], snap["segment"])
    for got, old in zip(state["cache"], snap["local"]):
        if old is not None:
            assert torch.equal(got["k"], old)
    for got, old in zip(state["archive_cache"], snap["archive"]):
        if old is not None:
            assert torch.equal(got["k"], old)


def test_rollout_boundary_and_readonly():
    torch.manual_seed(3)
    m = make()
    ctx = torch.randn(2, 3, 2, 8)
    acts = torch.randint(0, 3, (2, 8))
    state = m.rollout_init(ctx, acts[:, :3])
    assert state["archive_positions"].numel() == 0
    m.rollout_step(state, acts[:, 3], commit=True)  # position 3 completes [0,4)
    assert state["archive_positions"].tolist() == [3]
    assert state["segment_memory"].shape[1] == 0

    snap = _snapshot(state)
    m.rollout_step(state, acts[:, 4], commit=False)
    _assert_snapshot(state, snap)
    m.rollout_step(state, acts[:, 4], commit=True)  # non-boundary
    assert state["archive_positions"].tolist() == [3]
    assert state["segment_memory"].shape[1] == 1
    print("4. rollout boundary + read-only state immutability: PASS")


def test_deferred_compressor_vjp_matches_direct():
    torch.manual_seed(4)
    m = make()
    direct = copy.deepcopy(m.archive_compressor)
    deferred = copy.deepcopy(m.archive_compressor)
    src = torch.randn(2, 4, 2, 32)
    coeff1 = torch.randn(2, 2, 1, 32)
    coeff2 = torch.randn(2, 2, 1, 32)

    y = direct(src)
    ((y * coeff1).sum() / 7 + (y.square() * coeff2).sum() / 11).backward()

    with torch.no_grad():
        value = deferred(src)
    proxy = value.detach().requires_grad_()
    ((proxy * coeff1).sum() / 7).backward()
    ((proxy.square() * coeff2).sum() / 11).backward()
    real = deferred(src)
    real.backward(proxy.grad)

    for (n0, p0), (n1, p1) in zip(direct.named_parameters(), deferred.named_parameters()):
        assert n0 == n1
        assert torch.allclose(p0.grad, p1.grad, atol=2e-5, rtol=2e-5), (
            f"deferred VJP mismatch {n0}: {(p0.grad-p1.grad).abs().max():.3e}")
    print("5. deferred compressor VJP == direct one-shot gradient: PASS")


def test_long_rollout_backward_and_hiding():
    torch.manual_seed(5)
    m = make(archive_gate_init=0.1).train()
    z = torch.randn(2, 24, 2, 8)
    acts = torch.randint(0, 3, (2, 24))
    m.zero_grad(set_to_none=True)
    stats = archive_rollout_backward(
        m, z, acts, device="cpu", gen=torch.Generator().manual_seed(99),
        dense_tbptt_frames=8, max_frames=24, bootstrap=False, n_d_unlocked=1,
        force_mode="clean", force_fast_hide=True, force_hide_latents=True)
    comp_grad = sum(float(p.grad.abs().sum()) for p in m.archive_compressor.parameters()
                    if p.grad is not None)
    reader_grad = sum(float(p.grad.abs().sum()) for p in m.archive_readers.parameters()
                      if p.grad is not None)
    assert stats["n_slides"] == 4
    assert stats["n_archives"] == 6
    assert stats["n_archives_used"] > 0
    assert stats["fast_hide_frac"] == 1.0 and stats["hide_latents_frac"] == 1.0
    assert comp_grad > 0 and reader_grad > 0
    print("6. bounded blockwise rollout + archive-only hiding + long compressor gradient: PASS")


if __name__ == "__main__":
    test_compressor_shape_and_lane_isolation()
    test_reader_grouping_eligibility_and_cache_equivalence()
    test_gate_zero_matches_base_model()
    test_rollout_boundary_and_readonly()
    test_deferred_compressor_vjp_matches_direct()
    test_long_rollout_backward_and_hiding()
    print("\nALL HIERARCHICAL ARCHIVE SMOKE GATES PASSED")
