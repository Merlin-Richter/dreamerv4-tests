"""DynamicsModel gate tests (spec: models/dynamics_model.md).

Covers the contract that matters for the rebuild:
  * forward returns the right shape (x-prediction of the clean latents);
  * the shortcut-forcing loss runs (vanilla) and the FF9 sufficiency term's gradient reaches the
    memory-token construction (the load-bearing "gradient flows through the memory mechanism");
  * the carrying generate runs for vanilla AND memory models, finite + correct shape, through window
    eviction (n_generate > max_temporal_length);
  * long-context prefill: rollout_init/generate accept T_ctx > max_temporal_length (teacher-forced
    sliding commits), advance next_pos to T_ctx, keep the cache evicted to max_ctx, and stay finite;
  * a read-only branch (rollout_step commit=False) does NOT mutate the carried cache / next_pos
    (the invariant the recall eval relies on).

Run:  python src/tests/test_dynamics.py   (CPU only).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.dynamics_model import DynamicsModel, DynamicsModelConfig  # noqa: E402

BASE = dict(embedding_dim=64, depth=8, n_heads=8, mlp_ratio=2.0, n_latents=4, bottleneck_dim=16,
            max_temporal_length=8, max_sampling_steps=16, inference_steps=4, n_actions=2, n_registers=4)


def test_forward_shape():
    m = DynamicsModel(DynamicsModelConfig(**BASE)).eval()
    B, T, L, D = 2, 6, 4, 16
    z = torch.randn(B, T, L, D)
    tau = torch.randint(0, 16, (B, T))
    d = torch.randint(0, 5, (B, T))
    acts = m.action_features(torch.randint(0, 2, (B, T)))
    with torch.no_grad():
        out = m(z, tau, d, acts)
    assert out.shape == (B, T, L, D)
    print("[ok] forward shape (B,T,n_latents,bottleneck_dim)")


def test_loss_vanilla_and_ff9_gradient():
    B, T, L, D = 2, 6, 4, 16
    mv = DynamicsModel(DynamicsModelConfig(**BASE))
    lv = mv.loss(torch.randn(B, T, L, D), torch.randint(0, 2, (B, T)))
    assert torch.isfinite(lv) and lv.item() > 0
    cfg9 = dict(BASE); cfg9.update(n_memory=2, ff9_k=3)
    m9 = DynamicsModel(DynamicsModelConfig(**cfg9))
    l9, parts = m9.loss(torch.randn(B, T, L, D), torch.randint(0, 2, (B, T)), return_parts=True)
    l9.backward()
    g = m9.memory_tokens.grad
    assert g is not None and g.abs().sum() > 0, "FF9 gradient must reach the memory-token construction"
    assert "diffusion" in parts and "ff9" in parts
    print("[ok] loss runs (vanilla); FF9 gradient reaches memory_tokens")


def test_carrying_generate_and_eviction():
    B, Tctx, L, D = 2, 4, 4, 16
    n_gen = 12  # > max_temporal_length -> exercises sliding-window eviction
    for tag, extra in [("vanilla", {}), ("memory", dict(n_memory=2, ff9_k=2))]:
        c = dict(BASE); c.update(extra)
        m = DynamicsModel(DynamicsModelConfig(**c)).eval()
        ctx = torch.randn(B, Tctx, L, D)
        aidx = torch.randint(0, 2, (B, Tctx + n_gen))
        with torch.no_grad():
            out = m.generate(ctx, n_gen, action_idx=aidx)
        assert out.shape == (B, n_gen, L, D), f"{tag} {out.shape}"
        assert torch.isfinite(out).all(), f"{tag} produced non-finite latents"
    print("[ok] carrying generate finite + correct shape through eviction (vanilla + memory)")


def test_long_context_prefill():
    """T_ctx > max_temporal_length: first window in one forward, rest teacher-forced sliding commits."""
    B, Tctx, n_gen, L, D = 2, 20, 3, 4, 16  # Tctx=20 > W=8
    for tag, extra in [("vanilla", {}), ("memory", dict(n_memory=2, ff9_k=2))]:
        c = dict(BASE); c.update(extra)
        m = DynamicsModel(DynamicsModelConfig(**c)).eval()
        ctx = torch.randn(B, Tctx, L, D)
        aidx = torch.randint(0, 2, (B, Tctx + n_gen))
        with torch.no_grad():
            st = m.rollout_init(ctx, aidx[:, :Tctx])
            assert st["next_pos"] == Tctx, f"{tag}: next_pos {st['next_pos']} != T_ctx {Tctx}"
            for lc in st["cache"]:
                assert lc is None or lc["k"].shape[-2] <= st["max_ctx"], f"{tag}: cache not evicted"
            out = m.generate(ctx, n_gen, action_idx=aidx)
        assert out.shape == (B, n_gen, L, D), f"{tag} {out.shape}"
        assert torch.isfinite(out).all(), f"{tag} produced non-finite latents"
    print("[ok] long-context prefill (T_ctx > max_temporal_length) — vanilla + memory")


def test_readonly_branch_preserves_state():
    c = dict(BASE); c.update(n_memory=2, ff9_k=2)
    m = DynamicsModel(DynamicsModelConfig(**c)).eval()
    B, Tctx, L, D = 2, 4, 4, 16
    ctx = torch.randn(B, Tctx, L, D)
    aidx = torch.randint(0, 2, (B, Tctx + 10))
    with torch.no_grad():
        st = m.rollout_init(ctx, aidx[:, :Tctx])
        for i in range(5):
            m.rollout_step(st, aidx[:, Tctx + i:Tctx + i + 1], commit=True)
        pos_before = st["next_pos"]
        k_before = [None if lc is None else lc["k"].clone() for lc in st["cache"]]
        zb = m.rollout_step(st, torch.zeros(B, dtype=torch.long), commit=False)
        unchanged = all((a is None and lc is None) or torch.equal(a, lc["k"])
                        for a, lc in zip(k_before, st["cache"]))
    assert zb.shape == (B, 1, L, D)
    assert st["next_pos"] == pos_before, "read-only branch must not advance next_pos"
    assert unchanged, "read-only branch must not mutate the KV cache"
    print("[ok] read-only branch (commit=False) preserves carried cache + next_pos")


if __name__ == "__main__":
    test_forward_shape()
    test_loss_vanilla_and_ff9_gradient()
    test_carrying_generate_and_eviction()
    test_long_context_prefill()
    test_readonly_branch_preserves_state()
    print("\nALL DYNAMICS TESTS PASSED")
