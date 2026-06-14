"""C1 multi-step motion loss (D-027) smoke tests: identity-when-off (byte-identical, the V-T017-C1
C-D guard), finiteness + per-horizon parts, gradient flow, and the clip-length assert.

Run:  python src/D_dynamics_model/test_multistep_smoke.py   (or pytest)
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamics_model import DynamicsModel, DynamicsModelConfig  # noqa: E402


def _tiny_cfg(**kw) -> DynamicsModelConfig:
    base = dict(embedding_dim=64, n_heads=4, depth=4, max_temporal_length=8,
                max_sampling_steps=16, n_actions=2)
    base.update(kw)
    return DynamicsModelConfig(**base)


def _batch(cfg, B=2, T=8):
    z1 = torch.randn(B, T, cfg.n_latents, cfg.bottleneck_dim)
    actions = torch.randint(0, cfg.n_actions, (B, T))
    return z1, actions


def test_multistep_zero_is_identity():
    """multistep_h=0 must be byte-identical to the pre-C1 loss (no extra RNG draws, no new params)."""
    cfg = _tiny_cfg()
    assert cfg.multistep_h == 0
    model = DynamicsModel(cfg)
    assert not any("multistep" in n for n, _ in model.named_parameters())  # loss-only, no params
    z1, a = _batch(cfg)
    torch.manual_seed(7); l_default = model.loss(z1, a)
    torch.manual_seed(7); l_off = model.loss(z1, a, multistep_h=0)
    assert torch.equal(l_default, l_off), (l_default, l_off)


def test_multistep_changes_loss_and_has_per_horizon_parts():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = DynamicsModel(cfg)
    z1, a = _batch(cfg)
    torch.manual_seed(1); base = model.loss(z1, a)
    torch.manual_seed(1); total, parts = model.loss(z1, a, multistep_h=4, return_parts=True)
    assert torch.isfinite(total)
    assert not torch.equal(base, total)                       # the term is actually active
    assert "multistep" in parts
    assert {f"ms_h{j}" for j in (1, 2, 3, 4)} <= set(parts)   # per-horizon logged for the monitor
    assert all(torch.isfinite(v) for v in parts.values())


def test_multistep_lambda_scales():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = DynamicsModel(cfg)
    z1, a = _batch(cfg)
    torch.manual_seed(2); d = model.loss(z1, a)                              # diffusion only
    torch.manual_seed(2); t1, p1 = model.loss(z1, a, multistep_h=4, lambda_multistep=1.0, return_parts=True)
    torch.manual_seed(2); t2, _ = model.loss(z1, a, multistep_h=4, lambda_multistep=2.0, return_parts=True)
    # total = diffusion + lambda * multistep ; check the lambda scaling is exact.
    assert torch.allclose(t2 - t1, p1["multistep"], atol=1e-5), (t2 - t1, p1["multistep"])


def test_multistep_gradient_reaches_blocks():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = DynamicsModel(cfg)
    z1, a = _batch(cfg)
    model.loss(z1, a, multistep_h=4).backward()
    grads = [p.grad for p in model.blocks.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_multistep_unlabeled_ok():
    """Unlabeled (action_idx=None) must also work (act_in None path)."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(n_actions=0)
    model = DynamicsModel(cfg)
    z1 = torch.randn(2, 8, cfg.n_latents, cfg.bottleneck_dim)
    total = model.loss(z1, None, multistep_h=4)
    assert torch.isfinite(total)


def test_multistep_too_short_asserts():
    cfg = _tiny_cfg()   # max_temporal_length=8
    model = DynamicsModel(cfg)
    z1, a = _batch(cfg, T=8)
    try:
        model.loss(z1, a, multistep_h=6)   # seed(3)+6=9 > T=8 -> must assert
    except AssertionError:
        return
    raise AssertionError("multistep_h too large for clip should assert")


if __name__ == "__main__":
    fns = [v for n, v in sorted(globals().items()) if n.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} smoke tests passed.")
