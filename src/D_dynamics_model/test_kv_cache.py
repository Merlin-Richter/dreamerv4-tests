"""KV-cache (T-008, D-017) correctness tests for the dynamics model D.

The HOWTO (rope_kv_cache_caveat.md) requires that a cached rollout be bit-for-bit equal
to the uncached recompute over the same horizon — otherwise a positional-encoding bug
silently corrupts long rollouts and looks like a memory failure. These tests enforce that
at the deterministic forward level (generate adds RNG, so the forward is the real gate).

Run:  python src/D_dynamics_model/test_kv_cache.py   (or pytest)
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamics_model import DynamicsModel, DynamicsModelConfig  # noqa: E402

TOL = 1e-4


def _tiny_cfg(n_actions=0, **kw) -> DynamicsModelConfig:
    return DynamicsModelConfig(
        embedding_dim=64, n_heads=4, depth=4, max_temporal_length=8,
        max_sampling_steps=16, n_actions=n_actions, **kw,
    )


def _rand_inputs(model, cfg, B, T, device="cpu"):
    z = torch.randn(B, T, cfg.n_latents, cfg.bottleneck_dim, device=device)
    tau = torch.randint(0, model.K_max, (B, T), device=device)
    d = torch.randint(0, model.n_d, (B, T), device=device)
    return z, tau, d


def test_table_matches_onthefly():
    """The on-the-fly absolute-position RoPE (positions=arange(T)) must reproduce the fixed
    cos/sin table path (positions=None) for in-range positions, so the cache path and the
    training-default path agree."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = DynamicsModel(cfg).eval()
    z, tau, d = _rand_inputs(model, cfg, B=2, T=cfg.max_temporal_length)
    with torch.no_grad():
        out_table = model(z, tau, d)                                   # table path
        out_otf = model(z, tau, d, positions=torch.arange(z.shape[1]))  # on-the-fly path
    diff = (out_table - out_otf).abs().max().item()
    assert diff < TOL, f"on-the-fly RoPE diverges from table by {diff}"


def test_forward_cache_equivalence_in_range():
    """Frame-by-frame cached forward == full-sequence forward, T within the table size."""
    _check_incremental(T=8)


def test_forward_cache_equivalence_past_table():
    """Same, but T well beyond max_temporal_length (=8): exercises arbitrary absolute
    positions / the table-overflow long-rollout fix. The full reference uses explicit
    positions because the table path cannot index T>table."""
    _check_incremental(T=20)


def _check_incremental(T):
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = DynamicsModel(cfg).eval()
    B = 2
    z, tau, d = _rand_inputs(model, cfg, B=B, T=T)
    positions = torch.arange(T)
    with torch.no_grad():
        out_full = model(z, tau, d, positions=positions)  # (B, T, L, D)
        cache = model.new_kv_cache()
        outs = []
        for t in range(T):
            out_t = model(z[:, t:t + 1], tau[:, t:t + 1], d[:, t:t + 1],
                          positions=positions[t:t + 1], cache=cache, commit=True)
            outs.append(out_t)
        out_inc = torch.cat(outs, dim=1)
    diff = (out_full - out_inc).abs().max().item()
    assert diff < TOL, f"cached incremental forward diverges from full by {diff} (T={T})"


def test_generate_cached_matches_generate():
    """Seeded uncached generate == intra-frame-cached generate (same RNG -> same draws)."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = DynamicsModel(cfg).eval()
    B, T_ctx, n_gen = 2, 5, 7
    context = torch.randn(B, T_ctx, cfg.n_latents, cfg.bottleneck_dim)
    with torch.no_grad():
        torch.manual_seed(123)
        g_ref = model.generate(context, n_generate=n_gen, K=4)
        torch.manual_seed(123)
        g_cache = model.generate_cached(context, n_generate=n_gen, K=4)
    assert g_cache.shape == g_ref.shape
    diff = (g_ref - g_cache).abs().max().item()
    assert diff < TOL, f"generate_cached diverges from generate by {diff}"


def test_generate_cached_matches_generate_actions():
    """Same, with action conditioning (n_actions>0)."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(n_actions=2)
    model = DynamicsModel(cfg).eval()
    B, T_ctx, n_gen = 2, 5, 7
    context = torch.randn(B, T_ctx, cfg.n_latents, cfg.bottleneck_dim)
    action_idx = torch.randint(0, cfg.n_actions, (B, T_ctx + n_gen))
    with torch.no_grad():
        torch.manual_seed(123)
        g_ref = model.generate(context, n_generate=n_gen, K=4, action_idx=action_idx)
        torch.manual_seed(123)
        g_cache = model.generate_cached(context, n_generate=n_gen, K=4, action_idx=action_idx)
    diff = (g_ref - g_cache).abs().max().item()
    assert diff < TOL, f"action-conditioned generate_cached diverges by {diff}"


if __name__ == "__main__":
    fns = [v for n, v in sorted(globals().items()) if n.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} KV-cache tests passed.")
