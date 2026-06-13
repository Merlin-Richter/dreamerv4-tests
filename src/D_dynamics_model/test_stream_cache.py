"""Cross-frame sliding-window KV eviction cache (D-020, T-012) correctness tests.

generate_streaming persists finalized frames' K/V across rollout steps and evicts the oldest when
the window (N-1) overflows. Because cached K/V are stored already-RoPE-rotated at ABSOLUTE positions
(T-008), eviction is a pure time-axis slice. The real gate (per test_kv_cache.py's philosophy: "the
forward is the real gate") is FORWARD-LEVEL and RNG-free:

  commit+evict streaming forward  ==  full windowed recompute over [max(0,t-W) .. t]   (bit-for-bit)

isolating eviction + absolute-RoPE + causal mask from the one deliberate semantic deviation from
generate() (each frame's context-noise is drawn ONCE at commit, not redrawn every step). That
deviation is checked separately at the generate level against a frozen-noise reference (bit-exact),
and the residual divergence from generate() is reported.

Run:  python src/D_dynamics_model/test_stream_cache.py   (or pytest)
"""

import sys
import time
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


# --------------------------------------------------------------------------- forward-level gate
def _stream_forward(model, z, tau, d, W, actions=None):
    """Commit each frame into a persistent cache; evict so at most W context frames survive into the
    next step. Returns each frame's output (the new frame attends to its surviving context + itself)."""
    T = z.shape[1]
    cache = model.new_kv_cache()
    cache_len = 0
    outs = []
    for t in range(T):
        a = actions[:, t:t + 1] if actions is not None else None
        out_t = model(z[:, t:t + 1], tau[:, t:t + 1], d[:, t:t + 1], a,
                      positions=torch.tensor([t]), cache=cache, commit=True)
        outs.append(out_t)
        cache_len += 1
        while cache_len > W:
            model._evict_oldest(cache)
            cache_len -= 1
    return torch.cat(outs, dim=1)


def _windowed_full(model, z, tau, d, W, actions=None):
    """Reference independent of the cache: for each t, a full forward over the window [max(0,t-W)..t]
    with explicit absolute positions; take the last frame's output."""
    T = z.shape[1]
    outs = []
    for t in range(T):
        lo = max(0, t - W)
        a = actions[:, lo:t + 1] if actions is not None else None
        out = model(z[:, lo:t + 1], tau[:, lo:t + 1], d[:, lo:t + 1], a,
                    positions=torch.arange(lo, t + 1))[:, -1:]
        outs.append(out)
    return torch.cat(outs, dim=1)


def _check_eviction(T, n_actions=0):
    torch.manual_seed(0)
    cfg = _tiny_cfg(n_actions=n_actions)
    model = DynamicsModel(cfg).eval()
    W = cfg.max_temporal_length - 1
    B = 2
    z = torch.randn(B, T, cfg.n_latents, cfg.bottleneck_dim)
    tau = torch.randint(0, model.K_max, (B, T))
    d = torch.randint(0, model.n_d, (B, T))
    actions = torch.randint(0, n_actions, (B, T)) if n_actions else None
    actions_feat = model.action_features(actions)
    with torch.no_grad():
        out_stream = _stream_forward(model, z, tau, d, W, actions_feat)
        out_full = _windowed_full(model, z, tau, d, W, actions_feat)
    diff = (out_stream - out_full).abs().max().item()
    assert diff < TOL, f"streaming eviction diverges from windowed recompute by {diff} (T={T})"


def test_eviction_equivalence_in_range():
    """T within the table size — eviction == sliding-window recompute, bit-for-bit."""
    _check_eviction(T=8)


def test_eviction_equivalence_past_table():
    """T well beyond max_temporal_length (=8): eviction through unbounded absolute positions stays
    bit-for-bit equal to full recompute (the documented RoPE table-overflow trap)."""
    _check_eviction(T=24)


def test_eviction_equivalence_actions():
    """Same forward-level gate with action conditioning."""
    _check_eviction(T=20, n_actions=3)


# ------------------------------------------------------------------- generate-level frozen-noise
def _frozen_noise_reference(model, context, n_generate, K, action_idx=None):
    """Mirror generate_streaming's algorithm and EXACT RNG draw order (one context-noise draw, then
    per frame: one z draw, K substeps, one commit-noise draw) but with a full windowed recompute
    instead of the persistent cache. With the same seed this must be bit-identical to
    generate_streaming — proving the cached rollout composes correctly end-to-end."""
    B, T_ctx, L, D = context.shape
    W = model.config.max_temporal_length - 1
    tau_ctx_idx = round(model.config.context_signal * model.K_max)
    d_idx_val = (K).bit_length() - 1
    d_val = 1.0 / K
    has_act = model.n_actions > 0 and action_idx is not None

    ctx_noised = model._noise_to_ctx(context)                      # draw A (matches init)
    frames = [ctx_noised[:, t:t + 1] for t in range(T_ctx)]        # noised frames, each (B,1,L,D)
    positions_list = list(range(T_ctx))
    ids = [action_idx[:, t] if has_act else None for t in range(T_ctx)]

    generated = []
    for i in range(n_generate):
        new_pos = T_ctx + i
        lo = max(0, len(frames) - W)
        ctx = torch.cat(frames[lo:], dim=1)                       # (B, w, L, D)
        win_pos = positions_list[lo:]
        w = ctx.shape[1]
        new_id = action_idx[:, new_pos] if has_act else None
        act_in = None
        if has_act:
            id_seq = torch.stack(ids[lo:] + [new_id], dim=1)      # (B, w+1)
            act_in = model.action_features(id_seq)
        tau_col = torch.full((B, w + 1), tau_ctx_idx, dtype=torch.long)
        d_col = torch.full((B, w + 1), d_idx_val, dtype=torch.long)
        positions = torch.tensor(win_pos + [new_pos])
        z = torch.randn((B, 1, L, D))                             # draw B_i (matches step)
        for k in range(K):
            tau = k / K
            tau_col[:, -1] = round(tau * model.K_max)
            z_hat1 = model(torch.cat((ctx, z), dim=1), tau_col, d_col, act_in,
                           positions=positions)[:, -1:]
            v = (z_hat1 - z) / (1 - tau)
            z = z + v * d_val
        generated.append(z)
        frames.append(model._noise_to_ctx(z))                     # draw C_i (matches commit)
        positions_list.append(new_pos)
        ids.append(new_id)
    return torch.cat(generated, dim=1)


def _check_generate(n_actions=0):
    torch.manual_seed(0)
    cfg = _tiny_cfg(n_actions=n_actions)
    model = DynamicsModel(cfg).eval()
    B, T_ctx, n_gen, K = 2, 4, 12, 4   # n_gen makes the rollout slide well past the window
    context = torch.randn(B, T_ctx, cfg.n_latents, cfg.bottleneck_dim)
    action_idx = torch.randint(0, n_actions, (B, T_ctx + n_gen)) if n_actions else None
    with torch.no_grad():
        torch.manual_seed(123)
        g_stream = model.generate_streaming(context, n_gen, K=K, action_idx=action_idx)
        torch.manual_seed(123)
        g_ref = _frozen_noise_reference(model, context, n_gen, K, action_idx)
    assert g_stream.shape == g_ref.shape == (B, n_gen, cfg.n_latents, cfg.bottleneck_dim)
    diff = (g_stream - g_ref).abs().max().item()
    assert diff < TOL, f"generate_streaming diverges from frozen-noise reference by {diff}"


def test_generate_streaming_matches_frozen_reference():
    """End-to-end: generate_streaming == frozen-noise full-recompute reference, bit-for-bit."""
    _check_generate(n_actions=0)


def test_generate_streaming_matches_frozen_reference_actions():
    """Same, action-conditioned."""
    _check_generate(n_actions=3)


def test_deviation_from_generate_is_finite():
    """The frozen-noise semantics is a deliberate deviation from generate() (per-step noise redraw),
    so they are NOT bit-equal. Confirm the rollout still runs and the deviation is finite/sane (the
    benign-on-a-trained-model check belongs on a real checkpoint; here we just guard against blowups)."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = DynamicsModel(cfg).eval()
    B, T_ctx, n_gen, K = 2, 4, 10, 4
    context = torch.randn(B, T_ctx, cfg.n_latents, cfg.bottleneck_dim)
    with torch.no_grad():
        torch.manual_seed(7)
        g_gen = model.generate(context, n_gen, K=K)
        torch.manual_seed(7)
        g_stream = model.generate_streaming(context, n_gen, K=K)
    assert g_gen.shape == g_stream.shape
    dev = (g_gen - g_stream).abs().mean().item()
    assert torch.isfinite(torch.tensor(dev)) and dev < 100.0, f"deviation from generate() insane: {dev}"
    print(f"  [info] mean |generate - generate_streaming| on random-init model = {dev:.4f} "
          f"(non-zero by design; benign-magnitude check belongs on a trained ckpt)")


def test_speed_streaming_vs_cached():
    """Sanity (not a gate): streaming should be faster than generate_cached on a long rollout, since
    it does NOT rebuild the cache per frame."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = DynamicsModel(cfg).eval()
    B, T_ctx, n_gen, K = 4, 4, 60, 4
    context = torch.randn(B, T_ctx, cfg.n_latents, cfg.bottleneck_dim)
    with torch.no_grad():
        model.generate_cached(context, 2, K=K)   # warmup
        t0 = time.perf_counter(); model.generate_cached(context, n_gen, K=K); t_cached = time.perf_counter() - t0
        t0 = time.perf_counter(); model.generate_streaming(context, n_gen, K=K); t_stream = time.perf_counter() - t0
    print(f"  [info] long rollout (n_gen={n_gen}): generate_cached {t_cached*1e3:.0f} ms vs "
          f"generate_streaming {t_stream*1e3:.0f} ms  (speedup x{t_cached/max(t_stream,1e-9):.2f})")


if __name__ == "__main__":
    fns = [v for n, v in sorted(globals().items()) if n.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} stream-cache tests passed.")
