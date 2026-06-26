"""
INDEPENDENT verification probe (written from scratch, not derived from probe.py).

Claim under test:
  Production carrying KV-cached rollout (DynamicsModel.generate) is bit-exact (~1e-5) to an
  UNCACHED reference that recomputes the full current sliding window each step with cache=None
  & positions=None -- PROVIDED no eviction. Once the window evicts a committed frame, cached and
  uncached-current-window diverge materially (O(0.1-0.7)), and divergence grows with the number of
  stacked temporal layers. Discriminator: depth=3 (single temporal layer) should stay EXACT through
  eviction; depth>=6 should diverge.

Method:
  * I re-implement an uncached rollout from scratch. It runs the SAME algorithm as production
    (K shortcut steps + a near-clean "commit" re-presentation + sliding-window eviction), but every
    attention is computed by forward(cache=None, positions=None) over the whole current window.
  * Noise control: production draws global torch.randn in a fixed order. I seed the SAME value
    before production generate() and before my reference, and draw randn in the IDENTICAL order /
    shapes, so both consume identical noise. (Frame-0 / no-evict exact match validates this.)
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from models.dynamics_model import DynamicsModel, DynamicsModelConfig  # noqa: E402


# ----------------------------------------------------------------------------- uncached reference
@torch.no_grad()
def uncached_generate(m: DynamicsModel, context, n_generate, K=None, action_idx=None):
    """An uncached rollout that mirrors production's algorithm but recomputes the whole current
    sliding window through forward(cache=None, positions=None) at every step.

    Returns generated latents (B, n_generate, L, D). Draws randn in the SAME order/shape as
    production rollout_init/rollout_step so a shared seed gives identical noise.
    """
    cfg = m.config
    B, T_ctx, L, D = context.shape
    device = context.device
    K = K or cfg.inference_steps
    max_ctx = cfg.max_temporal_length - 1
    d_idx_val = K.bit_length() - 1
    tau_ctx_idx = min(round(cfg.context_signal * m.K_max), m.K_max - 1)

    ctx_act = action_idx[:, :T_ctx] if action_idx is not None else None
    ctx_act_feat = m.action_features(ctx_act)

    # --- committed-window store: lists of per-frame near-clean latent, written memory, action id.
    win_lat = []     # each (B,1,L,D) near-clean latent (the commit re-presentation input)
    win_mem = []     # each (B,1,M,E) written memory token, or None when n_memory==0
    win_act = []     # each (B,1) action id or None

    # ---- rollout_init reference: draw ONE randn_like(context) for the context near-clean. ----
    ctx_noised = (1 - cfg.context_signal) * torch.randn_like(context) + cfg.context_signal * context
    tau_col_ctx = torch.full((B, T_ctx), tau_ctx_idx, device=device, dtype=torch.long)
    d_col_ctx = torch.full((B, T_ctx), d_idx_val, device=device, dtype=torch.long)
    mem_in = None
    if m.n_memory > 0:
        positions = torch.arange(T_ctx, device=device)
        _, mem_in = m(ctx_noised, tau_col_ctx, d_col_ctx, ctx_act_feat,
                      positions=positions, return_memory=True)
    for t in range(T_ctx):
        win_lat.append(ctx_noised[:, t:t + 1])
        win_mem.append(None if mem_in is None else mem_in[:, t:t + 1])
        win_act.append(None if action_idx is None else action_idx[:, t:t + 1])
    # evict context to last max_ctx
    if len(win_lat) > max_ctx:
        win_lat = win_lat[-max_ctx:]; win_mem = win_mem[-max_ctx:]; win_act = win_act[-max_ctx:]

    out = []
    for i in range(n_generate):
        a_new = action_idx[:, T_ctx + i:T_ctx + i + 1] if action_idx is not None else None
        a_new_feat = m.action_features(a_new)

        # --- draw frame init noise (matches production rollout_step's first randn) ---
        z = torch.randn((B, 1, L, D), device=device)
        d_val = 1.0 / K
        written_mem = None
        mem_new_init = m.memory_tokens.expand(B, 1, -1, -1) if m.n_memory > 0 else None

        for step in range(K):
            tau = step / K
            tau_idx_new = round(tau * m.K_max)
            last = step == K - 1

            # Build the CURRENT window = committed frames (near-clean) + the new frame (denoising z).
            W = len(win_lat)
            lat_seq = torch.cat(win_lat + [z], dim=1)                       # (B, W+1, L, D)
            tau_seq = torch.full((B, W + 1), tau_ctx_idx, device=device, dtype=torch.long)
            tau_seq[:, -1] = tau_idx_new
            d_seq = torch.full((B, W + 1), d_idx_val, device=device, dtype=torch.long)

            mem_seq = None
            if m.n_memory > 0:
                mem_seq = torch.cat(win_mem + [mem_new_init], dim=1)        # (B, W+1, M, E)

            act_seq = None
            if action_idx is not None:
                act_ids = torch.cat(win_act + [a_new], dim=1)               # (B, W+1)
                act_seq = m.action_features(act_ids)

            # Uncached, within-window RoPE (positions=None -> indices 0..W).
            if last and m.n_memory > 0:
                z_hat_full, mem_full = m(lat_seq, tau_seq, d_seq, act_seq, memory_in=mem_seq,
                                         return_memory=True)
                written_mem = mem_full[:, -1:]                              # new frame's written mem
            else:
                z_hat_full = m(lat_seq, tau_seq, d_seq, act_seq, memory_in=mem_seq)
            z_hat1 = z_hat_full[:, -1:]                                     # only the new frame
            v = (z_hat1 - z) / (1 - tau)
            z = z + v * d_val

        out.append(z)

        # --- commit: re-present near-clean (draw randn_like(z), matching production). ---
        near_clean = (1 - cfg.context_signal) * torch.randn_like(z) + cfg.context_signal * z
        win_lat.append(near_clean)
        win_mem.append(None if written_mem is None else written_mem)
        win_act.append(None if action_idx is None else a_new)
        if len(win_lat) > max_ctx:
            win_lat = win_lat[-max_ctx:]; win_mem = win_mem[-max_ctx:]; win_act = win_act[-max_ctx:]

    return torch.cat(out, dim=1)


# ----------------------------------------------------------------------------- driver
def run_case(tag, cfg_extra, T_ctx, n_gen, seed=0, model_seed=0):
    torch.set_default_dtype(torch.float32)
    BASE = dict(embedding_dim=64, depth=8, n_heads=8, mlp_ratio=2.0, n_latents=4, bottleneck_dim=16,
                max_temporal_length=8, max_sampling_steps=16, inference_steps=4, n_actions=2,
                n_registers=4, drop_rate=0.0, att_drop_rate=0.0)
    c = dict(BASE); c.update(cfg_extra)
    torch.manual_seed(model_seed)
    m = DynamicsModel(DynamicsModelConfig(**c)).eval()
    B, L, D = 2, c["n_latents"], c["bottleneck_dim"]
    max_ctx = c["max_temporal_length"] - 1

    torch.manual_seed(1234)  # context + actions are the same for both paths
    ctx = torch.randn(B, T_ctx, L, D)
    aidx = torch.randint(0, c["n_actions"], (B, T_ctx + n_gen)) if c["n_actions"] > 0 else None

    K = c["inference_steps"]
    torch.manual_seed(seed)
    cached = m.generate(ctx, n_gen, K=K, action_idx=aidx)
    torch.manual_seed(seed)
    uncached = uncached_generate(m, ctx, n_gen, K=K, action_idx=aidx)

    # per-generated-frame max abs diff
    diffs = (cached - uncached).abs().amax(dim=(0, 2, 3))  # (n_gen,)
    # first frame whose ABSOLUTE commit position causes eviction:
    # committed count after generating frame i (0-based) = T_ctx + i + 1; window holds max_ctx.
    # eviction first drops a committed frame when committed > max_ctx during a STEP that reads it.
    # The first generated frame that READS an evicted-window (i.e. window already full at its start):
    # window length at start of gen-frame i = min(T_ctx + i, max_ctx). Eviction has happened once
    # T_ctx + i > max_ctx, i.e. i > max_ctx - T_ctx.
    first_evict_i = max_ctx - T_ctx + 1  # 0-based index of first gen frame after a drop
    return tag, c["depth"], T_ctx, n_gen, max_ctx, first_evict_i, diffs


def fmt(diffs):
    return " ".join(f"{d:.2e}" for d in diffs.tolist())


if __name__ == "__main__":
    print("=" * 100)
    print("INDEPENDENT cached-vs-uncached probe.  max_temporal_length=8 -> max_ctx=7")
    print("=" * 100)

    cases = [
        # tag, cfg_extra, T_ctx, n_gen
        ("vanilla d9 NO-EVICT", dict(depth=9, n_memory=0), 4, 3),    # 4+3=7 <= 8 window never evicts
        ("vanilla d9 EVICT",    dict(depth=9, n_memory=0), 4, 12),
        ("vanilla d6 EVICT",    dict(depth=6, n_memory=0), 4, 12),
        ("vanilla d3 EVICT",    dict(depth=3, n_memory=0), 4, 12),   # DISCRIMINATOR: 1 temporal layer
        ("memory  d9 NO-EVICT", dict(depth=9, n_memory=4, ff9_k=2), 4, 3),
        ("memory  d9 EVICT",    dict(depth=9, n_memory=4, ff9_k=2), 4, 12),
        ("memory  d6 EVICT",    dict(depth=6, n_memory=4, ff9_k=2), 4, 12),
        ("memory  d3 EVICT",    dict(depth=3, n_memory=4, ff9_k=2), 4, 12),  # DISCRIMINATOR
    ]
    rows = []
    for tag, extra, T_ctx, n_gen in cases:
        tag, depth, T_ctx, n_gen, max_ctx, fe, diffs = run_case(tag, extra, T_ctx, n_gen)
        pre = diffs[:max(fe, 0)].amax().item() if fe > 0 and diffs.numel() else float("nan")
        post = diffs[fe:].amax().item() if fe < diffs.numel() else float("nan")
        rows.append((tag, depth, fe, pre, post, diffs))
        print(f"\n[{tag}]  depth={depth}  T_ctx={T_ctx}  n_gen={n_gen}  first_evict_gen_i={fe}")
        print(f"  per-frame maxabsdiff: {fmt(diffs)}")
        print(f"  max diff PRE-evict (frames < {fe}): {pre:.3e}   POST-evict (frames >= {fe}): {post:.3e}")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for tag, depth, fe, pre, post, diffs in rows:
        print(f"{tag:22s} depth={depth} preEvict={pre:.2e} postEvict={post:.2e}")
