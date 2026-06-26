"""mem->mem training rollout (task: tasks/in-progress/test-new-memory-training.md).

The dynamics model already learns latents->memory (write, via the windowed pass) and memory->latents
(read, via the FF9 sufficiency loss). What it does NOT get is a signal to *construct a memory token
from previous memory tokens* — at training time the memory tokens in context are always the
learned-blank init, never real computed ones. This rollout supplies that signal.

Mechanism (no KV cache — this is training, full recompute each window):
  The model's temporal attention is causal *per token slot*, so a frame's MEMORY slot attends to the
  memory slots of earlier frames. If we (a) inject REAL, graph-attached memory tokens from a previous
  forward into the OLD half of a window, (b) let the model construct the NEW half's memory + denoise
  the NEW half's latents, and (c) put the flow + FF9 loss ONLY on the new half, then the new memory is
  built by attending to the old memory, and backprop flows new-memory -> old-memory CONSTRUCTION. We
  slide the window by n_ctx/2 and repeat, carrying the new half's memory forward as the next old half
  — a relay whose gradient reaches back through the memory chain (truncated at ~2N frames).

Noise modes (independent per batch element, the task's 50/50):
  * "clean": old half = near-clean GT latents, new half = sampled-signal noised latents.
  * "noise": ALL latents pure noise (tau=0) — the new half can ONLY be reconstructed from memory, so
    memory must carry the whole scene. This is the strongest mem->mem pressure.

Loss is on the NEW half only; the old half is rollout context that contributes no direct loss (its
gradient comes solely through being the memory/scene the new half reads).
"""
from __future__ import annotations

import torch


def _tau_d_consts(model):
    K_max, n_d = model.K_max, model.n_d
    tau_ctx_idx = min(round(model.config.context_signal * K_max), K_max - 1)
    d_idx_val = n_d - 1  # finest step -> pure flow (x-prediction) loss, no bootstrap
    return K_max, tau_ctx_idx, d_idx_val


def _flow_loss(model, z_hat_new, z1_new, tau_new):
    """x-prediction flow MSE on the new half, ramp-weighted by w(tau) (matches model.loss)."""
    w = (1 - model.config.ramp_min) * tau_new + model.config.ramp_min  # (B,h,1,1)
    return (w * (z_hat_new - z1_new) ** 2).mean()


@torch.no_grad()
def _sample_modes(B, device, gen, force_mode):
    if force_mode == "noise":
        return torch.ones(B, dtype=torch.bool, device=device)
    if force_mode == "clean":
        return torch.zeros(B, dtype=torch.bool, device=device)
    return torch.rand(B, device=device, generator=gen) < 0.5  # True -> full-noise


def mem2mem_rollout_loss(model, z1, actions_idx=None, *, n_ctx, device,
                         tbptt_frames=None, max_frames=None, gen=None, force_mode=None):
    """One mem->mem rollout over a long clip. Returns (total_loss, parts_dict).

    z1:          (B, T, L, D) clean latents (T should be long, e.g. up to 5*max_temporal_length).
    actions_idx: (B, T) long action ids, or None (unlabeled).
    n_ctx:       window size W (a power of two, 2 <= W <= max_temporal_length). SAME for the whole
                 batch (GPU-parallel requirement). The window slides by W/2.
    tbptt_frames: detach the carried memory once the relay graph is deeper than this many frames
                 (truncated BPTT; default 2*max_temporal_length). Bounds memory footprint.
    max_frames:  stop after this many frames from t=0 (default min(5*N, 10*W)).
    force_mode:  None -> per-element 50/50 clean/noise; "noise"/"clean" -> force (for tests).

    Loss = flow (new half) + FF9 sufficiency (new-half memories), normalized like model.loss.
    """
    assert model.n_memory > 0, "mem2mem needs n_memory > 0"
    N = model.config.max_temporal_length
    B, T, L, D = z1.shape
    W = int(n_ctx)
    assert W % 2 == 0 and 2 <= W <= N, f"n_ctx must be even in [2, {N}], got {W}"
    half = W // 2
    k = model.config.ff9_k
    tbptt_frames = 2 * N if tbptt_frames is None else tbptt_frames  # 0 is valid (detach every slide)
    max_frames = min(5 * N, 10 * W) if max_frames is None else max_frames
    end = min(T, max_frames)

    K_max, tau_ctx_idx, d_idx_val = _tau_d_consts(model)
    af_all = model.action_features(actions_idx)  # (B,T,n_act,E) or None
    blank_half = model.memory_tokens.expand(B, half, -1, -1)
    d_col_W = torch.full((B, W), d_idx_val, device=device, dtype=torch.long)

    def af(a, b):
        return af_all[:, a:b] if af_all is not None else None

    # ---- init window [0, W): near-clean latents, learned-blank memory -> construct initial memory ----
    zc = model._noise_to_ctx(z1[:, :W])
    tau_init = torch.full((B, W), tau_ctx_idx, device=device, dtype=torch.long)
    blank_W = model.memory_tokens.expand(B, W, -1, -1)
    _, mem_win = model(zc, tau_init, d_col_W, af(0, W), memory_in=blank_W,
                       positions=torch.arange(W, device=device), return_memory=True)
    old_mem = mem_win[:, half:]          # (B, half, M, E) — recent half, carried with graph
    old_constructed_at = half            # clip pos of the oldest carried memory's construction
    relay_depth = half                   # frames of graph currently in the relay

    total = z1.new_zeros(())
    n_terms = 0
    sum_flow = sum_ff9 = 0.0

    s = half  # next window starts here: window [s, s+W), old half [s, s+half) == carried frames
    while s + W <= end:
        new_a, new_b = s + half, s + W            # new-half clip positions [new_a, new_b)
        modes = _sample_modes(B, device, gen, force_mode)  # (B,) True=full-noise
        m = modes.view(B, 1, 1, 1).float()

        # truncated BPTT: detach the carried memory once the relay is deeper than tbptt_frames
        if relay_depth > tbptt_frames:
            old_mem = old_mem.detach()
            relay_depth = 0

        # ---- build the window's noised latents (B, W, L, D) + per-frame tau ----
        z1_win = z1[:, s:s + W]
        z0 = torch.randn(z1_win.shape, device=device, generator=gen)
        # old half: clean mode -> near-clean GT; noise mode -> pure noise.
        old_clean = model._noise_to_ctx(z1_win[:, :half])
        old_part = m[:, :1] * z0[:, :half] + (1 - m[:, :1]) * old_clean   # noise vs near-clean
        tau_old = torch.where(modes.view(B, 1), torch.zeros(B, half, device=device, dtype=torch.long),
                              torch.full((B, half), tau_ctx_idx, device=device, dtype=torch.long))
        # new half: clean mode -> sampled-signal noised; noise mode -> pure noise.
        tau_new_idx = torch.randint(0, K_max, (B, half), device=device, generator=gen)
        tau_new_idx = torch.where(modes.view(B, 1), torch.zeros_like(tau_new_idx), tau_new_idx)
        tau_new = (tau_new_idx.float() / K_max).view(B, half, 1, 1)
        new_part = (1 - tau_new) * z0[:, half:] + tau_new * z1_win[:, half:]

        z_tilde = torch.cat([old_part, new_part], dim=1)
        tau_col = torch.cat([tau_old, tau_new_idx], dim=1)
        memory_in = torch.cat([old_mem, blank_half], dim=1)   # old=real(graph), new=blank

        z_hat, mem_out = model(z_tilde, tau_col, d_col_W, af(s, s + W), memory_in=memory_in,
                               positions=torch.arange(W, device=device), return_memory=True)

        # ---- loss on NEW half only ----
        flow = _flow_loss(model, z_hat[:, half:], z1_win[:, half:], tau_new)
        new_mem = mem_out[:, half:]   # (B, half, M, E) — graph-attached; carried + FF9-scored
        if k > 0 and new_b + k <= T:
            z1_sub = z1[:, new_a:new_b + k]                          # (B, half+k, L, D)
            mem_sub = z1.new_zeros(B, half + k, new_mem.shape[-2], new_mem.shape[-1])
            mem_sub[:, :half] = new_mem
            ff9 = model._ff9_loss(z1_sub, mem_sub, af(new_a, new_b + k), k)
            scale = (flow.detach() / ff9.detach().clamp(min=1e-8))   # normalize like model.loss
            slide_loss = flow + scale * ff9
            sum_ff9 += float(ff9.detach())
        else:
            slide_loss = flow
        total = total + slide_loss
        sum_flow += float(flow.detach())
        n_terms += 1

        # ---- carry: next old half = current new-half memories (with near-clean GT latents next time) ----
        old_mem = new_mem
        old_constructed_at = new_a
        relay_depth += half
        s += half

    n_terms = max(1, n_terms)
    parts = {"flow": sum_flow / n_terms, "ff9": sum_ff9 / n_terms, "n_slides": float(n_terms),
             "n_ctx": float(W)}
    return total / n_terms, parts
