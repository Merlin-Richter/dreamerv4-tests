# SEEDED from experiments/mem2mem/rollout.py @ f405034 (the mem2mem sliding-rollout
# loss, the current best memory-training signal). EDITABLE LAYER — loop may modify.

"""mem->mem training rollout (task: tasks/in-progress/test-new-memory-training.md).

The dynamics model already learns latents->memory (write, via the windowed pass) and memory->latents
(read, via the FF9 sufficiency loss). What it does NOT get is a signal to *construct a memory token
from previous memory tokens* — at training time the memory tokens in context are always the
learned-blank init, never real computed ones. This rollout supplies that signal.

Mechanism (no KV cache — this is training, full recompute each window):
  The model's temporal attention is causal *per token slot*, so a frame's MEMORY slot attends to the
  memory slots of earlier frames. If we (a) inject REAL, graph-attached memory tokens from a previous
  forward into the OLD half of a window, (b) let the model construct the NEW half's memory + denoise
  the NEW half's latents, and (c) put the flow loss ONLY on the new half, then the new memory is
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


def _sample_tau_d(model, B, T, device, gen, n_d_unlocked=None):
    """model.sample_tau_d, but threaded through the rollout's generator for reproducibility, with an
    optional step-size CURRICULUM. d ~ U over the ``n_d_unlocked`` FINEST steps (d_idx in
    {n_d-n_d_unlocked .. n_d-1}); tau ~ U on the grid implied by d (snapped to the d_min grid).

    Why finest-first: a coarse step's bootstrap target is two d/2 (one-finer) steps of the model. If we
    unlock steps finest-first, the finer step a coarse step distils from is ALWAYS already trained, so we
    never distil from an untrained step. n_d_unlocked=1 => only d_min => pure flow (no bootstrap).
    None => all n_d steps (no curriculum). tau_idx=0 (step=0) is a valid grid point for every d, so
    forcing tau=0 later stays on-distribution."""
    n_d = model.n_d
    k = n_d if n_d_unlocked is None else max(1, min(n_d, int(n_d_unlocked)))
    off = torch.randint(0, k, (B, T), device=device, generator=gen)  # 0..k-1
    d_idx = (n_d - 1) - off                                          # the k finest steps
    K = torch.pow(2, d_idx)
    step = (torch.rand((B, T), device=device, generator=gen) * K).long()
    step = torch.minimum(step, K - 1)
    tau_idx = step * torch.pow(2, n_d - 1 - d_idx)  # snap to the d_min grid
    return tau_idx, d_idx


def _newhalf_loss(model, *, old_part, tau_old, new_part, tau_new_idx, d_new_idx, z1_new,
                  af_win, memory_in, positions, half, bootstrap):
    """Shortcut-forcing diffusion loss on the NEW half, holding the OLD half fixed as context.

    Mirrors ``DynamicsModel.loss``'s diffusion term, restricted to the new half: at the finest step
    (d_idx == n_d-1) it is the x-prediction flow MSE; at coarser steps it distils two d/2 steps of the
    model itself (stop-grad bootstrap target, Eq. 7) so the new half learns the full shortcut ladder,
    exactly like the normal windowed loss. The OLD half (context: near-clean GT or pure noise) keeps its
    fixed tau and the finest d across the main and both bootstrap sub-step forwards, so the new half reads
    a consistent scene+memory while only its own (tau, d) advance along the trajectory.

    Returns (loss, new_mem, flow_norm). new_mem is the graph-attached new-half memory from the MAIN
    forward. flow_norm is the ramp-weighted PURE x-prediction (d_min flow) loss over ALL new-half tokens
    regardless of whether each token's gradient comes from flow or bootstrap — i.e. the loss magnitude as
    if bootstrap were off. It is the FF9 normalizer basis the rollout-only (no-boot) winner used; using it
    keeps the FF9 term's effective weight invariant to enabling the bootstrap (whose smaller
    self-distillation term otherwise dilutes the mixed mean and silently down-weights the memory objective).
    """
    d_min_idx = model.n_d - 1
    tau_new = model._tau_value(tau_new_idx)[..., None, None]               # (B, half, 1, 1)
    d_old = torch.full_like(tau_old, d_min_idx)                            # context held at finest d
    z_tilde = torch.cat([old_part, new_part], dim=1)
    tau_col = torch.cat([tau_old, tau_new_idx], dim=1)
    d_col = torch.cat([d_old, d_new_idx], dim=1)

    z_hat, mem_out = model(z_tilde, tau_col, d_col, af_win, memory_in=memory_in,
                           positions=positions, return_memory=True)
    z_hat_new = z_hat[:, half:]
    new_mem = mem_out[:, half:]
    flow_loss = (z_hat_new - z1_new) ** 2

    # Skip the two bootstrap sub-step forwards when there is no coarse step in the batch (all frames at
    # d_min -> pure flow): the whole curriculum warmup, and any all-min batch. Saves 2 forwards/slide.
    do_boot = bootstrap and bool((d_new_idx != d_min_idx).any())
    if do_boot:
        with torch.no_grad():
            half_d_idx = (d_new_idx + 1).clamp(max=d_min_idx)             # one step finer (d/2)
            half_d = model._d_value(half_d_idx)[..., None, None]
            tau_inc = torch.pow(2, (model.n_d - 2 - d_new_idx).clamp(min=0))
            tau2_idx = (tau_new_idx + tau_inc).clamp(max=model.K_max - 1)
            tau2 = model._tau_value(tau2_idx)[..., None, None]
            d_col_half = torch.cat([d_old, half_d_idx], dim=1)
            # first d/2 step from the new half (old half fixed as context)
            y1 = model(z_tilde, tau_col, d_col_half, af_win, memory_in=memory_in,
                       positions=positions)[:, half:]
            b1 = (y1 - new_part) / (1 - tau_new)
            z_prime_new = new_part + b1 * half_d
            # second d/2 step from tau2 (advance only the new half)
            z_tilde2 = torch.cat([old_part, z_prime_new], dim=1)
            tau_col2 = torch.cat([tau_old, tau2_idx], dim=1)
            y2 = model(z_tilde2, tau_col2, d_col_half, af_win, memory_in=memory_in,
                       positions=positions)[:, half:]
            b2 = (y2 - z_prime_new) / (1 - tau2)
            v_target = (b1 + b2) / 2
        v_pred = (z_hat_new - new_part) / (1 - tau_new)
        boot_loss = (1 - tau_new) ** 2 * (v_pred - v_target) ** 2
        is_min = (d_new_idx == d_min_idx)[..., None, None]
        per_token = torch.where(is_min, flow_loss, boot_loss)
    else:
        per_token = flow_loss

    w = (1 - model.config.ramp_min) * tau_new + model.config.ramp_min     # ramp weight, Eq. 8
    flow_norm = (w * flow_loss).mean()   # pure-flow magnitude (bootstrap-independent) — FF9 norm basis
    return (w * per_token).mean(), new_mem, flow_norm


@torch.no_grad()
def _sample_modes(B, device, gen, force_mode):
    if force_mode == "noise":
        return torch.ones(B, dtype=torch.bool, device=device)
    if force_mode == "clean":
        return torch.zeros(B, dtype=torch.bool, device=device)
    return torch.rand(B, device=device, generator=gen) < 0.5  # True -> full-noise


def mem2mem_rollout_loss(model, z1, actions_idx=None, *, n_ctx, device,
                         tbptt_frames=None, max_frames=None, gen=None, force_mode=None,
                         bootstrap=True, n_d_unlocked=None, use_ff9=True, ff9_norm_flow=False,
                         relay_grad_clip=None):
    """One mem->mem rollout over a long clip. Returns (total_loss, parts_dict).

    z1:          (B, T, L, D) clean latents (T should be long, e.g. up to 5*max_temporal_length).
    actions_idx: (B, T) long action ids, or None (unlabeled).
    n_ctx:       window size W (a power of two, 2 <= W <= max_temporal_length). SAME for the whole
                 batch (GPU-parallel requirement). The window slides by W/2.
    tbptt_frames: detach the carried memory once the relay graph is deeper than this many frames
                 (truncated BPTT; default 2*max_temporal_length). Bounds memory footprint.
    max_frames:  stop after this many frames from t=0 (default min(5*N, 10*W)).
    force_mode:  None -> per-element 50/50 clean/noise; "noise"/"clean" -> force (for tests).
    bootstrap:   include the shortcut bootstrap distillation (coarser d/2 steps) in the new-half
                 denoising loss, exactly like the normal windowed model.loss. Applies to BOTH the
                 clean-context and the full-noise (memory-only) modes. False -> finest-step flow only.
    n_d_unlocked: step-size CURRICULUM — sample d only from the ``n_d_unlocked`` FINEST steps (the
                 trainer ramps this 1 -> n_d over training). 1 => d_min only => pure flow (the bootstrap
                 forwards are skipped). None => all steps. See _sample_tau_d for the finest-first rationale.
    use_ff9:     include the explicit FF9 sufficiency term. False -> memory is trained ONLY by the rollout
                 flow loss (in the 50% full-noise mode the new half can only be reconstructed from carried
                 memory, so that flow term IS the memory signal). Ablation: is FF9 needed at all?
    relay_grad_clip: None (default, OFF -> byte-identical) or a float C. Per-hop relay GRADIENT
                 normalizer: a backward hook on each carried memory tensor scales its gradient DOWN per
                 batch element so ||grad_b|| <= C (scale-down only; well-behaved/converged grads pass
                 untouched). Combats the measured backward explosion through the relay (~2-3x/hop at init,
                 catastrophic for small windows) without changing the forward pass or inference. Per-epoch
                 clip stats are stashed on model._relay_clip_stats for logging.
    ff9_norm_flow: normalize the FF9 term by the PURE d_min flow magnitude (``flow_norm``) instead of the
                 mixed flow+bootstrap diffusion mean. Keeps FF9's effective weight invariant to whether
                 the bootstrap is on — required for a fair bootstrap A/B (the mixed mean is diluted by the
                 smaller bootstrap self-distillation term, which silently down-weights memory). Default
                 False = mixed mean (faithful to model.loss).

    Loss = shortcut-forcing diffusion on the new half (flow at the finest step + bootstrap distillation
    at coarser steps) [+ FF9 sufficiency (new-half memories) when use_ff9], normalized like model.loss.
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

    # Per-hop relay gradient normalizer (OFF by default => no hook => byte-identical). Scales each
    # carried memory tensor's gradient DOWN per batch element to ||grad_b|| <= C, taming the backward
    # explosion through the relay. Stats stashed on the model for per-epoch logging.
    clip_stats = {"hooks": 0, "clipped": 0, "sum_norm": 0.0}
    model._relay_clip_stats = clip_stats

    def _relay_hook(grad):
        n = grad.flatten(1).norm(dim=1)                          # (B,) per-sequence grad norm
        clip_stats["hooks"] += int(n.numel())
        clip_stats["clipped"] += int((n > relay_grad_clip).sum())
        clip_stats["sum_norm"] += float(n.sum())
        scale = (relay_grad_clip / (n + 1e-12)).clamp(max=1.0)   # scale-down only
        return grad * scale.view(-1, *([1] * (grad.dim() - 1)))

    def _register_relay(t):
        if relay_grad_clip is not None and t.requires_grad:
            t.register_hook(_relay_hook)

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
    _register_relay(old_mem)             # per-hop relay grad normalizer (no-op when OFF)
    old_constructed_at = half            # clip pos of the oldest carried memory's construction
    relay_depth = half                   # frames of graph currently in the relay

    total = z1.new_zeros(())
    n_terms = 0
    sum_flow = sum_ff9 = sum_flow_norm = 0.0

    s = half  # next window starts here: window [s, s+W), old half [s, s+half) == carried frames
    while s + W <= end:
        new_a, new_b = s + half, s + W            # new-half clip positions [new_a, new_b)
        modes = _sample_modes(B, device, gen, force_mode)  # (B,) True=full-noise
        m = modes.view(B, 1, 1, 1).float()

        # truncated BPTT: detach the carried memory once the relay is deeper than tbptt_frames
        if relay_depth > tbptt_frames:
            old_mem = old_mem.detach()
            relay_depth = 0

        # ---- build the window's noised latents (B, W, L, D) + per-frame (tau, d) ----
        z1_win = z1[:, s:s + W]
        z0 = torch.randn(z1_win.shape, device=device, generator=gen)
        # old half: clean mode -> near-clean GT; noise mode -> pure noise.
        old_clean = model._noise_to_ctx(z1_win[:, :half])
        old_part = m[:, :1] * z0[:, :half] + (1 - m[:, :1]) * old_clean   # noise vs near-clean
        tau_old = torch.where(modes.view(B, 1), torch.zeros(B, half, device=device, dtype=torch.long),
                              torch.full((B, half), tau_ctx_idx, device=device, dtype=torch.long))
        # new half: snapped (tau, d) shortcut grid (clean mode) or pure noise tau=0 with sampled d
        # (noise mode -> memory is the only scene carrier; d still varies to train coarse steps-from-noise).
        tau_new_idx, d_new_idx = _sample_tau_d(model, B, half, device, gen, n_d_unlocked=n_d_unlocked)
        tau_new_idx = torch.where(modes.view(B, 1), torch.zeros_like(tau_new_idx), tau_new_idx)
        tau_new = model._tau_value(tau_new_idx)[..., None, None]
        new_part = (1 - tau_new) * z0[:, half:] + tau_new * z1_win[:, half:]

        memory_in = torch.cat([old_mem, blank_half], dim=1)   # old=real(graph), new=blank

        # ---- shortcut-forcing diffusion loss on the NEW half only (flow + bootstrap) ----
        flow, new_mem, flow_norm = _newhalf_loss(
            model, old_part=old_part, tau_old=tau_old, new_part=new_part,
            tau_new_idx=tau_new_idx, d_new_idx=d_new_idx, z1_new=z1_win[:, half:],
            af_win=af(s, s + W), memory_in=memory_in,
            positions=torch.arange(W, device=device), half=half, bootstrap=bootstrap)
        # new_mem: (B, half, M, E) — graph-attached; carried + FF9-scored
        if use_ff9 and k > 0 and new_b + k <= T:
            z1_sub = z1[:, new_a:new_b + k]                          # (B, half+k, L, D)
            mem_sub = z1.new_zeros(B, half + k, new_mem.shape[-2], new_mem.shape[-1])
            mem_sub[:, :half] = new_mem
            ff9 = model._ff9_loss(z1_sub, mem_sub, af(new_a, new_b + k), k)
            # FF9 normalizer basis: mixed diffusion (faithful to model.loss) by default; pure d_min flow
            # magnitude under ff9_norm_flow, so the bootstrap can't dilute FF9's effective weight.
            norm_basis = flow_norm if ff9_norm_flow else flow
            scale = (norm_basis.detach() / ff9.detach().clamp(min=1e-8))
            slide_loss = flow + scale * ff9
            sum_ff9 += float(ff9.detach())
        else:
            slide_loss = flow
        total = total + slide_loss
        sum_flow += float(flow.detach())
        sum_flow_norm += float(flow_norm.detach())
        n_terms += 1

        # ---- carry: next old half = current new-half memories (with near-clean GT latents next time) ----
        old_mem = new_mem
        _register_relay(old_mem)         # per-hop relay grad normalizer (no-op when OFF)
        old_constructed_at = new_a
        relay_depth += half
        s += half

    n_terms = max(1, n_terms)
    parts = {"flow": sum_flow / n_terms, "flow_norm": sum_flow_norm / n_terms,
             "ff9": sum_ff9 / n_terms, "n_slides": float(n_terms), "n_ctx": float(W)}
    return total / n_terms, parts
