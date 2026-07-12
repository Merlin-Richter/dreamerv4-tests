"""Long-clip rollout training for hierarchical sparse archive memory.

This differs from ``experiments/mem2mem/rollout.py`` in two load-bearing ways:

* the local window is fixed and uses absolute positions;
* dense relay graphs are backwarded/freed in bounded TBPTT blocks, while detached
  archive sources and leaf proxies retain compressor credit for the whole clip.

The function in this module performs backward itself.  The caller must zero gradients
before calling it and must not step model parameters until it returns (the deferred
compressor VJP is the final part of the call).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from model import DynamicsModelArchive


@dataclass
class ArchiveRecord:
    source: torch.Tensor       # detached (B,N,M,E)
    proxy: torch.Tensor        # leaf (B,M,R,E), accumulates dL/dS over TBPTT blocks
    end_pos: int


def _tau_d_consts(model):
    tau_ctx_idx = min(round(model.config.context_signal * model.K_max), model.K_max - 1)
    return tau_ctx_idx, model.n_d - 1


def _sample_tau_d(model, B, T, device, gen, n_d_unlocked=1):
    n_d = model.n_d
    k = n_d if n_d_unlocked is None else max(1, min(n_d, int(n_d_unlocked)))
    off = torch.randint(0, k, (B, T), device=device, generator=gen)
    d_idx = (n_d - 1) - off
    K = torch.pow(2, d_idx)
    step = (torch.rand((B, T), device=device, generator=gen) * K).long()
    step = torch.minimum(step, K - 1)
    tau_idx = step * torch.pow(2, n_d - 1 - d_idx)
    return tau_idx, d_idx


def _sample_base_modes(B, device, gen, force_mode):
    if force_mode == "noise":
        return torch.ones(B, dtype=torch.bool, device=device)
    if force_mode == "clean":
        return torch.zeros(B, dtype=torch.bool, device=device)
    if force_mode is not None:
        raise ValueError(f"force_mode must be clean/noise/None, got {force_mode!r}")
    return torch.rand(B, device=device, generator=gen) < 0.5


def _newhalf_loss(model, *, old_part, tau_old, new_part, tau_new_idx, d_new_idx, z1_new,
                  af_win, memory_in, positions, half, bootstrap, archive_bank,
                  archive_positions, archive_batch_mask):
    """Shortcut-forcing loss on the new half with raw differentiable archive proxies."""
    d_min_idx = model.n_d - 1
    tau_new = model._tau_value(tau_new_idx)[..., None, None]
    d_old = torch.full_like(tau_old, d_min_idx)
    z_tilde = torch.cat((old_part, new_part), dim=1)
    tau_col = torch.cat((tau_old, tau_new_idx), dim=1)
    d_col = torch.cat((d_old, d_new_idx), dim=1)
    archive_kw = dict(archive_bank=archive_bank, archive_positions=archive_positions,
                      archive_batch_mask=archive_batch_mask)

    z_hat, mem_out = model(
        z_tilde, tau_col, d_col, af_win, memory_in=memory_in,
        positions=positions, return_memory=True, **archive_kw)
    z_hat_new = z_hat[:, half:]
    new_mem = mem_out[:, half:]
    flow_loss = (z_hat_new - z1_new) ** 2

    do_boot = bootstrap and bool((d_new_idx != d_min_idx).any())
    if do_boot:
        with torch.no_grad():
            half_d_idx = (d_new_idx + 1).clamp(max=d_min_idx)
            half_d = model._d_value(half_d_idx)[..., None, None]
            tau_inc = torch.pow(2, (model.n_d - 2 - d_new_idx).clamp(min=0))
            tau2_idx = (tau_new_idx + tau_inc).clamp(max=model.K_max - 1)
            tau2 = model._tau_value(tau2_idx)[..., None, None]
            d_col_half = torch.cat((d_old, half_d_idx), dim=1)
            y1 = model(
                z_tilde, tau_col, d_col_half, af_win, memory_in=memory_in,
                positions=positions, **archive_kw)[:, half:]
            b1 = (y1 - new_part) / (1 - tau_new)
            z_prime_new = new_part + b1 * half_d
            z_tilde2 = torch.cat((old_part, z_prime_new), dim=1)
            tau_col2 = torch.cat((tau_old, tau2_idx), dim=1)
            y2 = model(
                z_tilde2, tau_col2, d_col_half, af_win, memory_in=memory_in,
                positions=positions, **archive_kw)[:, half:]
            b2 = (y2 - z_prime_new) / (1 - tau2)
            v_target = (b1 + b2) / 2
        v_pred = (z_hat_new - new_part) / (1 - tau_new)
        boot_loss = (1 - tau_new) ** 2 * (v_pred - v_target) ** 2
        per_token = torch.where((d_new_idx == d_min_idx)[..., None, None], flow_loss, boot_loss)
    else:
        per_token = flow_loss

    w = (1 - model.config.ramp_min) * tau_new + model.config.ramp_min
    return (w * per_token).mean(), new_mem, (w * flow_loss).mean()


def archive_rollout_backward(
        model: DynamicsModelArchive, z1: torch.Tensor, actions_idx: torch.Tensor = None, *,
        device, gen=None, dense_tbptt_frames: int = 64, max_frames: int = None,
        bootstrap: bool = False, n_d_unlocked: int | None = 1,
        fast_memory_hide_frac: float = 0.0, hide_latents_frac: float = 0.5,
        archive_drop_frac: float = 0.0, relay_grad_clip: float = None,
        force_mode: str = None, force_fast_hide: bool | None = None,
        force_hide_latents: bool | None = None,
        force_archive_drop: bool | None = None) -> dict:
    """Run one long archive rollout, backward bounded dense blocks, then deferred compressor VJP.

    The caller must call ``zero_grad`` first and performs global grad clipping / optimizer step after
    this function returns.  Model parameters remain fixed throughout the complete clip.
    """
    assert model.n_memory > 0
    B, T, _, _ = z1.shape
    W = model.config.max_temporal_length
    assert W % 2 == 0
    half = W // 2
    assert dense_tbptt_frames >= half and dense_tbptt_frames % half == 0
    assert 0 <= fast_memory_hide_frac <= 1
    assert 0 <= hide_latents_frac <= 1
    assert 0 <= archive_drop_frac <= 1
    end = min(T, max_frames) if max_frames is not None else T
    starts = list(range(half, end - W + 1, half))
    assert starts, f"clip length {end} must contain init W={W} plus one half-window"
    n_slides = len(starts)

    if gen is None:
        gen = torch.Generator(device=device)
        gen.manual_seed(0)

    tau_ctx_idx, d_min_idx = _tau_d_consts(model)
    d_col_W = torch.full((B, W), d_min_idx, device=device, dtype=torch.long)

    def af(a, b):
        # Parameter-derived action features must be rebuilt after every block backward; reusing one
        # graph-backed table projection across blocks would backward through a freed graph.
        return model.action_features(actions_idx[:, a:b]) if actions_idx is not None else None

    # Relay-gradient instrumentation (same scale-down-only policy as mem2mem).
    clip_stats = {"hooks": 0, "clipped": 0, "sum_norm": 0.0}

    def relay_hook(grad):
        n = grad.flatten(1).norm(dim=1)
        clip_stats["hooks"] += int(n.numel())
        clip_stats["clipped"] += int((n > relay_grad_clip).sum())
        clip_stats["sum_norm"] += float(n.sum())
        scale = (relay_grad_clip / (n + 1e-12)).clamp(max=1.0)
        return grad * scale.view(-1, *([1] * (grad.dim() - 1)))

    def register_relay(t):
        if relay_grad_clip is not None and t.requires_grad:
            t.register_hook(relay_hook)

    # Archive source/proxy state.  all_records survive to deferred VJP; active_records obey bank cap.
    all_records: list[ArchiveRecord] = []
    active_records: list[ArchiveRecord] = []
    pending = None
    pending_start = 0
    next_source_pos = 0

    def append_sources(mem: torch.Tensor, start_pos: int):
        nonlocal pending, pending_start, next_source_pos, active_records
        assert start_pos == next_source_pos, f"archive source gap: {start_pos} != {next_source_pos}"
        detached = mem.detach()
        if pending is None:
            pending = detached.new_empty((B, 0, model.n_memory, model.config.embedding_dim))
        pending = torch.cat((pending, detached), dim=1)
        next_source_pos += detached.shape[1]
        N = model.archive_interval
        while pending.shape[1] >= N:
            source = pending[:, :N]
            end_pos = pending_start + N - 1
            # Parameters must not step until the exact same compressor call is recomputed at clip end.
            with torch.no_grad():
                value = model.archive_compressor(source)
            proxy = value.detach().requires_grad_()
            rec = ArchiveRecord(source=source, proxy=proxy, end_pos=end_pos)
            all_records.append(rec)
            active_records.append(rec)
            if model.archive_max_sets > 0:
                active_records = active_records[-model.archive_max_sets:]
            pending = pending[:, N:]
            pending_start += N

    def raw_bank():
        if not active_records:
            return None, torch.empty(0, dtype=torch.long, device=device)
        bank = torch.stack([r.proxy for r in active_records], dim=1)
        pos = torch.tensor([r.end_pos for r in active_records], device=device, dtype=torch.long)
        return bank, pos

    # Initial window writes fast memories.  No archive can yet be eligible by construction.
    zc = model._noise_to_ctx(z1[:, :W])
    tau_init = torch.full((B, W), tau_ctx_idx, device=device, dtype=torch.long)
    blank_W = model.memory_tokens.expand(B, W, -1, -1)
    _, mem_win = model(
        zc, tau_init, d_col_W, af(0, W), memory_in=blank_W,
        positions=torch.arange(W, device=device), return_memory=True)
    append_sources(mem_win, 0)
    old_mem = mem_win[:, half:]
    register_relay(old_mem)

    block_loss = None
    block_frames = 0
    sum_flow = sum_flow_norm = 0.0
    n_clean = n_noise = n_fast_hide = n_hide_lat = n_archive_drop = 0

    for slide_i, s in enumerate(starts):
        new_a, new_b = s + half, s + W
        bank, bank_pos = raw_bank()
        first_new = new_a
        has_eligible = bool(bank_pos.numel() and
                            ((first_new - bank_pos) >=
                             (W - model.archive_interval + 1)).any())

        base_noise = _sample_base_modes(B, device, gen, force_mode)
        if force_fast_hide is None:
            fast_hide = ((torch.rand(B, device=device, generator=gen) < fast_memory_hide_frac)
                         if has_eligible else torch.zeros(B, dtype=torch.bool, device=device))
        else:
            fast_hide = torch.full((B,), bool(force_fast_hide and has_eligible),
                                   dtype=torch.bool, device=device)
        if force_hide_latents is None:
            hide_lat = fast_hide & (torch.rand(B, device=device, generator=gen) < hide_latents_frac)
        else:
            hide_lat = fast_hide & bool(force_hide_latents)

        # Fast-hide (a) forces clean latents; fast-hide (b) forces full noise.  Other examples retain
        # the existing independent 50/50 clean/noise curriculum.
        modes = torch.where(fast_hide, hide_lat, base_noise)  # True = all latents pure noise
        mode_f = modes.view(B, 1, 1, 1).to(z1.dtype)

        if force_archive_drop is None:
            archive_drop = torch.rand(B, device=device, generator=gen) < archive_drop_frac
        else:
            archive_drop = torch.full((B,), bool(force_archive_drop),
                                      dtype=torch.bool, device=device)
        # Never construct a pathless example: fast memory + latents hidden requires the archive.
        archive_drop = archive_drop & ~(fast_hide & hide_lat)
        archive_keep = ~archive_drop

        z1_win = z1[:, s:s + W]
        z0 = torch.randn(z1_win.shape, device=device, generator=gen)
        old_clean = model._noise_to_ctx(z1_win[:, :half])
        old_part = mode_f[:, :1] * z0[:, :half] + (1 - mode_f[:, :1]) * old_clean
        tau_old = torch.where(
            modes[:, None], torch.zeros(B, half, device=device, dtype=torch.long),
            torch.full((B, half), tau_ctx_idx, device=device, dtype=torch.long))

        tau_new_idx, d_new_idx = _sample_tau_d(
            model, B, half, device, gen, n_d_unlocked=n_d_unlocked)
        tau_new_idx = torch.where(modes[:, None], torch.zeros_like(tau_new_idx), tau_new_idx)
        tau_new = model._tau_value(tau_new_idx)[..., None, None]
        new_part = (1 - tau_new) * z0[:, half:] + tau_new * z1_win[:, half:]

        # Recreate the parameter view every slide so no ExpandBackward node crosses a completed
        # blockwise backward boundary.
        blank_half = model.memory_tokens.expand(B, half, -1, -1)
        old_input = torch.where(
            fast_hide.view(B, 1, 1, 1), blank_half, old_mem)
        memory_in = torch.cat((old_input, blank_half), dim=1)
        flow, new_mem, flow_norm = _newhalf_loss(
            model, old_part=old_part, tau_old=tau_old, new_part=new_part,
            tau_new_idx=tau_new_idx, d_new_idx=d_new_idx, z1_new=z1_win[:, half:],
            af_win=af(s, s + W), memory_in=memory_in,
            positions=torch.arange(s, s + W, device=device), half=half,
            bootstrap=bootstrap, archive_bank=bank, archive_positions=bank_pos,
            archive_batch_mask=archive_keep)

        normalized = flow / n_slides
        block_loss = normalized if block_loss is None else block_loss + normalized
        block_frames += half
        sum_flow += float(flow.detach())
        sum_flow_norm += float(flow_norm.detach())
        n_noise += int(modes.sum())
        n_clean += int((~modes).sum())
        n_fast_hide += int(fast_hide.sum())
        n_hide_lat += int(hide_lat.sum())
        n_archive_drop += int(archive_drop.sum())

        old_mem = new_mem
        register_relay(old_mem)
        append_sources(new_mem, new_a)

        is_last = slide_i == n_slides - 1
        if block_frames >= dense_tbptt_frames or is_last:
            assert block_loss is not None
            block_loss.backward()
            old_mem = old_mem.detach()
            register_relay(old_mem)
            block_loss = None
            block_frames = 0

    # Exact deferred chain rule through the tiny compressor.  Sources are detached, and model
    # parameters have not stepped, so recomputation matches the no-grad proxy values.
    used = [r for r in all_records if r.proxy.grad is not None]
    if used:
        source_batch = torch.cat([r.source for r in used], dim=0)
        grad_batch = torch.cat([r.proxy.grad for r in used], dim=0)
        archive_real = model.archive_compressor(source_batch)
        archive_real.backward(grad_batch)

    denom = max(1, n_slides * B)
    return {
        "loss": sum_flow / n_slides,
        "flow": sum_flow / n_slides,
        "flow_norm": sum_flow_norm / n_slides,
        "n_slides": float(n_slides),
        "n_archives": float(len(all_records)),
        "n_archives_used": float(len(used)),
        "clean_frac": n_clean / denom,
        "noise_frac": n_noise / denom,
        "fast_hide_frac": n_fast_hide / denom,
        "hide_latents_frac": n_hide_lat / denom,
        "archive_drop_frac": n_archive_drop / denom,
        "relay_clip_frac": clip_stats["clipped"] / max(1, clip_stats["hooks"]),
    }
