"""Sparse write-slots rollout training loss — fork of experiments/mem2mem/rollout.py's
mem2mem_rollout_loss, specialized to the v3 design (no FF9, no bootstrap, no curriculum):

- FIXED window W == 2*n_sparse (16 for n=8); slides advance W/2 == n_sparse, so window starts are
  write-aligned: write slots sit at window indices 0 (old half) and n (new half, index 0 of the
  new half). Asserted, not assumed.
- ABSOLUTE positions per slide (arange(s, s+W)) — the mask's phase must be clip-absolute (RoPE is
  relative, unaffected; the on-the-fly rope path handles arbitrary positions).
- CARRY = the new half's single WRITE-slot memory (graph-attached), not all frames. The old
  half's write slot gets the carried set injected; every other slot is phase-init (the model
  wrapper enforces scratch-init regardless of what is passed).
- 50/50 clean/noise mode per element per slide, d_min-only GT flow on the new half, ramp weight —
  identical to the no-bootstrap winner recipe (reuses `_newhalf_loss` and `_sample_tau_d` from
  the mem2mem experiment so the loss formula cannot drift).
- TBPTT identical: detach the carried memory once the relay graph exceeds tbptt_frames
  (default 2*max_temporal_length = Merlin's ~2x window ruling).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mem2mem"))
from rollout import _newhalf_loss, _sample_modes, _sample_tau_d, _tau_d_consts  # noqa: E402


def sparse_rollout_loss(model, z1, actions_idx=None, *, device, tbptt_frames=None,
                        max_frames=None, gen=None, force_mode=None):
    """One sparse write-slots rollout over a long clip. Returns (total_loss, parts_dict)."""
    n = model.SPARSE_N
    N = model.config.max_temporal_length
    B, T, L, D = z1.shape
    W = 2 * n
    assert W <= N, f"W={W} exceeds max_temporal_length {N}"
    half = W // 2
    assert half == n, "slide must equal n_sparse for write alignment"
    tbptt_frames = 2 * N if tbptt_frames is None else tbptt_frames
    max_frames = min(5 * N, 10 * W) if max_frames is None else max_frames
    end = min(T, max_frames)

    K_max, tau_ctx_idx, d_idx_val = _tau_d_consts(model)
    af_all = model.action_features(actions_idx)
    d_col_W = torch.full((B, W), d_idx_val, device=device, dtype=torch.long)

    def af(a, b):
        return af_all[:, a:b] if af_all is not None else None

    M, E = model.config.n_memory, model.config.embedding_dim

    def build_mem_in(carried, s):
        """(B, W, M, E): phase inits with the OLD-half write slot (window idx 0, abs pos s)
        replaced by the carried write set. The model wrapper re-forces scratch slots anyway."""
        pos = torch.arange(s, s + W, device=device)
        is_write = (pos % n) == 0
        base = model.mem_init2[(~is_write).long()].unsqueeze(0).expand(B, -1, -1, -1)
        if carried is None:
            return base
        out = base.clone()
        out[:, 0:1] = carried                      # window idx 0 == abs pos s, s % n == 0
        return out

    # ---- init window [0, W): near-clean latents, phase-init memory -> first write sets ----
    zc = model._noise_to_ctx(z1[:, :W])
    tau_init = torch.full((B, W), tau_ctx_idx, device=device, dtype=torch.long)
    _, mem_win = model(zc, tau_init, d_col_W, af(0, W), memory_in=None,
                       positions=torch.arange(W, device=device), return_memory=True)
    old_mem = mem_win[:, half:half + 1]            # the write set at abs pos n (next old half's write)
    relay_depth = half

    total = z1.new_zeros(())
    n_terms = 0
    sum_flow = sum_flow_norm = 0.0

    s = half
    while s + W <= end:
        assert s % n == 0
        modes = _sample_modes(B, device, gen, force_mode)   # (B,) True = full-noise
        m = modes.view(B, 1, 1, 1).float()

        if relay_depth > tbptt_frames:
            old_mem = old_mem.detach()
            relay_depth = 0

        z1_win = z1[:, s:s + W]
        z0 = torch.randn(z1_win.shape, device=device, generator=gen)
        old_clean = model._noise_to_ctx(z1_win[:, :half])
        old_part = m[:, :1] * z0[:, :half] + (1 - m[:, :1]) * old_clean
        tau_old = torch.where(modes.view(B, 1),
                              torch.zeros(B, half, device=device, dtype=torch.long),
                              torch.full((B, half), tau_ctx_idx, device=device, dtype=torch.long))
        # new half: d_min-only sampling (no-bootstrap winner recipe); noise mode forces tau=0.
        tau_new_idx, d_new_idx = _sample_tau_d(model, B, half, device, gen, n_d_unlocked=1)
        tau_new_idx = torch.where(modes.view(B, 1), torch.zeros_like(tau_new_idx), tau_new_idx)
        new_part = (1 - model._tau_value(tau_new_idx)[..., None, None]) * z0[:, half:] \
            + model._tau_value(tau_new_idx)[..., None, None] * z1_win[:, half:]

        memory_in = build_mem_in(old_mem, s)
        flow, new_mem, flow_norm = _newhalf_loss(
            model, old_part=old_part, tau_old=tau_old, new_part=new_part,
            tau_new_idx=tau_new_idx, d_new_idx=d_new_idx, z1_new=z1_win[:, half:],
            af_win=af(s, s + W), memory_in=memory_in,
            positions=torch.arange(s, s + W, device=device), half=half, bootstrap=False)

        total = total + flow
        sum_flow += float(flow.detach())
        sum_flow_norm += float(flow_norm.detach())
        n_terms += 1

        # carry the NEW half's write set (index 0 of the new half == abs pos s+half, % n == 0)
        old_mem = new_mem[:, 0:1]
        relay_depth += half
        s += half

    n_terms = max(1, n_terms)
    parts = {"flow": sum_flow / n_terms, "flow_norm": sum_flow_norm / n_terms,
             "n_slides": float(n_terms), "n_ctx": float(W)}
    return total / n_terms, parts
