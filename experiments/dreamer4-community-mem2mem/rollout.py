"""Rollout-only memory training for the pinned community Dreamer 4 model.

This module intentionally mirrors the community shortcut objective instead of
importing the in-repository dynamics loss.  A long clip is processed as
overlapping W-frame windows.  The first window only constructs grounded memory;
later windows score their new half exactly once and carry a dedicated commit
pass's written memory into the next window.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch


@dataclass
class RolloutResult:
    loss: Optional[torch.Tensor]
    mean_loss: float
    flow_mse: float
    bootstrap_mse: float
    n_slides: int
    n_segments: int
    scored_ranges: tuple[tuple[int, int], ...]
    memory_only_fraction: float
    initial_memory: torch.Tensor
    final_memory: torch.Tensor


def _emax(k_max: int) -> int:
    value = int(round(math.log2(k_max)))
    if (1 << value) != k_max:
        raise ValueError("k_max must be a power of two")
    return value


def _rand(shape, *, device, generator, dtype=torch.float32):
    return torch.rand(shape, device=device, generator=generator, dtype=dtype)


def _randn_like(x: torch.Tensor, *, generator) -> torch.Tensor:
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)


def _sample_new_half_conditioning(
    *, B: int, half: int, B_self: int, k_max: int, device, generator
):
    """Match the community empirical/self row split and shortcut grids."""
    B_emp = B - B_self
    emax = _emax(k_max)
    step_emp = torch.full((B_emp, half), emax, device=device, dtype=torch.long)
    if B_self:
        step_self = torch.randint(
            0, max(1, emax), (B_self, half), device=device, generator=generator
        )
        step = torch.cat([step_emp, step_self], dim=0)
    else:
        step_self = torch.zeros((0, half), device=device, dtype=torch.long)
        step = step_emp

    K = (1 << step).to(torch.long)
    j = torch.floor(_rand((B, half), device=device, generator=generator) * K).long()
    sigma = j.float() / K.float()
    scale = torch.div(torch.tensor(k_max, device=device), K, rounding_mode="floor")
    sigma_idx = j * scale
    return step, step_self, sigma, sigma_idx


def _row_slice(value, start: int):
    if value is None:
        return None
    if value.dim() == 1:
        return value
    return value[start:]


def community_newhalf_loss(
    dynamics: torch.nn.Module,
    *,
    z1_window: torch.Tensor,
    actions: Optional[torch.Tensor],
    act_mask: Optional[torch.Tensor],
    memory_in: torch.Tensor,
    memory_only: torch.Tensor,
    k_max: int,
    B_self: int,
    step: int,
    bootstrap_start: int,
    generator: Optional[torch.Generator],
):
    """Community flow/bootstrap loss, restricted to the window's new half.

    Old-half inputs are fixed across the main and both bootstrap forwards.
    Latent-present sequences use the community near-clean grid point; memory-only
    sequences use pure noise at tau=0 for both halves.  Targets and loss weights
    remain the community objective.
    """
    device = z1_window.device
    B, W = z1_window.shape[:2]
    if W % 2:
        raise ValueError("window length must be even")
    half = W // 2
    if not 0 <= B_self < B:
        raise ValueError(f"B_self must be in [0, B), got {B_self} for B={B}")
    if memory_only.shape != (B,):
        raise ValueError(f"memory_only must have shape {(B,)}, got {tuple(memory_only.shape)}")

    B_emp = B - B_self
    emax = _emax(k_max)
    step_new, step_self, sigma_new, sigma_idx_new = _sample_new_half_conditioning(
        B=B, half=half, B_self=B_self, k_max=k_max, device=device, generator=generator
    )

    # The load-bearing arm is deliberately evaluated from tau=0.  d still follows
    # the vanilla empirical/self split, so the shortcut objective itself is intact.
    mode_2d = memory_only[:, None]
    sigma_new = torch.where(mode_2d, torch.zeros_like(sigma_new), sigma_new)
    sigma_idx_new = torch.where(mode_2d, torch.zeros_like(sigma_idx_new), sigma_idx_new)

    context_idx = torch.where(
        mode_2d,
        torch.zeros((B, half), device=device, dtype=torch.long),
        torch.full((B, half), k_max - 1, device=device, dtype=torch.long),
    )
    context_sigma = context_idx.float() / float(k_max)
    step_context = torch.full((B, half), emax, device=device, dtype=torch.long)

    noise = _randn_like(z1_window, generator=generator)
    old_input = (
        (1.0 - context_sigma)[..., None, None] * noise[:, :half]
        + context_sigma[..., None, None] * z1_window[:, :half]
    )
    new_input = (
        (1.0 - sigma_new)[..., None, None] * noise[:, half:]
        + sigma_new[..., None, None] * z1_window[:, half:]
    )
    packed_input = torch.cat([old_input, new_input], dim=1)
    step_full = torch.cat([step_context, step_new], dim=1)
    signal_full = torch.cat([context_idx, sigma_idx_new], dim=1)

    pred, _ = dynamics(
        actions,
        step_full,
        signal_full,
        packed_input,
        act_mask=act_mask,
        agent_tokens=None,
        memory_in=memory_in,
    )
    pred_new = pred[:, half:]
    target_new = z1_window[:, half:]

    sigma_emp = sigma_new[:B_emp]
    flow_per = (pred_new[:B_emp].float() - target_new[:B_emp].float()).pow(2).mean((2, 3))
    loss_emp = (flow_per * (0.9 * sigma_emp + 0.1)).mean()

    zero = torch.zeros((), device=device, dtype=torch.float32)
    loss_self = zero
    boot_mse = zero
    if B_self and step >= bootstrap_start:
        sigma_self = sigma_new[B_emp:]
        sigma_idx_self = sigma_idx_new[B_emp:]
        d_self = 1.0 / (1 << step_self).float()
        d_half = d_self / 2.0
        step_half_new = step_self + 1
        sigma_plus = sigma_self + d_half
        sigma_idx_plus = sigma_idx_self + (k_max * d_half).long()

        step_half_full = torch.cat([step_context[B_emp:], step_half_new], dim=1)
        signal_half1 = torch.cat([context_idx[B_emp:], sigma_idx_self], dim=1)
        memory_self = memory_in[B_emp:]
        actions_self = _row_slice(actions, B_emp)
        mask_self = _row_slice(act_mask, B_emp)

        half1, _ = dynamics(
            actions_self,
            step_half_full,
            signal_half1,
            packed_input[B_emp:],
            act_mask=mask_self,
            agent_tokens=None,
            memory_in=memory_self,
        )
        half1 = half1[:, half:]
        self_input = new_input[B_emp:]
        b_prime = (half1.float() - self_input.float()) / (
            1.0 - sigma_self
        ).clamp_min(1e-6)[..., None, None]
        z_prime = self_input.float() + b_prime * d_half[..., None, None]

        input2 = torch.cat([old_input[B_emp:], z_prime.to(self_input.dtype)], dim=1)
        signal_half2 = torch.cat([context_idx[B_emp:], sigma_idx_plus], dim=1)
        half2, _ = dynamics(
            actions_self,
            step_half_full,
            signal_half2,
            input2,
            act_mask=mask_self,
            agent_tokens=None,
            memory_in=memory_self,
        )
        half2 = half2[:, half:]
        b_doubleprime = (half2.float() - z_prime.float()) / (
            1.0 - sigma_plus
        ).clamp_min(1e-6)[..., None, None]
        vhat = (pred_new[B_emp:].float() - self_input.float()) / (
            1.0 - sigma_self
        ).clamp_min(1e-6)[..., None, None]
        target_v = ((b_prime + b_doubleprime) / 2.0).detach()
        boot_per = (1.0 - sigma_self).pow(2) * (
            vhat - target_v
        ).pow(2).mean((2, 3))
        loss_self = (boot_per * (0.9 * sigma_self + 0.1)).mean()
        boot_mse = boot_per.mean()

    loss = (loss_emp * B_emp + loss_self * B_self) / B
    return loss, {
        "flow_mse": flow_per.mean().detach(),
        "bootstrap_mse": boot_mse.detach(),
        "loss_emp": loss_emp.detach(),
        "loss_self": loss_self.detach(),
        "sigma_mean": sigma_new.mean().detach(),
    }


def commit_window_memory(
    dynamics: torch.nn.Module,
    *,
    z1_window: torch.Tensor,
    actions: Optional[torch.Tensor],
    act_mask: Optional[torch.Tensor],
    memory_in: torch.Tensor,
    k_max: int,
    generator: Optional[torch.Generator],
    memory_only: Optional[torch.Tensor] = None,
    initialization: bool = False,
) -> torch.Tensor:
    """Dedicated near-clean write pass; never reuse a denoising intermediate.

    For load-bearing sequences after initialization, the old-half scene is
    ablated to tau=0 while the new half is teacher-forced near-clean.  This is
    causal teacher forcing for the state that inference will have generated and
    committed before the following window.
    """
    B, W = z1_window.shape[:2]
    half = W // 2
    device = z1_window.device
    emax = _emax(k_max)
    if initialization:
        context_idx = torch.full((B, W), k_max - 1, device=device, dtype=torch.long)
    else:
        if memory_only is None or memory_only.shape != (B,):
            raise ValueError("non-initial commit requires one memory_only flag per sequence")
        old_idx = torch.where(
            memory_only[:, None],
            torch.zeros((B, half), device=device, dtype=torch.long),
            torch.full((B, half), k_max - 1, device=device, dtype=torch.long),
        )
        new_idx = torch.full((B, half), k_max - 1, device=device, dtype=torch.long)
        context_idx = torch.cat([old_idx, new_idx], dim=1)

    tau = context_idx.float() / float(k_max)
    noise = _randn_like(z1_window, generator=generator)
    commit_input = (1.0 - tau)[..., None, None] * noise + tau[..., None, None] * z1_window
    step_idx = torch.full((B, W), emax, device=device, dtype=torch.long)
    _, _, written = dynamics(
        actions,
        step_idx,
        context_idx,
        commit_input,
        act_mask=act_mask,
        agent_tokens=None,
        memory_in=memory_in,
        return_memory=True,
    )
    if written is None:
        raise RuntimeError("memory-enabled forward did not return written memory")
    return written[:, half:]


def mem2mem_rollout(
    dynamics: torch.nn.Module,
    z1: Union[torch.Tensor, Callable[[int, int], torch.Tensor]],
    actions: Optional[torch.Tensor],
    act_mask: Optional[torch.Tensor],
    *,
    window: int = 32,
    clip_length: int = 128,
    tbptt_frames: int = 64,
    k_max: int = 8,
    B_self: int,
    step: int,
    bootstrap_start: int = 5_000,
    generator: Optional[torch.Generator] = None,
    force_mode: Optional[str] = None,
    backward_fn: Optional[Callable[[torch.Tensor], None]] = None,
    detach_boundaries: bool = True,
    detach_before_slide: Optional[int] = None,
) -> RolloutResult:
    """Run one long clip, optionally backpropagating at bounded TBPTT boundaries.

    Production passes ``backward_fn`` and performs one optimizer step after all
    segments.  Each segment loss is scaled by the total number of scored slides,
    so accumulated parameter gradients equal the correctly normalized truncated
    clip objective while completed graphs can be released immediately.
    """
    if getattr(dynamics, "n_memory", 0) <= 0:
        raise ValueError("mem2mem_rollout requires n_memory > 0")
    if window % 2 or clip_length < 2 * window:
        raise ValueError("expected an even window and a long clip of at least 2W")
    stride = window // 2
    if tbptt_frames <= 0 or tbptt_frames % stride:
        raise ValueError("tbptt_frames must be a positive multiple of W/2")
    if callable(z1):
        get_window = z1
        first_window = get_window(0, window)
    else:
        if clip_length != z1.shape[1]:
            raise ValueError("tensor z1 must contain exactly clip_length frames")
        get_window = lambda a, b: z1[:, a:b]
        first_window = get_window(0, window)
    B = first_window.shape[0]
    device = first_window.device

    if force_mode == "memory":
        memory_only = torch.ones(B, device=device, dtype=torch.bool)
    elif force_mode == "latent":
        memory_only = torch.zeros(B, device=device, dtype=torch.bool)
    elif force_mode is None:
        memory_only = _rand((B,), device=device, generator=generator) < 0.5
    else:
        raise ValueError("force_mode must be None, 'latent', or 'memory'")

    def slice_time(value, a: int, b: int):
        if value is None or value.dim() == 1:
            return value
        return value[:, a:b]

    blank_W = dynamics.blank_memory(
        B, window, device=device, dtype=first_window.dtype
    )
    initial_memory = commit_window_memory(
        dynamics,
        z1_window=first_window,
        actions=slice_time(actions, 0, window),
        act_mask=slice_time(act_mask, 0, window),
        memory_in=blank_W,
        k_max=k_max,
        generator=generator,
        initialization=True,
    )
    old_memory = initial_memory

    starts = tuple(range(stride, clip_length - window + 1, stride))
    if not starts:
        raise ValueError("clip has no scored rollout windows")
    n_slides = len(starts)
    slides_per_segment = tbptt_frames // stride
    scored_ranges = []
    flow_total = boot_total = loss_total = 0.0
    graph_total = None
    segment_loss = None
    n_segments = 0

    for slide_idx, start in enumerate(starts):
        if detach_before_slide == slide_idx:
            old_memory = old_memory.detach()
        blank_half = dynamics.blank_memory(
            B, stride, device=device, dtype=first_window.dtype
        )
        memory_in = torch.cat([old_memory, blank_half], dim=1)
        end = start + window
        z1_window = get_window(start, end)
        if z1_window.shape != first_window.shape:
            raise ValueError(
                f"window source returned {tuple(z1_window.shape)}, expected {tuple(first_window.shape)}"
            )
        slide_loss, aux = community_newhalf_loss(
            dynamics,
            z1_window=z1_window,
            actions=slice_time(actions, start, end),
            act_mask=slice_time(act_mask, start, end),
            memory_in=memory_in,
            memory_only=memory_only,
            k_max=k_max,
            B_self=B_self,
            step=step,
            bootstrap_start=bootstrap_start,
            generator=generator,
        )
        new_memory = commit_window_memory(
            dynamics,
            z1_window=z1_window,
            actions=slice_time(actions, start, end),
            act_mask=slice_time(act_mask, start, end),
            memory_in=memory_in,
            k_max=k_max,
            generator=generator,
            memory_only=memory_only,
        )
        old_memory = new_memory

        normalized = slide_loss / n_slides
        segment_loss = normalized if segment_loss is None else segment_loss + normalized
        loss_total += float(slide_loss.detach())
        flow_total += float(aux["flow_mse"])
        boot_total += float(aux["bootstrap_mse"])
        scored_ranges.append((start + stride, end))

        boundary = ((slide_idx + 1) % slides_per_segment == 0) or (slide_idx + 1 == n_slides)
        if boundary:
            n_segments += 1
            if backward_fn is None:
                graph_total = segment_loss if graph_total is None else graph_total + segment_loss
            else:
                backward_fn(segment_loss)
            segment_loss = None
            if detach_boundaries and slide_idx + 1 < n_slides:
                old_memory = old_memory.detach()

    return RolloutResult(
        loss=graph_total,
        mean_loss=loss_total / n_slides,
        flow_mse=flow_total / n_slides,
        bootstrap_mse=boot_total / n_slides,
        n_slides=n_slides,
        n_segments=n_segments,
        scored_ranges=tuple(scored_ranges),
        memory_only_fraction=float(memory_only.float().mean()),
        initial_memory=initial_memory if backward_fn is None else initial_memory.detach(),
        final_memory=old_memory.detach(),
    )
