"""Matched vanilla/memory autoregressive sampling for community Dreamer 4."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


def make_schedule(k_max: int, K: int):
    if K <= 0 or K > k_max or K & (K - 1) or k_max % K:
        raise ValueError(f"K must be a power of two dividing k_max={k_max}, got {K}")
    return {
        "K": K,
        "e": int(round(math.log2(K))),
        "dt": 1.0 / K,
        "tau": tuple(i / K for i in range(K)),
        "tau_idx": tuple(i * (k_max // K) for i in range(K)),
    }


@dataclass
class SampleResult:
    latent: torch.Tensor
    written_memory: Optional[torch.Tensor]


class CarryingSampler:
    """Sliding-window sampler with optional read-old/write-new memory.

    Denoising noise and commit-pass noise use separate generators.  Therefore a
    vanilla and memory player initialized with the same sample seed see exactly
    the same target-frame noise stream even though only the memory arm performs
    the dedicated fifth write pass.
    """

    def __init__(
        self,
        dynamics,
        *,
        k_max: int,
        K: int = 4,
        context_window: int = 31,
        sample_generator: Optional[torch.Generator] = None,
        commit_generator: Optional[torch.Generator] = None,
    ):
        self.dynamics = dynamics
        self.k_max = int(k_max)
        self.schedule = make_schedule(self.k_max, K)
        self.context_window = int(context_window)
        if self.context_window < 1:
            raise ValueError("context_window must be positive")
        self.sample_generator = sample_generator
        self.commit_generator = commit_generator
        self.latents: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.action_masks: list[torch.Tensor] = []
        self.memories: list[torch.Tensor] = []

    @property
    def has_memory(self):
        return int(getattr(self.dynamics, "n_memory", 0)) > 0

    def _near_clean_commit(
        self,
        packed: torch.Tensor,
        actions: torch.Tensor,
        masks: torch.Tensor,
        memory_in: torch.Tensor,
    ) -> torch.Tensor:
        B, T = packed.shape[:2]
        idx = torch.full((B, T), self.k_max - 1, device=packed.device, dtype=torch.long)
        tau = float(self.k_max - 1) / float(self.k_max)
        noise = torch.randn(
            packed.shape,
            device=packed.device,
            dtype=packed.dtype,
            generator=self.commit_generator,
        )
        commit_input = (1.0 - tau) * noise + tau * packed
        step = torch.full(
            (B, T), int(round(math.log2(self.k_max))),
            device=packed.device, dtype=torch.long,
        )
        _, _, written = self.dynamics(
            actions, step, idx, commit_input, act_mask=masks,
            agent_tokens=None, memory_in=memory_in, return_memory=True,
        )
        if written is None:
            raise RuntimeError("memory model did not return committed memory")
        return written

    @torch.no_grad()
    def initialize(
        self,
        packed_context: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
    ):
        if packed_context.dim() != 4 or packed_context.shape[0] != 1:
            raise ValueError("player sampler expects packed context shape (1,T,S,D)")
        if actions.shape[:2] != packed_context.shape[:2] or action_masks.shape != actions.shape:
            raise ValueError("context action/mask alignment mismatch")
        T = packed_context.shape[1]
        self.latents = [value.detach() for value in packed_context[0]]
        self.actions = [value.detach() for value in actions[0]]
        self.action_masks = [value.detach() for value in action_masks[0]]
        self.memories = []
        if self.has_memory:
            blank = self.dynamics.blank_memory(
                1, T, device=packed_context.device, dtype=packed_context.dtype
            )
            written = self._near_clean_commit(packed_context, actions, action_masks, blank)
            self.memories = [value.detach() for value in written[0]]

    @torch.no_grad()
    def sample_next(
        self,
        action: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        memory_override: Optional[torch.Tensor] = None,
        commit: bool = True,
    ) -> SampleResult:
        if not self.latents:
            raise RuntimeError("initialize must be called before sample_next")
        if action.shape != self.actions[0].shape or action_mask.shape != action.shape:
            raise ValueError("action and action_mask must match the initialized action width")

        end = len(self.latents)
        start = max(0, end - self.context_window)
        past = torch.stack(self.latents[start:end], dim=0).unsqueeze(0)
        past_actions = torch.stack(self.actions[start:end], dim=0)
        past_masks = torch.stack(self.action_masks[start:end], dim=0)
        actions = torch.cat([past_actions, action[None]], dim=0).unsqueeze(0)
        masks = torch.cat([past_masks, action_mask[None]], dim=0).unsqueeze(0)
        B, t = 1, past.shape[1]

        z = torch.randn(
            (B, 1, past.shape[2], past.shape[3]),
            device=past.device,
            dtype=past.dtype,
            generator=self.sample_generator,
        )
        emax = int(round(math.log2(self.k_max)))
        step_idx = torch.full((B, t + 1), emax, device=past.device, dtype=torch.long)
        step_idx[:, -1] = int(self.schedule["e"])
        signal_idx = torch.full(
            (B, t + 1), self.k_max - 1, device=past.device, dtype=torch.long
        )

        memory_in = None
        if self.has_memory:
            if memory_override is None:
                old_memory = torch.stack(self.memories[start:end], dim=0).unsqueeze(0)
            else:
                expected = (1, t, self.dynamics.n_memory, self.dynamics.d_model)
                if tuple(memory_override.shape) != expected:
                    raise ValueError(
                        f"memory_override must have shape {expected}, got {tuple(memory_override.shape)}"
                    )
                old_memory = memory_override
            blank = self.dynamics.blank_memory(
                1, 1, device=past.device, dtype=past.dtype
            )
            memory_in = torch.cat([old_memory, blank], dim=1)

        for i in range(int(self.schedule["K"])):
            tau = float(self.schedule["tau"][i])
            signal_idx[:, -1] = int(self.schedule["tau_idx"][i])
            sequence = torch.cat([past, z], dim=1)
            clean, _ = self.dynamics(
                actions, step_idx, signal_idx, sequence, act_mask=masks,
                agent_tokens=None, memory_in=memory_in,
            )
            velocity = (clean[:, -1:].float() - z.float()) / max(1e-4, 1.0 - tau)
            z = (z.float() + velocity * float(self.schedule["dt"])).to(past.dtype)

        written_target = None
        if self.has_memory and commit:
            committed = self._near_clean_commit(
                torch.cat([past, z], dim=1), actions, masks, memory_in
            )
            written_target = committed[0, -1].detach()

        latent = z[0, 0].detach()
        if commit:
            self.latents.append(latent)
            self.actions.append(action.detach())
            self.action_masks.append(action_mask.detach())
            if self.has_memory:
                self.memories.append(written_target)
        return SampleResult(latent=latent, written_memory=written_target)
