"""Grouped-Query Attention (GQA) dynamics model — experiment (NOT spec-backed).

Goal: cut the carrying rollout's KV-cache footprint 4x by sharing each K/V head across a group of
4 query heads (16 query heads -> 4 KV heads at the GridWorld config). The cache dicts keep the
same {'k','v'} layout with a kv-head-leading shape, so rollout_init / rollout_step / window
eviction (pure time-axis slices) work unchanged.

Faithful to the base Attention in every other respect and in exact order: QK-RMSNorm -> RoPE
(temporal; fixed table or absolute `positions`) -> cache prepend/commit -> learnable per-head
logit scale (still one per QUERY head) -> tanh soft-cap -> block-causal mask -> softmax ->
dropout. The grouped matmul broadcasts K/V over the group axis — repeated K/V is never
materialized, so peak activation memory also shrinks.

Objective: includes the tau0-anchor (p=0.5 forced (tau_idx=0, d=d_min) — the honest-baseline
objective from experiments/vanilla-honest-baseline/, copied inline so this experiment stands
alone). Comparison target is checkpoints/gridworld/dynamics_vanilla_tau0.pt: identical objective
and config, full-MHA attention — GQA is the single varying factor.

Usage (train_dynamics.py):
    --model-module experiments/gqa-dynamics/model.py:DynamicsModelGQA
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.dynamics_model import Attention, DynamicsModel, DynamicsModelConfig


class GQAAttention(Attention):
    """Attention with n_kv_heads = n_heads/groups shared K/V heads (GQA)."""

    def __init__(self, config: DynamicsModelConfig, is_temporal, groups: int = 4):
        super().__init__(config, is_temporal)  # builds norms, scales, RoPE tables, proj, dropouts
        assert self.n_heads % groups == 0, f"n_heads {self.n_heads} not divisible by groups {groups}"
        self.groups = groups
        self.n_kv_heads = self.n_heads // groups
        E = config.embedding_dim
        # Replace the fused qkv with separate q and (smaller) kv projections.
        del self.qkv
        self.q_proj = nn.Linear(E, E)
        self.kv_proj = nn.Linear(E, 2 * self.n_kv_heads * self.head_dim)

    def forward(self, x: torch.Tensor, positions: torch.Tensor = None,
                layer_cache: dict = None, commit: bool = False):
        B, T, N, C = x.shape
        H, KV, G, hd = self.n_heads, self.n_kv_heads, self.groups, self.head_dim

        q = self.q_proj(x).reshape(B, T, N, H, hd)
        kv = self.kv_proj(x).reshape(B, T, N, 2, KV, hd)
        if not self.is_temporal:
            q = q.permute(3, 0, 1, 2, 4)          # (H, B, T, N, hd)
            kv = kv.permute(3, 4, 0, 1, 2, 5)     # (2, KV, B, T, N, hd)
        else:
            q = q.permute(3, 0, 2, 1, 4)          # (H, B, N, T, hd)
            kv = kv.permute(3, 4, 0, 2, 1, 5)     # (2, KV, B, N, T, hd)
        k, v = kv[0], kv[1]
        q, k = self.q_norm(q), self.k_norm(k)

        if not self.is_temporal:
            mask = None
            k_all, v_all = k, v
        else:
            cos, sin = self._rope_cos_sin(T, positions, q.dtype, x.device)
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)
            if layer_cache is not None and layer_cache.get('k') is not None:
                k_all = torch.concat((layer_cache['k'], k), dim=-2)
                v_all = torch.concat((layer_cache['v'], v), dim=-2)
            else:
                k_all, v_all = k, v
            if commit and layer_cache is not None:
                layer_cache['k'] = k_all               # (KV, B, N, T_all, hd) — 1/groups the bytes
                layer_cache['v'] = v_all
            T_all = k_all.shape[-2]
            mask = torch.zeros((T, T_all), dtype=torch.bool, device=x.device)
            mask[:, T_all - T:] = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        # Grouped attention: q (KV, G, B, *, Tq, hd) against shared k/v (KV, 1, B, *, Tk, hd).
        q = q.reshape(KV, G, *q.shape[1:])
        scale = self.base_scale * torch.exp(self.logit_scale.clamp(max=self.max_logit_scale))
        scale = scale.reshape(KV, G, 1, 1, 1, 1)      # per-QUERY-head temperature, grouped view
        attn_scores = (q @ k_all.unsqueeze(1).transpose(-2, -1)) * scale
        attn_scores = self.soft_cap_act(attn_scores / self.soft_cap) * self.soft_cap
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.att_droput(attn)

        x = attn @ v_all.unsqueeze(1)                 # (KV, G, B, *, Tq, hd)
        x = x.reshape(H, *x.shape[2:])                # (H, B, *, Tq, hd)
        if not self.is_temporal:
            x = x.permute(1, 2, 3, 0, 4).reshape(B, T, N, C)
        else:
            x = x.permute(1, 3, 2, 0, 4).reshape(B, T, N, C)
        return self.dropout(self.proj(x))


class DynamicsModelGQA(DynamicsModel):
    """DynamicsModel with GQA attention (groups=4) + the tau0-anchor training objective."""

    GQA_GROUPS = 4
    P_ANCHOR = 0.5   # tau0-anchor (honest-baseline objective; see vanilla-honest-baseline)

    def __init__(self, config: DynamicsModelConfig):
        super().__init__(config)
        for i, block in enumerate(self.blocks):
            block.att = GQAAttention(config, is_temporal=(i % 3 == 1), groups=self.GQA_GROUPS)
        n_kv = config.n_heads // self.GQA_GROUPS
        print(f"[GQA] groups={self.GQA_GROUPS}: {config.n_heads} query heads -> {n_kv} kv heads "
              f"(KV cache 1/{self.GQA_GROUPS}); P_ANCHOR={self.P_ANCHOR}")

    def sample_tau_d(self, B: int, T: int, device):
        tau_idx, d_idx = super().sample_tau_d(B, T, device)
        if not self.training:               # val: default distribution (comparable val/loss)
            return tau_idx, d_idx
        anchor = torch.rand((B, T), device=device) < self.P_ANCHOR
        tau_idx = torch.where(anchor, torch.zeros_like(tau_idx), tau_idx)
        d_idx = torch.where(anchor, torch.full_like(d_idx, self.n_d - 1), d_idx)
        return tau_idx, d_idx
