"""Sparse write-slots dynamics model (design: tasks/drafts/sparse-memory-write-slots.md, v3).

Two changes over the base DynamicsModel, both memory-channel-only:

1. TEMPORAL MASK (SparseWSAttention, replaces the temporal blocks' attention): memory slots'
   queries attend ONLY to keys at absolute positions p with p % n_sparse == 0 (write slots),
   causally. Scratch frames' memory K/V are therefore never read by anyone — they are per-frame
   retrieval heads. All other slots keep the ordinary slot-wise causal mask.
   Key positions are RECONSTRUCTED (not stored): cached K/V columns are always consecutive
   positions ending right before the current query block (eviction is a tail slice), so
   pos_all = arange(end - T_all, end). This keeps the cache format byte-identical to base.

2. PHASE-INDEXED INIT (Merlin's (2, n_memory, E) design): `mem_init2[0]` = write-slot init,
   `mem_init2[1]` = scratch init. The forward wrapper ALWAYS constructs/overrides memory_in:
   scratch slots are FORCED to scratch-init (their injected value is semantically dead anyway —
   but their spatial mixing into the frame is not, so the input must be consistent); write slots
   use the provided memory_in (carried/written sets) or write-init if none. The base
   `memory_tokens` parameter is kept as an inert placeholder so rollout_step/_commit_context_frame
   run unchanged — the wrapper detects it by storage identity and treats it as "none provided".

Everything else (rollout_init/rollout_step/generate/eviction/recall eval) works unchanged.
Requires gqa_groups == 1 (the mask fork copies only the fused-qkv attention path).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.dynamics_model import Attention, DynamicsModel, DynamicsModelConfig


def sparse_write_mask(pos_q: torch.Tensor, pos_all: torch.Tensor, n_slots: int,
                      mem_start: int, mem_end: int, n_sparse: int) -> torch.Tensor:
    """(n_slots, T_q, T_all) bool mask, True = MASKED OUT.

    Non-memory slots: ordinary causal (key pos > query pos masked).
    Memory slots:     causal AND key must be a write slot (key pos % n_sparse == 0).
    Scratch queries do NOT see their own diagonal (their key pos % n != 0) — pure-fetch semantics.
    """
    causal = pos_all[None, :] > pos_q[:, None]                     # (T_q, T_all) True = masked
    not_write = (pos_all % n_sparse) != 0                          # (T_all,)
    mem_mask = causal | not_write[None, :]                         # (T_q, T_all)
    # Orphan fallback: a memory query with NO causally-visible write key (windows that start
    # mid-phase, e.g. phase-randomized training) attends its own diagonal instead of NaN-ing.
    orphan = mem_mask.all(dim=-1)                                  # (T_q,)
    if bool(orphan.any()):
        diag = pos_all[None, :] == pos_q[:, None]                  # (T_q, T_all)
        mem_mask = torch.where((orphan[:, None] & diag), torch.zeros_like(mem_mask), mem_mask)
    full = causal.unsqueeze(0).expand(n_slots, -1, -1).clone()
    full[mem_start:mem_end] = mem_mask
    return full


class SparseWSAttention(Attention):
    """Temporal attention with the sparse write-slot mask on the memory channel."""

    def __init__(self, config: DynamicsModelConfig, n_sparse: int, mem_start: int, mem_end: int):
        assert config.gqa_groups == 1, "sparse write-slots prototype requires gqa_groups=1"
        super().__init__(config, is_temporal=True)
        self.n_sparse = n_sparse
        self.mem_start, self.mem_end = mem_start, mem_end

    def forward(self, x: torch.Tensor, positions: torch.Tensor = None,
                layer_cache: dict = None, commit: bool = False):
        B, T, N, C = x.shape
        qkv = self.qkv(x).reshape((B, T, N, 3, self.n_heads, -1))
        qkv = qkv.permute((3, 4, 0, 2, 1, 5))            # (3, heads, B, N, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)

        cos, sin = self._rope_cos_sin(T, positions, q.dtype, x.device)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
        if layer_cache is not None and layer_cache.get('k') is not None:
            k_all = torch.concat((layer_cache['k'], k), dim=-2)
            v_all = torch.concat((layer_cache['v'], v), dim=-2)
        else:
            k_all, v_all = k, v
        if commit and layer_cache is not None:
            layer_cache['k'] = k_all
            layer_cache['v'] = v_all

        # Absolute positions: queries from `positions` (or 0..T-1); cached keys are always the
        # consecutive positions immediately BEFORE the query block (eviction = tail slice).
        pos_q = (positions if positions is not None
                 else torch.arange(T, device=x.device)).to(torch.long)
        T_all = k_all.shape[-2]
        end = int(pos_q.max().item()) + 1
        pos_all = torch.arange(end - T_all, end, device=x.device)
        mask = sparse_write_mask(pos_q, pos_all, N, self.mem_start, self.mem_end, self.n_sparse)

        scale = self.base_scale * torch.exp(self.logit_scale.clamp(max=self.max_logit_scale))
        attn_scores = (q @ k_all.transpose(-2, -1)) * scale       # (H, B, N, T, T_all)
        attn_scores = self.soft_cap_act(attn_scores / self.soft_cap) * self.soft_cap
        attn_scores = attn_scores.masked_fill(mask[None, None], float('-inf'))
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.att_droput(attn)
        x = attn @ v_all
        x = x.permute(1, 3, 2, 0, 4).reshape(B, T, N, C)
        return self.dropout(self.proj(x))


class DynamicsModelSparseWS(DynamicsModel):
    """DynamicsModel with sparse write-slots (n_sparse) + phase-indexed (2, n_memory, E) init."""

    SPARSE_N = 8

    def __init__(self, config: DynamicsModelConfig):
        assert config.n_memory > 0, "sparse write-slots needs n_memory > 0"
        assert config.max_temporal_length >= 2 * self.SPARSE_N, \
            "window must be >= 2*n so a write always sees the previous write"
        super().__init__(config)
        E = config.embedding_dim
        # Merlin's role encoding: one init for WRITE slots, one for SCRATCH slots.
        self.mem_init2 = nn.Parameter(0.05 * torch.rand((2, config.n_memory, E)))
        mem_start = config.n_action_tokens + config.n_latents + config.n_registers
        mem_end = mem_start + config.n_memory
        for i, block in enumerate(self.blocks):
            if i % 3 == 1:  # temporal blocks only
                block.att = SparseWSAttention(config, self.SPARSE_N, mem_start, mem_end)
        print(f"[SparseWS] n_sparse={self.SPARSE_N} mem_slots=[{mem_start},{mem_end}) "
              f"W={config.max_temporal_length} (>= 2n OK) init=(2,{config.n_memory},{E})")

    def _phase_memory_in(self, memory_in, B: int, T: int, positions, device):
        pos = (positions if positions is not None
               else torch.arange(T, device=device)).to(torch.long)
        is_write = (pos % self.SPARSE_N) == 0                          # (T,)
        base = self.mem_init2[(~is_write).long()]                      # (T, M, E)
        base = base.unsqueeze(0).expand(B, -1, -1, -1)
        placeholder = (memory_in is None
                       or memory_in.data_ptr() == self.memory_tokens.data_ptr())
        if placeholder:
            return base
        # provided memory (carried write sets / written commit mem): keep at WRITE slots only;
        # scratch slots are always forced to scratch-init.
        return torch.where(is_write.view(1, T, 1, 1), memory_in,
                           base.to(memory_in.dtype))

    def forward(self, z_tilde, tau_idx, d_idx, actions=None, memory_in=None,
                return_memory=False, positions=None, cache=None, commit=False):
        B, T = z_tilde.shape[:2]
        mem = self._phase_memory_in(memory_in, B, T, positions, z_tilde.device)
        return super().forward(z_tilde, tau_idx, d_idx, actions, memory_in=mem,
                               return_memory=return_memory, positions=positions,
                               cache=cache, commit=commit)
