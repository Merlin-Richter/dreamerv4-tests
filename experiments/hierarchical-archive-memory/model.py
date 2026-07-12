"""Hierarchical fast-memory + sparse segment archive dynamics model.

Experiment-local implementation of
``tasks/in-progress/hierarchical-archive-memory.md``.  The source-backed
``models.dynamics_model`` remains unchanged.

The ordinary per-frame memory tokens are the fast carrier.  Every N committed
written-memory sets are compressed, slot-wise, into R archive embeddings per
memory slot.  Each temporal block owns a grouped archive reader: memory slot m
can read only archive group m.  Inference stores pre-rotated per-layer archive
K/V; training supplies raw archive embeddings so every TBPTT block gets a fresh
projection graph.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from models.dynamics_model import DynamicsModel, DynamicsModelConfig


@dataclass
class ArchiveDynamicsConfig(DynamicsModelConfig):
    archive_interval: int = 16
    archive_per_memory: int = 1
    archive_compressor_depth: int = 1
    archive_compressor_mlp_ratio: float = 2.0
    archive_max_sets: int = 0             # 0 = retain every set
    archive_gate_init: float = 1e-3


def _apply_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply split-half RoPE; cos/sin must already broadcast over ``t``."""
    d = t.shape[-1] // 2
    first, second = t[..., :d], t[..., d:]
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


class QuerySwiGLU(nn.Module):
    """SwiGLU applied only to archive query tokens; compressor dropout is deliberately zero."""

    def __init__(self, dim: int, ratio: float):
        super().__init__()
        hidden = max(1, int(dim * ratio))
        self.up = nn.Linear(dim, hidden)
        self.gate = nn.Linear(dim, hidden)
        self.down = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.up(x) * torch.nn.functional.silu(self.gate(x)))


class ArchiveCompressorBlock(nn.Module):
    """Restricted slot-wise cross-attention: R queries read N immutable source memories."""

    def __init__(self, config: ArchiveDynamicsConfig):
        super().__init__()
        E, H = config.embedding_dim, config.n_heads
        assert E % H == 0
        self.n_heads = H
        self.head_dim = E // H
        self.q_norm_in = nn.RMSNorm(E)
        self.src_norm = nn.RMSNorm(E)
        self.q_proj = nn.Linear(E, E)
        self.kv_proj = nn.Linear(E, 2 * E)
        self.q_head_norm = nn.RMSNorm(self.head_dim)
        self.k_head_norm = nn.RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(E, E)
        self.mlp_norm = nn.RMSNorm(E)
        self.mlp = QuerySwiGLU(E, config.archive_compressor_mlp_ratio)
        self.logit_scale = nn.Parameter(torch.full((H, 1, 1), math.log(4.0)))
        self.max_logit_scale = math.log(100.0)
        self.base_scale = self.head_dim ** -0.5
        self.soft_cap = config.att_logit_soft_cap

        d_half = self.head_dim // 2
        freqs = 10_000 ** (-2 * torch.arange(d_half, dtype=torch.float32) / self.head_dim)
        self.register_buffer("rope_freqs", freqs, persistent=False)

    def forward(self, q: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        # q: (B, M, R, E); source: (B, M, N, E).  M stays a batch/group axis.
        B, M, R, E = q.shape
        N = source.shape[2]
        qh = self.q_proj(self.q_norm_in(q)).reshape(B, M, R, self.n_heads, self.head_dim)
        kv = self.kv_proj(self.src_norm(source)).reshape(
            B, M, N, 2, self.n_heads, self.head_dim)
        qh = qh.permute(0, 1, 3, 2, 4)                 # (B,M,H,R,D)
        kh = kv[:, :, :, 0].permute(0, 1, 3, 2, 4)    # (B,M,H,N,D)
        vh = kv[:, :, :, 1].permute(0, 1, 3, 2, 4)
        qh, kh = self.q_head_norm(qh), self.k_head_norm(kh)

        # Segment-local RoPE: source positions 0..N-1; all queries are at position N.
        src_pos = torch.arange(N, device=q.device, dtype=torch.float32)
        q_pos = torch.tensor([N], device=q.device, dtype=torch.float32)
        freq = self.rope_freqs.to(q.device)
        src_ang = torch.outer(src_pos, freq)
        q_ang = torch.outer(q_pos, freq)
        src_cos = torch.cos(src_ang).to(kh.dtype).view(1, 1, 1, N, -1)
        src_sin = torch.sin(src_ang).to(kh.dtype).view(1, 1, 1, N, -1)
        q_cos = torch.cos(q_ang).to(qh.dtype).view(1, 1, 1, 1, -1)
        q_sin = torch.sin(q_ang).to(qh.dtype).view(1, 1, 1, 1, -1)
        qh = _apply_rope(qh, q_cos, q_sin)
        kh = _apply_rope(kh, src_cos, src_sin)

        scale = self.base_scale * torch.exp(self.logit_scale.clamp(max=self.max_logit_scale))
        score = (qh @ kh.transpose(-2, -1)) * scale.view(1, 1, self.n_heads, 1, 1)
        score = torch.tanh(score / self.soft_cap) * self.soft_cap
        attn = torch.softmax(score, dim=-1)
        out = attn @ vh
        out = out.permute(0, 1, 3, 2, 4).reshape(B, M, R, E)
        q = q + self.out_proj(out)
        return q + self.mlp(self.mlp_norm(q))


class ArchiveCompressor(nn.Module):
    """Tiny shared compressor from (B,N,M,E) written memories to (B,M,R,E)."""

    def __init__(self, config: ArchiveDynamicsConfig):
        super().__init__()
        assert config.n_memory > 0
        assert config.archive_per_memory > 0
        E, M, R = config.embedding_dim, config.n_memory, config.archive_per_memory
        self.n_memory, self.n_archive = M, R
        self.memory_slot_embedding = nn.Parameter(0.05 * torch.rand(M, E))
        self.archive_queries = nn.Parameter(0.05 * torch.rand(R, E))
        self.blocks = nn.ModuleList(
            [ArchiveCompressorBlock(config) for _ in range(config.archive_compressor_depth)])

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        B, N, M, E = source.shape
        assert M == self.n_memory, f"source has M={M}, expected {self.n_memory}"
        # The caller owns stop-gradient policy.  Do not detach here: inference and VJP recomputation
        # use the same pure compressor, while training detaches source before invoking it.
        src = source.permute(0, 2, 1, 3)  # (B,M,N,E)
        q = self.memory_slot_embedding[:, None, :] + self.archive_queries[None, :, :]
        q = q.unsqueeze(0).expand(B, -1, -1, -1)
        for block in self.blocks:
            q = block(q, src)
        return q


class GroupedArchiveAttention(nn.Module):
    """Long-memory attention with one independent archive key channel per fast-memory slot."""

    def __init__(self, config: ArchiveDynamicsConfig):
        super().__init__()
        E, H = config.embedding_dim, config.n_heads
        assert E % H == 0
        assert H % config.gqa_groups == 0
        self.n_heads = H
        self.gqa_groups = config.gqa_groups
        self.n_kv_heads = H // config.gqa_groups
        self.head_dim = E // H
        self.n_memory = config.n_memory
        self.archive_per_memory = config.archive_per_memory
        self.min_archive_age = config.max_temporal_length - config.archive_interval + 1
        assert self.min_archive_age >= 1

        self.q_proj = nn.Linear(E, E)
        self.kv_proj = nn.Linear(E, 2 * self.n_kv_heads * self.head_dim)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(E, E)
        self.logit_scale = nn.Parameter(torch.full((H,), math.log(4.0)))
        self.max_logit_scale = math.log(100.0)
        self.base_scale = self.head_dim ** -0.5
        self.soft_cap = config.att_logit_soft_cap
        self.attn_dropout = nn.Dropout(config.att_drop_rate)
        self.out_dropout = nn.Dropout(config.drop_rate)

        d_half = self.head_dim // 2
        freqs = 10_000 ** (-2 * torch.arange(d_half, dtype=torch.float32) / self.head_dim)
        self.register_buffer("rope_freqs", freqs, persistent=False)

    def _cos_sin(self, positions: torch.Tensor, dtype: torch.dtype, device):
        ang = torch.outer(positions.to(device=device, dtype=torch.float32), self.rope_freqs.to(device))
        return torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)

    def project_archive(self, archive: torch.Tensor,
                        archive_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project raw archive sets to pre-rotated per-layer K/V.

        archive: (B,J,M,R,E); returns (kv_heads,B,M,J,R,head_dim).
        """
        B, J, M, R, _ = archive.shape
        assert M == self.n_memory and R == self.archive_per_memory
        kv = self.kv_proj(archive).reshape(
            B, J, M, R, 2, self.n_kv_heads, self.head_dim)
        k = kv[..., 0, :, :].permute(4, 0, 2, 1, 3, 5)
        v = kv[..., 1, :, :].permute(4, 0, 2, 1, 3, 5)
        k = self.k_norm(k)
        cos, sin = self._cos_sin(archive_positions, k.dtype, archive.device)
        cos = cos.view(1, 1, 1, J, 1, -1)
        sin = sin.view(1, 1, 1, J, 1, -1)
        return _apply_rope(k, cos, sin), v

    def forward(self, memory: torch.Tensor, query_positions: torch.Tensor, *,
                archive: torch.Tensor | None = None,
                archive_positions: torch.Tensor | None = None,
                archive_cache: dict | None = None,
                batch_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return the grouped archive-attention residual for (B,T,M,E) fast memory."""
        B, T, M, E = memory.shape
        assert M == self.n_memory
        if archive_positions is None or archive_positions.numel() == 0:
            return torch.zeros_like(memory)

        if archive_cache is not None and archive_cache.get("k") is not None:
            k, v = archive_cache["k"], archive_cache["v"]
            J, R = k.shape[3], k.shape[4]
        else:
            assert archive is not None, "raw archive or projected archive_cache required"
            k, v = self.project_archive(archive, archive_positions)
            J, R = archive.shape[1], archive.shape[3]

        q = self.q_proj(memory).reshape(B, T, M, self.n_heads, self.head_dim)
        q = q.permute(3, 0, 2, 1, 4)  # (H,B,M,T,D)
        q = self.q_norm(q)
        qcos, qsin = self._cos_sin(query_positions, q.dtype, memory.device)
        q = _apply_rope(q, qcos.view(1, 1, 1, T, -1), qsin.view(1, 1, 1, T, -1))

        # (kvH,B,M,J,R,D) -> flatten only archive set/subslot, never memory group.
        k = k.reshape(self.n_kv_heads, B, M, J * R, self.head_dim)
        v = v.reshape(self.n_kv_heads, B, M, J * R, self.head_dim)
        q = q.reshape(self.n_kv_heads, self.gqa_groups, B, M, T, self.head_dim)
        k, v = k.unsqueeze(1), v.unsqueeze(1)
        scale = self.base_scale * torch.exp(self.logit_scale.clamp(max=self.max_logit_scale))
        scale = scale.reshape(self.n_kv_heads, self.gqa_groups, 1, 1, 1, 1)
        score = (q @ k.transpose(-2, -1)) * scale
        score = torch.tanh(score / self.soft_cap) * self.soft_cap

        visible_set = ((query_positions[:, None] - archive_positions[None, :])
                       >= self.min_archive_age)  # (T,J)
        visible = visible_set.repeat_interleave(R, dim=1)       # (T,J*R)
        masked = ~visible
        score = score.masked_fill(masked.view(1, 1, 1, 1, T, J * R), float("-inf"))
        no_key = ~visible.any(dim=-1)                            # (T,)
        if bool(no_key.any()):
            score = torch.where(no_key.view(1, 1, 1, 1, T, 1),
                                torch.zeros_like(score), score)
        attn = torch.softmax(score, dim=-1)
        attn = attn.masked_fill(masked.view(1, 1, 1, 1, T, J * R), 0.0)
        attn = self.attn_dropout(attn)
        out = attn @ v
        out = out.reshape(self.n_heads, B, M, T, self.head_dim)
        out = out.permute(1, 3, 2, 0, 4).reshape(B, T, M, E)
        out = self.out_dropout(self.out_proj(out))
        # ``out_proj`` has a bias, so mask again after projection: a query with no eligible key
        # must be exactly inert rather than receiving that bias as a fake archive residual.
        out = out * visible.any(dim=-1).to(out.dtype).view(1, T, 1, 1)
        if batch_mask is not None:
            out = out * batch_mask.to(out.dtype).view(B, 1, 1, 1)
        return out


class DynamicsModelArchive(DynamicsModel):
    """Base dynamics model plus external segment compressor and per-temporal-layer archive readers."""

    def __init__(self, config: ArchiveDynamicsConfig):
        assert config.n_memory > 0, "archive memory requires fast memory tokens"
        assert config.archive_per_memory > 0, "use the base model when archive_per_memory=0"
        assert 1 <= config.archive_interval <= config.max_temporal_length
        assert config.archive_compressor_depth >= 1
        super().__init__(config)
        self.archive_interval = config.archive_interval
        self.archive_per_memory = config.archive_per_memory
        self.archive_max_sets = config.archive_max_sets
        self.archive_compressor = ArchiveCompressor(config)
        self.archive_readers = nn.ModuleDict()
        self.archive_norms = nn.ModuleDict()
        self.archive_gates = nn.ParameterDict()
        for i, block in enumerate(self.blocks):
            if block.att.is_temporal:
                key = str(i)
                self.archive_readers[key] = GroupedArchiveAttention(config)
                self.archive_norms[key] = nn.RMSNorm(config.embedding_dim)
                self.archive_gates[key] = nn.Parameter(torch.tensor(float(config.archive_gate_init)))

        self.mem_start = self.n_action_tokens + self.n_latents + self.n_registers
        self.mem_end = self.mem_start + self.n_memory

    # ------------------------------------------------------------------ main forward
    def forward(self, z_tilde: torch.Tensor, tau_idx: torch.Tensor, d_idx: torch.Tensor,
                actions: torch.Tensor = None, memory_in: torch.Tensor = None,
                return_memory: bool = False, positions: torch.Tensor = None,
                cache: list = None, commit: bool = False, *,
                archive_bank: torch.Tensor = None,
                archive_positions: torch.Tensor = None,
                archive_cache: list = None,
                archive_batch_mask: torch.Tensor = None):
        B, T, L, _ = z_tilde.shape
        device = z_tilde.device
        # Preserve the base model's fixed-table path when callers omit positions.  The archive reader
        # still needs an explicit query clock, while local attention must receive the original None for
        # exact disabled/warm-start behavior.
        local_positions = positions
        query_positions = (torch.arange(T, device=device) if positions is None else positions)

        lat = self.in_proj(z_tilde)
        shortcut = torch.cat((self.tau_embedding(tau_idx), self.d_embedding(d_idx)), dim=-1).unsqueeze(2)
        action = self.action_embedding.expand(B, T, -1, -1)
        if actions is not None:
            action = action + actions
        register = self.register_tokens.expand(B, T, -1, -1)
        memory = memory_in if memory_in is not None else self.memory_tokens.expand(B, T, -1, -1)
        x = torch.cat((action, lat, register, memory, shortcut), dim=2)

        have_archive = archive_positions is not None and archive_positions.numel() > 0
        for i, block in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None
            # Inline TransformerBlock.forward so the archive branch sits between local attention and MLP.
            x = x + block.att(block.ln1(x), positions=local_positions,
                              layer_cache=layer_cache, commit=commit)
            if block.att.is_temporal and have_archive:
                key = str(i)
                mem = x[:, :, self.mem_start:self.mem_end]
                ac = archive_cache[i] if archive_cache is not None else None
                delta = self.archive_readers[key](
                    self.archive_norms[key](mem), query_positions,
                    archive=archive_bank, archive_positions=archive_positions,
                    archive_cache=ac, batch_mask=archive_batch_mask)
                mem = mem + self.archive_gates[key] * delta
                x = torch.cat((x[:, :, :self.mem_start], mem, x[:, :, self.mem_end:]), dim=2)
            x = x + block.mlp(block.ln2(x))

        out = self.out_proj(self.out_norm(x[:, :, self.n_action_tokens:self.n_action_tokens + L]))
        if return_memory:
            return out, x[:, :, self.mem_start:self.mem_end]
        return out

    # ------------------------------------------------------------------ archive projection/cache helpers
    def new_archive_cache(self) -> list:
        return [({} if block.att.is_temporal else None) for block in self.blocks]

    def project_archive_bank(self, archive_bank: torch.Tensor,
                             archive_positions: torch.Tensor) -> list:
        out = self.new_archive_cache()
        for i, block in enumerate(self.blocks):
            if block.att.is_temporal:
                k, v = self.archive_readers[str(i)].project_archive(archive_bank, archive_positions)
                out[i] = {"k": k, "v": v}
        return out

    def _append_archive_set(self, state: dict, archive_set: torch.Tensor, end_pos: int) -> None:
        pos = torch.tensor([end_pos], device=state["device"], dtype=torch.long)
        raw = archive_set[:, None]  # (B,1,M,R,E)
        for i, block in enumerate(self.blocks):
            if not block.att.is_temporal:
                continue
            k, v = self.archive_readers[str(i)].project_archive(raw, pos)
            lc = state["archive_cache"][i]
            lc["k"] = k if lc.get("k") is None else torch.cat((lc["k"], k), dim=3)
            lc["v"] = v if lc.get("v") is None else torch.cat((lc["v"], v), dim=3)
        state["archive_positions"] = torch.cat((state["archive_positions"], pos))

        cap = self.archive_max_sets
        if cap > 0 and state["archive_positions"].numel() > cap:
            state["archive_positions"] = state["archive_positions"][-cap:]
            for lc in state["archive_cache"]:
                if lc is not None and lc.get("k") is not None:
                    lc["k"] = lc["k"][:, :, :, -cap:]
                    lc["v"] = lc["v"][:, :, :, -cap:]

    def _append_committed_memory(self, state: dict, memory: torch.Tensor, start_pos: int) -> None:
        """Append consecutive final written memories and compress every complete aligned segment."""
        assert memory.shape[1] > 0
        expected = state["segment_start"] + state["segment_memory"].shape[1]
        assert start_pos == expected, f"segment source gap/overlap: got {start_pos}, expected {expected}"
        state["segment_memory"] = torch.cat((state["segment_memory"], memory), dim=1)
        N = self.archive_interval
        while state["segment_memory"].shape[1] >= N:
            source = state["segment_memory"][:, :N]
            end_pos = state["segment_start"] + N - 1
            archive_set = self.archive_compressor(source)
            self._append_archive_set(state, archive_set, end_pos)
            state["segment_memory"] = state["segment_memory"][:, N:]
            state["segment_start"] += N

    # ------------------------------------------------------------------ archive-aware carrying rollout
    @torch.no_grad()
    def rollout_init(self, context: torch.Tensor, ctx_action_idx: torch.Tensor = None,
                     K: int = None, max_ctx: int = None) -> dict:
        K = K or self.config.inference_steps
        B, T_ctx = context.shape[:2]
        device = context.device
        max_ctx = (self.config.max_temporal_length - 1) if max_ctx is None else max_ctx
        assert max_ctx == self.config.max_temporal_length - 1, (
            "version-one archive eligibility is defined for the fixed native local window; "
            f"got max_ctx={max_ctx}, expected {self.config.max_temporal_length - 1}")
        d_idx_val = K.bit_length() - 1
        tau_ctx_idx = min(round(self.config.context_signal * self.K_max), self.K_max - 1)
        T0 = min(T_ctx, self.config.max_temporal_length)

        act = self.action_features(ctx_action_idx[:, :T0] if ctx_action_idx is not None else None)
        ctx_noised = self._noise_to_ctx(context[:, :T0])
        positions = torch.arange(T0, device=device)
        tau_col = torch.full((B, T0), tau_ctx_idx, device=device, dtype=torch.long)
        d_col = torch.full((B, T0), d_idx_val, device=device, dtype=torch.long)
        _, mem_in = self(ctx_noised, tau_col, d_col, act, positions=positions, return_memory=True)

        cache = self.new_kv_cache()
        self(ctx_noised, tau_col, d_col, act, memory_in=mem_in,
             positions=positions, cache=cache, commit=True)
        self._evict(cache, max_ctx)
        state = {
            "cache": cache, "next_pos": int(T0), "K": K, "max_ctx": max_ctx,
            "d_idx_val": d_idx_val, "tau_ctx_idx": tau_ctx_idx, "B": B, "device": device,
            "archive_cache": self.new_archive_cache(),
            "archive_positions": torch.empty(0, dtype=torch.long, device=device),
            "segment_memory": mem_in.new_empty((B, 0, self.n_memory, self.config.embedding_dim)),
            "segment_start": 0,
        }
        self._append_committed_memory(state, mem_in, 0)
        for t in range(T0, T_ctx):
            a = ctx_action_idx[:, t:t + 1] if ctx_action_idx is not None else None
            self._commit_context_frame(state, context[:, t:t + 1], a)
        return state

    def _commit_context_frame(self, state: dict, z: torch.Tensor,
                              action_idx: torch.Tensor = None) -> None:
        cache, pos = state["cache"], state["next_pos"]
        B, device = state["B"], state["device"]
        if action_idx is not None and action_idx.dim() == 1:
            action_idx = action_idx[:, None]
        act = self.action_features(action_idx)
        positions = torch.tensor([pos], device=device)
        d_col = torch.full((B, 1), state["d_idx_val"], device=device, dtype=torch.long)
        tau_col = torch.full((B, 1), state["tau_ctx_idx"], device=device, dtype=torch.long)
        zc = self._noise_to_ctx(z)
        mem_init = self.memory_tokens.expand(B, 1, -1, -1)
        _, written_mem = self(
            zc, tau_col, d_col, act, memory_in=mem_init, positions=positions,
            cache=cache, commit=False, return_memory=True,
            archive_cache=state["archive_cache"], archive_positions=state["archive_positions"])
        self(
            zc, tau_col, d_col, act, memory_in=written_mem, positions=positions,
            cache=cache, commit=True,
            archive_cache=state["archive_cache"], archive_positions=state["archive_positions"])
        self._evict(cache, state["max_ctx"])
        self._append_committed_memory(state, written_mem, pos)
        state["next_pos"] = pos + 1

    @torch.no_grad()
    def rollout_step(self, state: dict, action_idx: torch.Tensor = None,
                     commit: bool = True) -> torch.Tensor:
        cache, pos, K = state["cache"], state["next_pos"], state["K"]
        B, device = state["B"], state["device"]
        if action_idx is not None and action_idx.dim() == 1:
            action_idx = action_idx[:, None]
        act = self.action_features(action_idx)
        positions = torch.tensor([pos], device=device)
        d_col = torch.full((B, 1), state["d_idx_val"], device=device, dtype=torch.long)
        tau_col = torch.full((B, 1), state["tau_ctx_idx"], device=device, dtype=torch.long)
        mem_init = self.memory_tokens.expand(B, 1, -1, -1)
        z = torch.randn((B, 1, self.n_latents, self.bottleneck_dim), device=device)
        written_mem = None
        for step in range(K):
            tau = step / K
            tau_col[:, -1] = round(tau * self.K_max)
            if step == K - 1:
                z_hat, written_mem = self(
                    z, tau_col, d_col, act, memory_in=mem_init, positions=positions,
                    cache=cache, commit=False, return_memory=True,
                    archive_cache=state["archive_cache"], archive_positions=state["archive_positions"])
            else:
                z_hat = self(
                    z, tau_col, d_col, act, memory_in=mem_init, positions=positions,
                    cache=cache, commit=False,
                    archive_cache=state["archive_cache"], archive_positions=state["archive_positions"])
            z = z + (z_hat - z) / (1 - tau) * (1.0 / K)

        if commit:
            tau_col[:, -1] = state["tau_ctx_idx"]
            self(
                self._noise_to_ctx(z), tau_col, d_col, act, memory_in=written_mem,
                positions=positions, cache=cache, commit=True,
                archive_cache=state["archive_cache"], archive_positions=state["archive_positions"])
            self._evict(cache, state["max_ctx"])
            self._append_committed_memory(state, written_mem, pos)
            state["next_pos"] = pos + 1
        return z
