"""
Dreamer-4 latent dynamics model (spec: specs/models/dynamics_model.md).

A block-causal transformer over a window of per-frame tokenizer latents. For every frame it
consumes ``n_latents`` latent tokens of size ``bottleneck_dim`` (from the separately-trained,
frozen tokenizer) plus optional actions, and predicts the *clean* latents of that frame from a
noised version, using its causal history as context. Trained with **shortcut forcing** (diffusion
forcing + shortcut models) so each frame generates in K=4 forward passes without errors snowballing.
It never sees pixels — it lives entirely in the tokenizer's latent space.

Our addition over vanilla Dreamer-4: optional per-timestep **memory tokens** that should encode the
whole env state so hidden state survives past the latent window. They are NOT physically carried; the
model reads the old memory tokens (their cached K/V) and writes new ones each step, carrying state
through repeated read-and-write. The FF9 sufficiency loss (§5) trains memory to be a sufficient
full-state object; the carrying KV-cached rollout (§4) exercises the relay at inference.

Token layout per frame (sequence axis): ``[action | latents | registers | (memory) | shortcut]``.
Only the latent-token outputs are read out as the x-prediction of the clean representation.
"""

import math

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class DynamicsModelConfig:

    dtype: torch.dtype = torch.bfloat16

    # Must match the frozen tokenizer that produces the representations.
    bottleneck_dim: int = 64
    n_latents: int = 4

    embedding_dim: int = 256
    max_temporal_length: int = 16

    n_heads: int = 16
    # Grouped-query attention: every gqa_groups query heads share one K/V head (must divide
    # n_heads). 1 = plain multi-head attention (fused qkv projection, parameter-compatible with
    # pre-GQA checkpoints); >1 shrinks the across-time KV cache by exactly this factor.
    gqa_groups: int = 1
    mlp_ratio: float = 3.0
    depth: int = 9  # 3x[spatial, temporal, spatial]; temporal at i%3==1

    drop_rate: float = 0.1
    att_drop_rate: float = 0.1
    att_logit_soft_cap: float = 50

    n_action_tokens: int = 1
    n_registers: int = 4

    # Discrete action conditioning. 0 => unlabeled video (only the learned action embedding is
    # used). >0 => a lookup table maps each discrete action id to a per-frame action feature.
    n_actions: int = 0

    # Memory tokens (our extension). n_memory=0 => vanilla model (no memory tokens, byte-identical
    # to a memory-free Dreamer-4). ff9_k>0 (with n_memory>0) enables the FF9 sufficiency loss.
    n_memory: int = 0
    ff9_k: int = 0

    # Shortcut forcing schedule.
    max_sampling_steps: int = 128   # K_max; finest step d_min = 1/K_max. Must be a power of two.
    inference_steps: int = 4        # K used per frame at generation time (d = 1/K).
    context_signal: float = 0.9     # tau_ctx = SIGNAL level of context frames during rollout
                                    # (1.0 = clean, 0.0 = pure noise). Keep high; holding context
                                    # near-clean lets the model read it (EXP-008 / D-010).
    ramp_min: float = 0.1           # w(tau) = (1 - ramp_min) * tau + ramp_min.


class Attention(nn.Module):
    """Block-causal attention. Space layers attend fully within a frame; temporal layers
    attend causally across time and use RoPE on the time axis.

    The KV-cache path (``positions`` / ``layer_cache`` / ``commit``) drives the carrying rollout
    (§4): cached K/V is frozen at the rotation it got on entry, so we rotate every token at its
    ABSOLUTE rollout index (RoPE depends only on relative distance) and never re-index the window.
    The default path (positions=None, no cache) is the training/uncached forward.
    """

    def __init__(self, config: DynamicsModelConfig, is_temporal):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.embedding_dim / self.n_heads
        assert int(self.head_dim) == self.head_dim
        self.head_dim = int(self.head_dim)

        # Learnable per-head attention temperature (spec §2). q/k are RMSNorm'd below (QK-norm),
        # which caps |q.k|; with the textbook 1/sqrt(d) scale the logits stay ~O(1) over all keys
        # so softmax is near-uniform. Init ~4x sharper than 1/sqrt(d) to escape that basin and let
        # it adapt; clamped for stability. Mirrors the tokenizer's logit_scale.
        self.base_scale = 1 / (self.head_dim ** 0.5)
        self.logit_scale = nn.Parameter(torch.full((self.n_heads, 1, 1, 1, 1), math.log(4.0)))
        self.max_logit_scale = math.log(100.0)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.soft_cap_act = nn.Tanh()
        self.soft_cap = config.att_logit_soft_cap

        self.is_temporal = is_temporal
        self.dropout = nn.Dropout(config.drop_rate)
        self.att_droput = nn.Dropout(config.att_drop_rate)
        # GQA: every gqa_groups query heads share one K/V head. gqa_groups=1 keeps the fused qkv
        # projection (parameter-compatible with pre-GQA checkpoints); >1 uses separate q/kv
        # projections with n_kv_heads = n_heads / gqa_groups.
        self.gqa_groups = config.gqa_groups
        assert self.n_heads % self.gqa_groups == 0, \
            f"gqa_groups {self.gqa_groups} must divide n_heads {self.n_heads}"
        self.n_kv_heads = self.n_heads // self.gqa_groups
        if self.gqa_groups == 1:
            self.qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        else:
            self.q_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
            self.kv_proj = nn.Linear(config.embedding_dim, 2 * self.n_kv_heads * self.head_dim)
        self.proj = nn.Linear(config.embedding_dim, config.embedding_dim)

        # RoPE tables over the temporal axis. The fixed-size cos/sin tables drive the default
        # (training / uncached) path. `rope_freqs` (the base frequencies) lets the cached path
        # compute rotations on the fly at ARBITRARY absolute positions.
        d_half = self.head_dim // 2
        freqs = 10_000 ** (-2 * torch.arange(d_half, dtype=torch.float32) / self.head_dim)
        angles = torch.outer(torch.arange(config.max_temporal_length, dtype=torch.float32), freqs)
        self.register_buffer('cos', torch.cos(angles))
        self.register_buffer('sin', torch.sin(angles))
        self.register_buffer('rope_freqs', freqs, persistent=False)

    def _rope_cos_sin(self, T: int, positions: torch.Tensor, dtype, device):
        """RoPE cos/sin for the time axis. positions=None -> the fixed-table path (positions
        0..T-1). positions given -> rotations computed on the fly at those ABSOLUTE positions."""
        if positions is None:
            return self.cos[:T], self.sin[:T]
        ang = torch.outer(positions.to(device=device, dtype=torch.float32),
                          self.rope_freqs.to(device))  # (T, d_half)
        return torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)

    @staticmethod
    def _apply_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        d = t.shape[-1] // 2
        first, second = t[..., :d], t[..., d:]
        return torch.concat((first * cos + -second * sin, second * cos + first * sin), dim=-1)

    def forward(self, x: torch.Tensor, positions: torch.Tensor = None,
                layer_cache: dict = None, commit: bool = False):
        # x: (B, T, N, C). positions: optional (T,) absolute time indices of these frames.
        # layer_cache: optional {'k','v'} of this temporal layer's cached (rotated) K/V.
        # commit: append this call's K/V to layer_cache (extends the context cache).
        B, T, N, C = x.shape

        if self.gqa_groups == 1:
            qkv: torch.Tensor = self.qkv(x)
            qkv = qkv.reshape((B, T, N, 3, self.n_heads, -1))
            if not self.is_temporal:
                qkv = qkv.permute((3, 4, 0, 1, 2, 5))  # (3, heads, B, T, N, head_dim)
            else:
                qkv = qkv.permute((3, 4, 0, 2, 1, 5))  # (3, heads, B, N, T, head_dim)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            # GQA: q has n_heads heads, k/v only n_kv_heads (the cache stores the small shape).
            q = self.q_proj(x).reshape(B, T, N, self.n_heads, self.head_dim)
            kv = self.kv_proj(x).reshape(B, T, N, 2, self.n_kv_heads, self.head_dim)
            if not self.is_temporal:
                q = q.permute(3, 0, 1, 2, 4)           # (heads, B, T, N, head_dim)
                kv = kv.permute(3, 4, 0, 1, 2, 5)      # (2, kv_heads, B, T, N, head_dim)
            else:
                q = q.permute(3, 0, 2, 1, 4)           # (heads, B, N, T, head_dim)
                kv = kv.permute(3, 4, 0, 2, 1, 5)      # (2, kv_heads, B, N, T, head_dim)
            k, v = kv[0], kv[1]
        q, k = self.q_norm(q), self.k_norm(k)

        if not self.is_temporal:
            # Full self-attention within a frame: all token types exchange information.
            mask = None
            k_all, v_all = k, v
        else:
            cos, sin = self._rope_cos_sin(T, positions, q.dtype, x.device)
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)

            # Prepend cached (already-rotated) K/V from earlier frames, if any.
            if layer_cache is not None and layer_cache.get('k') is not None:
                k_all = torch.concat((layer_cache['k'], k), dim=-2)
                v_all = torch.concat((layer_cache['v'], v), dim=-2)
            else:
                k_all, v_all = k, v
            if commit and layer_cache is not None:
                layer_cache['k'] = k_all
                layer_cache['v'] = v_all

            # Causal mask (T_query, T_all): cached keys (the first T_all-T cols) are earlier
            # frames -> visible; new keys are causal among themselves. No cache -> triu(T,T).
            T_all = k_all.shape[-2]
            mask = torch.zeros((T, T_all), dtype=torch.bool, device=x.device)
            mask[:, T_all - T:] = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        scale = self.base_scale * torch.exp(self.logit_scale.clamp(max=self.max_logit_scale))
        if self.gqa_groups > 1:
            # Grouped view: q (kv_heads, groups, B, *, T, hd) against shared K/V broadcast over the
            # group axis (kv_heads, 1, B, *, T, hd) — repeated K/V is never materialized. The logit
            # scale stays per-QUERY-head (head h = kv_head * groups + g).
            q = q.reshape(self.n_kv_heads, self.gqa_groups, *q.shape[1:])
            k_all, v_all = k_all.unsqueeze(1), v_all.unsqueeze(1)
            scale = scale.reshape(self.n_kv_heads, self.gqa_groups, 1, 1, 1, 1)
        attn_scores = (q @ k_all.transpose(-2, -1)) * scale
        attn_scores = self.soft_cap_act(attn_scores / self.soft_cap) * self.soft_cap
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.att_droput(attn)

        x = attn @ v_all
        if self.gqa_groups > 1:
            x = x.reshape(self.n_heads, *x.shape[2:])  # merge (kv_heads, groups) -> heads
        if not self.is_temporal:
            x = x.permute(1, 2, 3, 0, 4).reshape(B, T, N, C)
        else:
            x = x.permute(1, 3, 2, 0, 4).reshape(B, T, N, C)

        return self.dropout(self.proj(x))


class SwiGLU(nn.Module):
    def __init__(self, config: DynamicsModelConfig) -> None:
        super().__init__()
        self.ll1_1 = nn.Linear(config.embedding_dim, int(config.embedding_dim * config.mlp_ratio))
        self.ll1_2 = nn.Linear(config.embedding_dim, int(config.embedding_dim * config.mlp_ratio))
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(config.drop_rate)
        self.ll2 = nn.Linear(int(config.embedding_dim * config.mlp_ratio), config.embedding_dim)

    def forward(self, x):
        x = self.ll1_1(x) * self.act(self.ll1_2(x))
        x = self.dropout(x)
        x = self.ll2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: DynamicsModelConfig, is_temporal) -> None:
        super().__init__()
        self.att = Attention(config, is_temporal)
        self.ln1 = nn.RMSNorm(config.embedding_dim)
        self.mlp = SwiGLU(config)
        self.ln2 = nn.RMSNorm(config.embedding_dim)

    def forward(self, x, positions: torch.Tensor = None,
                layer_cache: dict = None, commit: bool = False):
        x = x + self.att(self.ln1(x), positions=positions, layer_cache=layer_cache, commit=commit)
        x = x + self.mlp(self.ln2(x))
        return x


class DynamicsModel(nn.Module):
    """Block-causal transformer that denoises per-frame latents via shortcut forcing, with an
    optional memory channel relayed across time by the carrying KV-cached rollout."""

    def __init__(self, config: DynamicsModelConfig) -> None:
        super().__init__()
        self.config = config
        E = config.embedding_dim
        self.n_latents = config.n_latents
        self.bottleneck_dim = config.bottleneck_dim
        self.n_action_tokens = config.n_action_tokens
        self.n_registers = config.n_registers
        self.n_memory = config.n_memory

        self.K_max = config.max_sampling_steps
        assert (self.K_max & (self.K_max - 1)) == 0, "max_sampling_steps must be a power of two"
        self.n_d = self.K_max.bit_length()  # number of distinct step sizes: K in {1,2,...,K_max}

        # Latent <-> model-dim projections. Latents are read out for the x-prediction.
        self.in_proj = nn.Linear(config.bottleneck_dim, E)
        self.out_norm = nn.RMSNorm(E)
        self.out_proj = nn.Linear(E, config.bottleneck_dim)

        # Learned tokens. Kept in float32 (default) so the model also runs without autocast.
        self.action_embedding = nn.Parameter(0.05 * torch.rand((config.n_action_tokens, E)))
        self.register_tokens = nn.Parameter(0.05 * torch.rand((config.n_registers, E)))
        # Memory tokens (our extension). Only instantiated when n_memory>0 so n_memory=0 models are
        # byte-identical to a memory-free model. Placed AFTER registers so the register offset is fixed.
        if config.n_memory > 0:
            self.memory_tokens = nn.Parameter(0.05 * torch.rand((config.n_memory, E)))

        # Discrete action conditioning: map each action id to a per-frame feature added to the
        # learned action embedding. Absent (n_actions == 0) => unlabeled video.
        self.n_actions = config.n_actions
        if config.n_actions > 0:
            self.action_table = nn.Embedding(config.n_actions, config.n_action_tokens * E)

        # Discrete shortcut conditioning: tau and d each get half the channels, concatenated.
        assert E % 2 == 0
        self.tau_embedding = nn.Embedding(self.K_max, E // 2)
        self.d_embedding = nn.Embedding(self.n_d, E // 2)

        self.blocks = nn.ModuleList([
            # Layer cadence: 3x[spatial, temporal, spatial] -> temporal at i%3==1 (depth=9 = three groups).
            TransformerBlock(config, is_temporal=(i % 3 == 1))
            for i in range(config.depth)
        ])

    # ------------------------------------------------------------------ helpers
    def _d_value(self, d_idx: torch.Tensor) -> torch.Tensor:
        return torch.pow(2.0, -d_idx.float())  # d = 1 / K, with K = 2 ** d_idx.

    def _tau_value(self, tau_idx: torch.Tensor) -> torch.Tensor:
        return tau_idx.float() / self.K_max  # tau grid points are multiples of d_min = 1/K_max.

    def sample_tau_d(self, B: int, T: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample per-frame (tau_idx, d_idx). d ~ 1/U({1,...,K_max}); tau ~ U on the grid implied by d."""
        d_idx = torch.randint(0, self.n_d, (B, T), device=device)
        K = torch.pow(2, d_idx)
        step = (torch.rand((B, T), device=device) * K).long()
        step = torch.minimum(step, K - 1)
        tau_idx = step * torch.pow(2, self.n_d - 1 - d_idx)  # snap to the d_min grid
        return tau_idx, d_idx

    def action_features(self, action_idx: torch.Tensor) -> torch.Tensor:
        """(B, T) long action ids -> (B, T, n_action_tokens, E) feature tokens, or None when
        unlabeled / no ids (so callers can pass the result straight to ``forward``)."""
        if action_idx is None or self.n_actions == 0:
            return None
        B, T = action_idx.shape
        return self.action_table(action_idx).reshape(B, T, self.n_action_tokens, -1)

    def _noise_to_ctx(self, z: torch.Tensor) -> torch.Tensor:
        """Hold clean latents at signal level context_signal (near-clean) so the model can read them."""
        t = self.config.context_signal
        return (1 - t) * torch.randn_like(z) + t * z

    # ------------------------------------------------------------------ forward
    def forward(self, z_tilde: torch.Tensor, tau_idx: torch.Tensor, d_idx: torch.Tensor,
                actions: torch.Tensor = None, memory_in: torch.Tensor = None,
                return_memory: bool = False, positions: torch.Tensor = None,
                cache: list = None, commit: bool = False):
        """Predict clean latents (x-prediction) from noised latents.

        z_tilde:    (B, T, n_latents, bottleneck_dim) noised representations.
        tau_idx:    (B, T) long discrete signal level per frame.
        d_idx:      (B, T) long discrete step size per frame.
        actions:    optional (B, T, n_action_tokens, E) action features added to the learned action
                    embedding. None => unlabeled video.
        memory_in:  optional (B, T, n_memory, E) memory tokens to inject INSTEAD of the learned-init
                    tokens (FF9 sufficiency / carried-rollout commit). None => learned-init.
        return_memory: also return the final-layer memory states (B, T, n_memory, E) — the WRITTEN
                    memory tokens the carried rollout commits / the FF9 loss re-injects.
        positions:  optional (T,) absolute time indices (KV-cache / long rollout). None => 0..T-1 table.
        cache:      optional per-block list (len=depth) of {'k','v'} dicts (None for spatial blocks).
        commit:     append this call's temporal K/V to ``cache``.
        """
        B, T, L, _ = z_tilde.shape

        lat = self.in_proj(z_tilde)  # (B, T, L, E)
        shortcut = torch.concat(
            (self.tau_embedding(tau_idx), self.d_embedding(d_idx)), dim=-1
        ).unsqueeze(2)  # (B, T, 1, E)

        action = self.action_embedding.expand(B, T, -1, -1)
        if actions is not None:
            action = action + actions
        register = self.register_tokens.expand(B, T, -1, -1)

        # Token layout: [action | latents | registers | (memory) | shortcut].
        parts = [action, lat, register]
        if self.n_memory > 0:
            memory = memory_in if memory_in is not None else self.memory_tokens.expand(B, T, -1, -1)
            parts.append(memory)
        parts.append(shortcut)
        x = torch.concat(parts, dim=2)  # (B, T, N, E)

        for i, block in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None
            x = block(x, positions=positions, layer_cache=layer_cache, commit=commit)

        lat_start = self.n_action_tokens
        out = self.out_proj(self.out_norm(x[:, :, lat_start:lat_start + L, :]))
        if return_memory:
            mem_start = lat_start + L + self.n_registers
            return out, x[:, :, mem_start:mem_start + self.n_memory, :]
        return out

    # ------------------------------------------------------------------ loss
    def loss(self, z1: torch.Tensor, action_idx: torch.Tensor = None,
             return_parts: bool = False):
        """Shortcut forcing loss over a clip of clean representations, plus the FF9 memory
        sufficiency term when ``n_memory>0`` and ``config.ff9_k>0``.

        z1:         (B, T, n_latents, bottleneck_dim) clean tokenizer latents.
        action_idx: optional (B, T) long discrete action ids, aligned per frame.
        """
        device = z1.device
        actions = self.action_features(action_idx)
        B, T, L, _ = z1.shape
        # Training is confined to the temporal window (the RoPE table length); a longer clip would
        # index the fixed cos/sin table out of range on the default forward path.
        assert T <= self.config.max_temporal_length, (
            f"clip length {T} exceeds max_temporal_length {self.config.max_temporal_length}")

        tau_idx, d_idx = self.sample_tau_d(B, T, device)
        tau = self._tau_value(tau_idx)[..., None, None]  # (B, T, 1, 1)
        d = self._d_value(d_idx)[..., None, None]

        z0 = torch.randn_like(z1)
        z_tilde = (1 - tau) * z0 + tau * z1

        want_mem = self.n_memory > 0 and self.config.ff9_k > 0
        if want_mem:
            z_hat1, mem = self(z_tilde, tau_idx, d_idx, actions, return_memory=True)
        else:
            z_hat1 = self(z_tilde, tau_idx, d_idx, actions)

        is_min = (d_idx == self.n_d - 1)[..., None, None]  # finest step => pure flow loss

        # --- bootstrap targets: distill two d/2 steps (Eq. 7), stop-gradient. ---
        with torch.no_grad():
            half_d_idx = (d_idx + 1).clamp(max=self.n_d - 1)
            half_d = self._d_value(half_d_idx)[..., None, None]
            tau_inc = torch.pow(2, (self.n_d - 2 - d_idx).clamp(min=0))
            tau2_idx = (tau_idx + tau_inc).clamp(max=self.K_max - 1)
            tau2 = self._tau_value(tau2_idx)[..., None, None]

            y1 = self(z_tilde, tau_idx, half_d_idx, actions)
            b1 = (y1 - z_tilde) / (1 - tau)
            z_prime = z_tilde + b1 * half_d

            y2 = self(z_prime, tau2_idx, half_d_idx, actions)
            b2 = (y2 - z_prime) / (1 - tau2)
            v_target = (b1 + b2) / 2

        flow_loss = (z_hat1 - z1) ** 2
        v_pred = (z_hat1 - z_tilde) / (1 - tau)
        boot_loss = (1 - tau) ** 2 * (v_pred - v_target) ** 2
        per_token = torch.where(is_min, flow_loss, boot_loss)

        w = (1 - self.config.ramp_min) * tau + self.config.ramp_min  # ramp weight, Eq. 8
        diffusion = (w * per_token).mean()

        total = diffusion
        parts = {"diffusion": diffusion.detach()}
        if want_mem:
            ff9 = self._ff9_loss(z1, mem, actions, self.config.ff9_k)
            # Loss normalization: scale the FF9 term to the diffusion magnitude with a gradient-
            # detached scaler, so the memory-sufficiency gradient is balanced against the flow loss
            # regardless of raw scale (spec §5). The gradient still flows through ``ff9``.
            scale = (diffusion.detach() / ff9.detach().clamp(min=1e-8))
            total = total + scale * ff9
            parts["ff9"] = ff9.detach()
        if return_parts:
            return total, parts
        return total

    def _ff9_loss(self, z1: torch.Tensor, mem: torch.Tensor, actions: torch.Tensor,
                  k: int) -> torch.Tensor:
        """FF9 memory-only sufficiency loss (spec §5).

        For every frame t with k successors, a (k+1)-frame mini-window [t .. t+k] folded into the
        batch. Per window a horizon j ~ U{1..k} is sampled:
          * frames t .. t+j-1 (incl. the source t): signal level tau=0 -> latent slots are PURE
            NOISE, so NO ground-truth latent is anywhere on the path. memory_t (written by the main
            windowed pass) is injected at t; learned-init memory relays at t+1.. via the position-wise
            temporal memory channel. Memory is the ONLY scene carrier on the path.
          * frame t+j (terminal): a sampled tau so a well-posed target exists; low-tau samples force
            memory to be load-bearing. Frames > j are computed but masked (cannot affect frames <= j).
        Loss on frames t+1 .. t+j. Backprops through the injected memory_t into the windowed pass that
        wrote it (write-side credit), so the gradient flows through the memory-construction mechanism.
        """
        assert self.n_memory > 0, "ff9 requires n_memory > 0"
        B, T, L, D = z1.shape
        device = z1.device
        n_t = T - k
        assert n_t >= 1, f"clip length {T} too short for ff9_k={k}"
        M, E = mem.shape[-2], mem.shape[-1]
        BN = B * n_t

        idx = torch.arange(n_t, device=device)[:, None] + torch.arange(k + 1, device=device)[None, :]
        zw = z1[:, idx].reshape(BN, k + 1, L, D)

        mem0 = mem[:, :n_t].reshape(BN, 1, M, E)               # written memory at the source frame
        mem_rest = self.memory_tokens.expand(BN, k, -1, -1)    # learned-init elsewhere
        memory_in = torch.concat((mem0, mem_rest), dim=1)      # (BN, k+1, M, E)

        j = torch.randint(1, k + 1, (BN,), device=device)
        tau_idx = torch.zeros(BN, k + 1, dtype=torch.long, device=device)
        tau_term = torch.randint(0, self.K_max, (BN,), device=device)
        tau_idx[torch.arange(BN, device=device), j] = tau_term  # sampled tau on the terminal slot
        d_idx = torch.full((BN, k + 1), self.n_d - 1, dtype=torch.long, device=device)

        tau = self._tau_value(tau_idx)[..., None, None]
        z_tilde = (1 - tau) * torch.randn_like(zw) + tau * zw   # path frames (tau=0) => pure noise

        act_in = None
        if actions is not None:
            act_in = actions[:, idx].reshape(BN, k + 1, self.n_action_tokens, -1)

        z_hat = self(z_tilde, tau_idx, d_idx, act_in, memory_in=memory_in)

        flow = ((z_hat[:, 1:] - zw[:, 1:]) ** 2).mean(dim=(-1, -2))  # (BN, k) per-frame MSE
        frame_pos = torch.arange(1, k + 1, device=device)[None, :]
        mask = (frame_pos <= j[:, None]).float()
        return (flow * mask).sum() / mask.sum().clamp(min=1)

    # ------------------------------------------------------------------ carrying rollout (§4)
    def new_kv_cache(self) -> list:
        """A fresh per-block KV-cache list: an empty {} for each temporal block, None for spatial
        blocks (which attend within a frame and never cache across time)."""
        return [({} if block.att.is_temporal else None) for block in self.blocks]

    @staticmethod
    def _evict(cache: list, max_ctx: int) -> None:
        """Slide the window: keep only the most recent ``max_ctx`` committed time-columns. Cached K/V
        are pre-rotated at absolute positions, so eviction is a pure slice (no re-rotation)."""
        for lc in cache:
            if lc is not None and lc.get('k') is not None and lc['k'].shape[-2] > max_ctx:
                lc['k'] = lc['k'][..., -max_ctx:, :]
                lc['v'] = lc['v'][..., -max_ctx:, :]

    @torch.no_grad()
    def rollout_init(self, context: torch.Tensor, ctx_action_idx: torch.Tensor = None,
                     K: int = None, max_ctx: int = None) -> dict:
        """Prefill the carrying KV cache from observed context latents and return a rollout state.

        context:        (B, T_ctx, n_latents, bottleneck_dim) clean latents from the tokenizer.
        ctx_action_idx: optional (B, T_ctx) long action ids for the context frames.
        max_ctx:        committed time-columns kept in the sliding window (default
                        ``max_temporal_length-1``). Pass a smaller value to FORCE a shorter window than
                        the model trained with (e.g. probe memory at window 8 instead of 16).
        The context is committed at near-clean (signal=context_signal) with its WRITTEN memory tokens
        (the relay seed), at absolute positions 0..T_ctx-1, then evicted to the last max_ctx frames.

        T_ctx may EXCEED ``max_temporal_length`` (long-context prefill): the first window is committed
        in one forward exactly as above, then each remaining TRUE frame is teacher-forced one committed
        step at a time through the sliding window (the same near-clean commit pass as
        ``rollout_step(commit=True)``, written-memory relay included) — so a memory model absorbs
        pre-window context into its memory tokens; a vanilla model just slides.
        """
        K = K or self.config.inference_steps
        B, T_ctx = context.shape[:2]
        device = context.device
        max_ctx = (self.config.max_temporal_length - 1) if max_ctx is None else max_ctx
        d_idx_val = K.bit_length() - 1
        tau_ctx_idx = min(round(self.config.context_signal * self.K_max), self.K_max - 1)
        T0 = min(T_ctx, self.config.max_temporal_length)  # first (windowed) chunk

        act = self.action_features(ctx_action_idx[:, :T0] if ctx_action_idx is not None else None)
        ctx_noised = self._noise_to_ctx(context[:, :T0])
        positions = torch.arange(T0, device=device)
        tau_col = torch.full((B, T0), tau_ctx_idx, device=device, dtype=torch.long)
        d_col = torch.full((B, T0), d_idx_val, device=device, dtype=torch.long)

        mem_in = None
        if self.n_memory > 0:
            # Memory the model WRITES for the context (learned-init pass), used as the relay seed.
            _, mem_in = self(ctx_noised, tau_col, d_col, act, positions=positions, return_memory=True)

        cache = self.new_kv_cache()
        self(ctx_noised, tau_col, d_col, act, memory_in=mem_in,
             positions=positions, cache=cache, commit=True)
        self._evict(cache, max_ctx)
        state = {"cache": cache, "next_pos": int(T0), "K": K, "max_ctx": max_ctx,
                 "d_idx_val": d_idx_val, "tau_ctx_idx": tau_ctx_idx, "B": B, "device": device}
        for t in range(T0, T_ctx):  # teacher-forced prefill of the beyond-window context
            a = ctx_action_idx[:, t:t + 1] if ctx_action_idx is not None else None
            self._commit_context_frame(state, context[:, t:t + 1], a)
        return state

    def _commit_context_frame(self, state: dict, z: torch.Tensor,
                              action_idx: torch.Tensor = None) -> None:
        """Teacher-forced commit of one TRUE context latent at the next absolute position: the same
        near-clean commit pass as ``rollout_step(commit=True)`` (written-memory relay, K/V append,
        window eviction), with the provided latent in place of a generated one. Used by
        ``rollout_init`` to prefill context longer than the temporal window."""
        cache, pos = state["cache"], state["next_pos"]
        B, device = state["B"], state["device"]
        if action_idx is not None and action_idx.dim() == 1:
            action_idx = action_idx[:, None]
        act = self.action_features(action_idx)
        positions = torch.tensor([pos], device=device)
        d_col = torch.full((B, 1), state["d_idx_val"], device=device, dtype=torch.long)
        tau_col = torch.full((B, 1), state["tau_ctx_idx"], device=device, dtype=torch.long)
        zc = self._noise_to_ctx(z)
        written_mem = None
        if self.n_memory > 0:
            # A near-clean read of the frame against the carried cache WRITES this frame's memory.
            mem_in = self.memory_tokens.expand(B, 1, -1, -1)
            _, written_mem = self(zc, tau_col, d_col, act, memory_in=mem_in, positions=positions,
                                  cache=cache, commit=False, return_memory=True)
        self(zc, tau_col, d_col, act, memory_in=written_mem, positions=positions,
             cache=cache, commit=True)
        self._evict(cache, state["max_ctx"])
        state["next_pos"] = pos + 1

    @torch.no_grad()
    def rollout_step(self, state: dict, action_idx: torch.Tensor = None,
                     commit: bool = True) -> torch.Tensor:
        """Generate one frame from the carried state via K shortcut steps reading the cache.

        action_idx: optional (B,) or (B,1) long action id for the new frame.
        commit=True  -> append this frame to the cache (5th pass at near-clean + written memory),
                        evict, advance the rollout. This is the path the rollout actually takes.
        commit=False -> READ-ONLY branch: return the predicted latent WITHOUT mutating the carried
                        cache/memory (used by the recall eval's reveal branch). The frame is
                        predicted at the SAME absolute position the next committed frame would take.
        returns: (B, 1, n_latents, bottleneck_dim) predicted clean latent.
        """
        cache, pos, K = state["cache"], state["next_pos"], state["K"]
        max_ctx, d_idx_val, tau_ctx_idx = state["max_ctx"], state["d_idx_val"], state["tau_ctx_idx"]
        B, device = state["B"], state["device"]
        L, D = self.n_latents, self.bottleneck_dim

        if action_idx is not None and action_idx.dim() == 1:
            action_idx = action_idx[:, None]
        act = self.action_features(action_idx)  # (B,1,n_act,E) or None
        positions = torch.tensor([pos], device=device)
        d_col = torch.full((B, 1), d_idx_val, device=device, dtype=torch.long)
        tau_col = torch.full((B, 1), tau_ctx_idx, device=device, dtype=torch.long)
        mem_in = self.memory_tokens.expand(B, 1, -1, -1) if self.n_memory > 0 else None

        d_val = 1.0 / K
        z = torch.randn((B, 1, L, D), device=device)  # pure noise, tau = 0
        written_mem = None
        for step in range(K):
            tau = step / K
            tau_col[:, -1] = round(tau * self.K_max)
            last = step == K - 1
            if last and self.n_memory > 0:
                z_hat1, written_mem = self(z, tau_col, d_col, act, memory_in=mem_in,
                                           positions=positions, cache=cache, commit=False,
                                           return_memory=True)
            else:
                z_hat1 = self(z, tau_col, d_col, act, memory_in=mem_in,
                              positions=positions, cache=cache, commit=False)
            v = (z_hat1 - z) / (1 - tau)
            z = z + v * d_val

        if commit:
            # 5th pass: re-present the frame at near-clean with its written memory; commit its K/V.
            tau_col[:, -1] = tau_ctx_idx
            self(self._noise_to_ctx(z), tau_col, d_col, act, memory_in=written_mem,
                 positions=positions, cache=cache, commit=True)
            self._evict(cache, max_ctx)
            state["next_pos"] = pos + 1
        return z

    @torch.no_grad()
    def generate(self, context: torch.Tensor, n_generate: int, K: int = None,
                 action_idx: torch.Tensor = None, max_ctx: int = None) -> torch.Tensor:
        """Carrying autoregressive rollout: roll out ``n_generate`` frames after ``context``.

        context:    (B, T_ctx, n_latents, bottleneck_dim) clean latents from the tokenizer.
        action_idx: optional (B, T_ctx + n_generate) long action ids for context + generated frames
                    (required when action-conditioned). returns (B, n_generate, n_latents, bottleneck_dim).
        max_ctx:    forced sliding-window size (committed time-columns); default max_temporal_length-1.
        """
        T_ctx = context.shape[1]
        ctx_act = action_idx[:, :T_ctx] if action_idx is not None else None
        state = self.rollout_init(context, ctx_act, K, max_ctx=max_ctx)
        out = []
        for i in range(n_generate):
            a = action_idx[:, T_ctx + i:T_ctx + i + 1] if action_idx is not None else None
            out.append(self.rollout_step(state, a, commit=True))
        return torch.concat(out, dim=1)
