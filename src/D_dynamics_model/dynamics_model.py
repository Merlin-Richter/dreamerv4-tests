"""
Dreamer 4 interactive dynamics model (Section 3.2).

The dynamics model operates on the sequence of continuous representations produced by the
frozen causal tokenizer (``C_multi_image_auto_encoder``). For every frame it consumes
``n_latents`` latent tokens of size ``bottleneck_dim`` and predicts the *clean* latents of
that frame from a noised version of it, using its causal history as context. It is trained
with a **shortcut forcing** objective: diffusion forcing (a per-frame signal level) combined
with shortcut models (conditioning on the requested step size ``d``) so that frames can be
sampled with as few as ``K = 4`` forward passes at inference.

Key choices from the paper:
  * x-prediction: the network predicts the clean latents ``z1`` (not the velocity), which makes
    long autoregressive rollouts stable.
  * x-space flow loss at the finest step ``d_min = 1/max_sampling_steps``; bootstrap loss that
    distills two ``d/2`` steps for the larger step sizes (Eq. 7).
  * ramp loss weight ``w(tau) = 0.9*tau + 0.1`` to focus capacity on the high-signal levels.
  * signal level ``tau`` and step size ``d`` are discrete, encoded with embedding lookups whose
    channels are concatenated into a single shortcut token per frame.

For unlabeled video the action modality is just a learned embedding (the "unlabeled video"
case in the paper). For labeled data, set ``n_actions`` > 0: an embedding table maps each
discrete action id to a per-frame feature added to that learned embedding. ``loss`` and
``generate`` accept per-frame ``action_idx`` (the occluded-bouncing env emits curtain-state
ids); ``forward`` takes the pre-computed action features.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class DynamicsModelConfig():

    dtype: torch.dtype = torch.bfloat16

    # Must match the frozen tokenizer that produces the representations.
    bottleneck_dim: int = 64
    n_latents: int = 4

    embedding_dim: int = 256
    max_temporal_length: int = 16

    n_heads: int = 16
    mlp_ratio: float = 3.0
    depth: int = 8

    drop_rate: float = 0.1
    att_drop_rate: float = 0.1
    att_logit_soft_cap: float = 50

    n_action_tokens: int = 1
    n_registers: int = 4

    # Discrete action conditioning. 0 => unlabeled video (only the learned action embedding is
    # used). >0 => a lookup table maps each discrete action id to a per-frame action feature.
    n_actions: int = 0

    # Shortcut forcing schedule.
    max_sampling_steps: int = 128   # K_max; finest step d_min = 1/K_max. Must be a power of two.
    inference_steps: int = 4        # K used per frame at generation time (d = 1/K).
    context_signal: float = 0.9     # tau_ctx = SIGNAL level of context frames during rollout
                                    # (1.0 = clean, 0.0 = pure noise). Keep high; the old
                                    # default 0.1 meant 90% noise on the context -> the model
                                    # could not read ball color/position (EXP-008 / D-010).
    ramp_min: float = 0.1           # w(tau) = (1 - ramp_min) * tau + ramp_min.

    # FF7 register memory (D-014). When set, generate() dispatches to generate_memory():
    # a sequential window-1 rollout that carries each frame's final-layer register state and
    # injects it as the next step's context registers — the persistent channel that vanilla
    # generate() lacks (registers are re-expanded from the learned tokens every forward pass,
    # so only the latent sequence survives between steps; latents are pixel-bound).
    use_register_memory: bool = False
    ff7_k: int = 0                  # provenance: FF7 lookahead depth used in training (0 = off).


class Attention(nn.Module):
    """Block-causal attention. Space layers attend fully within a frame; temporal layers
    attend causally across time and use RoPE on the time axis."""

    def __init__(self, config: DynamicsModelConfig, is_temporal):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.embedding_dim / self.n_heads

        assert int(self.head_dim) == self.head_dim
        self.head_dim = int(self.head_dim)

        self.scale = 1 / (self.head_dim ** 0.5)

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.soft_cap_act = nn.Tanh()
        self.soft_cap = config.att_logit_soft_cap

        self.is_temporal = is_temporal
        self.dropout = nn.Dropout(config.drop_rate)
        self.att_droput = nn.Dropout(config.att_drop_rate)
        self.qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.proj = nn.Linear(config.embedding_dim, config.embedding_dim)

        # RoPE tables over the temporal axis. The fixed-size cos/sin tables drive the default
        # (training / uncached) path exactly as before. `rope_freqs` (the base frequencies) lets
        # the KV-cache inference path compute rotations on the fly for ARBITRARY absolute
        # positions — required because cached K/V is frozen at the rotation it got on entry and
        # must never be re-indexed when the window slides (HOWTO/rope_kv_cache_caveat.md).
        d_half = self.head_dim // 2
        freqs = 10_000 ** (-2 * torch.arange(d_half, dtype=torch.float32) / self.head_dim)
        angles = torch.outer(torch.arange(config.max_temporal_length, dtype=torch.float32), freqs)
        self.register_buffer('cos', torch.cos(angles))
        self.register_buffer('sin', torch.sin(angles))
        # Non-persistent: not in state_dict, so old checkpoints still load cleanly.
        self.register_buffer('rope_freqs', freqs, persistent=False)

    def _rope_cos_sin(self, T: int, positions: torch.Tensor, dtype, device):
        """RoPE cos/sin for the time axis. positions=None -> the exact fixed-table path
        (positions 0..T-1), byte-identical to the original. positions given -> rotations
        computed on the fly at those ABSOLUTE positions (KV-cache / long-rollout path)."""
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
        # layer_cache: optional {'k','v'} dict of this temporal layer's cached (rotated) K/V.
        # commit: append this call's K/V to layer_cache (used to extend the cache).
        B, T, N, C = x.shape

        qkv: torch.Tensor = self.qkv(x)
        qkv = qkv.reshape((B, T, N, 3, self.n_heads, -1))
        if not self.is_temporal:
            qkv = qkv.permute((3, 4, 0, 1, 2, 5))  # (3, heads, B, T, N, head_dim)
        else:
            qkv = qkv.permute((3, 4, 0, 2, 1, 5))  # (3, heads, B, N, T, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)

        if not self.is_temporal:
            # Full self-attention within a frame: latents, registers, actions and the
            # shortcut token all exchange information. No time axis -> no RoPE, no cache.
            mask = None
            k_all, v_all = k, v
        else:
            # RoPE on the time axis (absolute positions; default 0..T-1 via table).
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

            # Causal mask (T_query, T_all): cached keys (the first T_all-T cols) are all earlier
            # frames -> visible; the new keys are causal among themselves. With no cache this
            # reduces to the original triu(T, T).
            T_all = k_all.shape[-2]
            mask = torch.zeros((T, T_all), dtype=torch.bool, device=x.device)
            mask[:, T_all - T:] = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        attn_scores = (q @ k_all.transpose(-2, -1)) * self.scale
        attn_scores = self.soft_cap_act(attn_scores / self.soft_cap) * self.soft_cap
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.att_droput(attn)

        x = attn @ v_all
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
    """
    Block-causal transformer that denoises per-frame latents via shortcut forcing.

    Token layout per frame (along the spatial axis): ``[action tokens | latent tokens |
    register tokens | shortcut token]``. Only the latent-token outputs are read out as the
    x-prediction of the clean representation.
    """

    def __init__(self, config: DynamicsModelConfig) -> None:
        super().__init__()
        self.config = config
        E = config.embedding_dim
        self.n_latents = config.n_latents
        self.bottleneck_dim = config.bottleneck_dim
        self.n_action_tokens = config.n_action_tokens
        self.n_registers = config.n_registers

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

        # Discrete action conditioning: map each action id to a per-frame feature that is added
        # to the learned action embedding. Absent (n_actions == 0) => unlabeled video.
        self.n_actions = config.n_actions
        if config.n_actions > 0:
            self.action_table = nn.Embedding(config.n_actions, config.n_action_tokens * E)

        # Discrete shortcut conditioning: tau and d each get half the channels, concatenated.
        assert E % 2 == 0
        self.tau_embedding = nn.Embedding(self.K_max, E // 2)
        self.d_embedding = nn.Embedding(self.n_d, E // 2)

        self.blocks = nn.ModuleList([
            TransformerBlock(config, is_temporal=((i + 1) % 4 == 0))
            for i in range(config.depth)
        ])

    # ------------------------------------------------------------------ helpers
    def _d_value(self, d_idx: torch.Tensor) -> torch.Tensor:
        # d = 1 / K, with K = 2 ** d_idx.
        return torch.pow(2.0, -d_idx.float())

    def _tau_value(self, tau_idx: torch.Tensor) -> torch.Tensor:
        # tau grid points are integer multiples of d_min = 1 / K_max.
        return tau_idx.float() / self.K_max

    def sample_tau_d(self, B: int, T: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample per-frame (tau_idx, d_idx) following Eq. 4.

        d ~ 1/U({1, 2, ..., K_max})  ->  d_idx = log2(K) ~ U({0, ..., n_d-1}).
        tau ~ U({0, d, 2d, ..., 1-d}) on the grid implied by the sampled d.
        """
        d_idx = torch.randint(0, self.n_d, (B, T), device=device)
        K = torch.pow(2, d_idx)                               # steps for this frame
        step = (torch.rand((B, T), device=device) * K).long()  # uniform in {0, ..., K-1}
        step = torch.minimum(step, K - 1)
        # tau_idx = step * (K_max / K), i.e. snap to the d_min grid.
        tau_idx = step * torch.pow(2, self.n_d - 1 - d_idx)
        return tau_idx, d_idx

    def action_features(self, action_idx: torch.Tensor) -> torch.Tensor:
        """(B, T) long action ids -> (B, T, n_action_tokens, E) feature tokens.

        Returns ``None`` when the model is unlabeled or no ids are given, so callers can pass the
        result straight through to ``forward`` (which then uses only the learned embedding).
        """
        if action_idx is None or self.n_actions == 0:
            return None
        B, T = action_idx.shape
        return self.action_table(action_idx).reshape(B, T, self.n_action_tokens, -1)

    # ------------------------------------------------------------------ forward
    def forward(self, z_tilde: torch.Tensor, tau_idx: torch.Tensor, d_idx: torch.Tensor,
                actions: torch.Tensor = None, register_in: torch.Tensor = None,
                return_registers: bool = False, positions: torch.Tensor = None,
                cache: list = None, commit: bool = False) -> torch.Tensor:
        """Predict clean latents (x-prediction) from noised latents.

        z_tilde:     (B, T, n_latents, bottleneck_dim) noised representations.
        tau_idx:     (B, T) long, discrete signal level per frame.
        d_idx:       (B, T) long, discrete step size per frame.
        actions:     optional (B, T, n_action_tokens, E) action features to add to the learned
                     action embedding. ``None`` => unlabeled video.
        register_in: optional (B, T, n_registers, E) register embeddings used INSTEAD of the
                     learned register tokens (FF7 memory injection, D-014). Callers that
                     inject at only some frames build the full tensor with
                     ``self.register_tokens`` at the remaining frames.
        return_registers: also return the final-layer register states (B, T, n_registers, E)
                     — the carrier states that generate_memory()/the FF7 loss re-inject.
        positions:   optional (T,) absolute time indices for these frames (KV-cache / long
                     rollout). ``None`` => the default 0..T-1 table path (training).
        cache:       optional per-block list (len = depth) of {'k','v'} dicts (None entries for
                     spatial blocks), produced by ``new_kv_cache()``. Temporal blocks read cached
                     K/V and, when ``commit`` is set, append this call's K/V.
        commit:      append this call's K/V to ``cache`` (used to extend the context cache).
        returns:     (B, T, n_latents, bottleneck_dim) predicted clean latents
                     [, (B, T, n_registers, E) register states].
        """
        B, T, L, _ = z_tilde.shape

        lat = self.in_proj(z_tilde)  # (B, T, L, E)

        shortcut = torch.concat(
            (self.tau_embedding(tau_idx), self.d_embedding(d_idx)), dim=-1
        ).unsqueeze(2)  # (B, T, 1, E)

        action = self.action_embedding.expand(B, T, -1, -1)
        if actions is not None:
            action = action + actions
        if register_in is None:
            register = self.register_tokens.expand(B, T, -1, -1)
        else:
            register = register_in

        x = torch.concat((action, lat, register, shortcut), dim=2)  # (B, T, N, E)

        for i, block in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None
            x = block(x, positions=positions, layer_cache=layer_cache, commit=commit)

        lat_start = self.n_action_tokens
        out = x[:, :, lat_start:lat_start + L, :]
        out = self.out_proj(self.out_norm(out))
        if return_registers:
            regs = x[:, :, lat_start + L:lat_start + L + self.n_registers, :]
            return out, regs
        return out

    # ------------------------------------------------------------------ loss
    def loss(self, z1: torch.Tensor, action_idx: torch.Tensor = None, ff7_k: int = 0,
             lambda_ff7: float = 1.0, return_parts: bool = False) -> torch.Tensor:
        """Shortcut forcing loss over a clip of clean representations.

        z1:         (B, T, n_latents, bottleneck_dim) clean tokenizer latents.
        action_idx: optional (B, T) long discrete action ids, aligned per frame.
        ff7_k:      FF7 lookahead depth (D-014). 0 = vanilla loss. >0 adds the
                    single-timestep-sufficiency term: the registers built by THIS forward
                    pass must, injected alone next to the real latent, predict the next
                    ``ff7_k`` frames (see _ff7_loss).
        lambda_ff7: weight of the FF7 term in the total.
        return_parts: also return {"diffusion": .., "ff7": ..} detached components.
        """
        B, T, L, _ = z1.shape
        device = z1.device
        actions = self.action_features(action_idx)

        tau_idx, d_idx = self.sample_tau_d(B, T, device)
        tau = self._tau_value(tau_idx)[..., None, None]  # (B, T, 1, 1)
        d = self._d_value(d_idx)[..., None, None]

        z0 = torch.randn_like(z1)
        z_tilde = (1 - tau) * z0 + tau * z1

        if ff7_k > 0:
            # x-prediction (with gradient) + the register states the FF7 term re-injects.
            z_hat1, regs = self(z_tilde, tau_idx, d_idx, actions, return_registers=True)
        else:
            z_hat1 = self(z_tilde, tau_idx, d_idx, actions)

        is_min = (d_idx == self.n_d - 1)[..., None, None]  # finest step => pure flow loss

        # --- bootstrap targets: distill two d/2 steps (Eq. 7), stop-gradient. ---
        with torch.no_grad():
            half_d_idx = (d_idx + 1).clamp(max=self.n_d - 1)
            half_d = self._d_value(half_d_idx)[..., None, None]
            # tau + d/2 on the d_min grid; clamp keeps masked (is_min) entries in range.
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

        if ff7_k == 0:
            if return_parts:
                return diffusion, {"diffusion": diffusion.detach()}
            return diffusion

        ff7 = self._ff7_loss(z1, regs, actions, ff7_k)
        total = diffusion + lambda_ff7 * ff7
        if return_parts:
            return total, {"diffusion": diffusion.detach(), "ff7": ff7.detach()}
        return total

    def _ff7_loss(self, z1: torch.Tensor, regs: torch.Tensor, actions: torch.Tensor,
                  k: int) -> torch.Tensor:
        """FF7 single-timestep-sufficiency loss (D-014, IDEAS.md FF7 v1).

        For every frame t with k successors in the clip, run one extra forward over the
        (k+1)-frame sequence [t, t+1, ..., t+k], folded into the batch dim:
          * frame t ("context"): the REAL clean latent z1_t held at tau_ctx (overwrite-real:
            kills the latent color path during occlusion) + register_t INJECTED from the
            main windowed pass — the only carrier of off-screen state.
          * frames t+1..t+k: real latents noised at per-frame sampled tau, finest d, plain
            flow loss with the ramp weight (no bootstrap).
        The loss backprops through the injected registers into the windowed pass that wrote
        them, training the write side. k >= 2 additionally trains the in-pass register relay
        (frame t+j's register channel forwarding the injected state).
        """
        B, T, L, D = z1.shape
        device = z1.device
        n_t = T - k
        assert n_t >= 1, f"clip length {T} too short for ff7_k={k}"
        E = regs.shape[-1]

        # (n_t, k+1) window indices -> gather (B, n_t, k+1, ...) and fold n_t into batch.
        idx = torch.arange(n_t, device=device)[:, None] + torch.arange(k + 1, device=device)[None, :]
        zw = z1[:, idx].reshape(B * n_t, k + 1, L, D)

        reg0 = regs[:, :n_t].reshape(B * n_t, 1, self.n_registers, E)
        reg_rest = self.register_tokens.expand(B * n_t, k, -1, -1)
        register_in = torch.concat((reg0, reg_rest), dim=1)

        tau_ctx_idx = min(round(self.config.context_signal * self.K_max), self.K_max - 1)
        tau_idx = torch.randint(0, self.K_max, (B * n_t, k + 1), device=device)
        tau_idx[:, 0] = tau_ctx_idx
        d_idx = torch.full((B * n_t, k + 1), self.n_d - 1, device=device, dtype=torch.long)

        tau = self._tau_value(tau_idx)[..., None, None]
        z_tilde = (1 - tau) * torch.randn_like(zw) + tau * zw

        act_in = None
        if actions is not None:
            act_in = actions[:, idx].reshape(B * n_t, k + 1, self.n_action_tokens, -1)

        z_hat = self(z_tilde, tau_idx, d_idx, act_in, register_in=register_in)

        flow = (z_hat[:, 1:] - zw[:, 1:]) ** 2
        w = (1 - self.config.ramp_min) * tau[:, 1:] + self.config.ramp_min
        return (w * flow).mean()

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def _denoise_next(self, context: torch.Tensor, K: int,
                      actions: torch.Tensor = None) -> torch.Tensor:
        """Generate one clean frame conditioned on ``context`` (clean latents), using K
        shortcut steps. ``context`` is held at signal level tau_ctx (=context_signal,
        near 1.0 = near-clean) so the model can read it; see D-010 / EXP-008.

        actions: optional (B, T_ctx + 1, n_action_tokens, E) action features covering the
                 context frames and the new frame being generated.
        """
        B, T_ctx, L, _ = context.shape
        device = context.device
        d_val = 1.0 / K
        d_idx_val = (K).bit_length() - 1  # log2(K)

        tau_ctx = self.config.context_signal
        tau_ctx_idx = round(tau_ctx * self.K_max)
        # Hold the (clean) context at signal level tau_ctx for this frame's sampling:
        # ctx = tau_ctx * context + (1 - tau_ctx) * noise. High tau_ctx => near-clean.
        ctx_noised = (1 - tau_ctx) * torch.randn_like(context) + tau_ctx * context

        d_col = torch.full((B, T_ctx + 1), d_idx_val, device=device, dtype=torch.long)
        tau_col = torch.full((B, T_ctx + 1), tau_ctx_idx, device=device, dtype=torch.long)

        z = torch.randn((B, 1, L, self.bottleneck_dim), device=device)  # pure noise, tau = 0
        for k in range(K):
            tau = k / K
            tau_col[:, -1] = round(tau * self.K_max)
            inp = torch.concat((ctx_noised, z), dim=1)
            z_hat1 = self(inp, tau_col, d_col, actions)[:, -1:]  # x-prediction for the new frame
            v = (z_hat1 - z) / (1 - tau)
            z = z + v * d_val
        return z  # (B, 1, L, bottleneck_dim)

    @torch.no_grad()
    def generate(self, context: torch.Tensor, n_generate: int, K: int = None,
                 action_idx: torch.Tensor = None) -> torch.Tensor:
        """Autoregressively roll out ``n_generate`` frames after ``context``.

        context:    (B, T_ctx, n_latents, bottleneck_dim) clean latents from the tokenizer.
        action_idx: optional (B, T_ctx + n_generate) long action ids for every frame in the
                    context and the frames to generate (required when the model is action
                    conditioned, so the curtain state of each generated frame is known).
        returns:    (B, n_generate, n_latents, bottleneck_dim) generated clean latents.
        """
        if getattr(self.config, "use_register_memory", False):
            return self.generate_memory(context, n_generate, K, action_idx)
        K = K or self.config.inference_steps
        max_ctx = self.config.max_temporal_length - 1
        T_ctx = context.shape[1]
        act_feat = self.action_features(action_idx)  # (B, T_total, n_act, E) or None
        seq = context
        generated = []
        for i in range(n_generate):
            window = seq[:, -max_ctx:]
            w = window.shape[1]
            new_idx = T_ctx + i               # absolute index of the frame being generated
            act_window = None
            if act_feat is not None:
                # action features for [window frames ... new frame], matching _denoise_next input
                act_window = act_feat[:, new_idx - w : new_idx + 1]
            nxt = self._denoise_next(window, K, act_window)
            generated.append(nxt)
            seq = torch.concat((seq, nxt), dim=1)
        return torch.concat(generated, dim=1)

    # ------------------------------------------------------------------ KV cache (inference)
    def new_kv_cache(self) -> list:
        """A fresh per-block KV-cache list: an empty {} for each temporal block, None for
        spatial blocks (which attend within a frame and never cache across time)."""
        return [({} if block.att.is_temporal else None) for block in self.blocks]

    def _denoise_next_cached(self, context: torch.Tensor, K: int,
                             actions: torch.Tensor = None,
                             positions: torch.Tensor = None) -> torch.Tensor:
        """KV-cached equivalent of ``_denoise_next``. The context frames are held at tau_ctx and
        are causal, so their per-layer K/V is constant across the K shortcut substeps: encode
        them ONCE into a cache (commit=True), then run only the single new frame each substep
        against the cache. Bit-for-bit identical to ``_denoise_next`` (same ctx_noised + z draws),
        at ~K x fewer temporal-attention FLOPs.

        positions: absolute time indices of [context frames..., new frame], length T_ctx+1.
                   The new frame sits at the LAST position, so it must be RoPE-rotated there —
                   which is why the cached path needs explicit absolute positions, not the table.
        """
        B, T_ctx, L, _ = context.shape
        device = context.device
        d_val = 1.0 / K
        d_idx_val = (K).bit_length() - 1
        tau_ctx = self.config.context_signal
        tau_ctx_idx = round(tau_ctx * self.K_max)
        ctx_noised = (1 - tau_ctx) * torch.randn_like(context) + tau_ctx * context

        if positions is None:
            positions = torch.arange(T_ctx + 1, device=device)
        ctx_pos, new_pos = positions[:T_ctx], positions[T_ctx:T_ctx + 1]

        # Prefill: encode the context once, populating the per-layer KV cache.
        cache = self.new_kv_cache()
        d_col_ctx = torch.full((B, T_ctx), d_idx_val, device=device, dtype=torch.long)
        tau_col_ctx = torch.full((B, T_ctx), tau_ctx_idx, device=device, dtype=torch.long)
        act_ctx = actions[:, :T_ctx] if actions is not None else None
        self(ctx_noised, tau_col_ctx, d_col_ctx, act_ctx,
             positions=ctx_pos, cache=cache, commit=True)

        # Substeps: only the new frame, attending to the cached context (commit=False).
        d_col_new = torch.full((B, 1), d_idx_val, device=device, dtype=torch.long)
        tau_col_new = torch.full((B, 1), tau_ctx_idx, device=device, dtype=torch.long)
        act_new = actions[:, T_ctx:T_ctx + 1] if actions is not None else None
        z = torch.randn((B, 1, L, self.bottleneck_dim), device=device)  # pure noise, tau = 0
        for k in range(K):
            tau = k / K
            tau_col_new[:, -1] = round(tau * self.K_max)
            z_hat1 = self(z, tau_col_new, d_col_new, act_new,
                          positions=new_pos, cache=cache, commit=False)
            v = (z_hat1 - z) / (1 - tau)
            z = z + v * d_val
        return z  # (B, 1, L, bottleneck_dim)

    @torch.no_grad()
    def generate_cached(self, context: torch.Tensor, n_generate: int, K: int = None,
                        action_idx: torch.Tensor = None) -> torch.Tensor:
        """KV-cached rollout. Same signature/semantics as ``generate`` and bit-for-bit identical
        to it (given the same RNG state): the cache is rebuilt per generated frame, so each
        frame's window is re-noised exactly as in the uncached path. The saving is reusing the
        context window's K/V across the K shortcut substeps that denoise one frame (the context
        is constant + causal, so its K/V is identical every substep) — NOT cross-frame
        persistence and NOT anything within a frame's jointly-denoised tokens. The FF7
        register-memory path is already window-1 (no cache benefit) and is dispatched unchanged."""
        if getattr(self.config, "use_register_memory", False):
            return self.generate_memory(context, n_generate, K, action_idx)
        K = K or self.config.inference_steps
        max_ctx = self.config.max_temporal_length - 1
        T_ctx = context.shape[1]
        act_feat = self.action_features(action_idx)
        seq = context
        generated = []
        for i in range(n_generate):
            window = seq[:, -max_ctx:]
            w = window.shape[1]
            new_idx = T_ctx + i
            act_window = None
            if act_feat is not None:
                act_window = act_feat[:, new_idx - w: new_idx + 1]
            positions = torch.arange(w + 1, device=context.device)
            nxt = self._denoise_next_cached(window, K, act_window, positions=positions)
            generated.append(nxt)
            seq = torch.concat((seq, nxt), dim=1)
        return torch.concat(generated, dim=1)

    def _noise_to_ctx(self, z: torch.Tensor) -> torch.Tensor:
        """Hold clean latents at signal level context_signal (near-1 = near-clean) so the
        model can read them as context. Shared by _denoise_next and the memory rollout."""
        t = self.config.context_signal
        return (1 - t) * torch.randn_like(z) + t * z

    @torch.no_grad()
    def memory_rollout_init(self, context: torch.Tensor, ctx_action_idx: torch.Tensor = None,
                            K: int = None) -> dict:
        """Seed an FF7 register-carry rollout from a context window (the prefix pass of
        generate_memory). The last context frame's final-layer register state seeds the carry.

        context:        (B, T_ctx, L, D) clean latents.
        ctx_action_idx: (B, T_ctx) long action ids for the context frames, or None.
        Returns an opaque state dict to feed to memory_rollout_step (carries reg_prev, z_prev,
        the previous action id, and the sampling constants). This is the stateful, open-ended
        form of the same relay generate_memory runs in a closed loop.
        """
        K = K or self.config.inference_steps
        B, T_ctx = context.shape[0], context.shape[1]
        d_idx_val = (K).bit_length() - 1
        tau_ctx_idx = min(round(self.config.context_signal * self.K_max), self.K_max - 1)
        act_feat = self.action_features(ctx_action_idx)  # (B, T_ctx, n_act, E) or None
        tau_col = torch.full((B, T_ctx), tau_ctx_idx, device=context.device, dtype=torch.long)
        d_col = torch.full((B, T_ctx), d_idx_val, device=context.device, dtype=torch.long)
        _, regs = self(self._noise_to_ctx(context), tau_col, d_col, act_feat, return_registers=True)
        return {
            "reg_prev": regs[:, -1:],            # (B, 1, R, E) carried register state
            "z_prev": context[:, -1:],           # (B, 1, L, D) carried clean latent
            "prev_action": (ctx_action_idx[:, -1:] if ctx_action_idx is not None else None),
            "K": K, "d_idx_val": d_idx_val, "tau_ctx_idx": tau_ctx_idx,
        }

    @torch.no_grad()
    def memory_rollout_step(self, state: dict, action_id=None, K: int = None):
        """Advance an FF7 register-carry rollout by one frame. ``action_id`` is the action for
        the NEW frame: an int, or a (B,) / (B,1) long tensor, or None (unlabeled model).
        Returns (z_new (B, 1, L, D), new_state)."""
        K = K or state["K"]
        reg_prev, z_prev = state["reg_prev"], state["z_prev"]
        B, _, L, D = z_prev.shape
        device = z_prev.device
        d_idx_val, tau_ctx_idx = state["d_idx_val"], state["tau_ctx_idx"]

        new_action = None
        act2 = None
        if self.n_actions > 0 and action_id is not None:
            if torch.is_tensor(action_id):
                new_action = action_id.reshape(B, 1).to(device=device, dtype=torch.long)
            else:
                new_action = torch.full((B, 1), int(action_id), device=device, dtype=torch.long)
            prev = state["prev_action"]
            prev = new_action if prev is None else prev.reshape(B, 1)
            act2 = self.action_features(torch.cat((prev, new_action), dim=1))  # (B,2,n_act,E)

        reg_in = torch.concat((reg_prev, self.register_tokens.expand(B, 1, -1, -1)), dim=1)
        zc = self._noise_to_ctx(z_prev)
        tau2 = torch.full((B, 2), tau_ctx_idx, device=device, dtype=torch.long)
        d2 = torch.full((B, 2), d_idx_val, device=device, dtype=torch.long)

        z = torch.randn((B, 1, L, D), device=device)  # pure noise, tau = 0
        for kstep in range(K):
            tau = kstep / K
            tau2[:, -1] = round(tau * self.K_max)
            z_hat1 = self(torch.concat((zc, z), dim=1), tau2, d2, act2, register_in=reg_in)[:, -1:]
            v = (z_hat1 - z) / (1 - tau)
            z = z + v * (1.0 / K)

        # Extract the new frame's register state (its clean latent held at tau_ctx) for the carry.
        tau2[:, -1] = tau_ctx_idx
        inp = torch.concat((zc, self._noise_to_ctx(z)), dim=1)
        _, regs = self(inp, tau2, d2, act2, register_in=reg_in, return_registers=True)
        new_state = {**state, "reg_prev": regs[:, -1:], "z_prev": z,
                     "prev_action": new_action if new_action is not None else state["prev_action"]}
        return z, new_state

    @torch.no_grad()
    def generate_memory(self, context: torch.Tensor, n_generate: int, K: int = None,
                        action_idx: torch.Tensor = None) -> torch.Tensor:
        """FF7 register-memory rollout (D-014): sequential window-1 generation that carries
        each frame's final-layer register state and injects it as the next step's context
        registers — matching the FF7 training interface (one injected context frame at
        tau_ctx + the frame to denoise at finest d).

        Vanilla generate() has no persistent state beyond the latent window; this is the
        param-free inference change that lets the trained register relay actually run.
        Same signature/return as generate(). action_idx: (B, T_ctx + n_generate) ids or None.
        Thin closed loop over memory_rollout_init/step (the open-ended interactive form).
        """
        P = context.shape[1]
        ctx_ids = action_idx[:, :P] if action_idx is not None else None
        state = self.memory_rollout_init(context, ctx_ids, K)
        generated = []
        for i in range(n_generate):
            new_id = action_idx[:, P + i] if action_idx is not None else None
            z, state = self.memory_rollout_step(state, new_id, K)
            generated.append(z)
        return torch.concat(generated, dim=1)
