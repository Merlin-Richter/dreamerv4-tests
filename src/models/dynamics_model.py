"""
Dreamer 4 interactive dynamics model (Section 3.2).

The dynamics model operates on the sequence of continuous representations produced by the
frozen causal tokenizer (the frozen tokenizer (`models.tokenizer`)). For every frame it consumes
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

    # FF9 full-state memory (D-024). A DISTINCT memory-token type (registers revert to pure scratch)
    # trained by the FF9 v2 memory-only-sufficiency loss: from the memory tokens ALONE (path latents
    # withheld via signal level tau=0), predict the next 1..j frames. n_memory=0 => byte-identical to a
    # pre-D-024 model (no memory tokens added). use_full_state_memory dispatches generate() to the
    # memory-carry rollout (analog of generate_memory but on the memory slot).
    n_memory: int = 0
    use_full_state_memory: bool = False
    ff9_k: int = 0                  # provenance: FF9 max lookahead used in training (0 = off).
    ff9_ramp: bool = False          # FF9 term ramp-weighting. Default OFF (un-ramped) so the
                                    # memory-bearing low-tau samples are not down-weighted (V-T013).

    # C1 multi-step motion loss (D-027). Time-axis DAgger/scheduled-sampling term: roll the model h
    # self-steps from a short real seed and supervise each predicted successor against GT, with the
    # context built from the model's OWN DETACHED predictions (TBPTT-1). Fixes autoregressive
    # compounding (EXP-018). multistep_h=0 => byte-identical to a pre-D-027 model (no extra loss, no
    # new params, inference/probe/FF7/FF9 untouched). Loss-only; verified V-T017-C1.
    multistep_h: int = 0            # provenance: C1 lookahead depth used in training (0 = off).

    # FF9 rollout-training / memory->memory relay (op-3, D-048, EXP-029 design note). Trains the
    # cross-window memory relay the FF9 v2 sufficiency loss leaves un-gradiented (_ff9_loss fills the
    # intermediate frames with the learned-init placeholder -> "write mem_{t+1} <- mem_t" is on no
    # gradient path; V-T014 / EXP-028 decay is the production echo). Seeds memory from a real near-clean
    # prefix, then rolls h hops as differentiable 2-frame [source|new] windows mirroring the inference
    # relay (full_state_rollout_step) — the WRITTEN memory at each hop is carried (injected) into the
    # next, so the loss backprops through the memory-write chain TBPTT-k hops. Per step, prob p_hide the
    # source latent is HIDDEN (tau=0 => memory is the ONLY scene carrier, the relay gradient); else the
    # true latent near-clean (re-anchor). Provenance fields; the live knobs are loss() args.
    ff9_rollout_h: int = 0          # rollout hops used in training (0 = off). 0 => no extra RNG/loss.
    ff9_rollout_tbptt: int = 4      # detach the carried memory every k hops (TBPTT-k truncation depth).
    ff9_rollout_p_hide: float = 0.5 # per-step prob the source latent is hidden (iid mode only).
    ff9_rollout_hide_mode: str = "tail"  # "tail" (contiguous hidden run, mirrors occlusion) | "iid".


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

        # Distinct FF9 memory tokens (D-024). Only instantiated when n_memory > 0 so that n_memory=0
        # checkpoints are byte-identical to pre-D-024 models. Placed AFTER registers in the token layout
        # so the register read-out offset (lat_start + L) is unchanged.
        self.n_memory = config.n_memory
        if config.n_memory > 0:
            self.memory_tokens = nn.Parameter(0.05 * torch.rand((config.n_memory, E)))

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
                cache: list = None, commit: bool = False,
                memory_in: torch.Tensor = None, return_memory: bool = False) -> torch.Tensor:
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

        # Token layout: [action | latents | registers | (memory) | shortcut]. Memory tokens are inserted
        # AFTER registers (so the register offset is unchanged) and only when n_memory > 0.
        parts = [action, lat, register]
        if self.n_memory > 0:
            if memory_in is None:
                memory = self.memory_tokens.expand(B, T, -1, -1)
            else:
                memory = memory_in
            parts.append(memory)
        parts.append(shortcut)
        x = torch.concat(parts, dim=2)  # (B, T, N, E)

        for i, block in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None
            x = block(x, positions=positions, layer_cache=layer_cache, commit=commit)

        lat_start = self.n_action_tokens
        out = x[:, :, lat_start:lat_start + L, :]
        out = self.out_proj(self.out_norm(out))
        if return_registers or return_memory:
            extras = []
            if return_registers:
                extras.append(x[:, :, lat_start + L:lat_start + L + self.n_registers, :])
            if return_memory:
                mem_start = lat_start + L + self.n_registers
                extras.append(x[:, :, mem_start:mem_start + self.n_memory, :])
            return (out, *extras)
        return out

    # ------------------------------------------------------------------ loss
    def loss(self, z1: torch.Tensor, action_idx: torch.Tensor = None, ff7_k: int = 0,
             lambda_ff7: float = 1.0, ff9_k: int = 0, lambda_ff9: float = 1.0,
             multistep_h: int = 0, lambda_multistep: float = 1.0,
             ff9_rollout_h: int = 0, lambda_ff9_rollout: float = 1.0,
             ff9_rollout_tbptt: int = None, ff9_rollout_p_hide: float = None,
             ff9_rollout_hide_mode: str = None,
             return_parts: bool = False) -> torch.Tensor:
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
        device = z1.device
        act_full = self.action_features(action_idx)
        # The main terms (diffusion/ff7/ff9/multistep) run within the model's temporal window W; the
        # FF9 rollout term may use a LONGER clip (T>W) to train the memory relay to deeper hops (the
        # rollout is window-2 internally, so it never exceeds the RoPE table). When T==W (the usual
        # case) z_full==z1 and act_full==actions -> byte-identical to before.
        W = self.config.max_temporal_length
        z_full, z1 = z1, z1[:, :W]
        actions = act_full[:, :W] if act_full is not None else None
        B, T, L, _ = z1.shape

        tau_idx, d_idx = self.sample_tau_d(B, T, device)
        tau = self._tau_value(tau_idx)[..., None, None]  # (B, T, 1, 1)
        d = self._d_value(d_idx)[..., None, None]

        z0 = torch.randn_like(z1)
        z_tilde = (1 - tau) * z0 + tau * z1

        want_reg, want_mem = ff7_k > 0, ff9_k > 0
        if want_reg or want_mem:
            # x-prediction (with gradient) + the carrier states the FF7/FF9 term re-injects.
            out = self(z_tilde, tau_idx, d_idx, actions,
                       return_registers=want_reg, return_memory=want_mem)
            z_hat1, _e = out[0], 1
            if want_reg:
                regs = out[_e]; _e += 1
            if want_mem:
                mem = out[_e]; _e += 1
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

        total = diffusion
        parts = {"diffusion": diffusion.detach()}
        if ff7_k > 0:
            ff7 = self._ff7_loss(z1, regs, actions, ff7_k)
            total = total + lambda_ff7 * ff7
            parts["ff7"] = ff7.detach()
        if ff9_k > 0:
            ff9 = self._ff9_loss(z1, mem, actions, ff9_k)
            total = total + lambda_ff9 * ff9
            parts["ff9"] = ff9.detach()
        # C1 (D-027): time-axis multi-step DAgger loss. MUST be last so all its RNG draws come after
        # the existing ones -> multistep_h=0 is byte-identical (V-T017-C1 C-D).
        if multistep_h > 0:
            ms, per_j = self._multistep_loss(z1, actions, multistep_h)
            total = total + lambda_multistep * ms
            parts["multistep"] = ms.detach()
            for j, v in enumerate(per_j, start=1):     # per-horizon, for the prior-emission monitor
                parts[f"ms_h{j}"] = v
        # FF9 rollout-training (D-048): memory->memory relay (op-3). Appended LAST -> all its RNG draws
        # come after the existing ones, so ff9_rollout_h=0 is byte-identical to a pre-D-048 model.
        if ff9_rollout_h > 0:
            tbptt = ff9_rollout_tbptt if ff9_rollout_tbptt is not None else self.config.ff9_rollout_tbptt
            p_hide = ff9_rollout_p_hide if ff9_rollout_p_hide is not None else self.config.ff9_rollout_p_hide
            hmode = ff9_rollout_hide_mode if ff9_rollout_hide_mode is not None else self.config.ff9_rollout_hide_mode
            ff9r, per_h = self._ff9_rollout_loss(z_full, act_full, ff9_rollout_h, tbptt, p_hide, hmode)
            total = total + lambda_ff9_rollout * ff9r
            parts["ff9_rollout"] = ff9r.detach()
            for j, v in enumerate(per_h, start=1):
                parts[f"ff9r_h{j}"] = v
        if return_parts:
            return total, parts
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

    def _ff9_loss(self, z1: torch.Tensor, mem: torch.Tensor, actions: torch.Tensor,
                  k: int) -> torch.Tensor:
        """FF9 v2 memory-only sufficiency loss (D-024, IDEAS.md FF9 v2; Merlin 2026-06-13).

        For every frame t with k successors, a (k+1)-frame mini-forward [t, t+1, ..., t+k] folded into
        the batch. Per window a horizon j ~ U{1..k} is sampled:
          * frames t .. t+j-1 (incl. the memory-source frame t): signal level tau=0 -> the latent slots
            are PURE NOISE, so NO ground-truth latent is anywhere on the path. memory_t (from the main
            windowed pass) is injected at frame t; learned-init memory at t+1..t+j-1 relays via the
            position-wise temporal memory channel. Memory is the ONLY scene carrier on the path.
          * frame t+j (terminal): a sampled tau (any level) so a well-posed denoising target exists and
            the memory-conditioned denoiser is calibrated; the low-tau samples force memory to be
            load-bearing (V-T013). It has no successors -> its signal leaks to nothing.
          * loss on frames t+1 .. t+j (each a memory-sufficiency target at its horizon), un-ramped by
            default. Frames > j are computed but masked out (causally they cannot affect frames <= j).
        Backprops through the injected memory_t into the windowed pass that wrote it (write-side credit,
        TBPTT-1). Within one window frame t+j attends DIRECTLY to frame t's memory -> trains memory as a
        sufficient full-state object, NOT the cross-window relay (that is option A, layered on next).
        """
        assert self.n_memory > 0, "ff9 requires n_memory > 0"
        B, T, L, D = z1.shape
        device = z1.device
        n_t = T - k
        assert n_t >= 1, f"clip length {T} too short for ff9_k={k}"
        M, E = mem.shape[-2], mem.shape[-1]
        BN = B * n_t

        # (n_t, k+1) window indices -> gather (B, n_t, k+1, ...) and fold n_t into the batch.
        idx = torch.arange(n_t, device=device)[:, None] + torch.arange(k + 1, device=device)[None, :]
        zw = z1[:, idx].reshape(BN, k + 1, L, D)

        # memory injected at frame 0 (the source), learned-init elsewhere.
        mem0 = mem[:, :n_t].reshape(BN, 1, M, E)
        mem_rest = self.memory_tokens.expand(BN, k, -1, -1)
        memory_in = torch.concat((mem0, mem_rest), dim=1)            # (BN, k+1, M, E)

        # per-window horizon j in {1..k}; tau=0 on the path, sampled tau on the terminal frame j.
        j = torch.randint(1, k + 1, (BN,), device=device)            # (BN,) in [1, k]
        tau_idx = torch.zeros(BN, k + 1, dtype=torch.long, device=device)
        tau_term = torch.randint(0, self.K_max, (BN,), device=device)
        tau_idx[torch.arange(BN, device=device), j] = tau_term       # scatter to the terminal slot
        d_idx = torch.full((BN, k + 1), self.n_d - 1, dtype=torch.long, device=device)  # finest d

        tau = self._tau_value(tau_idx)[..., None, None]
        z_tilde = (1 - tau) * torch.randn_like(zw) + tau * zw        # path frames (tau=0) => pure noise

        act_in = None
        if actions is not None:
            act_in = actions[:, idx].reshape(BN, k + 1, self.n_action_tokens, -1)

        z_hat = self(z_tilde, tau_idx, d_idx, act_in, memory_in=memory_in)

        # loss on frames 1..j (mask out frames > j); un-ramped by default.
        flow = ((z_hat[:, 1:] - zw[:, 1:]) ** 2).mean(dim=(-1, -2))  # (BN, k) per-frame MSE
        frame_pos = torch.arange(1, k + 1, device=device)[None, :]   # output positions = frames 1..k
        mask = (frame_pos <= j[:, None]).float()                     # (BN, k) active where frame <= j
        if self.config.ff9_ramp:
            flow = ((1 - self.config.ramp_min) * tau[:, 1:, 0, 0] + self.config.ramp_min) * flow
        return (flow * mask).sum() / mask.sum().clamp(min=1)

    def _ff9_rollout_loss(self, z1: torch.Tensor, actions: torch.Tensor, h: int,
                          tbptt_k: int, p_hide: float, hide_mode: str = "tail"):
        """FF9 rollout-training: train the memory->memory relay (op-3) — D-048, EXP-029 design (C1).

        `_ff9_loss` credits writing memory_t from latents (op-1) and reading memory_t j hops later
        (op-2) but fills the intermediate frames with the learned-init placeholder (line ~576), so the
        cross-window WRITE "memory_{t+1} <- memory_t" is on NO gradient path. This loss puts a REAL
        chain of memory writes on the gradient path:

          * SEED: write an initial memory token from a real near-clean prefix window (frames 0..seed-1),
            differentiable (also trains op-1).
          * ROLL h hops. Hop j predicts true frame seed+j from a 2-frame [source|new] window that
            MIRRORS inference (full_state_rollout_step): the carried memory injected at the SOURCE
            frame, learned-init memory at the NEW frame (it reads the carry via the position-wise
            temporal memory channel). The NEW frame's latent slot is pure noise (tau=0) and is the
            x-prediction flow target (newest-frame flow / memory-sufficiency). Per (sample, hop) a
            Bernoulli(p_hide) coin HIDES the source latent (tau=0, pure noise => memory is the ONLY
            scene carrier -> the memory-only relay gradient) or shows the true source latent near-clean
            (re-anchor to ground truth so a wrong guess can't compound forever). GT context is
            teacher-forced (the source is the true previous frame) so memory is the only recurrent
            element — isolating the relay from open-loop latent drift (V-T014 reader-anchor logic).
          * CARRY: the WRITTEN memory at the new frame becomes the next hop's injected memory — this is
            the op-3 relay, ON the gradient path. The carry is DETACHED every tbptt_k hops (TBPTT-k:
            keep at most k hops of memory-write graph; k from the EXP-029 P1 sweep, not guessed).

        Returns (mean_loss, per_hop detached). Deterministic GridWorld => a wrong prediction is always a
        genuine error (no valid-but-wrong branch), so the full-downstream loss is correct signal
        (design note 4: butterfly is a non-issue here). Requires n_memory>0; gated off by h=0.
        """
        assert self.n_memory > 0, "ff9_rollout requires n_memory > 0"
        B, T, L, D = z1.shape
        device = z1.device
        maxctx = self.config.max_temporal_length - 1
        seed = min(maxctx, 3)
        assert seed >= 2 and seed + h <= T, \
            f"clip length {T} too short for ff9_rollout_h={h} (need seed({seed})+h<=T)"
        tau_ctx_idx = min(round(self.config.context_signal * self.K_max), self.K_max - 1)
        d_fine = self.n_d - 1

        # --- SEED: write the initial memory from the real near-clean prefix (differentiable) ---
        win = z1[:, :seed]                                              # (B, seed, L, D)
        w = win.shape[1]
        tau_seed = torch.full((B, w), tau_ctx_idx, device=device, dtype=torch.long)
        d_seed = torch.full((B, w), d_fine, device=device, dtype=torch.long)
        act_seed = actions[:, :seed] if actions is not None else None
        _, mem = self(self._noise_to_ctx(win), tau_seed, d_seed, act_seed, return_memory=True)
        mem_carry = mem[:, -1:]                                        # (B, 1, M, E) initial state

        # --- ROLL h hops, each a 2-frame [source|new] window with the carried memory injected ---
        d2 = torch.full((B, 2), d_fine, device=device, dtype=torch.long)
        zero_tau = torch.zeros(B, dtype=torch.long, device=device)
        ctx_tau = torch.full((B,), tau_ctx_idx, dtype=torch.long, device=device)
        # Which source latents are HIDDEN (tau=0 => memory is the only carrier). Two modes:
        #  - "tail": a few visible anchor steps then a CONTIGUOUS hidden tail (start r~U[1,h-1]).
        #    Mirrors the recall eval (observe a prefix, then a contiguous occluded run to the reveal)
        #    and forces the relay to hold across many consecutive memory-only hops (the real test).
        #  - "iid": i.i.d. Bernoulli(p_hide) per step (design-note default; frequent re-anchoring,
        #    more stable but mostly short hidden runs).
        if hide_mode == "tail":
            r = torch.randint(1, max(2, h), (B,), device=device)      # 1..h-1 visible anchor steps
            hide = torch.arange(h, device=device)[None, :] >= r[:, None]
        else:
            hide = torch.rand(B, h, device=device) < p_hide           # per (sample, hop) hide coin
        learned_mem = self.memory_tokens.expand(B, 1, -1, -1)
        per_j = []
        for jj in range(h):
            tgt_pos = seed + jj
            hcoin = hide[:, jj][:, None, None, None]
            src_true = self._noise_to_ctx(z1[:, tgt_pos - 1:tgt_pos])  # true prev frame, near-clean
            src = torch.where(hcoin, torch.randn_like(src_true), src_true)
            tau_src = torch.where(hide[:, jj], zero_tau, ctx_tau)      # hidden=>0, visible=>near-clean
            z1_new = z1[:, tgt_pos:tgt_pos + 1]
            new_tilde = torch.randn_like(z1_new)                       # pure-noise target slot (tau=0)
            inp = torch.cat((src, new_tilde), dim=1)                   # (B, 2, L, D)
            tau2 = torch.stack((tau_src, zero_tau), dim=1)            # (B, 2)
            mem_in = torch.cat((mem_carry, learned_mem), dim=1)       # (B, 2, M, E)
            act2 = actions[:, tgt_pos - 1:tgt_pos + 1] if actions is not None else None
            z_hat, mem_out = self(inp, tau2, d2, act2, memory_in=mem_in, return_memory=True)
            per_j.append(((z_hat[:, -1:] - z1_new) ** 2).mean())      # newest-frame flow (tau=0)
            mem_carry = mem_out[:, -1:]                                # op-3 relay: written mem -> next
            if (jj + 1) % tbptt_k == 0:
                mem_carry = mem_carry.detach()                        # TBPTT-k truncation
        losses = torch.stack(per_j)                                   # (h,)
        return losses.mean(), losses.detach()

    def _multistep_loss(self, z1: torch.Tensor, actions: torch.Tensor, h: int):
        """C1 (D-027): time-axis multi-step DAgger / scheduled-sampling loss.

        For each anchor, seed a short REAL context (~the eval prefix length; >=2 frames so velocity is
        inferable), then roll the model h self-steps: each step predicts the TRUE successor z1[t+j]
        from a pure-noise (tau=0) target slot, GIVEN the model's OWN DETACHED self-generated context
        held near-clean at context_signal. The detach makes it TBPTT-1: the step-j gradient is
        bit-identical to teacher-forcing the map at the state the rollout actually visited (verified
        V-T017-C1 C-C(ii)), so this supervises the off-trajectory states open-loop visits and corrects
        autoregressive compounding (EXP-018). NB: the mechanism is on-policy distribution correction
        (DAgger), NOT a contraction map (V-T017-C1 C-A) — it helps iff the deficit is off-manifold
        ACCURACY, which EXP-018 shows is the ff7/ff9 case. Returns (mean_loss, per_horizon detached).
        """
        B, T, L, D = z1.shape
        device = z1.device
        maxctx = self.config.max_temporal_length - 1
        seed = min(maxctx, 3)                       # real seed frames/anchor (>=2 velocity; ~eval P=3)
        assert seed >= 2 and seed + h <= T, \
            f"clip length {T} too short for multistep_h={h} (need seed({seed})+h<=T)"
        n_a = T - seed - h + 1
        a = torch.arange(n_a, device=device)
        win_idx = a[:, None] + torch.arange(seed + h, device=device)[None, :]   # (n_a, seed+h)
        seq = z1[:, win_idx].reshape(B * n_a, seed + h, L, D)                   # (Bn, seed+h, L, D)
        Bn = seq.shape[0]
        act_seq = None
        if actions is not None:
            act_seq = actions[:, win_idx].reshape(Bn, seed + h, self.n_action_tokens, actions.shape[-1])
        tau_ctx_idx = min(round(self.config.context_signal * self.K_max), self.K_max - 1)
        d_fine = self.n_d - 1

        ctx = [seq[:, i:i + 1] for i in range(seed)]   # detached context buffer (real clean seed)
        per_j = []
        for jj in range(h):
            tgt_pos = seed + jj
            win = torch.cat(ctx[-maxctx:], dim=1)      # (Bn, w, L, D), all detached (TBPTT-1)
            w = win.shape[1]
            z_ctx = self._noise_to_ctx(win)            # hold context near-clean at context_signal
            tgt = torch.randn(Bn, 1, L, D, device=device)            # pure-noise target slot (tau=0)
            inp = torch.cat((z_ctx, tgt), dim=1)       # (Bn, w+1, L, D)
            tau_idx = torch.zeros(Bn, w + 1, dtype=torch.long, device=device)
            tau_idx[:, :w] = tau_ctx_idx
            d_idx = torch.full((Bn, w + 1), d_fine, dtype=torch.long, device=device)
            act_in = act_seq[:, tgt_pos - w:tgt_pos + 1] if act_seq is not None else None
            z_hat = self(inp, tau_idx, d_idx, act_in)[:, -1:]        # x-prediction of frame tgt_pos
            gt = seq[:, tgt_pos:tgt_pos + 1]
            per_j.append(((z_hat - gt) ** 2).mean())
            ctx.append(z_hat.detach())                 # detach + append -> next step reads self-output
        losses = torch.stack(per_j)                    # (h,)
        return losses.mean(), losses.detach()

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
        if getattr(self.config, "use_full_state_memory", False):
            return self.generate_full_state_memory(context, n_generate, K, action_idx)
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
                        action_idx: torch.Tensor = None, plain: bool = False) -> torch.Tensor:
        """KV-cached rollout. Same signature/semantics as ``generate`` and bit-for-bit identical
        to it (given the same RNG state): the cache is rebuilt per generated frame, so each
        frame's window is re-noised exactly as in the uncached path. The saving is reusing the
        context window's K/V across the K shortcut substeps that denoise one frame (the context
        is constant + causal, so its K/V is identical every substep) — NOT cross-frame
        persistence and NOT anything within a frame's jointly-denoised tokens. The FF7
        register-memory path is already window-1 (no cache benefit) and is dispatched unchanged.

        ``plain=True`` runs the standard sliding-window rollout EVEN for memory models, skipping the
        dispatch to generate_memory / generate_full_state_memory. The per-frame memory tokens are still
        present (memory_in=None -> learned-init each frame) and are carried across frames via the
        position-wise temporal attention / KV cache — i.e. the NORMAL rollout where memory tokens flow
        in the window like the frame latents (the intended FF9 inference), NOT the frozen-snapshot
        special case."""
        if not plain and getattr(self.config, "use_register_memory", False):
            return self.generate_memory(context, n_generate, K, action_idx)
        if not plain and getattr(self.config, "use_full_state_memory", False):
            return self.generate_full_state_memory(context, n_generate, K, action_idx)
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

    @torch.no_grad()
    def full_state_rollout_init(self, context: torch.Tensor, ctx_action_idx: torch.Tensor = None,
                                K: int = None, source_tau_idx: int = 0) -> dict:
        """Seed an FF9 v2 full-state-memory rollout (D-024). WRITE the memory snapshot ONCE from the
        observed prefix window (held near-clean at context_signal): the LAST frame's memory tokens
        become a static ``mem_carry`` (the model is trained to write a full-state object from a window
        of real latents; see ``_ff9_loss``). The snapshot is then carried frozen — this is the
        open-ended, stateful form of the closed loop in ``generate_full_state_memory`` (A1+B1,
        V-T013-eval). It draws no rollout noise itself (only the WRITE's context-noise).

        context:        (B, T_ctx, L, D) clean latents (the observed prefix; up to max_ctx are used).
        ctx_action_idx: (B, T_ctx) long action ids for the prefix frames, or None (unlabeled model).
        Returns an opaque state dict for ``full_state_rollout_step``.
        """
        assert self.n_memory > 0, "full_state_rollout_init requires n_memory > 0"
        K = K or self.config.inference_steps
        B, T_ctx, L, Dd = context.shape
        device = context.device
        d_idx_val = (K).bit_length() - 1
        tau_ctx_idx = min(round(self.config.context_signal * self.K_max), self.K_max - 1)
        max_ctx = self.config.max_temporal_length - 1

        # --- WRITE mem_carry ONCE from the observed prefix window (near-clean) ---
        win = context[:, -max_ctx:]
        w = win.shape[1]
        act_win = None
        if ctx_action_idx is not None and self.n_actions > 0:
            act_win = self.action_features(ctx_action_idx[:, -w:])
        tau_col = torch.full((B, w), tau_ctx_idx, device=device, dtype=torch.long)
        d_col = torch.full((B, w), d_idx_val, device=device, dtype=torch.long)
        _, mem = self(self._noise_to_ctx(win), tau_col, d_col, act_win, return_memory=True)
        mem_carry = mem[:, -1:]                                    # (B, 1, M, E) frozen snapshot

        # memory injected at the source frame; learned-init at the new frame.
        mem_in = torch.concat((mem_carry, self.memory_tokens.expand(B, 1, -1, -1)), dim=1)  # (B,2,M,E)
        s_val = float(self._tau_value(torch.tensor(source_tau_idx, device=device))) if source_tau_idx > 0 else 0.0
        return {
            "mem_in": mem_in, "K": K, "d_idx_val": d_idx_val,
            "source_tau_idx": source_tau_idx, "s_val": s_val,
            "prev_id": (ctx_action_idx[:, -1:] if ctx_action_idx is not None else None),
            "prev_lat": context[:, -1:], "L": L, "Dd": Dd,
        }

    @torch.no_grad()
    def full_state_rollout_step(self, state: dict, action_id=None, K: int = None):
        """Advance an FF9 v2 full-state-memory rollout by one frame. A 2-frame window
        ``[source | new]`` with the frozen ``mem_carry`` injected at the SOURCE frame (the new frame
        reads it through the position-wise temporal memory channel) and learned-init memory at the new
        frame; the source latent at signal ``source_tau_idx`` (default 0 = pure noise, **A1**, matching
        training — the scene comes ONLY from memory), the new frame denoised over K shortcut steps.
        ``mem_carry`` is NEVER updated (static carry, **B1**; re-extracting it is the untrained
        memory->memory relay (op-3) and drifts past the trained horizon — see V-T013-eval / V-T014).

        ``action_id`` is the action for the NEW frame: an int, a (B,) / (B,1) long tensor, or None.
        Returns (z_new (B, 1, L, D), new_state)."""
        K = K or state["K"]
        mem_in = state["mem_in"]
        B = mem_in.shape[0]
        L, Dd = state["L"], state["Dd"]
        device = mem_in.device
        d_idx_val, source_tau_idx = state["d_idx_val"], state["source_tau_idx"]

        act2, new_id = None, None
        if self.n_actions > 0 and action_id is not None:
            if torch.is_tensor(action_id):
                new_id = action_id.reshape(B, 1).to(device=device, dtype=torch.long)
            else:
                new_id = torch.full((B, 1), int(action_id), device=device, dtype=torch.long)
            prev_id = state["prev_id"]
            prev_id = new_id if prev_id is None else prev_id.reshape(B, 1)
            act2 = self.action_features(torch.concat((prev_id, new_id), dim=1))  # (B,2,n_act,E)

        d2 = torch.full((B, 2), d_idx_val, device=device, dtype=torch.long)
        # source frame at signal source_tau_idx (default 0 => pure noise, A1); new frame denoised.
        src = torch.randn((B, 1, L, Dd), device=device)
        if source_tau_idx > 0:                                     # A2: near-clean prev frame + noise
            src = (1 - state["s_val"]) * src + state["s_val"] * state["prev_lat"]
        tau2 = torch.zeros((B, 2), device=device, dtype=torch.long)
        tau2[:, 0] = source_tau_idx
        z = torch.randn((B, 1, L, Dd), device=device)
        for kstep in range(K):
            tau = kstep / K
            tau2[:, -1] = round(tau * self.K_max)
            z_hat1 = self(torch.concat((src, z), dim=1), tau2, d2, act2, memory_in=mem_in)[:, -1:]
            v = (z_hat1 - z) / (1 - tau)
            z = z + v * (1.0 / K)
        new_state = {**state, "prev_lat": z,
                     "prev_id": (new_id if new_id is not None else state["prev_id"])}
        return z, new_state

    @torch.no_grad()
    def generate_full_state_memory(self, context: torch.Tensor, n_generate: int, K: int = None,
                                   action_idx: torch.Tensor = None,
                                   source_tau_idx: int = 0) -> torch.Tensor:
        """FF9 v2 full-state-memory rollout (D-024). Beyond-window inference that carries a single
        memory snapshot, matching the FF9 v2 READ op exactly (verdict V-T013-eval = A1+B1). WRITE the
        snapshot once from the observed prefix, then carry it frozen and denoise each new frame from a
        2-frame ``[source | new]`` window with the snapshot injected at the source — see
        ``full_state_rollout_init`` / ``full_state_rollout_step`` for the per-op semantics.

        Carries STATIC hidden state (e.g. ball color) indefinitely past the N-frame window; precise
        dynamic position is NOT tracked (the snapshot is frozen) — that is the sequential relay's job
        (op-3, T-014). ``source_tau_idx``: 0 = A1 (FF9-faithful, the dispatch default); pass
        ``tau_ctx_idx`` for the A2 shape-matched-to-FF7 secondary. Same signature/return as generate().
        Thin closed loop over full_state_rollout_init/step (the open-ended interactive form).
        """
        assert self.n_memory > 0, "generate_full_state_memory requires n_memory > 0"
        P = context.shape[1]
        ctx_ids = action_idx[:, :P] if action_idx is not None else None
        state = self.full_state_rollout_init(context, ctx_ids, K, source_tau_idx)
        generated = []
        for i in range(n_generate):
            new_id = action_idx[:, P + i] if action_idx is not None else None
            z, state = self.full_state_rollout_step(state, new_id, K)
            generated.append(z)
        return torch.concat(generated, dim=1)

    @torch.no_grad()
    def generate_updating_memory(self, context: torch.Tensor, n_generate: int, K: int = None,
                                 action_idx: torch.Tensor = None) -> torch.Tensor:
        """FF9 rollout-training inference (D-048): the UPDATING memory relay (op-3 / B2), the
        inference `_ff9_rollout_loss` actually trains. Like generate_full_state_memory (A1) the scene
        comes ONLY from memory (source latent = pure noise, tau=0, matching the rollout-loss's hidden
        steps), BUT the memory is RE-WRITTEN every step instead of frozen: after generating frame t the
        new frame's written memory token becomes the carry for t+1. That is exactly the chain the
        rollout loss puts on the gradient path, so a model trained with it should track DYNAMIC hidden
        state past the window where the frozen snapshot (B1) only holds static state and the untrained
        relay (V-T013-eval B2) drifts. Same signature/return as generate().

        The memory carry is extracted at the tau=0 shortcut substep (new latent slot = pure noise),
        which is the exact configuration the rollout loss writes memory from — so train and inference
        write the carry identically. The frame itself is still denoised over the full K shortcut steps.
        """
        assert self.n_memory > 0, "generate_updating_memory requires n_memory > 0"
        K = K or self.config.inference_steps
        B, T_ctx, L, Dd = context.shape
        device = context.device
        d_idx_val = (K).bit_length() - 1
        tau_ctx_idx = min(round(self.config.context_signal * self.K_max), self.K_max - 1)
        max_ctx = self.config.max_temporal_length - 1
        act_feat = self.action_features(action_idx) if action_idx is not None else None

        # --- SEED the memory from the observed prefix window (near-clean), as in training's seed write
        win = context[:, -max_ctx:]
        w = win.shape[1]
        act_win = act_feat[:, T_ctx - w:T_ctx] if act_feat is not None else None
        tau_col = torch.full((B, w), tau_ctx_idx, device=device, dtype=torch.long)
        d_col = torch.full((B, w), d_idx_val, device=device, dtype=torch.long)
        _, mem = self(self._noise_to_ctx(win), tau_col, d_col, act_win, return_memory=True)
        mem_carry = mem[:, -1:]                                        # (B, 1, M, E)

        learned = self.memory_tokens.expand(B, 1, -1, -1)
        d2 = torch.full((B, 2), d_idx_val, device=device, dtype=torch.long)
        generated = []
        for i in range(n_generate):
            new_idx = T_ctx + i
            act2 = act_feat[:, new_idx - 1:new_idx + 1] if act_feat is not None else None  # [prev, new]
            mem_in = torch.cat((mem_carry, learned), dim=1)           # inject carry at SOURCE
            src = torch.randn((B, 1, L, Dd), device=device)           # A1: source = pure noise (tau=0)
            tau2 = torch.zeros((B, 2), device=device, dtype=torch.long)
            z = torch.randn((B, 1, L, Dd), device=device)
            new_mem = None
            for kstep in range(K):
                tau = kstep / K
                tau2[:, -1] = round(tau * self.K_max)
                if kstep == 0:                                        # tau=0 substep: write the carry
                    z_hat1, mem_out = self(torch.cat((src, z), dim=1), tau2, d2, act2,
                                           memory_in=mem_in, return_memory=True)
                    z_hat1 = z_hat1[:, -1:]
                    new_mem = mem_out[:, -1:]                         # new frame's written memory (op-3)
                else:
                    z_hat1 = self(torch.cat((src, z), dim=1), tau2, d2, act2, memory_in=mem_in)[:, -1:]
                v = (z_hat1 - z) / (1 - tau)
                z = z + v * (1.0 / K)
            generated.append(z)
            mem_carry = new_mem                                       # B2: UPDATE the carry each step
        return torch.concat(generated, dim=1)

    # ----------------------------------------- cross-frame sliding-window KV cache (D-020, T-012)
    @staticmethod
    def _make_noise_fn(seed: int):
        """A deterministic per-frame noise source addressed by (role, frame_id) — NOT by RNG call
        order. role 0 = a frame's generation seed (pure noise z), role 1 = a frame's context-noise.
        Because the draw is keyed on the absolute frame id, the cached rollout and the uncached
        windowed rollout get IDENTICAL noise for the same frame regardless of loop structure or
        caching — so a seeded cached-vs-uncached comparison isolates the cache (any divergence is a
        cache/eviction/RoPE bug, not a noise mismatch). See test_stream_cache.py."""
        def nf(role: int, frame_id: int, shape, device):
            g = torch.Generator(device=device)
            g.manual_seed(((int(seed) & 0xffffffff) * 1000003 + int(frame_id) * 9176
                           + int(role) * 131 + 1) & 0x7fffffffffffffff)
            return torch.randn(shape, generator=g, device=device)
        return nf

    def _ctx_noise_frame(self, z: torch.Tensor, frame_id: int, noise_fn) -> torch.Tensor:
        """Hold a clean frame at signal level context_signal, drawing its context-noise ONCE. With
        ``noise_fn`` (seeded) the draw is deterministic per frame_id; else a fresh randn_like. Same
        formula as _noise_to_ctx, but the noise is injectable so cached/uncached paths can match."""
        s = self.config.context_signal
        noise = torch.randn_like(z) if noise_fn is None else noise_fn(1, frame_id, z.shape, z.device)
        return (1 - s) * noise + s * z

    @staticmethod
    def _evict_oldest(cache: list) -> None:
        """Drop the oldest time-column from every temporal block's persistent KV cache. Because
        cached K/V are stored ALREADY-RoPE-rotated at absolute positions (T-008), eviction is a
        pure slice on the time axis (dim -2) — the surviving columns keep their correct absolute
        phase, no re-rotation. Spatial blocks (None) are skipped."""
        for lc in cache:
            if lc is not None and lc.get('k') is not None:
                lc['k'] = lc['k'][..., 1:, :]
                lc['v'] = lc['v'][..., 1:, :]

    @torch.no_grad()
    def stream_rollout_init(self, context: torch.Tensor, action_idx: torch.Tensor = None,
                            K: int = None, noise_seed: int = None) -> dict:
        """Seed a cross-frame sliding-window cached rollout (D-020) for efficient open-ended /
        continuous generation. Unlike generate_cached (which rebuilds the cache every frame), this
        PERSISTS finalized frames' K/V across rollout steps and evicts the oldest when the window
        (N-1) overflows — so an arbitrarily long rollout costs O(1) attention per step instead of
        re-encoding the whole window. Prefill: commit the context frames' K/V ONCE at context_signal.

        context:    (B, T_ctx, L, D) clean latents. action_idx: (B, T_ctx) ids for them, or None.
        Returns an opaque state dict for stream_rollout_step. NOTE on semantics: each frame's
        context-noise is drawn ONCE at commit (not redrawn every step like generate()) — a
        documented, defensible deviation (a committed frame's representation is fixed once
        generated). So NOT bit-identical to generate() at the rollout level; bit-identical to
        generate_windowed (the uncached twin) at the rollout level and to a full windowed recompute
        at the forward level.
        """
        K = K or self.config.inference_steps
        B, T_ctx, L, D = context.shape
        device = context.device
        W = self.config.max_temporal_length - 1            # max context frames in the window
        tau_ctx_idx = round(self.config.context_signal * self.K_max)
        d_idx_val = (K).bit_length() - 1

        nf = self._make_noise_fn(noise_seed) if noise_seed is not None else None
        cache = self.new_kv_cache()
        act_feat = self.action_features(action_idx)        # (B, T_ctx, n_act, E) or None
        if nf is None:
            ctx_noised = self._noise_to_ctx(context)       # default: one draw over all context frames
        else:                                              # seeded: per-frame, addressed by frame id
            ctx_noised = torch.cat([self._ctx_noise_frame(context[:, t:t + 1], t, nf)
                                    for t in range(T_ctx)], dim=1)
        tau_col = torch.full((B, T_ctx), tau_ctx_idx, device=device, dtype=torch.long)
        d_col = torch.full((B, T_ctx), d_idx_val, device=device, dtype=torch.long)
        positions = torch.arange(T_ctx, device=device)
        # Bulk-commit the context once (per-frame temporal K/V are independent, so bulk == one-by-one).
        self(ctx_noised, tau_col, d_col, act_feat, positions=positions, cache=cache, commit=True)
        cache_len = T_ctx
        while cache_len > W:                               # keep only the last W context frames
            self._evict_oldest(cache)
            cache_len -= 1
        return {
            "cache": cache, "next_pos": T_ctx, "cache_len": cache_len, "W": W,
            "K": K, "d_idx_val": d_idx_val, "tau_ctx_idx": tau_ctx_idx,
            "B": B, "L": L, "D": D, "device": device, "noise_fn": nf,
        }

    @torch.no_grad()
    def stream_rollout_step(self, state: dict, action_id=None, K: int = None):
        """Advance the cross-frame cached rollout by one frame (D-020). Denoise the new frame via
        K shortcut substeps against the persistent cache (commit=False), then commit the finalized
        frame's K/V at context_signal (commit=True) and evict the oldest if the window overflows.
        ``action_id`` is the NEW frame's action (int / (B,) / (B,1) tensor / None).
        Returns (z (B, 1, L, D), new_state). Mirrors _denoise_next_cached's substep math."""
        K = K or state["K"]
        cache, next_pos, W = state["cache"], state["next_pos"], state["W"]
        cache_len = state["cache_len"]
        d_idx_val, tau_ctx_idx = state["d_idx_val"], state["tau_ctx_idx"]
        B, L, D, device = state["B"], state["L"], state["D"], state["device"]
        nf = state.get("noise_fn")
        d_val = 1.0 / K

        new_pos = torch.tensor([next_pos], device=device)
        act_new = None
        if self.n_actions > 0 and action_id is not None:
            if torch.is_tensor(action_id):
                ids = action_id.reshape(B, 1).to(device=device, dtype=torch.long)
            else:
                ids = torch.full((B, 1), int(action_id), device=device, dtype=torch.long)
            act_new = self.action_features(ids)            # (B, 1, n_act, E)

        d_col = torch.full((B, 1), d_idx_val, device=device, dtype=torch.long)
        tau_col = torch.full((B, 1), tau_ctx_idx, device=device, dtype=torch.long)
        z = (torch.randn((B, 1, L, D), device=device) if nf is None
             else nf(0, next_pos, (B, 1, L, D), device))   # pure noise, tau = 0 (one draw / frame)
        for k in range(K):
            tau = k / K
            tau_col[:, -1] = round(tau * self.K_max)
            # commit=False: the new frame attends to the cached context + itself, cache untouched.
            z_hat1 = self(z, tau_col, d_col, act_new, positions=new_pos, cache=cache, commit=False)
            v = (z_hat1 - z) / (1 - tau)
            z = z + v * d_val

        # Commit the finalized frame at context_signal so it becomes context for the next step.
        tau_col[:, -1] = tau_ctx_idx
        commit_noised = (self._noise_to_ctx(z) if nf is None
                         else self._ctx_noise_frame(z, next_pos, nf))
        self(commit_noised, tau_col, d_col, act_new, positions=new_pos, cache=cache, commit=True)
        cache_len += 1
        while cache_len > W:
            self._evict_oldest(cache)
            cache_len -= 1
        new_state = {**state, "next_pos": next_pos + 1, "cache_len": cache_len}
        return z, new_state

    @torch.no_grad()
    def generate_streaming(self, context: torch.Tensor, n_generate: int, K: int = None,
                           action_idx: torch.Tensor = None, noise_seed: int = None) -> torch.Tensor:
        """Cross-frame sliding-window CACHED rollout (D-020): a thin loop over
        stream_rollout_init/step for efficient open-ended / continuous generation. Same
        signature/return as generate(); persists finalized frames' K/V across steps (no per-frame
        cache rebuild), so a long rollout is O(1) attention per step.
        Frozen per-frame context-noise (see stream_rollout_init) => NOT bit-identical to generate()
        but bit-identical to generate_windowed (the uncached twin) under the same ``noise_seed``
        (test_stream_cache.py — that comparison isolates the cache). ``noise_seed`` makes the rollout
        deterministic/reproducible; None = global-RNG draws. The FF7 register-memory path is window-1
        already and is dispatched to generate_memory unchanged."""
        if getattr(self.config, "use_register_memory", False):
            return self.generate_memory(context, n_generate, K, action_idx)
        if getattr(self.config, "use_full_state_memory", False):
            return self.generate_full_state_memory(context, n_generate, K, action_idx)
        P = context.shape[1]
        ctx_ids = action_idx[:, :P] if action_idx is not None else None
        state = self.stream_rollout_init(context, ctx_ids, K, noise_seed=noise_seed)
        generated = []
        for i in range(n_generate):
            new_id = action_idx[:, P + i] if action_idx is not None else None
            z, state = self.stream_rollout_step(state, new_id, K)
            generated.append(z)
        return torch.concat(generated, dim=1)

    @torch.no_grad()
    def generate_windowed(self, context: torch.Tensor, n_generate: int, K: int = None,
                          action_idx: torch.Tensor = None, noise_seed: int = None) -> torch.Tensor:
        """UNCACHED sliding-window rollout with the SAME frozen per-frame context-noise semantics as
        generate_streaming, implemented by full windowed recompute (no persistent cache). This is the
        non-cache twin used to validate the cache: with a shared ``noise_seed`` both paths draw
        identical per-frame noise (addressed by absolute frame id), so generate_streaming == this,
        bit-for-bit — any divergence is a cache/eviction/RoPE bug, not a noise mismatch. Independent
        stepping logic (a list of finalized frames, not a cache) so it also cross-checks the rollout
        bookkeeping. ``noise_seed=None`` still freezes each frame's context-noise (drawn once)."""
        if getattr(self.config, "use_register_memory", False):
            return self.generate_memory(context, n_generate, K, action_idx)
        if getattr(self.config, "use_full_state_memory", False):
            return self.generate_full_state_memory(context, n_generate, K, action_idx)
        K = K or self.config.inference_steps
        B, T_ctx, L, D = context.shape
        device = context.device
        W = self.config.max_temporal_length - 1
        tau_ctx_idx = round(self.config.context_signal * self.K_max)
        d_idx_val = (K).bit_length() - 1
        d_val = 1.0 / K
        nf = self._make_noise_fn(noise_seed) if noise_seed is not None else None
        has_act = self.n_actions > 0 and action_idx is not None

        # Finalized context-noised frames, indexed by absolute frame id (== list index).
        frames = [self._ctx_noise_frame(context[:, t:t + 1], t, nf) for t in range(T_ctx)]
        generated = []
        for i in range(n_generate):
            p = T_ctx + i                                  # absolute id of the frame being generated
            lo = max(0, len(frames) - W)                   # window start (== p - w)
            ctx = torch.concat(frames[lo:], dim=1)         # (B, w, L, D)
            w = ctx.shape[1]
            positions = torch.arange(p - w, p + 1, device=device)
            act_in = self.action_features(action_idx[:, p - w:p + 1]) if has_act else None
            tau_col = torch.full((B, w + 1), tau_ctx_idx, device=device, dtype=torch.long)
            d_col = torch.full((B, w + 1), d_idx_val, device=device, dtype=torch.long)
            z = (torch.randn((B, 1, L, D), device=device) if nf is None
                 else nf(0, p, (B, 1, L, D), device))
            for k in range(K):
                tau = k / K
                tau_col[:, -1] = round(tau * self.K_max)
                z_hat1 = self(torch.concat((ctx, z), dim=1), tau_col, d_col, act_in,
                              positions=positions)[:, -1:]
                v = (z_hat1 - z) / (1 - tau)
                z = z + v * d_val
            generated.append(z)
            frames.append(self._ctx_noise_frame(z, p, nf))  # freeze this frame's context-noise once
        return torch.concat(generated, dim=1)
