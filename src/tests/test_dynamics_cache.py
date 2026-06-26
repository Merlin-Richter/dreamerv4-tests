"""Cache-equivalence gate tests for the carrying KV-cached rollout (spec: models/dynamics_model.md §4).

The only inference path (`generate` / `rollout_init` / `rollout_step`) always uses an across-time KV
cache: a frame's K diffusion steps only READ committed past K/V, and a 5th pass commits the frame's K/V
(near-clean + written memory). RoPE is rotated at the ABSOLUTE rollout index, so the cached path should
equal an uncached recompute of the *current sliding window* through `forward(cache=None)` — that is the
property these tests pin down, against a from-scratch uncached reference with matched noise.

WHAT WE FOUND (and what these tests now lock in):
  * WITHIN the window (rollout never exceeds `max_temporal_length`): cached == uncached up to fp
    (~1e-6). The absolute-index RoPE + KV cache is a CORRECT optimization here. This is the hard gate.
  * ACROSS sliding-window EVICTION: cached and the uncached current-window recompute DIVERGE
    materially (O(0.1-0.7), not fp). Root cause: with >=2 stacked temporal layers, each committed
    frame's frozen deep-temporal-layer K/V encodes its COMMIT-TIME receptive field; after eviction the
    current window no longer holds some of those frames, so a fresh windowed recompute cannot reproduce
    the frozen K/V. So the cache is NOT a pure optimization once the rollout passes the window — it
    implements sliding-window-attention-with-frozen-cache semantics, not windowed-recompute semantics.
    The recall eval runs long occluded rollouts that ALWAYS exceed the window, so it lives in this
    regime. These tests CHARACTERIZE the divergence (exact before the first post-eviction frame,
    material after) rather than masking it with a loose tolerance. See agent/EXPERIMENTS.md (cache-equiv)
    and HOWTO/rope_kv_cache_caveat.md. Flagged to Merlin: design decision, not silently "fixed" here.

Run:  python src/tests/test_dynamics_cache.py   (CPU only, fp32).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.dynamics_model import DynamicsModel, DynamicsModelConfig  # noqa: E402

# Small, fp32, dropout off. depth=6 -> temporal at i%3==1 = {1,4} (2 temporal layers, enough to
# exhibit the stacked-layer effect); depth=9 = production cadence (3 temporal layers).
SMALL = dict(embedding_dim=32, n_heads=4, mlp_ratio=2.0, n_latents=2, bottleneck_dim=8,
             n_registers=2, max_sampling_steps=4, inference_steps=2, drop_rate=0.0, att_drop_rate=0.0)
WITHIN_TOL = 1e-3   # fp headroom for within-window equivalence (observed ~5e-6).
DIVERGE_MIN = 1e-2  # post-eviction divergence is orders above this (observed >0.1).


def make(**ov):
    cfg = dict(SMALL); cfg.update(ov)
    return DynamicsModel(DynamicsModelConfig(**cfg)).eval()


class UncachedRollout:
    """Reference rollout that mirrors rollout_init / rollout_step but uses NO cache: each step it
    recomputes the full current sliding window through forward(cache=None) with within-window RoPE
    (positions=None) and the same sliding-window eviction. It draws noise in the SAME order as the
    production path (init: one randn_like(context); per step: randn(frame) then, on commit, the
    commit randn_like via _noise_to_ctx), so seeding both paths with one manual_seed makes the noise
    identical — the cache is then the ONLY difference between the two rollouts."""

    def __init__(self, model: DynamicsModel, context: torch.Tensor, K=None,
                 ctx_action_idx: torch.Tensor = None):
        self.m = model
        cfg = model.config
        self.K = K or cfg.inference_steps
        self.B, T_ctx = context.shape[:2]
        self.device = context.device
        self.max_ctx = cfg.max_temporal_length - 1
        self.d_idx_val = self.K.bit_length() - 1
        self.tau_ctx_idx = min(round(cfg.context_signal * model.K_max), model.K_max - 1)

        ctx_noised = model._noise_to_ctx(context)                      # DRAW: randn_like(context)
        tau_col = torch.full((self.B, T_ctx), self.tau_ctx_idx, device=self.device, dtype=torch.long)
        d_col = torch.full((self.B, T_ctx), self.d_idx_val, device=self.device, dtype=torch.long)
        mem_ctx = None
        if model.n_memory > 0:
            _, mem_ctx = model(ctx_noised, tau_col, d_col, model.action_features(ctx_action_idx),
                               positions=None, return_memory=True)
        self.win_repr = [ctx_noised[:, t:t + 1] for t in range(T_ctx)]
        self.win_mem = [(mem_ctx[:, t:t + 1] if mem_ctx is not None else None) for t in range(T_ctx)]
        self.win_act = [(ctx_action_idx[:, t:t + 1] if ctx_action_idx is not None else None)
                        for t in range(T_ctx)]
        self._evict()

    def _evict(self):
        if len(self.win_repr) > self.max_ctx:
            self.win_repr = self.win_repr[-self.max_ctx:]
            self.win_mem = self.win_mem[-self.max_ctx:]
            self.win_act = self.win_act[-self.max_ctx:]

    def _run_window(self, z_new, tau_new_idx, a_new_idx, return_memory):
        m, B = self.m, self.B
        W = len(self.win_repr)
        reprs = torch.cat(self.win_repr + [z_new], dim=1)
        taus = torch.full((B, W + 1), self.tau_ctx_idx, device=self.device, dtype=torch.long)
        taus[:, -1] = tau_new_idx
        ds = torch.full((B, W + 1), self.d_idx_val, device=self.device, dtype=torch.long)
        if m.n_memory > 0:
            learned = m.memory_tokens.expand(B, 1, -1, -1)
            mems = torch.cat(self.win_mem + [learned], dim=1)
        else:
            mems = None
        if m.n_actions > 0 and a_new_idx is not None:
            acts = m.action_features(torch.cat(self.win_act + [a_new_idx], dim=1))
        else:
            acts = None
        out = m(reprs, taus, ds, acts, memory_in=mems, positions=None, return_memory=return_memory)
        if return_memory:
            zh, mo = out
            return zh[:, -1:], mo[:, -1:]
        return out[:, -1:]

    def step(self, a_new_idx=None, commit=True):
        m = self.m
        z = torch.randn((self.B, 1, m.n_latents, m.bottleneck_dim), device=self.device)  # DRAW: frame
        written_mem = None
        for s in range(self.K):
            tau = s / self.K
            tau_idx = round(tau * m.K_max)
            if s == self.K - 1 and m.n_memory > 0:
                z_hat1, written_mem = self._run_window(z, tau_idx, a_new_idx, return_memory=True)
            else:
                z_hat1 = self._run_window(z, tau_idx, a_new_idx, return_memory=False)
            z = z + (z_hat1 - z) / (1 - tau) * (1.0 / self.K)
        if commit:
            self.win_repr.append(m._noise_to_ctx(z))                   # DRAW: commit randn_like(z)
            self.win_mem.append(written_mem)
            self.win_act.append(a_new_idx)
            self._evict()
        return z


def _drive(model, ctx, n_gen, action_idx, commit_flags, seed=1234):
    """Run cached (production) and uncached (reference) rollouts over the same step schedule and
    return (cached_outs, uncached_outs) as (B, n_gen, L, D). commit_flags[i] selects commit vs the
    read-only branch for generated frame i. One manual_seed per path => identical noise."""
    T_ctx = ctx.shape[1]
    ctx_act = action_idx[:, :T_ctx] if action_idx is not None else None

    torch.manual_seed(seed)
    state = model.rollout_init(ctx, ctx_act, K=None)
    cached = []
    for i in range(n_gen):
        a = action_idx[:, T_ctx + i:T_ctx + i + 1] if action_idx is not None else None
        cached.append(model.rollout_step(state, a, commit=commit_flags[i]))

    torch.manual_seed(seed)
    ref = UncachedRollout(model, ctx, K=None, ctx_action_idx=ctx_act)
    uncached = []
    for i in range(n_gen):
        a = action_idx[:, T_ctx + i:T_ctx + i + 1] if action_idx is not None else None
        uncached.append(ref.step(a, commit=commit_flags[i]))
    return torch.cat(cached, dim=1), torch.cat(uncached, dim=1)


CONFIGS = [
    ("vanilla d6", dict(depth=6, max_temporal_length=6), 0),
    ("vanilla d9", dict(depth=9, max_temporal_length=6), 0),
    ("memory  d6", dict(depth=6, max_temporal_length=6, n_memory=2, ff9_k=1), 0),
    ("labeled d6", dict(depth=6, max_temporal_length=6, n_actions=3), 3),
]


@torch.no_grad()
def test_within_window_exact():
    """No eviction (T_ctx + n_gen <= max_temporal_length): cached == uncached up to fp. The cache +
    absolute-index RoPE is a correct optimization within a window. Vanilla, memory, labeled."""
    B, T_ctx = 2, 2
    for tag, ov, n_act in CONFIGS:
        m = make(**ov)
        n_gen = m.config.max_temporal_length - T_ctx  # exactly fills the window, never evicts
        L, D = m.n_latents, m.bottleneck_dim
        ctx = torch.randn(B, T_ctx, L, D)
        aidx = torch.randint(0, n_act, (B, T_ctx + n_gen)) if n_act > 0 else None
        cached, uncached = _drive(m, ctx, n_gen, aidx, [True] * n_gen)
        md = (cached - uncached).abs().max().item()
        assert md < WITHIN_TOL, f"{tag}: within-window cached!=uncached, maxdiff={md:.3e}"
    print(f"[ok] within-window: cached == uncached up to fp ({len(CONFIGS)} configs, tol {WITHIN_TOL})")


@torch.no_grad()
def test_readonly_branch_matches_uncached():
    """The read-only branch rollout_step(commit=False) (used by the recall reveal) predicts the next
    frame WITHOUT mutating the carried cache. Within the window it must equal the uncached read-only
    prediction, AND must not desync the rollout (subsequent committed frames still match)."""
    B, T_ctx = 2, 2
    for tag, ov, n_act in CONFIGS:
        m = make(**ov)
        n_gen = m.config.max_temporal_length - T_ctx
        ro_at = n_gen // 2  # a read-only peek partway through, still within the window
        flags = [i != ro_at for i in range(n_gen)]  # commit everywhere except the read-only frame
        L, D = m.n_latents, m.bottleneck_dim
        ctx = torch.randn(B, T_ctx, L, D)
        aidx = torch.randint(0, n_act, (B, T_ctx + n_gen)) if n_act > 0 else None
        cached, uncached = _drive(m, ctx, n_gen, aidx, flags)
        md = (cached - uncached).abs().max().item()
        assert md < WITHIN_TOL, f"{tag}: read-only/commit mix cached!=uncached, maxdiff={md:.3e}"
    print(f"[ok] read-only branch (commit=False) == uncached read-only, no desync ({len(CONFIGS)} configs)")


@torch.no_grad()
def test_eviction_divergence_characterized():
    """Through sliding-window eviction the cached rollout and an uncached current-window recompute
    DIVERGE (documented finding). Lock in the exact structure: bit-exact for every frame generated
    up to and including the one whose commit first triggers eviction, material divergence after."""
    B, T_ctx = 2, 2
    for tag, ov, n_act in CONFIGS:
        m = make(**ov)
        max_ctx = m.config.max_temporal_length - 1
        n_gen = max_ctx + 6  # well past the window
        i0 = max_ctx - T_ctx  # gen[i0] is the last exact frame; gen[i0+1] is first post-eviction frame
        L, D = m.n_latents, m.bottleneck_dim
        ctx = torch.randn(B, T_ctx, L, D)
        aidx = torch.randint(0, n_act, (B, T_ctx + n_gen)) if n_act > 0 else None
        cached, uncached = _drive(m, ctx, n_gen, aidx, [True] * n_gen)
        per_frame = (cached - uncached).abs().amax(dim=(0, 2, 3))  # (n_gen,)
        pre = per_frame[:i0 + 1].max().item()
        post = per_frame[i0 + 1:].max().item()
        assert pre < WITHIN_TOL, f"{tag}: pre-eviction not exact, maxdiff={pre:.3e}"
        assert post > DIVERGE_MIN, (
            f"{tag}: expected post-eviction divergence (>{DIVERGE_MIN}); got {post:.3e}. If this "
            f"dropped to ~0 the rollout semantics changed (now a true windowed recompute?) — update "
            f"this test and the finding in agent/EXPERIMENTS.md / HOWTO/rope_kv_cache_caveat.md.")
    print(f"[ok] eviction: exact through first-evicting frame, then diverges (characterized, {len(CONFIGS)} configs)")


@torch.no_grad()
def test_single_temporal_layer_stays_exact_through_eviction():
    """Root-cause discriminator: with ONE temporal layer (depth=3 -> temporal only at i=1), each
    frame's K/V at that layer is a function of its own spatial-only representation (window-independent),
    so the frozen-receptive-field effect cannot occur and cached == uncached EVEN through eviction.
    This pins the multi-temporal-layer stacking as the cause of the divergence above."""
    B, T_ctx = 2, 2
    m = make(depth=3, max_temporal_length=5)  # temporal at i=1 only
    max_ctx = m.config.max_temporal_length - 1
    n_gen = max_ctx + 6
    L, D = m.n_latents, m.bottleneck_dim
    ctx = torch.randn(B, T_ctx, L, D)
    cached, uncached = _drive(m, ctx, n_gen, None, [True] * n_gen)
    md = (cached - uncached).abs().max().item()
    assert md < WITHIN_TOL, (
        f"single-temporal-layer model should stay exact through eviction, got maxdiff={md:.3e}")
    print(f"[ok] single temporal layer (depth=3): cached == uncached THROUGH eviction (maxdiff {md:.1e})")


if __name__ == "__main__":
    test_within_window_exact()
    test_readonly_branch_matches_uncached()
    test_eviction_divergence_characterized()
    test_single_temporal_layer_stays_exact_through_eviction()
    print("\nALL CACHE-EQUIVALENCE TESTS PASSED")
