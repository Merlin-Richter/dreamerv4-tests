"""Probe: does the production carrying KV-cached rollout (generate) equal an uncached
recompute-the-current-window-each-step reference, with matched noise?

Prints per-generated-frame max-abs-diff so we can see WHERE (if anywhere) they diverge --
in particular whether sliding-window eviction + stacked temporal layers breaks bit-equivalence.
Not a gate test; a measurement harness to decide the real test's assertions.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from models.dynamics_model import DynamicsModel, DynamicsModelConfig  # noqa: E402


def make(**ov):
    base = dict(embedding_dim=32, n_heads=4, mlp_ratio=2.0, n_latents=2, bottleneck_dim=8,
                n_registers=2, max_sampling_steps=4, inference_steps=2, drop_rate=0.0,
                att_drop_rate=0.0)
    base.update(ov)
    m = DynamicsModel(DynamicsModelConfig(**base)).eval()
    return m


@torch.no_grad()
def uncached_generate(model, context, n_generate, K=None, action_idx=None):
    """Recompute the full sliding window through forward(cache=None) each step, within-window RoPE
    (positions=None), sliding-window eviction. Draws noise in the SAME order as production generate()
    so a shared manual_seed makes the noise identical (no cache is the ONLY difference)."""
    cfg = model.config
    K = K or cfg.inference_steps
    B, T_ctx = context.shape[:2]
    L, D = model.n_latents, model.bottleneck_dim
    device = context.device
    max_ctx = cfg.max_temporal_length - 1
    d_idx_val = K.bit_length() - 1
    tau_ctx_idx = min(round(cfg.context_signal * model.K_max), model.K_max - 1)

    ctx_act_idx = action_idx[:, :T_ctx] if action_idx is not None else None
    ctx_noised = model._noise_to_ctx(context)                      # DRAW: randn_like(context)
    tau_col = torch.full((B, T_ctx), tau_ctx_idx, device=device, dtype=torch.long)
    d_col = torch.full((B, T_ctx), d_idx_val, device=device, dtype=torch.long)
    mem_ctx = None
    if model.n_memory > 0:
        _, mem_ctx = model(ctx_noised, tau_col, d_col, model.action_features(ctx_act_idx),
                           positions=None, return_memory=True)

    win_repr = [ctx_noised[:, t:t + 1] for t in range(T_ctx)]
    win_mem = [(mem_ctx[:, t:t + 1] if mem_ctx is not None else None) for t in range(T_ctx)]
    win_act = [(ctx_act_idx[:, t:t + 1] if ctx_act_idx is not None else None) for t in range(T_ctx)]
    win_repr, win_mem, win_act = win_repr[-max_ctx:], win_mem[-max_ctx:], win_act[-max_ctx:]

    learned_mem = model.memory_tokens.expand(B, 1, -1, -1) if model.n_memory > 0 else None

    def run_window(z_new, tau_new_idx, a_new_idx, return_memory):
        W = len(win_repr)
        reprs = torch.cat(win_repr + [z_new], dim=1)
        taus = torch.full((B, W + 1), tau_ctx_idx, device=device, dtype=torch.long)
        taus[:, -1] = tau_new_idx
        ds = torch.full((B, W + 1), d_idx_val, device=device, dtype=torch.long)
        mems = torch.cat(win_mem + [learned_mem], dim=1) if model.n_memory > 0 else None
        if action_idx is not None and model.n_actions > 0:
            a_idx = torch.cat(win_act + [a_new_idx], dim=1)
            acts = model.action_features(a_idx)
        else:
            acts = None
        out = model(reprs, taus, ds, acts, memory_in=mems, positions=None,
                    return_memory=return_memory)
        if return_memory:
            zh, mo = out
            return zh[:, -1:], mo[:, -1:]
        return out[:, -1:]

    out_frames = []
    for i in range(n_generate):
        a_new = action_idx[:, T_ctx + i:T_ctx + i + 1] if action_idx is not None else None
        z = torch.randn((B, 1, L, D), device=device)              # DRAW: frame init
        written_mem = None
        for step in range(K):
            tau = step / K
            tau_idx = round(tau * model.K_max)
            last = step == K - 1
            if last and model.n_memory > 0:
                z_hat1, written_mem = run_window(z, tau_idx, a_new, return_memory=True)
            else:
                z_hat1 = run_window(z, tau_idx, a_new, return_memory=False)
            v = (z_hat1 - z) / (1 - tau)
            z = z + v * (1.0 / K)
        frame_noised = model._noise_to_ctx(z)                     # DRAW: commit noise
        win_repr.append(frame_noised)
        win_mem.append(written_mem)
        win_act.append(a_new)
        if len(win_repr) > max_ctx:
            win_repr, win_mem, win_act = win_repr[-max_ctx:], win_mem[-max_ctx:], win_act[-max_ctx:]
        out_frames.append(z)
    return torch.cat(out_frames, dim=1)


def probe(tag, depth, max_temporal_length, n_actions=0, T_ctx=2, n_gen=10, extra=None):
    extra = extra or {}
    model = make(depth=depth, max_temporal_length=max_temporal_length, n_actions=n_actions, **extra)
    B = 2
    L, D = model.n_latents, model.bottleneck_dim
    max_ctx = model.config.max_temporal_length - 1

    ctx = torch.randn(B, T_ctx, L, D)
    aidx = torch.randint(0, max(1, n_actions), (B, T_ctx + n_gen)) if n_actions > 0 else None

    torch.manual_seed(1234)
    cached = model.generate(ctx, n_gen, action_idx=aidx)
    torch.manual_seed(1234)
    uncached = uncached_generate(model, ctx, n_gen, action_idx=aidx)

    print(f"\n=== {tag}  (max_ctx={max_ctx}, T_ctx={T_ctx}, n_gen={n_gen}, depth={depth}, "
          f"n_memory={model.n_memory}, n_actions={n_actions}) ===")
    per_frame = (cached - uncached).abs().amax(dim=(0, 2, 3))  # (n_gen,)
    evict_onset = max_ctx - T_ctx  # gen-frame index at which the window first drops a committed frame
    for i, d in enumerate(per_frame.tolist()):
        marker = "  <-- first frame generated AFTER an eviction" if i == evict_onset + 1 else ""
        print(f"  gen[{i:2d}] maxdiff={d:.3e}{marker}")
    print(f"  OVERALL maxdiff={per_frame.max().item():.3e}")
    return per_frame, evict_onset


if __name__ == "__main__":
    probe("vanilla depth6 (2 temporal)", depth=6, max_temporal_length=6)
    probe("vanilla depth9 (3 temporal)", depth=9, max_temporal_length=6)
    probe("memory depth6", depth=6, max_temporal_length=6, extra=dict(n_memory=2, ff9_k=1))
    probe("labeled depth6", depth=6, max_temporal_length=6, n_actions=3)
    # Fully within the window for the whole rollout (T_ctx + n_gen <= max_temporal_length): never evicts.
    pf, _ = probe("vanilla NO eviction (stays in window)", depth=9, max_temporal_length=14, n_gen=8)
    print(f"  -> no-eviction OVERALL maxdiff = {pf.max().item():.3e} (expect ~0)")
