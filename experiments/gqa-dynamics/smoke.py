"""GQA correctness smoke (run BEFORE training): the attention forward was rewritten, so verify
the three properties that would silently corrupt results if wrong.

1. CAUSALITY: perturbing frame t's inputs must not change any output at frames < t (eval mode).
2. CACHE EQUIVALENCE: rollout-style committed incremental forwards must equal the one-shot
   uncached forward on the same near-clean window (fp32, within-window — the regime V-cache-equiv
   proved exact for the base model).
3. FOOTPRINT: the carried KV cache after rollout_init must be exactly 1/4 the bytes of the base
   (MHA) model's cache at the identical config/context.

Usage:  venv/Scripts/python.exe -u experiments/gqa-dynamics/smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "gqa-dynamics"))
from model import DynamicsModelGQA  # noqa: E402
from models.dynamics_model import DynamicsModel, DynamicsModelConfig  # noqa: E402

torch.manual_seed(0)
CFG = DynamicsModelConfig(n_actions=2)  # GridWorld-like: 256/9/16, W=16, L=4, D=64


def cache_bytes(state):
    return sum(t.numel() * t.element_size()
               for lc in state["cache"] if lc is not None
               for t in (lc["k"], lc["v"]) if t is not None)


def main():
    B, T, L, D = 2, 8, CFG.n_latents, CFG.bottleneck_dim
    m = DynamicsModelGQA(CFG).eval()
    z = torch.randn(B, T, L, D)
    tau = torch.full((B, T), 100, dtype=torch.long)
    d = torch.full((B, T), m.n_d - 1, dtype=torch.long)
    act = m.action_features(torch.zeros((B, T), dtype=torch.long))

    # --- 1. causality: perturb frame t_p, outputs at frames < t_p must be identical ---
    with torch.no_grad():
        y0 = m(z, tau, d, act)
        for t_p in (3, 6):
            z2 = z.clone()
            z2[:, t_p:] = torch.randn_like(z2[:, t_p:])
            y1 = m(z2, tau, d, act)
            diff_past = (y0[:, :t_p] - y1[:, :t_p]).abs().max().item()
            diff_fut = (y0[:, t_p:] - y1[:, t_p:]).abs().max().item()
            assert diff_past == 0.0, f"CAUSALITY VIOLATED: past diff {diff_past} at t_p={t_p}"
            assert diff_fut > 0, "perturbation had no effect at all — test is vacuous"
        print("1. causality: PASS (past outputs bit-identical under future perturbation)")

    # --- 2. cache equivalence: committed incremental forwards == one-shot forward ---
    with torch.no_grad():
        tau_c = torch.full((B, T), int(round(CFG.context_signal * m.K_max)), dtype=torch.long)
        zc = z  # any fixed input; equivalence is a property of masking+cache, not of the values
        ref = m(zc, tau_c, d, act)                              # one-shot, no cache
        cache = [{} if blk.att.is_temporal else None for blk in m.blocks]
        outs = []
        for t in range(T):
            o = m(zc[:, t:t + 1], tau_c[:, t:t + 1], d[:, t:t + 1], act[:, t:t + 1],
                  positions=torch.tensor([t]), cache=cache, commit=True)
            outs.append(o)
        inc = torch.cat(outs, dim=1)
        md = (ref - inc).abs().max().item()
        assert md < 2e-5, f"CACHE MISMATCH: maxdiff {md}"
        print(f"2. cache equivalence: PASS (maxdiff {md:.2e})")

    # --- 3. footprint: GQA cache bytes == base cache bytes / 4 ---
    with torch.no_grad():
        base = DynamicsModel(CFG).eval()
        ctx = torch.randn(B, T, L, D)
        a_ctx = torch.zeros((B, T), dtype=torch.long)
        s_gqa = m.rollout_init(ctx, a_ctx, K=4)
        s_base = base.rollout_init(ctx, a_ctx, K=4)
        bg, bb = cache_bytes(s_gqa), cache_bytes(s_base)
        print(f"3. footprint: GQA {bg / 1e6:.2f} MB vs base {bb / 1e6:.2f} MB "
              f"-> ratio {bb / bg:.2f}x")
        assert abs(bb / bg - 4.0) < 1e-6, "cache ratio is not exactly 4x"

    # --- bonus: loss backward + a 3-frame generate run ---
    m.train()
    loss, parts = m.loss(z, torch.zeros((B, T), dtype=torch.long), return_parts=True)
    loss.backward()
    assert torch.isfinite(loss), "loss not finite"
    m.eval()
    with torch.no_grad():
        gen = m.generate(z[:, :4], n_generate=3, K=4,
                         action_idx=torch.zeros((B, 7), dtype=torch.long))
    assert gen.shape == (B, 3, L, D)
    print(f"4. loss backward + generate: PASS (loss {loss.item():.4f}, gen {tuple(gen.shape)})")
    print("ALL GQA SMOKE CHECKS PASS")


if __name__ == "__main__":
    main()
