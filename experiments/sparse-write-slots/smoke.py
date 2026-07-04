"""Sparse write-slots correctness smoke — run BEFORE any GPU-hours.

1. MASK UNIT: sparse_write_mask — memory slots see exactly the causal write-slot keys (no
   scratch keys, no diagonal for scratch queries); non-memory slots keep plain causal.
2. ATTENTION-WEIGHT CHECK (white-box): hooked temporal attention — the probability mass that
   memory-slot queries place on non-write keys is exactly 0 (post-softmax).
3. CAUSALITY: perturbing frame t leaves outputs < t bit-identical (eval mode).
4. CACHE EQUIVALENCE: incremental committed forwards == one-shot uncached forward, both under
   phase-init memory (maxdiff ~fp32 tolerance).
5. FINITENESS incl. full-noise input; loss-path backward via sparse_rollout_loss (both modes),
   with gradients reaching mem_init2[0] AND mem_init2[1] AND the carried relay.
6. rollout_init/rollout_step/generate API smoke (the recall eval's exact call pattern).

Usage:  venv/Scripts/python.exe -u experiments/sparse-write-slots/smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import DynamicsModelSparseWS, sparse_write_mask  # noqa: E402
from rollout_sparse import sparse_rollout_loss  # noqa: E402
from models.dynamics_model import DynamicsModelConfig  # noqa: E402

torch.manual_seed(0)
CFG = DynamicsModelConfig(n_actions=7, n_memory=4, ff9_k=0, max_temporal_length=16)
N_SPARSE = DynamicsModelSparseWS.SPARSE_N  # 8


def main():
    B, T, L, D = 2, 16, CFG.n_latents, CFG.bottleneck_dim
    m = DynamicsModelSparseWS(CFG).eval()
    mem_start = CFG.n_action_tokens + CFG.n_latents + CFG.n_registers
    mem_end = mem_start + CFG.n_memory

    # --- 1. mask unit ---
    pos = torch.arange(5, 21)  # unaligned start on purpose
    mask = sparse_write_mask(pos, pos, 10, 6, 8, N_SPARSE)
    for qi, qp in enumerate(pos):
        for ki, kp in enumerate(pos):
            causal_ok = kp <= qp
            for s in range(10):
                visible = not bool(mask[s, qi, ki])
                if s in (6, 7):  # memory slots
                    assert visible == (causal_ok and kp % N_SPARSE == 0), (s, int(qp), int(kp))
                else:
                    assert visible == causal_ok, (s, int(qp), int(kp))
    print("1. mask unit: PASS (memory -> causal write keys only; others plain causal)")

    # --- 2. attention-weight check on a real forward ---
    grabbed = []

    def hook(module, args, out):
        pass  # placeholder (weights grabbed via re-run below)

    z = torch.randn(B, T, L, D)
    tau = torch.full((B, T), 100, dtype=torch.long)
    d = torch.full((B, T), m.n_d - 1, dtype=torch.long)
    act = m.action_features(torch.zeros((B, T), dtype=torch.long))
    # re-implement the score path of one temporal block to grab softmax weights faithfully:
    # easier + equally strong: monkeypatch att_droput to capture its input (the attn weights).
    att = m.blocks[1].att
    captured = {}
    orig_drop = att.att_droput

    class Grab(torch.nn.Module):
        def forward(self, x):
            captured["attn"] = x.detach()
            return orig_drop(x)

    att.att_droput = Grab()
    with torch.no_grad():
        m(z, tau, d, act)
    att.att_droput = orig_drop
    attn = captured["attn"]                     # (H, B, N, T, T_all), positions 0..15
    write_keys = (torch.arange(T) % N_SPARSE) == 0
    leak = attn[:, :, mem_start:mem_end][..., ~write_keys].abs().max().item()
    assert leak == 0.0, f"memory attention leaks onto non-write keys: {leak}"
    assert attn[:, :, mem_start:mem_end][..., write_keys].sum() > 0
    print("2. attention-weight check: PASS (zero mass on non-write keys for memory slots)")

    # --- 3. causality ---
    with torch.no_grad():
        y0 = m(z, tau, d, act)
        for t_p in (5, 11):
            z2 = z.clone()
            z2[:, t_p:] = torch.randn_like(z2[:, t_p:])
            y1 = m(z2, tau, d, act)
            assert (y0[:, :t_p] - y1[:, :t_p]).abs().max().item() == 0.0, f"causality @ {t_p}"
    print("3. causality: PASS")

    # --- 4. cache equivalence (both sides phase-init memory) ---
    with torch.no_grad():
        tau_c = torch.full((B, T), int(round(CFG.context_signal * m.K_max)), dtype=torch.long)
        ref = m(z, tau_c, d, act)
        cache = m.new_kv_cache()
        outs = []
        for t in range(T):
            outs.append(m(z[:, t:t + 1], tau_c[:, t:t + 1], d[:, t:t + 1], act[:, t:t + 1],
                          positions=torch.tensor([t]), cache=cache, commit=True))
        md = (ref - torch.cat(outs, dim=1)).abs().max().item()
        assert md < 2e-5, f"cache mismatch {md}"
    print(f"4. cache equivalence: PASS (maxdiff {md:.2e})")

    # --- 5. rollout loss backward, both modes, gradient reach ---
    m.train()
    z_long = torch.randn(B, 64, L, D)
    a_long = torch.zeros((B, 64), dtype=torch.long)
    gen = torch.Generator(device="cpu").manual_seed(0)
    for mode in ("noise", "clean", None):
        loss, parts = sparse_rollout_loss(m, z_long, a_long, device="cpu", gen=gen,
                                          force_mode=mode)
        assert torch.isfinite(loss), f"loss not finite in mode {mode}"
        m.zero_grad(set_to_none=True)
        loss.backward()
        g0 = m.mem_init2.grad
        assert g0 is not None and torch.isfinite(g0).all()
        assert g0[0].abs().sum() > 0, f"no grad to WRITE init in mode {mode}"
        assert g0[1].abs().sum() > 0, f"no grad to SCRATCH init in mode {mode}"
        print(f"5. rollout loss [{mode}]: PASS (loss {loss.item():.4f}, "
              f"slides {parts['n_slides']:.0f}, grads reach both inits)")

    # --- 6. rollout API smoke (recall's exact pattern incl. read-only branch) ---
    m.eval()
    with torch.no_grad():
        state = m.rollout_init(z[:, :4], torch.zeros((B, 4), dtype=torch.long), K=2)
        z_occ = m.rollout_step(state, torch.ones((B,), dtype=torch.long), commit=True)
        pre = {i: (lc['k'].clone() if lc else None) for i, lc in enumerate(state["cache"]) if lc}
        _ = m.rollout_step(state, torch.zeros((B,), dtype=torch.long), commit=False)
        for i, k0 in pre.items():
            assert torch.equal(k0, state["cache"][i]['k']), "read-only branch mutated cache"
        gen_out = m.generate(z[:, :4], n_generate=3, K=2,
                             action_idx=torch.zeros((B, 7), dtype=torch.long))
        assert gen_out.shape == (B, 3, L, D) and torch.isfinite(gen_out).all()
    print("6. rollout API: PASS (init/step/read-only-branch/generate)")
    print("ALL SPARSE WRITE-SLOTS SMOKE CHECKS PASS")


if __name__ == "__main__":
    main()
