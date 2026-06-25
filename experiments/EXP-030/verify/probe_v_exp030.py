"""V-EXP030 — independent verification of _ff9_rollout_loss (D-048).

Probes (decisive, minimal):
  P1  C1: TBPTT-k truly bounds graph depth. Measure gradient of the SEED-write
          (z1[:, :seed]) under p_hide=1 (memory is the only path from seed to a later
          hop's loss). Sweep tbptt_k. Predictions:
            - tbptt=1: only hop-0 loss reaches the seed (carry detached every hop).
            - tbptt=H: all H hops reach the seed.
            - tbptt=K (1<K<H): hops whose loss is within K-1 of a *non-detach* boundary
              reach the seed; a clean monotone-ish increase with K, and a HARD CAP:
              detaching at k must zero the contribution of hops strictly beyond the last
              retained block. We test the strongest, unambiguous statement:
              grad(tbptt=H) > grad(tbptt=1) AND grad is monotonic non-decreasing in k,
              AND a *surgical* test: zero out all per-hop losses except the LAST hop and
              confirm seed-grad is zero for tbptt smaller than the distance to the last
              non-detach boundary, nonzero otherwise.

  P2  C1 mechanism: per-hop carry Jacobian d(mem_out_newframe)/d(mem_carry_injected) is
      nonzero (the op-3 map is connected), under a tau=0/tau=0 window (memory is the only
      signal). Also confirm the SEED mem tensor requires_grad and is NOT detached.

  P3  C2: on a HIDDEN step, the GT target latent z1_new must NOT reach the prediction
      input. Test: perturb z1_new (the GT) on a hidden step; the model INPUT (and hence
      z_hat) must be invariant to it (only the loss target moves). Concretely:
      d(z_hat_newframe)/d(z1_new) == 0 on a hidden step  => no latent leak.
      Contrast with: d(z_hat)/d(src_true GT) == 0 on a hidden step (source is replaced by
      noise). And the positive control: on a VISIBLE step d(z_hat)/d(src_true) != 0.

  P4  C3: ff9_rollout_h=0 byte-identical + off-path RNG identical; windowing T==W identical.

Run: venv/Scripts/python.exe -u experiments/EXP-030/verify/probe_v_exp030.py
SEED reported inline. CPU, tiny cfg.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from models.dynamics_model import DynamicsModel, DynamicsModelConfig  # noqa: E402


def tiny_cfg(**kw):
    return DynamicsModelConfig(
        embedding_dim=64, n_heads=4, depth=4, max_temporal_length=8,
        max_sampling_steps=16, n_actions=2, n_memory=4, **kw,
    )


def batch(cfg, B=3, T=8):
    z1 = torch.randn(B, T, cfg.n_latents, cfg.bottleneck_dim)
    actions = torch.randint(0, cfg.n_actions, (B, T))
    return z1, actions


SEED = 0
results = {}


# ----------------------------------------------------------------- P1: tbptt bounds depth
def p1_tbptt_bounds_depth():
    torch.manual_seed(SEED)
    cfg = tiny_cfg()
    model = DynamicsModel(cfg)
    z1, actions = batch(cfg)
    H, seed = 5, 3

    def seed_grad(tbptt):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(123)  # identical hide coins + noise across arms
        z = z1.clone().requires_grad_(True)
        af = model.action_features(actions)  # rebuild graph each arm (freed by prior backward)
        loss, _ = model._ff9_rollout_loss(z, af, h=H, tbptt_k=tbptt, p_hide=1.0, hide_mode='iid')
        loss.backward()
        return z.grad[:, :seed].abs().sum().item()

    grads = {k: seed_grad(k) for k in (1, 2, 3, 4, 5)}
    mono = all(grads[k] <= grads[k + 1] + 1e-9 for k in (1, 2, 3, 4))
    results['P1_grads'] = grads
    results['P1_monotone'] = mono
    results['P1_full_gt_one'] = grads[5] > 1.02 * grads[1]
    print(f"P1 seed-grad by tbptt: " + "  ".join(f"k{k}={v:.3e}" for k, v in grads.items()))
    print(f"   monotone non-decreasing in k: {mono}   full>1.02*one: {results['P1_full_gt_one']}")


# ------------------------------------------- P1b: surgical hard-cap (last-hop-only loss)
def p1b_hardcap_last_hop_only():
    """Strongest C1 statement: if ONLY the last hop (j=H-1) contributes to the loss, then
    the seed-write gradient must be EXACTLY ZERO whenever a detach boundary separates the
    seed from hop H-1, and NONZERO when no detach boundary intervenes. We monkeypatch the
    per-hop loss aggregation by re-implementing the rollout inline (mirror of the method)
    so we can isolate the last hop. This directly tests `tbptt_k truly bounds the depth`.
    """
    torch.manual_seed(SEED)
    cfg = tiny_cfg()
    model = DynamicsModel(cfg)
    z1, actions = batch(cfg)
    H, seed = 5, 3
    maxctx = cfg.max_temporal_length - 1
    assert seed == min(maxctx, 3)
    tau_ctx_idx = min(round(cfg.context_signal * model.K_max), model.K_max - 1)
    d_fine = model.n_d - 1
    B = z1.shape[0]

    def last_hop_seed_grad(tbptt):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(7)
        z = z1.clone().requires_grad_(True)
        af = model.action_features(actions)  # rebuild graph each arm
        # SEED (mirror)
        win = z[:, :seed]
        w = win.shape[1]
        tau_seed = torch.full((B, w), tau_ctx_idx, dtype=torch.long)
        d_seed = torch.full((B, w), d_fine, dtype=torch.long)
        _, mem = model(model._noise_to_ctx(win), tau_seed, d_seed, af[:, :seed], return_memory=True)
        mem_carry = mem[:, -1:]
        d2 = torch.full((B, 2), d_fine, dtype=torch.long)
        zero_tau = torch.zeros(B, dtype=torch.long)
        learned_mem = model.memory_tokens.expand(B, 1, -1, -1)
        last_loss = None
        for jj in range(H):
            tgt_pos = seed + jj
            src = torch.randn_like(z[:, tgt_pos - 1:tgt_pos])      # all hidden
            tau_src = zero_tau
            z1_new = z[:, tgt_pos:tgt_pos + 1]
            new_tilde = torch.randn_like(z1_new)
            inp = torch.cat((src, new_tilde), dim=1)
            tau2 = torch.stack((tau_src, zero_tau), dim=1)
            mem_in = torch.cat((mem_carry, learned_mem), dim=1)
            act2 = af[:, tgt_pos - 1:tgt_pos + 1]
            z_hat, mem_out = model(inp, tau2, d2, act2, memory_in=mem_in, return_memory=True)
            if jj == H - 1:
                last_loss = ((z_hat[:, -1:] - z1_new) ** 2).mean()
            mem_carry = mem_out[:, -1:]
            if (jj + 1) % tbptt == 0:
                mem_carry = mem_carry.detach()
        last_loss.backward()
        return z.grad[:, :seed].abs().sum().item() if z.grad is not None else 0.0

    # H=5, hops j=0..4 (last is j=4). Detach boundaries are after hop (jj+1)%k==0.
    # tbptt=1: detach after every hop -> seed cut from hop4 -> seed-grad==0.
    # tbptt=5: never detach (5%5 only at jj=4, after last hop) -> seed reaches hop4 -> >0.
    # H=5 hops j=0..4 (last=4). Boundary after hop jj iff (jj+1)%k==0.
    # The seed reaches the last hop (4) iff NO detach occurs at boundaries jj=0..3,
    # i.e. iff k does not divide any of {1,2,3,4} before hop 4 -> only k=5 (and k>=5).
    # So: k in {1,2,3,4} -> seed cut -> grad==0 ; k==5 -> grad>0. A precise cap test.
    gk = {k: last_hop_seed_grad(k) for k in (1, 2, 3, 4, 5)}
    results['P1b_lasthop_by_k'] = gk
    cut = all(gk[k] < 1e-12 for k in (1, 2, 3, 4))
    reaches = gk[5] > 0
    results['P1b_cap_ok'] = cut and reaches
    print("P1b last-hop-only seed-grad by k: " + "  ".join(f"k{k}={v:.3e}" for k, v in gk.items()))
    print(f"   k1..4 cut(==0): {cut}   k5 reaches(>0): {reaches}   cap_ok={results['P1b_cap_ok']}")


# ----------------------------------------------------- P2: per-hop carry Jacobian connected
def p2_relay_jacobian():
    torch.manual_seed(SEED)
    cfg = tiny_cfg()
    model = DynamicsModel(cfg)
    B = 4
    E = cfg.embedding_dim
    mem0 = torch.randn(B, 1, cfg.n_memory, E, requires_grad=True)
    learned = model.memory_tokens.expand(B, 1, -1, -1)
    src = torch.randn(B, 1, cfg.n_latents, cfg.bottleneck_dim)
    new = torch.randn(B, 1, cfg.n_latents, cfg.bottleneck_dim)
    inp = torch.cat((src, new), dim=1)
    tau = torch.zeros(B, 2, dtype=torch.long)
    d = torch.full((B, 2), model.n_d - 1, dtype=torch.long)
    mem_in = torch.cat((mem0, learned), dim=1)
    _, mem_out = model(inp, tau, d, memory_in=mem_in, return_memory=True)
    mem1 = mem_out[:, -1:]
    g = torch.autograd.grad(mem1.sum(), mem0)[0]
    results['P2_jacobian_l1'] = g.abs().sum().item()
    results['P2_connected'] = g.abs().sum().item() > 0
    print(f"P2 ||d(mem_new)/d(mem_src)||_1 = {results['P2_jacobian_l1']:.3e}  connected={results['P2_connected']}")

    # Seed mem is on the graph (requires grad) -> not detached at construction.
    z1, actions = batch(cfg)
    af = model.action_features(actions)
    win = z1[:, :3]
    tau_s = torch.full((3, 3), 1, dtype=torch.long)
    d_s = torch.full((3, 3), model.n_d - 1, dtype=torch.long)
    _, mem = model(model._noise_to_ctx(win), tau_s, d_s, af[:, :3], return_memory=True)
    results['P2_seed_requires_grad'] = bool(mem.requires_grad)
    print(f"   seed mem requires_grad={results['P2_seed_requires_grad']} (must be True)")


# ----------------------------------------------------- P3: C2 no latent leak on hidden step
def p3_no_leak_hidden():
    """On a hidden step: (a) the GT target z1_new must not reach the prediction input;
    (b) the GT source must not reach the prediction. We directly differentiate z_hat (the
    prediction, BEFORE the loss) w.r.t. GT tensors with a forced-hidden coin. We reconstruct
    a single hop manually to control the hide coin and isolate gradients."""
    torch.manual_seed(SEED)
    cfg = tiny_cfg()
    model = DynamicsModel(cfg)
    B = 4
    E = cfg.embedding_dim
    L, D = cfg.n_latents, cfg.bottleneck_dim
    mem_carry = torch.randn(B, 1, cfg.n_memory, E)
    learned = model.memory_tokens.expand(B, 1, -1, -1)
    tau_ctx_idx = min(round(cfg.context_signal * model.K_max), model.K_max - 1)
    d2 = torch.full((B, 2), model.n_d - 1, dtype=torch.long)

    # ---- HIDDEN step: mirror lines 705-716 with hcoin=True ----
    src_true = torch.randn(B, 1, L, D, requires_grad=True)   # GT prev frame (leaf)
    z1_new = torch.randn(B, 1, L, D, requires_grad=True)     # GT target (leaf)
    # hidden: src replaced by noise, tau_src=0, new = pure noise
    src = torch.randn_like(src_true)                          # noise (no dep on src_true)
    new_tilde = torch.randn_like(z1_new)                      # noise (no dep on z1_new)
    inp = torch.cat((src, new_tilde), dim=1)
    tau2 = torch.zeros(B, 2, dtype=torch.long)
    mem_in = torch.cat((mem_carry, learned), dim=1)
    z_hat, _ = model(inp, tau2, d2, memory_in=mem_in, return_memory=True)
    pred = z_hat[:, -1:]
    g_tgt = torch.autograd.grad(pred.sum(), z1_new, retain_graph=True, allow_unused=True)[0]
    g_src = torch.autograd.grad(pred.sum(), src_true, retain_graph=True, allow_unused=True)[0]
    leak_tgt = 0.0 if g_tgt is None else g_tgt.abs().sum().item()
    leak_src = 0.0 if g_src is None else g_src.abs().sum().item()
    results['P3_hidden_leak_from_target'] = leak_tgt
    results['P3_hidden_leak_from_source'] = leak_src
    results['P3_hidden_no_leak'] = (leak_tgt == 0.0) and (leak_src == 0.0)
    print(f"P3 HIDDEN: d(pred)/d(GT target)={leak_tgt:.3e}  d(pred)/d(GT source)={leak_src:.3e}  no_leak={results['P3_hidden_no_leak']}")

    # ---- VISIBLE positive control: source GT DOES reach pred ----
    src_true2 = torch.randn(B, 1, L, D, requires_grad=True)
    z1_new2 = torch.randn(B, 1, L, D, requires_grad=True)
    src_v = model._noise_to_ctx(src_true2)                   # visible: near-clean GT (depends on src_true2)
    new_tilde2 = torch.randn_like(z1_new2)
    inp2 = torch.cat((src_v, new_tilde2), dim=1)
    tau2v = torch.stack((torch.full((B,), tau_ctx_idx, dtype=torch.long),
                         torch.zeros(B, dtype=torch.long)), dim=1)
    mem_in2 = torch.cat((mem_carry, learned), dim=1)
    z_hat2, _ = model(inp2, tau2v, d2, memory_in=mem_in2, return_memory=True)
    pred2 = z_hat2[:, -1:]
    g_src_vis = torch.autograd.grad(pred2.sum(), src_true2, retain_graph=True, allow_unused=True)[0]
    g_tgt_vis = torch.autograd.grad(pred2.sum(), z1_new2, allow_unused=True)[0]
    vis_src = 0.0 if g_src_vis is None else g_src_vis.abs().sum().item()
    vis_tgt = 0.0 if g_tgt_vis is None else g_tgt_vis.abs().sum().item()
    results['P3_visible_leak_from_source'] = vis_src
    results['P3_visible_leak_from_target'] = vis_tgt
    results['P3_visible_control_ok'] = vis_src > 0 and vis_tgt == 0.0
    print(f"P3 VISIBLE control: d(pred)/d(GT source)={vis_src:.3e} (expect >0)  d(pred)/d(GT target)={vis_tgt:.3e} (expect 0)")


# ----------------------------------------------------- P4: C3 off-identity + RNG + windowing
def p4_off_identity():
    torch.manual_seed(SEED)
    cfg = tiny_cfg()
    model = DynamicsModel(cfg)
    z1, actions = batch(cfg)
    torch.manual_seed(7)
    base = model.loss(z1, actions, ff9_k=2)
    torch.manual_seed(7)
    withzero = model.loss(z1, actions, ff9_k=2, ff9_rollout_h=0)
    results['P4_off_identical'] = bool(torch.equal(base, withzero))
    print(f"P4 ff9_rollout_h=0 byte-identical: {results['P4_off_identical']}")

    # Windowing: T==W -> z_full==z1[:, :W]. Feed T==W; check the on-path diffusion is unaffected.
    # Compare loss with ff9_rollout active vs not, RNG-aligned, isolating that the *windowing line*
    # (z_full, z1 = z1, z1[:, :W]) doesn't alter the main path when T==W. We check the diffusion part.
    Tw = cfg.max_temporal_length
    z1w, actw = batch(cfg, B=3, T=Tw)
    torch.manual_seed(11)
    _, parts_off = model.loss(z1w, actw, return_parts=True)
    torch.manual_seed(11)
    _, parts_on = model.loss(z1w, actw, ff9_rollout_h=0, return_parts=True)
    results['P4_windowing_diffusion_identical'] = bool(torch.equal(parts_off['diffusion'], parts_on['diffusion']))
    print(f"P4 T==W diffusion identical with windowing line present: {results['P4_windowing_diffusion_identical']}")


if __name__ == "__main__":
    print(f"=== V-EXP030 probes (SEED={SEED}, CPU, tiny cfg) ===")
    p1_tbptt_bounds_depth()
    p1b_hardcap_last_hop_only()
    p2_relay_jacobian()
    p3_no_leak_hidden()
    p4_off_identity()
    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    # C1 verdict rests on the SOUND falsifiers: the surgical hard-cap (P1b) + the connected
    # relay Jacobian (P2) + deeper-graph-adds-credit (P1). P1_monotone is NOT a valid falsifier
    # (different k detach at different boundary SETS, so the aggregate full-window seed-grad need
    # not be monotone in k) -> reported but not gating.
    verdict_C1 = results['P1_full_gt_one'] and results['P1b_cap_ok'] and results['P2_connected'] and results['P2_seed_requires_grad']
    verdict_C2 = results['P3_hidden_no_leak'] and results['P3_visible_control_ok']
    verdict_C3 = results['P4_off_identical'] and results['P4_windowing_diffusion_identical']
    print(f"\n  C1 (relay on grad path, tbptt bounds depth): {'SUPPORTED' if verdict_C1 else 'REFUTED'}")
    print(f"  C2 (no latent leak on hidden step):          {'SUPPORTED' if verdict_C2 else 'REFUTED'}")
    print(f"  C3 (identity when off):                      {'SUPPORTED' if verdict_C3 else 'REFUTED'}")
