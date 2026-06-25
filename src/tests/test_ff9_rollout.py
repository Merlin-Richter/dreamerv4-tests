"""FF9 rollout-training (D-048, EXP-029 C1) tests: the memory->memory relay is on the gradient
path and TBPTT-k truncation controls how deep the credit flows.

Decisive test (test_tbptt_truncation_controls_graph_depth): under p_hide=1.0 (every source latent
HIDDEN => pure noise), the seed-window latents z1[:, :seed] enter the loss ONLY through the initial
memory write, whose influence on a later hop's prediction must travel the carried-memory chain. So
gradient w.r.t. z1[:, :seed] is a direct measurement of how far the memory relay carries credit:
full graph (tbptt>=h) >> tbptt=1 (carry detached every hop). If they were equal, the relay would NOT
be on the gradient path (the FF9 v2 bug this loss exists to fix).

Run:  venv/Scripts/python.exe -u src/tests/test_ff9_rollout.py   (or pytest)
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.dynamics_model import DynamicsModel, DynamicsModelConfig  # noqa: E402


def _tiny_cfg(**kw) -> DynamicsModelConfig:
    return DynamicsModelConfig(
        embedding_dim=64, n_heads=4, depth=4, max_temporal_length=8,
        max_sampling_steps=16, n_actions=2, n_memory=4, **kw,
    )


def _batch(cfg, B=3, T=8):
    z1 = torch.randn(B, T, cfg.n_latents, cfg.bottleneck_dim)
    actions = torch.randint(0, cfg.n_actions, (B, T))
    return z1, actions


def test_off_is_byte_identical():
    """ff9_rollout_h=0 draws no RNG and adds no term -> loss identical to not passing the args."""
    cfg = _tiny_cfg()
    model = DynamicsModel(cfg)
    z1, actions = _batch(cfg)
    torch.manual_seed(7)
    base = model.loss(z1, actions, ff9_k=2)
    torch.manual_seed(7)
    withzero = model.loss(z1, actions, ff9_k=2, ff9_rollout_h=0)
    assert torch.equal(base, withzero), (base, withzero)


def test_finite_and_parts():
    cfg = _tiny_cfg()
    model = DynamicsModel(cfg)
    z1, actions = _batch(cfg)
    total, parts = model.loss(z1, actions, ff9_rollout_h=4, ff9_rollout_tbptt=2,
                              ff9_rollout_p_hide=0.5, return_parts=True)
    assert torch.isfinite(total), total
    assert "ff9_rollout" in parts and torch.isfinite(parts["ff9_rollout"])
    assert {"ff9r_h1", "ff9r_h2", "ff9r_h3", "ff9r_h4"} <= set(parts)
    assert all(torch.isfinite(parts[k]) for k in parts)


def test_gradient_reaches_memory_and_blocks():
    """The rollout term alone must backprop through the memory tokens and the transformer blocks."""
    cfg = _tiny_cfg()
    model = DynamicsModel(cfg)
    z1, actions = _batch(cfg)
    loss, _ = model._ff9_rollout_loss(z1, model.action_features(actions), h=5, tbptt_k=5, p_hide=1.0, hide_mode='iid')
    loss.backward()
    assert model.memory_tokens.grad is not None and model.memory_tokens.grad.abs().sum() > 0
    blk_grads = [p.grad for p in model.blocks.parameters() if p.grad is not None]
    assert blk_grads and all(torch.isfinite(g).all() for g in blk_grads)


def test_relay_jacobian_is_connected():
    """DECISIVE (mechanism): the memory->memory write map mem_{t+1} <- mem_t is differentiable and
    connected. Inject a memory token at the source of a 2-frame [noise|noise] window (so the ONLY
    signal is memory) and require d(written memory at the new frame)/d(injected memory) to be
    non-zero. This is exactly the op-3 map _ff9_loss leaves un-gradiented; if this Jacobian were
    zero, no rollout training could ever credit the relay."""
    cfg = _tiny_cfg()
    model = DynamicsModel(cfg)
    B = 4
    E = cfg.embedding_dim
    mem0 = torch.randn(B, 1, cfg.n_memory, E, requires_grad=True)   # injected memory (leaf)
    learned = model.memory_tokens.expand(B, 1, -1, -1)
    src = torch.randn(B, 1, cfg.n_latents, cfg.bottleneck_dim)      # noise source latent
    new = torch.randn(B, 1, cfg.n_latents, cfg.bottleneck_dim)      # noise new-frame slot
    inp = torch.cat((src, new), dim=1)
    tau = torch.zeros(B, 2, dtype=torch.long)                       # both pure noise
    d = torch.full((B, 2), model.n_d - 1, dtype=torch.long)
    mem_in = torch.cat((mem0, learned), dim=1)
    _, mem_out = model(inp, tau, d, memory_in=mem_in, return_memory=True)
    mem1 = mem_out[:, -1:]                                          # written memory at the new frame
    g = torch.autograd.grad(mem1.sum(), mem0)[0]
    print(f"  ||d(mem_{{t+1}})/d(mem_t)||_1 = {g.abs().sum().item():.3e}")
    assert g is not None and g.abs().sum() > 0, "relay map mem_t -> mem_{t+1} is disconnected"


def test_tbptt_truncation_controls_graph_depth():
    """The multi-hop carried-memory chain ADDS gradient to the seed write: full graph > tbptt=1
    (which cuts the carry every hop). NB on an UNTRAINED model the deep-hop credit is small (the
    relay attenuates gradient fast — the V-T014 vanishing this loss + a measured TBPTT-k exist to
    fix); the test asserts the chain propagates SOME credit, not its trained magnitude."""
    cfg = _tiny_cfg()
    model = DynamicsModel(cfg)
    z1, actions = _batch(cfg)
    H, seed = 5, 3

    def seed_grad(tbptt):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(123)                 # same hide coins + noise across arms
        z = z1.clone().requires_grad_(True)
        loss, _ = model._ff9_rollout_loss(z, model.action_features(actions), h=H, tbptt_k=tbptt, p_hide=1.0, hide_mode='iid')
        loss.backward()
        return z.grad[:, :seed].abs().sum().item()

    g_full = seed_grad(H)     # full graph: all H hops reach the seed write
    g_one = seed_grad(1)      # detach every hop: only hop-0 reaches the seed write
    print(f"  seed-grad full(tbptt={H})={g_full:.3e}  tbptt=1={g_one:.3e}  ratio={g_full/max(g_one,1e-12):.2f}")
    assert g_full > 0, "seed write gets no gradient even with the full graph -> relay NOT on grad path"
    assert g_full > 1.02 * g_one, (
        f"deeper TBPTT did not add seed-write credit (full {g_full:.3e} vs k=1 {g_one:.3e}) "
        "-> the carried memory chain is not propagating gradient at all")


def test_p_hide_zero_no_leak_to_target_via_memory_only():
    """Sanity: with p_hide=0 (all visible) the source latent carries the scene (re-anchor mode);
    with p_hide=1 (all hidden) memory is the only carrier -> hidden loss should be >= visible loss
    on an untrained model (harder task), confirming the hide coin actually removes the latent path."""
    cfg = _tiny_cfg()
    model = DynamicsModel(cfg).eval()
    z1, actions = _batch(cfg, B=64)
    with torch.no_grad():
        torch.manual_seed(1)
        vis, _ = model._ff9_rollout_loss(z1, model.action_features(actions), h=5, tbptt_k=5, p_hide=0.0, hide_mode='iid')
        torch.manual_seed(1)
        hid, _ = model._ff9_rollout_loss(z1, model.action_features(actions), h=5, tbptt_k=5, p_hide=1.0, hide_mode='iid')
    print(f"  visible(p_hide=0) loss={vis.item():.4f}  hidden(p_hide=1) loss={hid.item():.4f}")
    assert hid >= vis, "hiding the source latent did not make the task harder -> latent path not removed"


def test_generate_carries_memory():
    """generate() for a memory model threads each frame's WRITTEN memory forward (not recomputed):
    (a) it runs + right shape + finite; (b) the carried memory is actually READ (perturbing the
    injected context memory changes the prediction); (c) the carrying rollout differs from the
    recompute path (generate_cached plain=True), i.e. carrying is actually active."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(use_full_state_memory=True)         # n_memory=4
    model = DynamicsModel(cfg).eval()
    B, P, n_gen = 2, 4, 6
    ctx = torch.randn(B, P, cfg.n_latents, cfg.bottleneck_dim)
    act = torch.randint(0, cfg.n_actions, (B, P + n_gen))
    out = model.generate(ctx, n_gen, K=4, action_idx=act)
    assert out.shape == (B, n_gen, cfg.n_latents, cfg.bottleneck_dim), out.shape
    assert torch.isfinite(out).all()

    # (b) the carried context memory is READ: perturbing it changes the new-frame prediction.
    E = cfg.embedding_dim
    window = ctx
    act_w = model.action_features(act[:, :P + 1])
    mem = model._written_memory(ctx, model.action_features(act[:, :P]))
    torch.manual_seed(7); a = model._denoise_next(window, 4, act_w, context_mem=mem)
    torch.manual_seed(7); b = model._denoise_next(window, 4, act_w, context_mem=mem + 5.0)
    assert not torch.allclose(a, b, atol=1e-5), "carried memory is ignored by the rollout"

    # (c) carrying != recompute (generate_cached plain re-inits memory every step).
    torch.manual_seed(3); carried = model.generate(ctx, n_gen, K=4, action_idx=act)
    torch.manual_seed(3); recomputed = model.generate_cached(ctx, n_gen, K=4, action_idx=act, plain=True)
    assert not torch.allclose(carried, recomputed, atol=1e-5), \
        "carrying memory produced the same rollout as recomputing it -> carry not active"


if __name__ == "__main__":
    fns = [v for n, v in sorted(globals().items()) if n.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} ff9-rollout tests passed.")
