"""Autograd correctness for the mem->mem relay (the must-have check from the task).

Autograd can silently break here: a stray no_grad / in-place op on the carried memory would zero the
relay gradient with NO error, and the only symptom would be that the eval never moves. So we prove the
gradient actually flows from a later window's loss back to the CONSTRUCTION of a memory token whose
grounding latent has been evicted.

Construction: run the rollout in forced FULL-NOISE mode. Then every window after the init has pure-noise
latents, so the only scene information anywhere is the memory carried from the init window. We make z1 a
leaf with requires_grad and check the grad on frame 0's latent (present ONLY in the init window, never in
a loss-bearing new half): it can receive gradient ONLY through the memory relay (init -> memory -> later
window loss). It must be nonzero WITH the relay, and ~zero WITH the relay detached (tbptt_frames=0).

Run:  python -u experiments/mem2mem/test_autograd.py   (CPU OK).
"""
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.dynamics_model import DynamicsModel, DynamicsModelConfig  # noqa: E402
from rollout import mem2mem_rollout_loss                              # noqa: E402

CFG = dict(embedding_dim=32, n_heads=4, mlp_ratio=2.0, depth=6, n_latents=2, bottleneck_dim=8,
           n_registers=2, max_temporal_length=8, max_sampling_steps=4, inference_steps=4,
           n_memory=2, ff9_k=1, n_actions=2, drop_rate=0.0, att_drop_rate=0.0)
B, T, L, D = 2, 12, 2, 8
W = 4  # window; init [0,4), first loss-bearing window [2,6) -> frame 0 evicted from all new halves


def _grad_on_frame0(tbptt_frames):
    torch.manual_seed(0)
    model = DynamicsModel(DynamicsModelConfig(**CFG)).eval()  # eval -> dropout off (graph clean)
    z1 = torch.randn(B, T, L, D, requires_grad=True)
    actions = torch.randint(0, 2, (B, T))
    gen = torch.Generator().manual_seed(123)
    total, parts = mem2mem_rollout_loss(model, z1, actions, n_ctx=W, device="cpu",
                                        tbptt_frames=tbptt_frames, force_mode="noise", gen=gen)
    assert total.requires_grad, "rollout loss is detached from the graph!"
    model.zero_grad(set_to_none=True)
    total.backward()
    assert z1.grad is not None, "no grad reached z1 at all"
    g0 = z1.grad[:, 0].norm().item()           # frame 0: relayed only via memory
    g_new = z1.grad[:, W:].norm().item()        # later frames: have direct (ff9-target) grad paths
    return g0, g_new, parts


def test_relay_gradient_reaches_evicted_construction():
    g0_relay, g_new, parts = _grad_on_frame0(tbptt_frames=10_000)  # no detach -> relay intact
    g0_detach, _, _ = _grad_on_frame0(tbptt_frames=0)              # detach relay every slide
    print(f"slides={parts['n_slides']:.0f}  flow={parts['flow']:.4f}  ff9={parts['ff9']:.4f}")
    print(f"|grad z1[frame0]|  relay-on={g0_relay:.3e}   relay-detached={g0_detach:.3e}")
    print(f"|grad z1[frames>=W]| (direct paths, sanity) = {g_new:.3e}")

    assert g0_relay > 1e-8, (
        "FAIL: frame-0 latent got NO gradient through the memory relay — mem->mem is training nothing "
        "(a no_grad / in-place op likely severed the carried memory).")
    assert g0_detach < g0_relay * 1e-3, (
        f"FAIL: detaching the relay (tbptt=0) did NOT remove frame-0's gradient "
        f"({g0_detach:.3e} vs relay {g0_relay:.3e}) — the grad is not actually flowing via memory.")
    assert g_new > 1e-8, "sanity: later frames should have direct gradient paths"
    print("[ok] mem->mem relay gradient reaches the evicted memory token's construction; "
          "detaching the relay removes it.")


if __name__ == "__main__":
    test_relay_gradient_reaches_evicted_construction()
    print("\nMEM2MEM AUTOGRAD TEST PASSED")
