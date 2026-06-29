"""Training-scale relay-gradient probe for the FAIR no-FF9 ablation.

The 411270 "FF9 is necessary" result is under suspicion: conceptually the 50% full-noise rollout mode
should train memory to carry hidden state even without the explicit FF9 term, because the new-half flow
loss can ONLY be satisfied from carried memory. Merlin's worry: maybe the gradient doesn't actually
flow back through the memory relay behind the current window, so the noise-mode signal trains nothing.

test_autograd.py already falsifies that on a TINY model with use_ff9=True. This probe re-checks it at the
REAL training config (DynamicsModelConfig defaults, n_memory=4, ff9_k=3) and, crucially, with
**use_ff9=False, bootstrap=False, n_d_unlocked=1** - i.e. EXACTLY the clean re-run's loss (winner config
minus FF9). It also measures how the relay gradient decays with relay depth (vanishing-through-time?).

Setup: forced full-noise mode => every window after the init has pure-noise latents, so the only scene
info anywhere is the memory carried from the init window. z1 is a leaf; the grad on an EARLY frame's
latent (present only in the init window, never in a loss-bearing new half) can reach it ONLY via the
memory relay. Nonzero with the relay on, ~0 with it detached => the noise-mode loss DOES train memory.

Run:  venv/Scripts/python.exe -u experiments/mem2mem-rollout-noff9-fair/probe_relay_grad.py   (GPU or CPU)
"""
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "experiments" / "mem2mem"))
from models.dynamics_model import DynamicsModel, DynamicsModelConfig  # noqa: E402
from rollout import mem2mem_rollout_loss                              # noqa: E402


def run(tbptt_frames, *, W, T, device):
    torch.manual_seed(0)
    # REAL training config: defaults + the mem2mem architecture knobs the trainer uses.
    cfg = DynamicsModelConfig(n_actions=2, n_memory=4, ff9_k=3)
    model = DynamicsModel(cfg).to(device).eval()  # eval -> dropout off, clean graph
    B, L, D = 2, cfg.n_latents, cfg.bottleneck_dim
    z1 = torch.randn(B, T, L, D, device=device, requires_grad=True)
    actions = torch.randint(0, 2, (B, T), device=device)
    gen = torch.Generator(device=device).manual_seed(123)
    total, parts = mem2mem_rollout_loss(
        model, z1, actions, n_ctx=W, device=device, tbptt_frames=tbptt_frames,
        force_mode="noise", gen=gen,
        bootstrap=False, n_d_unlocked=1, use_ff9=False)   # <-- the clean re-run's exact loss
    assert total.requires_grad, "rollout loss detached from graph!"
    model.zero_grad(set_to_none=True)
    total.backward()
    g = z1.grad                                            # (B, T, L, D)
    per_frame = g.flatten(2).norm(dim=2).mean(0)           # (T,) mean over batch+latent
    return total.detach(), parts, per_frame.cpu()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = DynamicsModelConfig().max_temporal_length
    W = N                  # deepest relay: largest valid window
    T = 64                 # the trainer's default clip_len
    print(f"device={device}  N(max_temporal_length)={N}  W={W}  T={T}  "
          f"loss=noise-mode flow ONLY (use_ff9=False, bootstrap=False, d_min)")

    loss_on, parts, pf_on = run(10_000, W=W, T=T, device=device)     # relay intact
    _,       _,    pf_off = run(0,      W=W, T=T, device=device)     # detach relay every slide
    half = W // 2
    print(f"slides={parts['n_slides']:.0f}  flow={parts['flow']:.4f}  ff9={parts['ff9']:.4f}\n")

    # Early frames (< half) live ONLY in the init window -> reachable ONLY via the relay.
    early_on = pf_on[:half].norm().item()
    early_off = pf_off[:half].norm().item()
    print(f"|grad| on init-only frames [0,{half}) : relay-on={early_on:.3e}  relay-detached={early_off:.3e}")
    print(f"ratio detached/on = {early_off/max(early_on,1e-12):.2e}  (~=0 => those frames get grad ONLY via memory)\n")

    print("per-frame |grad z1| (relay ON) - watch decay from late (direct loss) to early (relay-only):")
    for t in range(T):
        tag = ""
        if t < half: tag = "init-only (relay)"
        elif t >= T - half: tag = "last new half (direct)"
        print(f"  t={t:2d}  {pf_on[t]:.3e}  {tag}")

    ok_relay = early_on > 1e-8
    ok_detach = early_off < early_on * 1e-3
    print(f"\n[{'ok' if ok_relay else 'FAIL'}] noise-mode flow loss (NO FF9) sends gradient to evicted memory construction")
    print(f"[{'ok' if ok_detach else 'FAIL'}] detaching the relay removes it (grad really flows via memory, not a leak)")
    if ok_relay and ok_detach:
        print("\nVERDICT: the relay gradient is healthy at training scale WITHOUT FF9 - the noise-mode "
              "signal trains memory. So a no-FF9 collapse is NOT a severed-gradient bug; it is optimization "
              "(signal weak/slow) or the 411270 confounds (bootstrap+curriculum+instability+36ep).")


if __name__ == "__main__":
    main()
