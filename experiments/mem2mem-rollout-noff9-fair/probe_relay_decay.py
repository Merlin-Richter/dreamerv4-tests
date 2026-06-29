"""Does the memory-construction gradient VANISH or EXPLODE as it propagates back through the relay?

The mem->mem rollout is BPTT through a chain: loss at slide s -> read old_mem -> its construction at
slide s-1 -> ... -> init. Like any RNN-style BPTT, the per-hop Jacobian can shrink (<1, vanishing) or
grow (>1, exploding) the gradient geometrically. This probe measures that factor directly, at the real
training config and the clean no-FF9 loss (noise-mode flow only).

Method (forced full-noise so the ONLY signal path is the relay):
  Replicate mem2mem_rollout_loss's loop, retaining grad on each carried memory tensor m_i
  (m_0 = init memory; m_i = new_mem written at slide i). NO tbptt detach (we want the raw chain).
  - "last-only":  loss = ONLY the final slide's flow loss. Then |grad m_{S-1-h}| as a function of relay
     hops h is the CLEAN per-hop propagation curve. ratio r_h = |grad m_{S-1-h}| / |grad m_{S-h}|.
  - "full":       loss = mean over all slides (what training sees). Shows the effective profile.
Also reports the FORWARD norm |m_i| per slide (does the carried activation itself grow/shrink?).

Run:  venv/Scripts/python.exe -u experiments/mem2mem-rollout-noff9-fair/probe_relay_decay.py
"""
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "experiments" / "mem2mem"))
from models.dynamics_model import DynamicsModel, DynamicsModelConfig  # noqa: E402
import rollout as R                                                   # noqa: E402


def relay(model, z1, actions, *, W, device, gen, last_only):
    """Mirror mem2mem_rollout_loss (noise mode, no-ff9, no-boot, d_min), retaining grad per hop."""
    N = model.config.max_temporal_length
    B, T, L, D = z1.shape
    half = W // 2
    _, tau_ctx_idx, d_idx_val = R._tau_d_consts(model)
    af_all = model.action_features(actions)
    blank_half = model.memory_tokens.expand(B, half, -1, -1)
    d_col_W = torch.full((B, W), d_idx_val, device=device, dtype=torch.long)
    pos = torch.arange(W, device=device)

    def af(a, b):
        return af_all[:, a:b] if af_all is not None else None

    # init window: near-clean latents, learned-blank memory -> construct initial carried memory
    zc = model._noise_to_ctx(z1[:, :W])
    tau_init = torch.full((B, W), tau_ctx_idx, device=device, dtype=torch.long)
    blank_W = model.memory_tokens.expand(B, W, -1, -1)
    _, mem_win = model(zc, tau_init, d_col_W, af(0, W), memory_in=blank_W,
                       positions=pos, return_memory=True)
    old_mem = mem_win[:, half:]
    old_mem.retain_grad()
    carried = [old_mem]
    losses = []

    s = half
    end = min(T, min(5 * N, 10 * W))
    while s + W <= end:
        z1_win = z1[:, s:s + W]
        z0 = torch.randn(z1_win.shape, device=device, generator=gen)
        old_part = z0[:, :half]                                   # pure noise
        tau_old = torch.zeros(B, half, device=device, dtype=torch.long)
        tau_new_idx, d_new_idx = R._sample_tau_d(model, B, half, device, gen, n_d_unlocked=1)
        tau_new_idx = torch.zeros_like(tau_new_idx)              # noise: tau=0
        tau_new = model._tau_value(tau_new_idx)[..., None, None]
        new_part = (1 - tau_new) * z0[:, half:] + tau_new * z1_win[:, half:]
        memory_in = torch.cat([old_mem, blank_half], dim=1)
        loss, new_mem, _ = R._newhalf_loss(
            model, old_part=old_part, tau_old=tau_old, new_part=new_part,
            tau_new_idx=tau_new_idx, d_new_idx=d_new_idx, z1_new=z1_win[:, half:],
            af_win=af(s, s + W), memory_in=memory_in, positions=pos, half=half, bootstrap=False)
        new_mem.retain_grad()
        carried.append(new_mem)
        losses.append(loss)
        old_mem = new_mem
        s += half

    total = losses[-1] if last_only else sum(losses) / len(losses)
    model.zero_grad(set_to_none=True)
    total.backward()
    gnorm = [c.grad.flatten(1).norm(dim=1).mean().item() if c.grad is not None else 0.0 for c in carried]
    fnorm = [c.detach().flatten(1).norm(dim=1).mean().item() for c in carried]
    return gnorm, fnorm, len(losses)


def _per_hop_factor(g_last):
    """Geometric-mean per-hop backward factor (deeper/shallower) over hops that carry gradient."""
    nz = [g for g in g_last if g > 0]
    if len(nz) < 2:
        return float("nan"), []
    ratios = [nz[k - 1] / max(nz[k], 1e-12) for k in range(len(nz) - 1, 0, -1)]
    geo = (torch.tensor(ratios).log().mean().exp()).item()
    return geo, ratios


def _load(ckpt_path, device):
    blob = torch.load(ckpt_path, map_location=device)
    cfg = DynamicsModelConfig(**blob["config"])
    model = DynamicsModel(cfg).to(device).eval()
    model.load_state_dict(blob["model_state_dict"])
    return model, cfg


def measure(model, cfg, device, *, W, T=64, B=4, seed=7):
    z1 = torch.randn(B, T, cfg.n_latents, cfg.bottleneck_dim, device=device,
                     generator=torch.Generator(device=device).manual_seed(seed))
    actions = torch.randint(0, 2, (B, T), device=device,
                            generator=torch.Generator(device=device).manual_seed(seed + 1))
    gen = torch.Generator(device=device).manual_seed(seed + 2)
    g_last, f_norm, S = relay(model, z1, actions, W=W, device=device, gen=gen, last_only=True)
    geo, _ = _per_hop_factor(g_last)
    fr = [f_norm[i + 1] / max(f_norm[i], 1e-12) for i in range(len(f_norm) - 1)]
    fwd_geo = (torch.tensor(fr).log().mean().exp()).item() if fr else float("nan")
    N = cfg.max_temporal_length
    tbptt_hops = (2 * N) // (W // 2)   # relay hops kept before the trainer's default tbptt detach (2N frames)
    return dict(W=W, S=S, fwd_geo=fwd_geo, bwd_geo=geo, tbptt_hops=tbptt_hops,
                g_last=g_last, f_norm=f_norm)


def report(tag, r):
    print(f"\n=== {tag}  W={r['W']} (slides={r['S']}) ===")
    print(f"  FORWARD |m_i|: {[round(f,1) for f in r['f_norm']]}  geo ratio/hop={r['fwd_geo']:.3f}")
    print(f"  BACKWARD (last-only) |grad m_i| newest->oldest: "
          f"{[f'{g:.2e}' for g in r['g_last']]}")
    print(f"  >>> per-hop BACKWARD factor (deeper/shallower) = {r['bwd_geo']:.3f}  "
          f"[<1 vanishing, >1 exploding, ~1 stable]")
    print(f"  >>> training keeps ~{r['tbptt_hops']} relay hops (tbptt=2N) -> "
          f"max compounding ~ {r['bwd_geo']:.2f}^{r['tbptt_hops']} = {r['bwd_geo']**r['tbptt_hops']:.1f}x")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Trained dynamics ckpt to load (else random init). The winner is "
                         "checkpoints/gridworld/dynamics_mem2mem_rollout.pt")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    if args.checkpoint is not None:
        model, cfg = _load(args.checkpoint, device)
        tag = f"TRAINED ({args.checkpoint.name})"
    else:
        cfg = DynamicsModelConfig(n_actions=2, n_memory=4, ff9_k=3)
        model = DynamicsModel(cfg).to(device).eval()
        tag = "RANDOM-INIT"
    print(f"device={device}  model={tag}  N={cfg.max_temporal_length}  "
          f"(tbptt OFF in the probe -> raw chain; forced full-noise; no-ff9/no-boot/d_min loss)")
    print("carried memory: m0=init, m_i=new_mem written at slide i; sweep window W in {4,8,16}")

    for W in (16, 8, 4):
        if W > cfg.max_temporal_length:
            continue
        report(tag, measure(model, cfg, device, W=W))


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
