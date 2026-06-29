"""Measure the per-hop carried-memory gradient scale on REAL encoded latents at init, to set the
per-hop relay grad-clip cap for the normalized no-FF9 run. Reports per-batch-element |grad| of each
carried memory tensor (the quantity the clip caps), under the clean no-FF9 loss (full 50/50 + forced
noise), W in {4,8,16}. Random init = the START of training, which is where the explosion lives.
"""
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "experiments" / "mem2mem"))
from models.dynamics_model import DynamicsModel, DynamicsModelConfig          # noqa: E402
from training.train_dynamics import load_tokenizer, encode_frames            # noqa: E402
import rollout as R                                                          # noqa: E402


def carried_grad_norms(model, z1, actions, *, W, device, gen, force_mode):
    """Per-batch-element |grad| of each carried memory tensor under the no-ff9/no-boot/d_min loss."""
    N = model.config.max_temporal_length
    B, T, L, D = z1.shape
    half = W // 2
    _, tau_ctx_idx, d_idx_val = R._tau_d_consts(model)
    af_all = model.action_features(actions)
    blank_half = model.memory_tokens.expand(B, half, -1, -1)
    d_col_W = torch.full((B, W), d_idx_val, device=device, dtype=torch.long)
    pos = torch.arange(W, device=device)
    af = lambda a, b: (af_all[:, a:b] if af_all is not None else None)

    zc = model._noise_to_ctx(z1[:, :W])
    tau_init = torch.full((B, W), tau_ctx_idx, device=device, dtype=torch.long)
    blank_W = model.memory_tokens.expand(B, W, -1, -1)
    _, mem_win = model(zc, tau_init, d_col_W, af(0, W), memory_in=blank_W, positions=pos, return_memory=True)
    old_mem = mem_win[:, half:]; old_mem.retain_grad()
    carried = [old_mem]; losses = []
    s = half; end = min(T, min(5 * N, 10 * W))
    while s + W <= end:
        z1_win = z1[:, s:s + W]
        z0 = torch.randn(z1_win.shape, device=device, generator=gen)
        modes = R._sample_modes(B, device, gen, force_mode); m = modes.view(B, 1, 1, 1).float()
        old_clean = model._noise_to_ctx(z1_win[:, :half])
        old_part = m[:, :1] * z0[:, :half] + (1 - m[:, :1]) * old_clean
        tau_old = torch.where(modes.view(B, 1), torch.zeros(B, half, device=device, dtype=torch.long),
                              torch.full((B, half), tau_ctx_idx, device=device, dtype=torch.long))
        tau_new_idx, d_new_idx = R._sample_tau_d(model, B, half, device, gen, n_d_unlocked=1)
        tau_new_idx = torch.where(modes.view(B, 1), torch.zeros_like(tau_new_idx), tau_new_idx)
        tau_new = model._tau_value(tau_new_idx)[..., None, None]
        new_part = (1 - tau_new) * z0[:, half:] + tau_new * z1_win[:, half:]
        memory_in = torch.cat([old_mem, blank_half], dim=1)
        loss, new_mem, _ = R._newhalf_loss(model, old_part=old_part, tau_old=tau_old, new_part=new_part,
            tau_new_idx=tau_new_idx, d_new_idx=d_new_idx, z1_new=z1_win[:, half:], af_win=af(s, s + W),
            memory_in=memory_in, positions=pos, half=half, bootstrap=False)
        new_mem.retain_grad(); carried.append(new_mem); losses.append(loss)
        old_mem = new_mem; s += half
    (sum(losses) / len(losses)).backward()
    # per-batch-element grad norm of each carried tensor (this is what the hook caps)
    return [c.grad.flatten(1).norm(dim=1) for c in carried if c.grad is not None]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    raw = np.load(_ROOT / "data" / "gridworld.npy", mmap_mode="r")
    acts_np = np.load(_ROOT / "data" / "gridworld_actions.npy")
    n_actions = int(acts_np.max()) + 1
    B, T = 8, 64
    frames = torch.from_numpy(np.ascontiguousarray(raw[:B, :T]).astype(np.float32) / 255.0).to(device)
    actions = torch.from_numpy(acts_np[:B, :T]).long().to(device)
    tok = load_tokenizer(_ROOT / "checkpoints" / "gridworld" / "tokenizer.pt", device)
    tok_T = int(getattr(getattr(tok, "config", None), "max_temporal_length", 16))
    with torch.no_grad():
        z1 = torch.cat([encode_frames(tok, frames[:, i:i + tok_T]) for i in range(0, T, tok_T)], dim=1).float()
    print(f"device={device} z1 {tuple(z1.shape)} |z1|/elem rms={z1.pow(2).mean().sqrt():.3f} n_actions={n_actions}\n")

    cfg = DynamicsModelConfig(n_actions=n_actions, n_memory=4, ff9_k=3)
    for W in (16, 8, 4):
        torch.manual_seed(0)
        model = DynamicsModel(cfg).to(device).eval()
        gen = torch.Generator(device=device).manual_seed(7)
        gns = carried_grad_norms(model, z1, actions, W=W, device=device, gen=gen, force_mode=None)  # 50/50
        allv = torch.cat([g for g in gns])
        print(f"W={W}: hops={len(gns)}  per-batch carried |grad| "
              f"min={allv.min():.2e} median={allv.median():.2e} max={allv.max():.2e}  "
              f"deepest-hop mean={gns[0].mean():.2e} nearest-hop mean={gns[-1].mean():.2e}")


if __name__ == "__main__":
    main()
