"""V-T013-eval Probe 1 — Is the FF9 v2 read op in-distribution at tau=0 (A1) vs tau_ctx (A2),
and is the injected memory LOAD-BEARING under each?

Reconstructs the exact FF9 read op from _ff9_loss on the TRAINED checkpoint:
  1. Write mem_t from a real causal window (the W op): run the full clip forward with all real
     latents noised at random per-frame tau (exactly loss()'s main forward), return_memory.
  2. Build a (k+1)-frame mini-window [t..t+k]; inject mem_t at frame 0, learned-init at 1..k.
  3. Predict frame t+1 latent. Compare:
       A1: source frame 0 at tau=0 (pure-noise latent) -- training-faithful read.
       A2: source frame 0 at tau_ctx (near-clean latent) -- generate_memory-style.
     Both vs an ablation where the injected memory is REPLACED with the learned-init memory
     (memory removed). If prediction degrades a lot when memory is removed, memory is
     load-bearing; if not, the model is reading the latent and memory is inert.

Decisive numbers:
  - mse_with_mem  : latent-MSE of predicted t+1 with the real written memory injected.
  - mse_no_mem    : same but memory replaced by learned-init (memory removed).
  - load_bearing  := mse_no_mem - mse_with_mem  (>0 means memory helps; ~0 means inert).
We report this for A1 and A2 separately, and for terminal-frame target = t+1 (j=1).

Seed 0. Run with venv/Scripts/python.exe -u.
"""
from __future__ import annotations
import pathlib, sys
import numpy as np
import torch

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
for _p in (_SRC, _SRC / "probe", _SRC / "C_multi_image_auto_encoder", _SRC / "D_dynamics_model"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dataclasses import fields
from video_auto_encoder import AutoEncoder, AutoEncoderConfig
from dynamics_model import DynamicsModel, DynamicsModelConfig
from probe_env import make_probe_batch

_ROOT = _SRC.parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def _cfg(d, cls):
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in allowed})


def load():
    tp = torch.load(_ROOT / "trained_autoencoder.pt", map_location=DEV, weights_only=False)
    tok = AutoEncoder(_cfg(tp["config"], AutoEncoderConfig))
    tok.load_state_dict(tp["model_state_dict"]); tok = tok.to(DEV).float().eval()
    for p in tok.parameters(): p.requires_grad_(False)
    dp = torch.load(_ROOT / "experiments/EXP-017/ff9v2_s0.pt", map_location=DEV, weights_only=False)
    dcfg = _cfg(dp["config"], DynamicsModelConfig)
    dyn = DynamicsModel(dcfg); dyn.load_state_dict(dp["model_state_dict"])
    dyn = dyn.to(DEV).float().eval()
    for p in dyn.parameters(): p.requires_grad_(False)
    return tok, dyn, dcfg


@torch.no_grad()
def encode_clip(tok, frames_u8):
    x = torch.from_numpy(frames_u8.astype(np.float32) / 255.0).unsqueeze(0).to(DEV)
    return tok.encoder(x)  # (1, T, L, d)


@torch.no_grad()
def write_memory(dyn, z1):
    """The W op: main windowed forward over real latents noised at random per-frame tau,
    return memory states. Matches loss()'s main forward (the only place memory is written)."""
    B, T, L, _ = z1.shape
    tau_idx, d_idx = dyn.sample_tau_d(B, T, z1.device)
    tau = dyn._tau_value(tau_idx)[..., None, None]
    z0 = torch.randn_like(z1)
    z_tilde = (1 - tau) * z0 + tau * z1
    out = dyn(z_tilde, tau_idx, d_idx, actions=None, return_memory=True)
    return out[1]  # (B, T, M, E)


@torch.no_grad()
def read_predict(dyn, mem_t, zw, source_tau_idx, use_real_mem, K_d_idx):
    """Inject memory (or learned-init) at frame 0 of mini-window zw, predict frame 1 latent
    by running the K-step shortcut denoiser on frame 1 (frames>=1 latents start as pure noise).
    Mirrors the inference read: source frame held at source_tau (=0 for A1, tau_ctx for A2),
    target frame 1 denoised from noise. Returns predicted z1 for frame 1 (1,L,d)."""
    B, W, L, D = zw.shape
    M, E = mem_t.shape[-2], mem_t.shape[-1]
    if use_real_mem:
        mem0 = mem_t.reshape(B, 1, M, E)
    else:
        mem0 = dyn.memory_tokens.expand(B, 1, -1, -1)
    mem_rest = dyn.memory_tokens.expand(B, W - 1, -1, -1)
    memory_in = torch.cat((mem0, mem_rest), dim=1)

    # source frame 0 latent held at source_tau (clean zw[:,0] mixed with noise)
    src_tau = dyn._tau_value(torch.tensor([source_tau_idx], device=zw.device)).item()
    src_noise = torch.randn_like(zw[:, :1])
    src = (1 - src_tau) * src_noise + src_tau * zw[:, :1]

    d_idx_val = K_d_idx
    K = 2 ** d_idx_val
    d_col = torch.full((B, W), d_idx_val, device=zw.device, dtype=torch.long)
    tau_col = torch.zeros((B, W), device=zw.device, dtype=torch.long)
    tau_col[:, 0] = source_tau_idx
    # frame 1 = target; frames>=2 stay tau=0 pure noise (causally irrelevant to frame 1)
    z = torch.randn((B, W - 1, L, D), device=zw.device)  # frames 1..W-1 start as noise
    for k in range(K):
        tau = k / K
        tau_col[:, 1] = round(tau * dyn.K_max)
        inp = torch.cat((src, z), dim=1)
        z_hat1 = dyn(inp, tau_col, d_col, actions=None, memory_in=memory_in)
        z_hat1_f1 = z_hat1[:, 1:2]
        v = (z_hat1_f1 - z[:, :1]) / (1 - tau)
        z = torch.cat((z[:, :1] + v * (1.0 / K), z[:, 1:]), dim=1)
    return z[:, :1]  # predicted frame-1 latent (1,L,d)


def main():
    tok, dyn, dcfg = load()
    K = dcfg.inference_steps
    K_d_idx = (K).bit_length() - 1
    k = dcfg.ff9_k
    tau_ctx_idx = round(dcfg.context_signal * dyn.K_max)
    n_ep = 48
    # episodes: short visible clips so all latents are real & informative (curtain UP throughout)
    eps = make_probe_batch(k=0, n_seeds=n_ep, P=2, R=k + 2, seed0=3000)  # T = k+4

    res = {"A1": {"with": [], "no": []}, "A2": {"with": [], "no": []}}
    for ep in eps:
        z1 = encode_clip(tok, ep.frames)  # (1, T, L, d)
        T = z1.shape[1]
        # pick source frame t = P-1 region; use t such that window [t..t+k] fits
        t = 2
        mem = write_memory(dyn, z1)         # (1,T,M,E)
        mem_t = mem[:, t]                   # (1,M,E)
        zw = z1[:, t:t + k + 1]             # (1,k+1,L,d)
        gt1 = z1[:, t + 1]                  # (1,L,d) GT next-frame latent
        for label, src_tau_idx in (("A1", 0), ("A2", tau_ctx_idx)):
            p_with = read_predict(dyn, mem_t, zw, src_tau_idx, True, K_d_idx)
            p_no = read_predict(dyn, mem_t, zw, src_tau_idx, False, K_d_idx)
            res[label]["with"].append(float(((p_with[:, 0] - gt1) ** 2).mean()))
            res[label]["no"].append(float(((p_no[:, 0] - gt1) ** 2).mean()))

    print(f"== Probe 1: FF9 read-op load-bearing test (n={n_ep}, K={K}, ff9_k={k}, "
          f"tau_ctx_idx={tau_ctx_idx}/{dyn.K_max}) ==")
    for label in ("A1", "A2"):
        mw = float(np.mean(res[label]["with"])); mn = float(np.mean(res[label]["no"]))
        print(f"  {label}: mse_with_mem={mw:.4f}  mse_no_mem={mn:.4f}  "
              f"load_bearing(no-with)={mn - mw:+.4f}  ratio={mn / max(mw,1e-9):.2f}x")
    # also a pure-noise-source no-memory baseline = chance for the read (A1, no mem already ~that)
    np.save(_ROOT / "experiments/verify-T013-eval/probe1_raw.npy",
            {k: {kk: np.array(vv) for kk, vv in v.items()} for k, v in res.items()},
            allow_pickle=True)


if __name__ == "__main__":
    main()
