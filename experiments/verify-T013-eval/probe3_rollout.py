"""V-T013-eval Probe 3 — Full beyond-window rollout: {A1,A2} x {B1,B2} color recall vs n_occ.

Builds the actual eval rollout for generate_full_state_memory under each of the 4 designs and
measures color dRGB at the reveal frame as a function of occlusion length n_occ. Window N=8,
prefix P=3. For n_occ>=N-1 the visible prefix has scrolled out of the latent window, so recall
must come from the carried memory.

Designs:
  source-tau:  A1 = tau=0 source ; A2 = tau_ctx source (near-clean).
  carry:       B1 = static: write mem_carry ONCE from the observed prefix, inject UNCHANGED each step.
               B2 = re-extract: each step, re-extract the new frame's memory (return_memory) and
                    carry it forward (the untrained memory->memory relay).

Rollout shape (per step): a 2-frame window [source_prev_latent | new_frame], memory injected at
frame 0 (source), new frame denoised at frame 1 with K shortcut steps. The carried/generated
latent of the previous step becomes the next step's source latent. Sliding the source latent
forward matches generate_memory's z_prev carry. Action-aligned.

Reference: vanilla cliff-to-chance and FF7 relay are reported elsewhere (EXP-010/012); here we
compare the four FF9 designs against each other and the no-memory floor.

Seed 0. venv/Scripts/python.exe -u.
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
from revisit_probe import detect_ball, _decode_frame

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
    dyn = DynamicsModel(dcfg); dyn.load_state_dict(dp["model_state_dict"]); dyn = dyn.to(DEV).float().eval()
    for p in dyn.parameters(): p.requires_grad_(False)
    return tok, dyn, dcfg


@torch.no_grad()
def encode_clip(tok, frames_u8, lo, hi):
    x = torch.from_numpy(frames_u8[lo:hi].astype(np.float32) / 255.0).unsqueeze(0).to(DEV)
    return tok.encoder(x)


@torch.no_grad()
def write_memory_from_prefix(dyn, z_prefix):
    B, T, L, _ = z_prefix.shape
    tau_ctx_idx = round(dyn.config.context_signal * dyn.K_max)
    tau_idx = torch.full((B, T), tau_ctx_idx, device=z_prefix.device, dtype=torch.long)
    d_idx = torch.full((B, T), dyn.n_d - 1, device=z_prefix.device, dtype=torch.long)
    tau = dyn._tau_value(tau_idx)[..., None, None]
    z_tilde = (1 - tau) * torch.randn_like(z_prefix) + tau * z_prefix
    out = dyn(z_tilde, tau_idx, d_idx, actions=None, return_memory=True)
    return out[1][:, -1]  # (1,M,E)


@torch.no_grad()
def step(dyn, mem_carry, src_lat, src_act, nxt_act, src_tau_idx, K, want_new_mem):
    """One rollout step. Returns (new_latent (1,1,L,d), new_mem (1,M,E) or None)."""
    B, _, L, D = src_lat.shape
    M, E = mem_carry.shape[-2], mem_carry.shape[-1]
    mem_in = torch.cat((mem_carry.reshape(B, 1, M, E),
                        dyn.memory_tokens.expand(B, 1, -1, -1)), dim=1)
    src_tau = dyn._tau_value(torch.tensor([src_tau_idx], device=src_lat.device)).item()
    src = (1 - src_tau) * torch.randn_like(src_lat) + src_tau * src_lat
    d_idx_val = (K).bit_length() - 1
    d_col = torch.full((B, 2), d_idx_val, device=src_lat.device, dtype=torch.long)
    tau_col = torch.zeros((B, 2), device=src_lat.device, dtype=torch.long)
    tau_col[:, 0] = src_tau_idx
    act_in = None
    if dyn.n_actions > 0 and src_act is not None:
        ids = torch.tensor([[int(src_act), int(nxt_act)]], device=src_lat.device, dtype=torch.long)
        act_in = dyn.action_features(ids)
    z = torch.randn((B, 1, L, D), device=src_lat.device)
    for kk in range(K):
        tau = kk / K
        tau_col[:, 1] = round(tau * dyn.K_max)
        inp = torch.cat((src, z), dim=1)
        z_hat1 = dyn(inp, tau_col, d_col, act_in, memory_in=mem_in)[:, 1:2]
        v = (z_hat1 - z) / (1 - tau)
        z = z + v * (1.0 / K)
    new_mem = None
    if want_new_mem:
        # re-extract: forward with the generated frame held near-clean at frame 1, read its memory
        tau_ctx_idx = round(dyn.config.context_signal * dyn.K_max)
        tau2 = torch.full((B, 2), tau_ctx_idx, device=src_lat.device, dtype=torch.long)
        tau2[:, 0] = src_tau_idx
        zc = (1 - dyn._tau_value(torch.tensor([tau_ctx_idx], device=src_lat.device)).item()) \
             * torch.randn_like(z) + dyn._tau_value(torch.tensor([tau_ctx_idx], device=src_lat.device)).item() * z
        inp2 = torch.cat((src, zc), dim=1)
        out = dyn(inp2, tau2, d_col, act_in, memory_in=mem_in, return_memory=True)
        new_mem = out[1][:, -1]  # (1,M,E)
    return z, new_mem


@torch.no_grad()
def rollout(dyn, tok, ep, P, src_tau_idx, carry_mode, K):
    """carry_mode: 'B1' static or 'B2' re-extract. Returns dRGB at reveal frame or nan."""
    z_prefix = encode_clip(tok, ep.frames, 0, P)
    mem_carry = write_memory_from_prefix(dyn, z_prefix)
    src_lat = z_prefix[:, -1:]           # last visible frame latent
    n_steps = ep.frames.shape[0] - P     # generate frames P..end
    has_act = dyn.n_actions > 0
    last = None
    for i in range(n_steps):
        cur_idx = P + i                  # frame being generated
        src_act = ep.actions[cur_idx - 1] if has_act else None
        nxt_act = ep.actions[cur_idx] if has_act else None
        want = (carry_mode == "B2")
        z, new_mem = step(dyn, mem_carry, src_lat, src_act, nxt_act, src_tau_idx, K, want)
        if carry_mode == "B2":
            mem_carry = new_mem
        src_lat = z                      # generated latent becomes next source
        last = z
    f, x, y, color = detect_ball(_decode_frame(tok, last[:, 0]))
    if not f:
        return float("nan")
    return float(np.abs(color.astype(np.float32) - ep.ball_color.astype(np.float32)).mean())


def main():
    tok, dyn, dcfg = load()
    K = dcfg.inference_steps
    tau_ctx_idx = round(dcfg.context_signal * dyn.K_max)
    P = 3
    n_ep = 32
    occ_grid = [2, 6, 8, 12, 16, 24]
    designs = [("A1", 0, "B1"), ("A2", tau_ctx_idx, "B1"),
               ("A1", 0, "B2"), ("A2", tau_ctx_idx, "B2")]
    print(f"== Probe 3: rollout color dRGB vs n_occ (P={P}, K={K}, actions={dyn.n_actions>0}, "
          f"tau_ctx_idx={tau_ctx_idx}/{dyn.K_max}, n_ep={n_ep}) ==")
    print(f"{'n_occ':>5} " + " ".join(f"{a}+{b}" .rjust(8) for a, _, b in designs))
    results = {}
    for n_occ in occ_grid:
        eps = make_probe_batch(k=n_occ, n_seeds=n_ep, P=P, R=1, seed0=5000 + n_occ)
        row = []
        for label, st, cm in designs:
            vals = [rollout(dyn, tok, ep, P, st, cm, K) for ep in eps]
            vals = [v for v in vals if np.isfinite(v)]
            m = float(np.mean(vals)) if vals else float("nan")
            row.append(m)
            results[(n_occ, label, cm)] = m
        print(f"{n_occ:>5} " + " ".join(f"{v:8.1f}" for v in row))
    np.save(_ROOT / "experiments/verify-T013-eval/probe3_raw.npy", results, allow_pickle=True)


if __name__ == "__main__":
    main()
