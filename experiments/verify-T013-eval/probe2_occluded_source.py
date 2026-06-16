"""V-T013-eval Probe 2 — In the REAL eval regime the source latent is an OCCLUDED (curtain)
frame with no ball/color. Does A2 (near-clean source) still let the model read color from
injected memory, or does the near-clean curtain latent (no color) leave memory load-bearing too?

This is the fair test of A2's confound: probe1 used visible source frames (latent HAS color, so
A2 cheated via the latent). Here the source frame the memory is injected at is a CURTAIN frame.
We:
  1. Build [P up | k down] episodes; encode.
  2. Write mem from the VISIBLE prefix window (the W op on the prefix latents) -> mem carries color.
  3. Inject mem at a CURTAIN source frame; predict the NEXT curtain frame; decode and read color.
     A1: curtain source at tau=0.  A2: curtain source at tau_ctx (near-clean curtain latent).
  4. Compare decoded color dRGB to GT ball color, and a no-memory ablation (learned-init mem).

If A2 still recalls color (low dRGB) AND memory is load-bearing under A2 here, then the inertness
seen in probe1 was specific to visible sources and A2 is NOT inert in the eval regime. If A2's
color recall collapses or memory is inert, A1 is the faithful read.

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
    """W op on the visible prefix window (held near-clean, as at inference the observed prefix
    is real). Use tau_ctx so memory is written from a near-clean observed window (most faithful
    to 'write from observations'). Returns last-frame memory (1,M,E)."""
    B, T, L, _ = z_prefix.shape
    tau_ctx_idx = round(dyn.config.context_signal * dyn.K_max)
    tau_idx = torch.full((B, T), tau_ctx_idx, device=z_prefix.device, dtype=torch.long)
    d_idx = torch.full((B, T), dyn.n_d - 1, device=z_prefix.device, dtype=torch.long)
    noise = torch.randn_like(z_prefix)
    tau = dyn._tau_value(tau_idx)[..., None, None]
    z_tilde = (1 - tau) * noise + tau * z_prefix
    out = dyn(z_tilde, tau_idx, d_idx, actions=None, return_memory=True)
    return out[1][:, -1]  # (1,M,E)


@torch.no_grad()
def read_one(dyn, mem_t, src_latent, src_action, nxt_action, src_tau_idx, use_real_mem, K):
    """Inject mem at a single source frame, predict the next frame (2-frame window).
    src_latent: (1,1,L,d) clean source frame latent (a curtain frame in eval). actions optional."""
    B, _, L, D = src_latent.shape
    M, E = mem_t.shape[-2], mem_t.shape[-1]
    mem0 = mem_t.reshape(B, 1, M, E) if use_real_mem else dyn.memory_tokens.expand(B, 1, -1, -1)
    mem_in = torch.cat((mem0, dyn.memory_tokens.expand(B, 1, -1, -1)), dim=1)  # (1,2,M,E)
    src_tau = dyn._tau_value(torch.tensor([src_tau_idx], device=src_latent.device)).item()
    src = (1 - src_tau) * torch.randn_like(src_latent) + src_tau * src_latent
    d_idx_val = (K).bit_length() - 1
    d_col = torch.full((B, 2), d_idx_val, device=src_latent.device, dtype=torch.long)
    tau_col = torch.zeros((B, 2), device=src_latent.device, dtype=torch.long)
    tau_col[:, 0] = src_tau_idx
    act_in = None
    if dyn.n_actions > 0 and src_action is not None:
        ids = torch.tensor([[int(src_action), int(nxt_action)]], device=src_latent.device, dtype=torch.long)
        act_in = dyn.action_features(ids)
    z = torch.randn((B, 1, L, D), device=src_latent.device)
    for k in range(K):
        tau = k / K
        tau_col[:, 1] = round(tau * dyn.K_max)
        inp = torch.cat((src, z), dim=1)
        z_hat1 = dyn(inp, tau_col, d_col, act_in, memory_in=mem_in)[:, 1:2]
        v = (z_hat1 - z) / (1 - tau)
        z = z + v * (1.0 / K)
    return z  # (1,1,L,d)


def main():
    tok, dyn, dcfg = load()
    K = dcfg.inference_steps
    tau_ctx_idx = round(dcfg.context_signal * dyn.K_max)
    has_act = dyn.n_actions > 0
    P, k = 3, 8
    n_ep = 48
    # k occluded frames then a reveal (R=1). reveal_index = P+k (curtain UP).
    eps = make_probe_batch(k=k, n_seeds=n_ep, P=P, R=1, seed0=4000)

    # Inject memory at the LAST curtain frame (index P+k-1, curtain DOWN) and predict the REVEAL
    # frame (index P+k, curtain UP) so the ball is rendered and color is readable. The source
    # latent is a curtain frame -> NO color in the latent -> recall must come from memory.
    out = {("A1", True): [], ("A1", False): [], ("A2", True): [], ("A2", False): []}
    found = {("A1", True): 0, ("A1", False): 0, ("A2", True): 0, ("A2", False): 0}
    for ep in eps:
        z_prefix = encode_clip(tok, ep.frames, 0, P)           # (1,P,L,d) visible
        mem_t = write_memory_from_prefix(dyn, z_prefix)        # (1,M,E) color carrier
        src_idx = P + k - 1                                     # last curtain frame
        rev_idx = P + k                                         # reveal frame (curtain UP)
        src_lat = encode_clip(tok, ep.frames, 0, src_idx + 1)[:, -1:]  # (1,1,L,d) curtain
        src_act = ep.actions[src_idx] if has_act else None
        nxt_act = ep.actions[rev_idx] if has_act else None
        for label, st in (("A1", 0), ("A2", tau_ctx_idx)):
            for um in (True, False):
                pred = read_one(dyn, mem_t, src_lat, src_act, nxt_act, st, um, K)
                f, x, y, color = detect_ball(_decode_frame(tok, pred[:, 0]))
                if f:
                    out[(label, um)].append(float(np.abs(color.astype(np.float32) -
                                            ep.ball_color.astype(np.float32)).mean()))
                    found[(label, um)] += 1
    print(f"== Probe 2: occluded-source read (inject mem at curtain frame, predict next curtain) "
          f"n={n_ep} K={K} actions={has_act} tau_ctx_idx={tau_ctx_idx}/{dyn.K_max} ==")
    print("  (color dRGB vs GT ball color; lower=better recall; found=ball visible in prediction)")
    for label in ("A1", "A2"):
        for um, name in ((True, "real_mem"), (False, "no_mem")):
            vals = out[(label, um)]
            mean = float(np.mean(vals)) if vals else float("nan")
            print(f"  {label} {name:8s}: dRGB={mean:6.1f}  found={found[(label,um)]}/{n_ep}")
        lb = ((float(np.mean(out[(label,False)])) if out[(label,False)] else float('nan'))
              - (float(np.mean(out[(label,True)])) if out[(label,True)] else float('nan')))
        print(f"  {label} load_bearing(no-real dRGB)= {lb:+.1f}")


if __name__ == "__main__":
    main()
