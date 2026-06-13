"""
Probe for T-013 FF9 verifier audit — is memory LOAD-BEARING in the FF9 loss?

Crux: in _ff9_loss, frames t+1..t+k are supplied with their OWN real latents noised at
uniformly-random tau (per _ff7_loss:464-466), and the loss is a flow loss on those frames.
If the model can predict z1[t+j] well from its own noised latent alone (no memory), then
mem_t is non-load-bearing and the FF9 objective does NOT force memory to encode hidden state.

We quantify, on REAL occluded-env latents, the achievable flow loss under two context regimes,
using the trained dynamics model my_dynamics.pt as a competent denoiser:

  L_self   : predict z1[t+1] from its own noised latent z_tilde[t+1] with an UNINFORMATIVE
             context frame at position 0 (latent = mean/absent placeholder, learned-init regs).
             == the loss FF9 can reach with EMPTY memory.
  L_oracle : predict z1[t+1] with the FULL real clean previous frame present at position 0.
             == best any context mechanism (incl. perfect memory) could do.

memory's MAXIMUM possible contribution to reducing the loss = L_self - L_oracle, as a function
of the successor's own noise level tau. If that gap is small at the tau values _ff7_loss
actually samples (uniform over [0,1)), memory is weakly pressured -> shortcut risk SUPPORTED.

Also reports the loss when the successor frame is itself heavily noised (low tau) to test the
note's own mitigation idea (open question #1: "noise frames 1..k near-fully so memory must
supply the content").

Run:
  venv/Scripts/python.exe experiments/verify-T013/probe_memory_loadbearing.py
"""
import sys, os
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "D_dynamics_model"))
sys.path.insert(0, str(ROOT / "src" / "C_multi_image_auto_encoder"))
sys.path.insert(0, str(ROOT / "src"))

from dynamics_model import DynamicsModel, DynamicsModelConfig
from video_auto_encoder import AutoEncoder, AutoEncoderConfig

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


def cfg_from(d, cls):
    import dataclasses
    fields = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in fields})


# ---- load tokenizer + dynamics ----
tok_p = torch.load(ROOT / "trained_autoencoder.pt", map_location=device, weights_only=False)
tok = AutoEncoder(cfg_from(tok_p["config"], AutoEncoderConfig)).to(device).eval()
tok.load_state_dict(tok_p["model_state_dict"])
for p in tok.parameters(): p.requires_grad_(False)

dyn_p = torch.load(ROOT / "my_dynamics.pt", map_location=device, weights_only=False)
dcfg = cfg_from(dyn_p["config"], DynamicsModelConfig)
model = DynamicsModel(dcfg).to(device).eval()
model.load_state_dict(dyn_p["model_state_dict"])
for p in model.parameters(): p.requires_grad_(False)
K_max = model.K_max
print(f"device={device} K_max={K_max} n_actions={dcfg.n_actions}")

# ---- encode some occluded clips (prefer clips that contain occlusion) ----
frames = np.load(ROOT / "occluded.npy", mmap_mode="r")
acts = np.load(ROOT / "occluded_actions.npy", mmap_mode="r")
N, T = frames.shape[0], frames.shape[1]
L = dcfg.max_temporal_length
B = 64
# pick clips where the curtain is DOWN at the successor frame (hidden state matters there)
clips, clip_acts = [], []
rng = np.random.default_rng(SEED)
tries = 0
while len(clips) < B and tries < 5000:
    tries += 1
    ep = int(rng.integers(0, N)); st = int(rng.integers(0, T - L))
    a = acts[ep, st:st+L]
    # want frame index 1 (the first successor in a t=0 / t+1 window) to be occluded
    if a[1] == 1 and a[0] == 0:   # prev visible, successor occluded == the hard memory case
        clips.append(frames[ep, st:st+L].astype(np.float32)/255.0)
        clip_acts.append(a.astype(np.int64))
if len(clips) < B:
    # fall back: any clips
    while len(clips) < B:
        ep = int(rng.integers(0, N)); st = int(rng.integers(0, T - L))
        clips.append(frames[ep, st:st+L].astype(np.float32)/255.0)
        clip_acts.append(acts[ep, st:st+L].astype(np.int64))
x = torch.from_numpy(np.stack(clips)).to(device)
action_idx = torch.from_numpy(np.stack(clip_acts)).to(device)
print(f"clips={x.shape} occluded-successor cases={tries and len(clips)}")

with torch.no_grad():
    z1 = tok.encoder(x)  # (B, L, n_lat, dim)
print("z1", z1.shape, "z1 std", z1.std().item())

n_lat, dim = z1.shape[2], z1.shape[3]
mean_lat = z1.mean(dim=(0,1), keepdim=True)  # (1,1,n_lat,dim) absent placeholder ~ data mean

actions_feat = model.action_features(action_idx)  # (B,L,n_act,E) or None


def flow_loss_at_tau(tau_succ, with_real_context):
    """Two-frame window [ctx_frame, succ_frame]. Measure flow loss on succ from its own
    noised latent at signal tau_succ. ctx frame is either the REAL previous clean latent
    (oracle) or an UNINFORMATIVE absent placeholder (empty memory).
    We average over many (t) positions: use frame pairs (t, t+1) for t in 0..L-2.
    """
    losses = []
    for t in range(L - 1):
        z_prev = z1[:, t:t+1]      # (B,1,n_lat,dim) real previous frame
        z_succ = z1[:, t+1:t+2]    # (B,1,n_lat,dim) successor (target)
        if with_real_context:
            ctx = z_prev
        else:
            ctx = mean_lat.expand(z_prev.shape).clone()  # absent / empty memory
        # build 2-frame window
        tau_ctx = model.config.context_signal
        tau_ctx_idx = min(round(tau_ctx * K_max), K_max - 1)
        tau_succ_idx = min(round(tau_succ * K_max), K_max - 1)
        tau_idx = torch.tensor([[tau_ctx_idx, tau_succ_idx]], device=device).expand(z1.shape[0], 2).contiguous()
        d_idx = torch.full((z1.shape[0], 2), model.n_d - 1, device=device, dtype=torch.long)
        zw = torch.cat([ctx, z_succ], dim=1)  # (B,2,n_lat,dim)
        tau = model._tau_value(tau_idx)[..., None, None]
        noise = torch.randn_like(zw)
        z_tilde = (1 - tau) * noise + tau * zw
        act_in = None
        if actions_feat is not None:
            act_in = torch.cat([actions_feat[:, t:t+1], actions_feat[:, t+1:t+2]], dim=1)
        with torch.no_grad():
            z_hat = model(z_tilde, tau_idx, d_idx, act_in)
        l = ((z_hat[:, 1] - zw[:, 1]) ** 2).mean().item()
        losses.append(l)
    return float(np.mean(losses))


print("\n=== Flow loss on successor frame vs its own signal level tau_succ ===")
print(" tau_succ |  L_self(empty mem) |  L_oracle(real ctx) |  gap = max mem benefit | gap/L_self")
taus = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
rows = []
for ts in taus:
    Ls = flow_loss_at_tau(ts, with_real_context=False)
    Lo = flow_loss_at_tau(ts, with_real_context=True)
    gap = Ls - Lo
    rows.append((ts, Ls, Lo, gap, gap/Ls if Ls>0 else 0))
    print(f"  {ts:5.2f}   |   {Ls:10.5f}      |   {Lo:10.5f}       |   {gap:10.5f}        | {gap/Ls if Ls>0 else 0:6.2%}")

# Loss the FF7/FF9 objective ACTUALLY samples: tau_succ ~ U[0,1). Integrate over the grid.
print("\n=== Expected loss under the ACTUAL _ff7_loss tau sampling (tau_succ ~ U[0,1)) ===")
many = []
for ts in np.linspace(0.0, 1.0, 21)[1:]:  # avoid tau=0 (degenerate /(1-tau) not used here, plain flow)
    Ls = flow_loss_at_tau(float(ts), with_real_context=False)
    Lo = flow_loss_at_tau(float(ts), with_real_context=True)
    many.append((ts, Ls, Lo))
many = np.array(many)
ELs, ELo = many[:,1].mean(), many[:,2].mean()
print(f"  E[L_self]   (empty memory, FF9 reachable w/o mem) = {ELs:.5f}")
print(f"  E[L_oracle] (perfect context/memory)             = {ELo:.5f}")
print(f"  E[max memory benefit] = {ELs-ELo:.5f}   ({(ELs-ELo)/ELs:.1%} of E[L_self])")
print(f"\n  total raw variance of z1 (E[z1^2] about 0)        = {(z1**2).mean().item():.5f}")
print(f"  variance of z1 about its mean                    = {((z1-mean_lat)**2).mean().item():.5f}")

# ============================================================================
# FF9-SPECIFIC: does WITHHOLDING frame-0's latent (FF9) vs keeping real frame-0
# (FF7) change the SUCCESSOR loss when memory is uninformative?  If both give the
# same successor loss, then frame-0 withholding is INERT w.r.t. the gradient on the
# successor's own denoiser, and memory's only job is identical to FF7's.
# Here we compare k=1 successor loss under (a) FF7-style real frame0, (b) FF9-style
# absent frame0 -- both with NON-informative memory/registers (learned init).
# ============================================================================
print("\n=== FF9 frame-0 withholding effect on successor loss (empty memory) ===")
print(" tau_succ | succ-loss FF7(real f0) | succ-loss FF9(absent f0) | delta")
for ts in [0.1, 0.3, 0.5, 0.7, 0.9]:
    Lff7 = flow_loss_at_tau(ts, with_real_context=True)   # real prev frame present (FF7 puts real latent at f0)
    Lff9 = flow_loss_at_tau(ts, with_real_context=False)  # absent placeholder at f0 (FF9)
    print(f"  {ts:5.2f}   |     {Lff7:10.5f}        |      {Lff9:10.5f}         | {Lff9-Lff7:+.5f}")
print("(NB: with_real_context=True here = FF7's real f0; =False = FF9's absent f0. The successor")
print(" still always carries its OWN noised latent in both. delta = what memory must recover.)")
