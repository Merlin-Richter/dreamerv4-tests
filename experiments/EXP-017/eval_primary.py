"""
EXP-017 FF9 v2 PRIMARY readouts (independent of the beyond-window rollout design):

  1. No-regression diffusion: FF9 v2's diffusion-only loss on the EXACT held-out val split
     (replicates train_dynamics_model.py: Generator(0), 5%, chunk_len=16, start_offset=0),
     vs vanilla_s0's val ~0.0066. Tripwire D-024: base dynamics must not regress.

  2. Memory sufficiency L(mem) << L(no-mem): a faithful replication of the FF9 v2 READ op
     (`_ff9_loss`): write each frame's memory from a near-clean window, then predict frame
     t+j from a pure-noise (tau=0) path with the written memory injected at the source frame
     vs learned-init (no memory) vs chance/copy-last. The gap = memory's contribution; if
     L(mem) << L(no-mem) ~ chance, the memory tokens encode a load-bearing full-state object.

Run: venv/Scripts/python.exe -u experiments/EXP-017/eval_primary.py
"""
import sys, json, dataclasses
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
CKPT = ROOT / "experiments" / "EXP-017" / "ff9v2_s0.pt"
TOK = ROOT / "trained_autoencoder.pt"
FRAMES = ROOT / "occluded.npy"
ACTIONS = ROOT / "occluded_actions.npy"
OUT = ROOT / "experiments" / "EXP-017" / "primary.json"


def cfg_from(d, cls):
    fields = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in fields})


# ---------------- load ----------------
tok_p = torch.load(TOK, map_location=device, weights_only=False)
tok = AutoEncoder(cfg_from(tok_p["config"], AutoEncoderConfig)).to(device).eval()
tok.load_state_dict(tok_p["model_state_dict"])
for p in tok.parameters(): p.requires_grad_(False)

dyn_p = torch.load(CKPT, map_location=device, weights_only=False)
dcfg = cfg_from(dyn_p["config"], DynamicsModelConfig)
model = DynamicsModel(dcfg).to(device).eval()
model.load_state_dict(dyn_p["model_state_dict"])
for p in model.parameters(): p.requires_grad_(False)
K_max, n_d = model.K_max, model.n_d
M = dcfg.n_memory
L_clip = dcfg.max_temporal_length            # 16
ff9_k = dcfg.ff9_k
print(f"device={device} K_max={K_max} n_memory={M} ff9_k={ff9_k} L={L_clip} n_actions={dcfg.n_actions}")

frames = np.load(FRAMES, mmap_mode="r")
acts = np.load(ACTIONS, mmap_mode="r")
N, T = frames.shape[0], frames.shape[1]


@torch.no_grad()
def encode(clip_u8):
    x = torch.from_numpy(np.asarray(clip_u8).astype(np.float32) / 255.0).to(device)
    return tok.encoder(x)


# ============================================================
# PART 1 — no-regression diffusion on the exact val split
# ============================================================
def val_split_indices():
    n = N
    n_val = min(max(1, int(round(n * 0.05))), n - 1)
    g = torch.Generator().manual_seed(0)          # train script uses fixed seed 0 for the split
    perm = torch.randperm(n, generator=g)
    return perm[:n_val].numpy()


@torch.no_grad()
def part1_no_regression(n_passes=3, batch=64):
    val_idx = val_split_indices()
    # clips: each val episode, frames[ep, 0:L] (start_offset=0), with actions
    print(f"\n[part1] val episodes={len(val_idx)} (chunk_len={L_clip}, start_offset=0)")
    accum = {"diffusion": [], "ff9": [], "total": []}
    for _pass in range(n_passes):
        tot = {"diffusion": 0.0, "ff9": 0.0, "total": 0.0}
        nb = 0
        for i in range(0, len(val_idx), batch):
            ep = val_idx[i:i + batch]
            clip = np.stack([frames[e, 0:L_clip] for e in ep])
            a = torch.from_numpy(np.stack([np.asarray(acts[e, 0:L_clip]) for e in ep]).astype(np.int64)).to(device)
            z1 = encode(clip)
            total, parts = model.loss(z1, a, ff9_k=ff9_k, lambda_ff9=1.0, return_parts=True)
            tot["total"] += float(total); tot["diffusion"] += float(parts["diffusion"])
            tot["ff9"] += float(parts.get("ff9", 0.0)); nb += 1
        for k in tot: accum[k].append(tot[k] / nb)
    res = {k: float(np.mean(v)) for k, v in accum.items()}
    res["diffusion_std"] = float(np.std(accum["diffusion"]))
    print(f"[part1] val diffusion = {res['diffusion']:.5f} (+/-{res['diffusion_std']:.5f}, {n_passes} passes) "
          f"| ff9 = {res['ff9']:.5f} | total = {res['total']:.5f}")
    print(f"[part1] vanilla_s0 val ref ~0.0066 -> {'NO REGRESSION' if res['diffusion'] < 0.0066*1.5 else 'CHECK'}")
    return res


# ============================================================
# PART 2 — memory sufficiency: L(mem) vs L(no-mem) vs chance
# faithful replication of _ff9_loss READ op.
# ============================================================
@torch.no_grad()
def part2_memory_sufficiency(n_clips=512, src_positions=(4, 8, 11), tau_terms=(0.0, 0.1, 0.3, 0.5, 0.9)):
    rng = np.random.default_rng(SEED)
    # sample clips of length L_clip
    eps = rng.integers(0, N, size=n_clips)
    starts = rng.integers(0, T - L_clip, size=n_clips)
    clip = np.stack([frames[e, s:s + L_clip] for e, s in zip(eps, starts)])
    a_np = np.stack([np.asarray(acts[e, s:s + L_clip]) for e, s in zip(eps, starts)]).astype(np.int64)
    z1 = encode(clip)                                   # (B, L, Lat, dim)
    a = torch.from_numpy(a_np).to(device)
    act_feat = model.action_features(a)                 # (B, L, n_act, E) or None
    B, _, Lat, dim = z1.shape

    # write memory from a near-clean full window
    tau_ctx = model.config.context_signal
    tau_ctx_idx = min(round(tau_ctx * K_max), K_max - 1)
    tau_col = torch.full((B, L_clip), tau_ctx_idx, device=device, dtype=torch.long)
    d_col = torch.full((B, L_clip), n_d - 1, device=device, dtype=torch.long)
    tauc = model._tau_value(tau_col)[..., None, None]
    z_near = (1 - tauc) * torch.randn_like(z1) + tauc * z1
    _, mem = model(z_near, tau_col, d_col, act_feat, return_memory=True)   # (B, L, M, E)

    # references
    mean_lat = z1.mean(dim=(0, 1), keepdim=True)
    chance = float(((z1 - mean_lat) ** 2).mean())

    def read_mse(t, j, tau_term, inject):
        """MSE predicting frame t+j from a tau=0 path; inject=mem at source frame 0 or learned-init."""
        w = j + 1
        zw = z1[:, t:t + w]
        tau_idx = torch.zeros(B, w, dtype=torch.long, device=device)
        tau_idx[:, j] = min(round(tau_term * K_max), K_max - 1)
        d_idx = torch.full((B, w), n_d - 1, dtype=torch.long, device=device)
        tau = model._tau_value(tau_idx)[..., None, None]
        z_tilde = (1 - tau) * torch.randn_like(zw) + tau * zw
        if inject:
            memory_in = model.memory_tokens.expand(B, w, -1, -1).clone()
            memory_in[:, 0] = mem[:, t]
        else:
            memory_in = None
        act_in = act_feat[:, t:t + w] if act_feat is not None else None
        z_hat = model(z_tilde, tau_idx, d_idx, act_in, memory_in=memory_in)
        return float(((z_hat[:, j] - zw[:, j]) ** 2).mean())

    out = {"chance": chance, "by_j": {}}
    print(f"\n[part2] memory sufficiency (B={B}, src t in {src_positions}); chance(var)={chance:.4f}")
    for j in range(1, ff9_k + 1):
        # copy-last: freeze source latent
        cl = float(np.mean([((z1[:, t + j] - z1[:, t]) ** 2).mean().item() for t in src_positions]))
        rows = {}
        for tt in tau_terms:
            Lm = float(np.mean([read_mse(t, j, tt, inject=True) for t in src_positions]))
            Ln = float(np.mean([read_mse(t, j, tt, inject=False) for t in src_positions]))
            rows[f"{tt:.2f}"] = {"mem": Lm, "no_mem": Ln, "gap": Ln - Lm,
                                 "gap_frac": (Ln - Lm) / Ln if Ln > 0 else 0.0}
        out["by_j"][j] = {"copy_last": cl, "tau_term": rows}
        print(f"  j={j}  copy_last={cl:.4f}")
        print(f"    tau_term |  L(mem)  | L(no_mem) |   gap   | gap/no_mem")
        for tt in tau_terms:
            r = rows[f"{tt:.2f}"]
            print(f"      {tt:4.2f}    | {r['mem']:.4f}  |  {r['no_mem']:.4f}  | {r['gap']:+.4f} |  {r['gap_frac']:6.1%}")
    return out


if __name__ == "__main__":
    results = {"ckpt": str(CKPT.relative_to(ROOT)), "part1_no_regression": part1_no_regression(),
               "part2_memory_sufficiency": part2_memory_sufficiency()}
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
