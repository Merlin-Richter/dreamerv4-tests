"""Probe 1: n_memory=0 => new forward byte-identical to src_old forward (same state_dict + input).
Seed 0. Loads both DynamicsModel impls by file path (same module name) and compares.
"""
import importlib.util, sys, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

new = load(ROOT / "src/models/dynamics_model.py", "dyn_new")
old = load(ROOT / "src_old/models/dynamics_model.py", "dyn_old")

torch.manual_seed(0)
cfg_kw = dict(embedding_dim=32, n_heads=4, depth=8, bottleneck_dim=8, n_latents=2,
              n_registers=2, max_temporal_length=8, max_sampling_steps=16, n_memory=0,
              n_actions=0, drop_rate=0.0, att_drop_rate=0.0)

cnew = new.DynamicsModelConfig(**cfg_kw)
cold = old.DynamicsModelConfig(**cfg_kw)
mnew = new.DynamicsModel(cnew).eval()
mold = old.DynamicsModel(cold).eval()
mold.load_state_dict(mnew.state_dict())  # identical params => same names for n_memory=0

print("state_dict keys equal:", set(mnew.state_dict()) == set(mold.state_dict()))

B, T, L, D = 3, 6, 2, 8
torch.manual_seed(1)
z = torch.randn(B, T, L, D)
tau = torch.randint(0, 16, (B, T))
d = torch.randint(0, cnew.max_sampling_steps.bit_length(), (B, T))

with torch.no_grad():
    on = mnew(z, tau, d)
    oo = mold(z, tau, d)
print("shapes:", on.shape, oo.shape)
print("max abs diff:", (on - oo).abs().max().item())
print("bit-identical:", torch.equal(on, oo))

# also action-conditioned variant
cfg_kw2 = dict(cfg_kw); cfg_kw2["n_actions"] = 3
torch.manual_seed(2)
mna = new.DynamicsModel(new.DynamicsModelConfig(**cfg_kw2)).eval()
moa = old.DynamicsModel(old.DynamicsModelConfig(**cfg_kw2)).eval()
moa.load_state_dict(mna.state_dict())
aidx = torch.randint(0, 3, (B, T))
with torch.no_grad():
    an = mna.loss(z if False else torch.randn(B, T, L, D))  # exercise loss path no-crash
    fa_n = mna(z, tau, d, mna.action_features(aidx))
    fa_o = moa(z, tau, d, moa.action_features(aidx))
print("action-cond bit-identical:", torch.equal(fa_n, fa_o), "maxdiff", (fa_n-fa_o).abs().max().item())
