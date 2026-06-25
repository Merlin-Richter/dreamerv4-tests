"""Probe 2: rollout_step(commit=False) is read-only — leaves cache K/V AND next_pos unmutated,
and a subsequent commit=True step is identical whether or not a read-only branch preceded it
(carried *deterministic* state intact). Uses a memory model (n_memory>0). Seed 0.
"""
import importlib.util, sys, torch, copy
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("dyn", ROOT / "src/models/dynamics_model.py")
dyn = importlib.util.module_from_spec(spec); sys.modules["dyn"] = dyn; spec.loader.exec_module(dyn)

torch.manual_seed(0)
cfg = dyn.DynamicsModelConfig(embedding_dim=32, n_heads=4, depth=8, bottleneck_dim=8, n_latents=2,
                              n_registers=2, n_memory=3, ff9_k=0, max_temporal_length=6,
                              max_sampling_steps=16, n_actions=2, drop_rate=0.0, att_drop_rate=0.0)
m = dyn.DynamicsModel(cfg).eval()

B = 2
ctx = torch.randn(B, 4, 2, 8)
ctx_a = torch.zeros(B, 4, dtype=torch.long)

def snapshot(cache):
    return [None if lc is None or lc.get('k') is None
            else (lc['k'].clone(), lc['v'].clone()) for lc in cache]

def cache_equal(cache, snap):
    for lc, s in zip(cache, snap):
        if s is None:
            if lc is not None and lc.get('k') is not None: return False
            continue
        if not (torch.equal(lc['k'], s[0]) and torch.equal(lc['v'], s[1])): return False
    return True

# Build a state and advance a few committed steps so the cache is non-trivial.
torch.manual_seed(1)
state = m.rollout_init(ctx, ctx_a, K=4)
for _ in range(3):
    m.rollout_step(state, torch.ones(B, dtype=torch.long), commit=True)

snap = snapshot(state["cache"]); pos0 = state["next_pos"]
# Many read-only reveal branches.
for _ in range(5):
    m.rollout_step(state, torch.zeros(B, dtype=torch.long), commit=False)
print("cache unchanged after 5 read-only branches:", cache_equal(state["cache"], snap))
print("next_pos unchanged:", state["next_pos"] == pos0, "(", pos0, "->", state["next_pos"], ")")

# Determinism: committed step identical with vs without an interleaved read-only branch
# (fix RNG around each so only the branch's side-effects could differ).
sA = m.rollout_init(ctx, ctx_a, K=4)
for _ in range(3): m.rollout_step(sA, torch.ones(B, dtype=torch.long), commit=True)
sB = copy.deepcopy(sA)
g = torch.Generator().manual_seed(99)
torch.manual_seed(7); _ = m.rollout_step(sB, torch.zeros(B, dtype=torch.long), commit=False)  # branch first
torch.manual_seed(123); zA = m.rollout_step(sA, torch.ones(B, dtype=torch.long), commit=True)
torch.manual_seed(123); zB = m.rollout_step(sB, torch.ones(B, dtype=torch.long), commit=True)
print("committed step identical w/ vs w/o preceding read-only branch:", torch.equal(zA, zB),
      "maxdiff", (zA-zB).abs().max().item())
print("caches identical after that commit:", cache_equal(sA["cache"], snapshot(sB["cache"])))
