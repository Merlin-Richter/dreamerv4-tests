"""
EXP-029 P1 — DYNAMIC-state relay credit probe + tbptt-k sweep.

Extends experiments/verify-T014/probe_detached_relay_v2.py from a STATIC secret to a DYNAMIC one
(the GridWorld-relevant case: position that must be INTEGRATED each hop, not just preserved), and
sweeps tbptt-k to find the minimum gradient depth that extrapolates.

Question 1 (capacity-vs-credit): can ANY relay (full BPTT ceiling) carry DYNAMIC state across many
  hops? If BPTT fails -> the deficit is capacity/representability, not credit -> widen memory first.
Question 2 (min depth): what is the smallest tbptt-k that matches full BPTT past the training depth?
  That k is the truncation horizon for the FF9 rollout-training loss (C1 in the design note).

Task: a 1-D bouncing position on [0, GRID). Each hop the position steps by a velocity that reflects
off the walls (deterministic transition, like GridWorld). The secret (initial pos+vel) is supplied
to the writer at hop 0 ONLY; thereafter the input carries only the (known) action-free dynamics flag
(pure noise in the secret channels). The reader must output the CURRENT position at every hop -> the
relay MUST apply the bounce transition each hop, not copy a static code. Determinism => a wrong
output is always a genuine error (no valid-but-wrong branch; mirrors GridWorld, see design note §4).

Modes: no_relay (floor), tbptt1/2/4/8/16 (grad kept k hops, value carried faithfully), bptt (ceiling).

Run (Windows): venv/Scripts/python.exe experiments/EXP-029-design/probe_dynamic_relay.py
Reuses the V-T014 harness shape; ~30-60 min on CPU/4070. SEED reported.
"""
import json
from pathlib import Path
import torch
import torch.nn as nn

SEED = 0
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(__file__).resolve().parent

MEM_DIM, HID = 32, 64
X_DIM = 8                      # input channels; first 2 carry the secret (pos, vel) at hop 0
GRID = 6.0                     # 1-D world width (GridWorld is 6x6; 1-D suffices for the credit test)
HOPS, EVAL_HOPS = 32, 200      # train depth 32, extrapolate to 200 (6x)
BATCH, STEPS, LR = 256, 4000, 3e-3
TBPTT_KS = [1, 2, 4, 8, 16]


class Writer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.GRUCell(X_DIM, MEM_DIM)
        self.init_mem = nn.Parameter(0.05 * torch.randn(MEM_DIM))

    def forward(self, m_prev, x):
        return self.cell(x, m_prev)


class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(MEM_DIM, HID), nn.SiLU(), nn.Linear(HID, 1))

    def forward(self, m):
        return self.net(m)  # predict CURRENT position


def make_episode(batch, hops, gen):
    """Deterministic 1-D bounce. Returns (positions (batch,hops), x (batch,hops,X_DIM))."""
    pos0 = GRID * torch.rand(batch, generator=gen, device=device)
    vel = (2.0 * torch.rand(batch, generator=gen, device=device) - 1.0)  # in [-1,1] units/hop
    pos = torch.empty(batch, hops, device=device)
    p, v = pos0.clone(), vel.clone()
    for t in range(hops):
        pos[:, t] = p
        p = p + v
        # reflect off [0, GRID]
        over = p > GRID
        under = p < 0.0
        p = torch.where(over, 2 * GRID - p, p)
        p = torch.where(under, -p, p)
        v = torch.where(over | under, -v, v)
    x = 0.5 * torch.randn(batch, hops, X_DIM, generator=gen, device=device)
    x[:, 0, 0] = pos0           # secret: initial position
    x[:, 0, 1] = vel            # secret: initial velocity
    x[:, 0, -1] = 1.0           # hop-0 flag
    x[:, 1:, -1] = 0.0
    return pos, x


def run(writer, reader, pos, x, mode, tbptt_k=None):
    B, H, _ = x.shape
    m = writer.init_mem.expand(B, MEM_DIM)
    total, cnt = 0.0, 0
    per_hop = torch.zeros(H, device=device)
    for t in range(H):
        if mode == "no_relay":
            m_prev = writer.init_mem.expand(B, MEM_DIM)
        else:
            m_prev = m  # carried value (detach cadence handled below for tbptt)
        m = writer(m_prev, x[:, t])
        pred = reader(m).squeeze(-1)
        err = ((pred - pos[:, t]) ** 2).mean()
        total = total + err; cnt += 1
        with torch.no_grad():
            per_hop[t] = err.item()
        # tbptt-k: detach the carry every k hops so grad flows back at most k hops
        if mode == "tbptt" and ((t + 1) % tbptt_k == 0):
            m = m.detach()
    return total / cnt, per_hop


def train(mode, tbptt_k=None):
    torch.manual_seed(SEED)
    gen = torch.Generator(device=device); gen.manual_seed(SEED + 1)
    w, r = Writer().to(device), Reader().to(device)
    opt = torch.optim.Adam(list(w.parameters()) + list(r.parameters()), lr=LR)
    for step in range(STEPS):
        pos, x = make_episode(BATCH, HOPS, gen)
        loss, _ = run(w, r, pos, x, mode, tbptt_k)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0 or step == STEPS - 1:
            tag = f"{mode}{'' if tbptt_k is None else f'-{tbptt_k}'}"
            print(f"  [{tag}] step {step:4d} loss {loss.item():.4f}", flush=True)
    return w, r


@torch.no_grad()
def evaluate(w, r, hops):
    gen = torch.Generator(device=device); gen.manual_seed(SEED + 777)
    pos, x = make_episode(BATCH, hops, gen)
    _, per_hop = run(w, r, pos, x, "bptt")  # carry faithfully at eval (same values for all carry modes)
    return per_hop.cpu()


def main():
    # chance ~ Var of position over the bounce; estimate empirically
    gen = torch.Generator(device=device); gen.manual_seed(SEED + 777)
    pos, _ = make_episode(BATCH, EVAL_HOPS, gen)
    chance = ((pos - pos.mean()) ** 2).mean().item()
    print(f"chance (predict-mean MSE) ~ {chance:.3f}\n", flush=True)

    res, curves = {}, {}
    arms = [("no_relay", None), ("bptt", None)] + [("tbptt", k) for k in TBPTT_KS]
    for mode, k in arms:
        name = mode if k is None else f"tbptt{k}"
        print(f"=== {name} ===", flush=True)
        w, r = train(mode, k)
        err = evaluate(w, r, EVAL_HOPS)
        depths = [1, 2, 4, 8, 16, 31, 50, 100, 150, 199]
        res[name] = {d: round(err[d].item(), 4) for d in depths}
        curves[name] = err.tolist()
        print(f"  recovery MSE by depth: {res[name]}\n", flush=True)

    print("========= SUMMARY (chance ~ %.3f) =========" % chance, flush=True)
    hdr = "depth | no_relay |   bptt   | " + " | ".join(f"tbptt{k:<2d}" for k in TBPTT_KS)
    print(hdr, flush=True)
    for d in [1, 16, 31, 50, 100, 199]:
        row = f"{d:5d} | {res['no_relay'][d]:8.4f} | {res['bptt'][d]:8.4f} | "
        row += " | ".join(f"{res[f'tbptt{k}'][d]:7.4f}" for k in TBPTT_KS)
        print(row, flush=True)

    (OUT / "dynamic_relay_results.json").write_text(
        json.dumps({"chance": chance, "results": res, "curves": curves}, indent=2))
    print(f"\nwrote {OUT/'dynamic_relay_results.json'}", flush=True)

    def deepavg(m): return sum(res[m][d] for d in [100, 150, 199]) / 3
    bptt_deep = deepavg("bptt")
    print(f"\nDEEP avg(100/150/199): no_relay={deepavg('no_relay'):.3f} bptt={bptt_deep:.3f} " +
          " ".join(f"tbptt{k}={deepavg(f'tbptt{k}'):.3f}" for k in TBPTT_KS), flush=True)
    print(f"\nQ1 BPTT carries DYNAMIC state (deep << chance)? {bptt_deep < 0.3 * chance} "
          f"(deep {bptt_deep:.3f} vs chance {chance:.3f})", flush=True)
    # smallest k whose deep-avg is within 1.5x of full BPTT
    kstar = next((k for k in TBPTT_KS if deepavg(f'tbptt{k}') < 1.5 * bptt_deep), None)
    print(f"Q2 min tbptt-k that extrapolates (deep < 1.5x BPTT): k* = {kstar}", flush=True)


if __name__ == "__main__":
    main()
