"""
T-014 verifier probe v2 — robust version. Same falsifiable question as v1 but with a design
that GUARANTEES the full-BPTT control can solve it (so the comparison is valid), and a GRU-style
gated carrier (the realistic capacity a transformer memory-token relay has).

Question (unchanged): can a DETACHED-CARRY, PER-STEP-SUFFICIENCY relay learn to preserve a
hop-0 secret across many hops, matching full-BPTT, or does the detach break write-credit so it
drifts/collapses / fails to learn preservation?

Design fixes vs v1:
  * GRU cell as the writer (gated -> can learn near-identity carry; the copy task is reachable).
  * Train rollout HOPS=32, eval extrapolates to 200 (tests stability/drift beyond training depth).
  * Reveal probe at EVERY hop during training (dense supervision; isolates the credit question,
    not a sparse-probe confound).
  * Secret is the FIRST few channels of x at hop 0 with an explicit write gate; pure-noise after.

Modes:
  no_relay : memory reset to learned init each hop (can't carry) -> chance lower bound.
  detached : Mode B -- m_t = GRU(detach(m_prev), x_t), per-step loss only.
  bptt     : m_t = GRU(m_prev, x_t), per-step loss, full BPTT through the rollout (control upper bd).
  tbptt1   : carry value faithfully but attach only ONE step of grad (IDEAS.md option B) -- a
             middle ground the note mentions; included to see if 1-hop grad rescues write-credit.

Run: venv/Scripts/python.exe experiments/verify-T014/probe_detached_relay_v2.py
"""
import json
from pathlib import Path
import torch
import torch.nn as nn

SEED = 0
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(__file__).resolve().parent

MEM_DIM, SECRET_DIM, X_DIM = 32, 4, 8
HID = 64
HOPS, EVAL_HOPS = 32, 200
BATCH, STEPS, LR = 256, 3000, 3e-3


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
        self.net = nn.Sequential(nn.Linear(MEM_DIM, HID), nn.SiLU(), nn.Linear(HID, SECRET_DIM))

    def forward(self, m):
        return self.net(m)


def make_episode(batch, hops, gen):
    s = torch.randn(batch, SECRET_DIM, generator=gen, device=device)
    x = 0.5 * torch.randn(batch, hops, X_DIM, generator=gen, device=device)
    x[:, 0, :SECRET_DIM] = s
    x[:, 0, -1] = 1.0
    x[:, 1:, -1] = 0.0
    return s, x


def run(writer, reader, s, x, mode):
    B, H, _ = x.shape
    m = writer.init_mem.expand(B, MEM_DIM)
    total, cnt = 0.0, 0
    per_hop = torch.zeros(H, device=device)
    for t in range(H):
        if mode == "no_relay":
            m_prev = writer.init_mem.expand(B, MEM_DIM)
        elif mode == "detached":
            m_prev = m.detach()
        elif mode == "tbptt1":
            m_prev = m  # attached this hop; previous hops already detached at their own step
        else:  # bptt
            m_prev = m
        m = writer(m_prev, x[:, t])
        if mode == "tbptt1":
            # keep grad only one hop: after computing m_t's loss, the carried value into t+1 is detached
            m_for_carry = m.detach()
        pred = reader(m)
        err = ((pred - s) ** 2).mean()
        total = total + err; cnt += 1
        with torch.no_grad():
            per_hop[t] = err.item()
        if mode == "tbptt1":
            m = m_for_carry
    return total / cnt, per_hop


def train(mode):
    torch.manual_seed(SEED)
    gen = torch.Generator(device=device); gen.manual_seed(SEED + 1)
    w, r = Writer().to(device), Reader().to(device)
    opt = torch.optim.Adam(list(w.parameters()) + list(r.parameters()), lr=LR)
    for step in range(STEPS):
        s, x = make_episode(BATCH, HOPS, gen)
        loss, _ = run(w, r, s, x, mode)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == STEPS - 1:
            print(f"  [{mode}] step {step:4d} loss {loss.item():.4f}", flush=True)
    return w, r


@torch.no_grad()
def evaluate(w, r, mode, hops):
    gen = torch.Generator(device=device); gen.manual_seed(SEED + 777)
    s, x = make_episode(BATCH, hops, gen)
    # eval mode uses the SAME carry rule, but no grad needed
    em = "bptt" if mode in ("bptt", "tbptt1") else mode  # carry faithfully at eval (detached==same values)
    _, per_hop = run(w, r, s, x, em if mode != "detached" else "bptt")
    return per_hop.cpu()


def main():
    res, curves = {}, {}
    for mode in ["no_relay", "detached", "tbptt1", "bptt"]:
        print(f"\n=== mode={mode} ===", flush=True)
        w, r = train(mode)
        err = evaluate(w, r, mode, EVAL_HOPS)
        depths = [1, 2, 4, 8, 16, 31, 50, 100, 150, 199]
        res[mode] = {d: round(err[d].item(), 4) for d in depths}
        curves[mode] = err.tolist()
        print(f"  recovery MSE by depth: {res[mode]}", flush=True)

    print("\n========= SUMMARY (chance~1.00) =========", flush=True)
    print("depth | no_relay | detached | tbptt1 | bptt", flush=True)
    for d in [1, 2, 4, 8, 16, 31, 50, 100, 150, 199]:
        print(f"{d:5d} | {res['no_relay'][d]:8.4f} | {res['detached'][d]:8.4f} | "
              f"{res['tbptt1'][d]:6.4f} | {res['bptt'][d]:.4f}", flush=True)

    (OUT / "results_v2.json").write_text(json.dumps({"results": res, "curves": curves}, indent=2))
    print(f"\nwrote {OUT/'results_v2.json'}", flush=True)

    def deepavg(m): return sum(res[m][d] for d in [100, 150, 199]) / 3
    print(f"\nDEEP avg(100/150/199): no_relay={deepavg('no_relay'):.3f} "
          f"detached={deepavg('detached'):.3f} tbptt1={deepavg('tbptt1'):.3f} bptt={deepavg('bptt'):.3f}",
          flush=True)
    print(f"  BPTT control solves task (deep << chance)?  {deepavg('bptt') < 0.3}", flush=True)
    print(f"  detached matches bptt (detach harmless)?    {deepavg('detached') < 1.5*deepavg('bptt')}", flush=True)
    print(f"  detached >> bptt (detach broke credit)?     {deepavg('detached') > 2*deepavg('bptt')}", flush=True)
    print(f"  detached drift vs training depth (collapse)? d199/d16 = "
          f"{res['detached'][199]/max(res['detached'][16],1e-6):.2f}", flush=True)


if __name__ == "__main__":
    main()
