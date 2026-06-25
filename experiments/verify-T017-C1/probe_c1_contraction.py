"""
V-T017-C1 probe: does the C1 learning rule (self-fed rollout context, DETACHED context,
ANCHORED ground-truth successor target, single-step forward) actually reduce open-loop
multi-step error-compounding relative to a teacher-forced-only control?

This isolates the *learning rule* of the C1 design from the dynamics-model plumbing. The C1
design's defining trio is:
  (a) target slot has NO ground-truth (in the design: tau=0 pure-noise slot) -> prediction
      comes only from context  -> here: the predictor maps context -> next state, no GT in input.
  (b) the context fed at step j>1 is the model's OWN (detached) prediction of step j-1 -> the
      model is optimized on the state distribution its own rollout visits (DAgger / scheduled
      sampling on-policy states).
  (c) the per-step target is the GROUND-TRUTH next state z1[t+j], and the loss is plain flow MSE.
      Context detached each step (TBPTT-1).

Contrast with the T-014 relay (REFUTED): there the carrier had NO ground-truth anchor (a
self-consistency condition), so it could drift. HERE every step has a GT anchor, so the
question is whether detaching the context still leaves enough gradient to fix compounding.

Toy: a 1-D scalar dynamics with a stable fixed point but a region of LOCAL EXPANSION, so a
naive one-step-fit model has gain>1 somewhere and its own errors COMPOUND in open loop -- the
exact phenomenon (exposure bias) the design targets. We give the predictor extra capacity /
mild label noise so a pure one-step fit is NOT identical to the truth (otherwise there is no
exposure bias to fix and the test is vacuous).

We compare three trained predictors, identical architecture/seed/optim, differing ONLY in the
loss:
  TF   : teacher-forced one-step MSE on GT pairs (the current vanilla diffusion analog).
  C1   : the design's rule -- h-step self-rollout, detached context, GT successor target.
  C1ng : C1 but context NOT detached (full BPTT through the rollout) -- to see whether the
         detach (TBPTT-1) is what carries the contraction fix or whether grad-through-time is
         needed (tests claim C-C(ii)).
Metric: OPEN-LOOP rollout error vs horizon, averaged over held-out initial conditions.

Run:  python experiments/verify-T017-C1/probe_c1_contraction.py
"""
import torch, torch.nn as nn, math, json, os

SEED = 0
torch.manual_seed(SEED)
DEV = "cpu"

# ---- true dynamics: f(x) = x + dt*(x - x^3) (a cubic with unstable point at 0, stable at +-1)
# Near x=0 the map x_{t+1}=x+dt*x has gain (1+dt) > 1 -> LOCAL EXPANSION -> small errors grow
# until the trajectory saturates near +-1. This is a clean compounding/exposure-bias testbed.
DT = 0.20
def f_true(x):
    return x + DT * (x - x**3)

def make_traj(x0, T):
    xs = [x0]
    for _ in range(T):
        xs.append(f_true(xs[-1]))
    return torch.stack(xs, dim=1)  # (B, T+1)

class Net(nn.Module):
    # CAPACITY-LIMITED one-step predictor x_t -> x_{t+1}. The tiny hidden width makes a perfect
    # one-step fit unattainable, so the trained map has STRUCTURED residual error that compounds
    # in open loop (exposure bias) -- the regime the C1 design targets. (With a wide net the
    # one-step map is fit to ~1e-7 and there is no exposure bias to fix; the test would be vacuous.)
    def __init__(self, hid=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hid), nn.Tanh(),
                                 nn.Linear(hid, 1))
    def forward(self, x):  # x: (...,1)
        return x + self.net(x)  # residual param (mirrors x-prediction's "predict the state")

def clone_init(seed):
    torch.manual_seed(seed)
    return Net().to(DEV)

# ---- data: random initial conditions, mostly in the expanding region near 0
def sample_x0(B):
    # Start in the EXPANDING CORE near 0 (|x0| small) so the trajectory has to travel out to the
    # stable point -- this is where one-step residual error compounds the most (gain>1). Avoid
    # exactly 0 (the unstable equilibrium, no motion).
    s = (torch.rand(B, 1, device=DEV) * 0.3 + 0.05)
    sign = torch.where(torch.rand(B, 1, device=DEV) < 0.5, -1.0, 1.0)
    return s * sign  # |x0| in (0.05, 0.35)

NTRAIN, NVAL, T_TRAIN, H = 2048, 512, 24, 6
EPOCHS = 3000
x0_tr = sample_x0(NTRAIN)
x0_va = sample_x0(NVAL)
traj_tr = make_traj(x0_tr, T_TRAIN)   # (NTRAIN, T_TRAIN+1)
traj_va = make_traj(x0_va, T_TRAIN)

# Label noise on the one-step targets => a perfect one-step fit is unattainable, so the trained
# map has residual error that CAN compound (otherwise the test is vacuous). Same noise for all
# regimes (drawn once, fixed) so the comparison is apples-to-apples.
torch.manual_seed(123)
NOISE = 0.02
tr_pairs_x = traj_tr[:, :-1].reshape(-1, 1)
tr_pairs_y = traj_tr[:, 1:].reshape(-1, 1) + NOISE * torch.randn_like(traj_tr[:, 1:]).reshape(-1, 1)

def open_loop_err(model, x0, horizon):
    """mean |x_pred - x_true| per horizon over a fresh true trajectory from x0."""
    model.eval()
    with torch.no_grad():
        true = make_traj(x0, horizon)  # (B, horizon+1)
        x = x0
        errs = []
        for h in range(1, horizon + 1):
            x = model(x)
            errs.append((x - true[:, h:h+1]).abs().mean().item())
    return errs

def train_tf(seed, epochs=EPOCHS, lr=2e-3):
    m = clone_init(seed); opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        pred = m(tr_pairs_x)
        loss = ((pred - tr_pairs_y) ** 2).mean()
        loss.backward(); opt.step()
    return m

def train_c1(seed, detach_ctx=True, epochs=EPOCHS, lr=2e-3):
    """C1 rule: from each anchor state, roll the model forward H steps feeding its OWN output
    as the next input (context). Loss at each step = MSE to the GT successor. If detach_ctx,
    detach the fed-forward state each step (TBPTT-1)."""
    m = clone_init(seed); opt = torch.optim.Adam(m.parameters(), lr=lr)
    anchors = traj_tr[:, :T_TRAIN - H + 1]                     # states that have H successors
    A = anchors.shape[1]
    for _ in range(epochs):
        opt.zero_grad()
        # also keep the plain one-step term (design: "existing per-frame diffusion loss UNCHANGED")
        one_step = ((m(tr_pairs_x) - tr_pairs_y) ** 2).mean()
        x = anchors.reshape(-1, 1)                              # (NTRAIN*A, 1) current self-state
        ms = 0.0
        for j in range(1, H + 1):
            x = m(x)                                            # self-rollout step (grad enabled)
            # GT successor target for each anchor at offset j
            tgt = traj_tr[:, j:j + A].reshape(-1, 1)            # noiseless GT (anchored target)
            ms = ms + ((x - tgt) ** 2).mean()
            if detach_ctx:
                x = x.detach()                                  # TBPTT-1: cut grad through context
        ms = ms / H
        (one_step + ms).backward(); opt.step()
    return m

def evaluate(tag, model, horizon=24):
    errs = open_loop_err(model, x0_va, horizon)
    return {"tag": tag, "errs": errs,
            "h1": errs[0], "h6": errs[5], "h12": errs[11], "h24": errs[23]}

if __name__ == "__main__":
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    results = []
    m_tf = train_tf(SEED)
    results.append(evaluate("TF (teacher-forced one-step only)", m_tf))
    m_c1 = train_c1(SEED, detach_ctx=True)
    results.append(evaluate("C1 (self-rollout, DETACHED ctx, GT target)", m_c1))
    m_c1ng = train_c1(SEED, detach_ctx=False)
    results.append(evaluate("C1ng (self-rollout, GRAD-through ctx)", m_c1ng))

    # copy-last baseline (predict no change) open-loop
    class Copy(nn.Module):
        def forward(self, x): return x
    results.append(evaluate("copy-last (zero motion)", Copy()))

    print(f"\nTrue dynamics gain near 0 = {1+DT:.2f} (local expansion); label noise={NOISE}")
    print(f"Train horizon T={T_TRAIN}, C1 lookahead H={H}, seed={SEED}\n")
    print(f"{'regime':<46} {'h1':>8} {'h6':>8} {'h12':>8} {'h24':>8}")
    for r in results:
        print(f"{r['tag']:<46} {r['h1']:8.4f} {r['h6']:8.4f} {r['h12']:8.4f} {r['h24']:8.4f}")
    with open(os.path.join(os.path.dirname(__file__), "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nsaved results.json")
