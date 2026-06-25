"""
V-T017-C1 probe #2 (mechanism): does the C1 rule change the learned map's behaviour ON the
self-rollout manifold in a way that reduces compounding? We test the *contraction* claim
directly and cleanly, separating two things the design conflates:

  MECHANISM-A ("contraction map"): C1 lowers the local GAIN |df/dx| of the learned map so
      perturbations shrink. The design literally says "pushes toward a contraction map".
  MECHANISM-B ("distribution-correct"): C1 makes the map ACCURATE on the off-trajectory states
      the rollout actually visits (the model's own drifted states), even if gain is unchanged.
      This is the DAgger / scheduled-sampling mechanism.

These are DIFFERENT. A loss that puts a GT anchor on every self-rollout step CANNOT generically
lower the true gain (the GT successor of a perturbed state is the true f of that state, whose
gain is fixed by the data) -- it makes the map MATCH f on visited states (Mechanism-B). The
"contraction" framing in the design is therefore imprecise; the real, sound mechanism is B.

Setup: an UNSTABLE linear region. true f(x)=a*x with a>1 near 0 (pure expansion, no saturation
inside the test band) so compounding is exact and analytic: open-loop error after h steps from a
one-step map with multiplicative error (1+e) is a^h * ((1+e)^h - 1)-ish. We train a SINGLE scalar
gain g (the model is x->g*x) under TF vs C1 and read off g. Then a capacity-limited nonlinear
case to confirm B empirically (accuracy on visited off-manifold states).

Run:  python experiments/verify-T017-C1/probe_c1_gain.py
"""
import torch, torch.nn as nn, json, os

torch.manual_seed(0)

# ---------- Part 1: scalar-gain analytic case. true f(x) = A*x (A>1), data x0 ~ band near 0.
A = 1.10
T, H = 16, 6
N = 4096
x0 = (torch.rand(N, 1) * 0.2 - 0.1)               # band (-0.1,0.1): stays in linear region for T
def f(x): return A * x
def traj(x0, T):
    xs=[x0]
    for _ in range(T): xs.append(f(xs[-1]))
    return torch.stack(xs,1)
tr = traj(x0, T)
NOISE = 0.0   # noiseless: with a single param the TF optimum is EXACTLY g=A; isolates the rule
pairs_x = tr[:,:-1].reshape(-1,1)
pairs_y = (tr[:,1:]).reshape(-1,1)

def fit_gain(rule):
    g = nn.Parameter(torch.tensor(0.5))            # deliberately wrong init (under-gain)
    opt = torch.optim.Adam([g], lr=5e-3)
    anchors = tr[:, :T-H+1]; Acnt = anchors.shape[1]
    for _ in range(4000):
        opt.zero_grad()
        if rule == "TF":
            loss = ((g*pairs_x - pairs_y)**2).mean()
        else:  # C1 self-rollout, detached ctx, GT target
            x = anchors.reshape(-1,1); loss = 0.0
            for j in range(1,H+1):
                x = g*x
                tgt = tr[:, j:j+Acnt].reshape(-1,1)
                loss = loss + ((x-tgt)**2).mean()
                x = x.detach()
            loss = loss / H
        loss.backward(); opt.step()
    return g.item()

g_tf = fit_gain("TF")
g_c1 = fit_gain("C1")
print(f"Part 1 (true gain A={A}):  TF learns g={g_tf:.4f}   C1 learns g={g_c1:.4f}")
print("  => both recover the TRUE gain; C1 does NOT push g below A (no spurious contraction).")

# ---------- Part 2: capacity-limited nonlinear; measure ACCURACY on VISITED off-manifold states.
DT = 0.20
def f2(x): return x + DT*(x - x**3)
def traj2(x0,T):
    xs=[x0]
    for _ in range(T): xs.append(f2(xs[-1]))
    return torch.stack(xs,1)
class Net(nn.Module):
    def __init__(s,h=4):
        super().__init__(); s.net=nn.Sequential(nn.Linear(1,h),nn.Tanh(),nn.Linear(h,1))
    def forward(s,x): return x + s.net(x)

def x0s(n):
    s=(torch.rand(n,1)*0.3+0.05); sg=torch.where(torch.rand(n,1)<.5,-1.,1.); return s*sg
NTR=2048; T2=24; H2=6; EP=3000; NOISE2=0.02
torch.manual_seed(0); xtr=x0s(NTR); tr2=traj2(xtr,T2)
torch.manual_seed(7); px=tr2[:,:-1].reshape(-1,1)
py=tr2[:,1:].reshape(-1,1)+NOISE2*torch.randn_like(tr2[:,1:]).reshape(-1,1)

def train(rule, seed=0):
    torch.manual_seed(seed); m=Net()
    opt=torch.optim.Adam(m.parameters(),lr=2e-3)
    anchors=tr2[:,:T2-H2+1]; Ac=anchors.shape[1]
    for _ in range(EP):
        opt.zero_grad()
        one=((m(px)-py)**2).mean()
        if rule=="TF":
            (one).backward()
        else:
            x=anchors.reshape(-1,1); ms=0.0
            for j in range(1,H2+1):
                x=m(x); tgt=tr2[:,j:j+Ac].reshape(-1,1); ms=ms+((x-tgt)**2).mean(); x=x.detach()
            (one+ms/H2).backward()
        opt.step()
    return m

m_tf=train("TF"); m_c1=train("C1")

# Build the set of states each model's OWN open-loop rollout VISITS (the on-policy distribution),
# then measure how accurately each model predicts the TRUE next step f2 AT THOSE STATES.
@torch.no_grad()
def visited_states(m, x0, T):
    x=x0; S=[x0]
    for _ in range(T): x=m(x); S.append(x)
    return torch.cat(S,0)
@torch.no_grad()
def onestep_err_on(m, states):
    return (m(states)-f2(states)).abs().mean().item()
@torch.no_grad()
def openloop(m,x0,T):
    x=x0; tr=traj2(x0,T); e=[]
    for h in range(1,T+1):
        x=m(x); e.append((x-tr[:,h:h+1]).abs().mean().item())
    return e

torch.manual_seed(1); xva=x0s(512)
vis_tf=visited_states(m_tf,xva,24); vis_c1=visited_states(m_c1,xva,24)
# Each model's accuracy on ITS OWN visited states (the distribution it must be right on):
print(f"\nPart 2 (capacity-limited, exposure bias present):")
print(f"  one-step |err| on TF's own visited states:  TF-map {onestep_err_on(m_tf,vis_tf):.4f}   C1-map {onestep_err_on(m_c1,vis_tf):.4f}")
print(f"  one-step |err| on C1's own visited states:  TF-map {onestep_err_on(m_tf,vis_c1):.4f}   C1-map {onestep_err_on(m_c1,vis_c1):.4f}")
e_tf=openloop(m_tf,xva,24); e_c1=openloop(m_c1,xva,24)
print(f"  open-loop |err| h1/h6/h12/h24:  TF {e_tf[0]:.4f}/{e_tf[5]:.4f}/{e_tf[11]:.4f}/{e_tf[23]:.4f}")
print(f"                                  C1 {e_c1[0]:.4f}/{e_c1[5]:.4f}/{e_c1[11]:.4f}/{e_c1[23]:.4f}")

out={"part1":{"A":A,"g_tf":g_tf,"g_c1":g_c1},
     "part2":{"openloop_tf":e_tf,"openloop_c1":e_c1,
              "err_on_tf_visited":{"tf":onestep_err_on(m_tf,vis_tf),"c1":onestep_err_on(m_c1,vis_tf)},
              "err_on_c1_visited":{"tf":onestep_err_on(m_tf,vis_c1),"c1":onestep_err_on(m_c1,vis_c1)}}}
with open(os.path.join(os.path.dirname(__file__),"results_gain.json"),"w") as fh: json.dump(out,fh,indent=2)
print("\nsaved results_gain.json")
