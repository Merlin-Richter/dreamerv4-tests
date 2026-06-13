"""T-011 scorer-validity probe (critical-claim-verifier).

We implement the PROPOSED scorer from tasks/T-011.md and test whether its
GT-floor / forgetting-separation claims (C2, C3) actually hold on synthetic
belief trajectories. No trained model is needed: the scorer operates on a
sequence of believed (x,y) positions, so we feed it constructed beliefs.

Scorer under test (T-011 §"The score"):
  Part 2 (headline, GT-FREE): best-fit constant-speed billiard residual.
    Fit a billiard trajectory (free x0,y0,theta0; FIXED speed S; known walls/clamp)
    to believed positions {b_k}; report mean_k ||b_k - fit_k||.
  Part 1 (onset anchor): believed (x,y) & believed velocity vs GT over early
    in-window occluded steps.

Walls/clamp replicate occluded_bouncing.OccludedBouncingEnv._advance_physics exactly.

Run:  python scorer_probe.py
"""
from __future__ import annotations
import sys, pathlib, json
import numpy as np
from scipy.optimize import minimize

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
from data_generators.occluded_bouncing import OccludedBouncingEnv  # noqa

IMG = 64
RADIUS = 10
WALL_LO = RADIUS                # 10
WALL_HI = IMG - 1 - RADIUS      # 53


def billiard_rollout(x0, y0, vx0, vy0, n):
    """Replicate env physics EXACTLY (advance-then-reflect-and-clamp). Returns (n,2) positions
    at frames 1..n (i.e. AFTER each advance), matching how the env records state per step."""
    x, y, vx, vy = float(x0), float(y0), float(vx0), float(vy0)
    out = np.empty((n, 2), dtype=np.float64)
    for t in range(n):
        x += vx
        y += vy
        if x - RADIUS < 0:
            vx = abs(vx); x = float(RADIUS)
        elif x + RADIUS > IMG - 1:
            vx = -abs(vx); x = float(IMG - 1 - RADIUS)
        if y - RADIUS < 0:
            vy = abs(vy); y = float(RADIUS)
        elif y + RADIUS > IMG - 1:
            vy = -abs(vy); y = float(IMG - 1 - RADIUS)
        out[t] = (x, y)
    return out


def billiard_from_params(x0, y0, theta0, S, n):
    """Constant-speed billiard. x0,y0 = position at the FIRST believed step (out[0]);
    we treat (x0,y0) as the post-advance position of step 0 and propagate forward.
    To align fit_k with b_k (k=0..n-1): fit[0]=(x0,y0); subsequent steps advance with speed S."""
    vx0 = S * np.cos(theta0)
    vy0 = S * np.sin(theta0)
    # out[0] should be (x0,y0); propagate from there for the remaining n-1 steps.
    rest = billiard_rollout(x0, y0, vx0, vy0, n - 1) if n > 1 else np.zeros((0, 2))
    out = np.empty((n, 2))
    out[0] = (x0, y0)
    if n > 1:
        out[1:] = rest
    return out


def best_fit_billiard_residual(beliefs, S, n_theta=72, n_restart_pos=1):
    """mean_k ||b_k - fit_k|| minimized over (x0,y0,theta0), speed fixed = S.
    Grid over theta0, init x0,y0 at b_0, then local refine. Returns (residual, params)."""
    beliefs = np.asarray(beliefs, dtype=np.float64)
    n = len(beliefs)

    def resid(p):
        x0, y0, th = p
        fit = billiard_from_params(x0, y0, th, S, n)
        return np.mean(np.linalg.norm(beliefs - fit, axis=1))

    best = (np.inf, None)
    for th0 in np.linspace(0, 2 * np.pi, n_theta, endpoint=False):
        x0g, y0g = beliefs[0]
        r = minimize(resid, x0=[x0g, y0g, th0], method="Nelder-Mead",
                     options=dict(xatol=1e-3, fatol=1e-3, maxiter=2000))
        if r.fun < best[0]:
            best = (float(r.fun), r.x)
    return best


def onset_anchor_error(beliefs, gt_states, n_anchor):
    """Mean over the first n_anchor steps of believed-pos vs GT-pos error, plus
    believed-velocity (finite diff) vs GT velocity error. Returns (pos_err, vel_err)."""
    beliefs = np.asarray(beliefs, dtype=np.float64)
    gt_pos = gt_states[:, :2]
    gt_vel = gt_states[:, 2:4]
    pos_e = np.mean(np.linalg.norm(beliefs[:n_anchor] - gt_pos[:n_anchor], axis=1))
    # believed velocity via finite diff
    bvel = np.diff(beliefs, axis=0)
    vel_e = np.mean(np.linalg.norm(bvel[:n_anchor] - gt_vel[:n_anchor], axis=1)) if n_anchor <= len(bvel) else np.nan
    return float(pos_e), float(vel_e)


# ---------------------------------------------------------------- belief constructors
def gt_belief(states):
    return states[:, :2].copy()


def frozen_belief(states):
    b = np.repeat(states[0:1, :2], len(states), axis=0)
    return b


def shuffled_belief(states, rng):
    b = states[:, :2].copy()
    idx = rng.permutation(len(b))
    return b[idx]


def smooth_random_drift(states, rng, S):
    """Smooth-but-random: a random-walk in heading at env speed S, clamped to box.
    No wall reflection logic — a plausible 'lost the object' smooth wander."""
    n = len(states)
    b = np.empty((n, 2))
    b[0] = states[0, :2]
    th = rng.uniform(0, 2 * np.pi)
    for t in range(1, n):
        th += rng.normal(0, 0.6)  # smooth heading drift
        x = b[t - 1, 0] + S * np.cos(th)
        y = b[t - 1, 1] + S * np.sin(th)
        x = np.clip(x, WALL_LO, WALL_HI)
        y = np.clip(y, WALL_LO, WALL_HI)
        b[t] = (x, y)
    return b


def hallucinated_physical(states, rng, S):
    """A COHERENT billiard at env speed S but a MADE-UP initial state unrelated to GT.
    This is the C2 adversary."""
    n = len(states)
    x0 = rng.uniform(WALL_LO, WALL_HI)
    y0 = rng.uniform(WALL_LO, WALL_HI)
    th = rng.uniform(0, 2 * np.pi)
    return billiard_from_params(x0, y0, th, S, n)


def f2_bounce_desync(states, S):
    """F2: true physics but ONE bounce timing error. Start from GT onset state, run the
    correct billiard, but offset the start by a tiny epsilon so a bounce lands one frame
    off -> still-physical trajectory that diverges from GT. Should PASS."""
    x0, y0, vx0, vy0 = states[0, 0], states[0, 1], states[0, 2], states[0, 3]
    # nudge starting position slightly so a wall bounce happens a frame early/late
    return billiard_rollout(x0 + 1.5, y0, vx0, vy0, len(states))


def make_gt_episode(seed, n_occ):
    """Drive the env: prefix already advanced; we just take an occluded run of n_occ states."""
    env = OccludedBouncingEnv(img_size=IMG, radius=RADIUS).reset(seed=seed)
    states = np.empty((n_occ, 5), dtype=np.float64)
    for t in range(n_occ):
        f, s = env.step(1)  # curtain down; physics advances
        states[t] = s
    S = float(np.hypot(states[0, 2], states[0, 3]))
    return states, S


def run(n_occ=16, n_seeds=40, n_anchor=3, detector_noise_px=0.65, seed_master=0):
    rng = np.random.default_rng(seed_master)
    cats = ["gt", "f2_bounce", "hallucinated", "frozen", "shuffled", "smooth_drift"]
    resid = {c: [] for c in cats}
    onset_pos = {c: [] for c in cats}
    onset_vel = {c: [] for c in cats}

    for i in range(n_seeds):
        states, S = make_gt_episode(seed=1000 + i, n_occ=n_occ)

        def add_noise(b):
            return b + rng.normal(0, detector_noise_px, size=b.shape)

        beliefs = {
            "gt": add_noise(gt_belief(states)),
            "f2_bounce": add_noise(f2_bounce_desync(states, S)),
            "hallucinated": add_noise(hallucinated_physical(states, rng, S)),
            "frozen": add_noise(frozen_belief(states)),
            "shuffled": add_noise(shuffled_belief(states, rng)),
            "smooth_drift": add_noise(smooth_random_drift(states, rng, S)),
        }
        for c, b in beliefs.items():
            r, _ = best_fit_billiard_residual(b, S)
            resid[c].append(r)
            pe, ve = onset_anchor_error(b, states, n_anchor)
            onset_pos[c].append(pe)
            onset_vel[c].append(ve)

    def summ(d):
        return {c: dict(mean=float(np.mean(v)), p50=float(np.median(v)),
                        p90=float(np.percentile(v, 90)), min=float(np.min(v)))
                for c, v in d.items()}

    out = dict(
        config=dict(n_occ=n_occ, n_seeds=n_seeds, n_anchor=n_anchor,
                    detector_noise_px=detector_noise_px, seed_master=seed_master),
        billiard_residual=summ(resid),
        onset_pos_err=summ(onset_pos),
        onset_vel_err=summ(onset_vel),
    )
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_occ", type=int, default=16)
    ap.add_argument("--n_seeds", type=int, default=40)
    ap.add_argument("--n_anchor", type=int, default=3)
    args = ap.parse_args()
    res = run(n_occ=args.n_occ, n_seeds=args.n_seeds, n_anchor=args.n_anchor)
    print(json.dumps(res, indent=2))
    outp = pathlib.Path(__file__).parent / f"result_nocc{args.n_occ}.json"
    outp.write_text(json.dumps(res, indent=2))
    print("wrote", outp)
