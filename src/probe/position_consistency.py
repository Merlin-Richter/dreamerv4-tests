"""Position-memory CONSISTENCY metric (T-011 / D-018) — the H3 *position* yardstick.

Replaces the open-loop GT-matched position error (rejected, ESC-008): that metric punished
both the non-tracker (F1) and the accurate-but-butterfly-desynced tracker (F2). This one
measures whether the model retains a *coherent, physically-evolving belief* about the hidden
ball — crediting F2 (late GT-divergence under chaos), penalizing F1 (forgetting).

Framing LOCKED (Merlin, 2026-06-13): the metric certifies "a coherent physical belief, at env
speed, anchored to the real ball at occlusion onset" — NOT correct dead-reckoning past the
bounce-ambiguity horizon (unmeasurable under chaos). See tasks/T-011.md.

Readout (Merlin, ESC-008): the model's belief at occluded step k = "what would it predict if
revealed NOW" — inject the absolute curtain-UP action for one frame, decode, run the gated
`detect_ball`. Sweeping the reveal horizon k=1..K (fixed seed => shared physics) yields the
believed trajectory at consecutive hidden timesteps. NO state-probe (sidesteps transfer risk).

Score (verifier-audited, V-T011):
  1. ONSET ANCHOR — believed (x,y)+velocity vs GT over the in-window early steps (pins the
     belief to the real ball; catches F1-static via wrong speed and forgetting).
  2. BEST-FIT BILLIARD RESIDUAL (headline, GT-free) — fit a constant-speed billiard
     (free x0,y0,theta0; speed FIXED at env S; exact env walls/clamp) to the believed
     positions; mean residual. Low <=> physical belief at env speed. **Trust only at n_occ>=8**
     (3-param fit over-flexible on short windows — V-T011 C3). Speed MUST stay fixed at S
     (freeing it lets wrong-speed coherent balls pass — V-T011).
  3. GT-TRACKING HORIZON (report-only) — first step believed-vs-GT exceeds a threshold; the
     only near-GT constraint (the metric forgives divergence past it by design).
Non-degeneracy gate: a too-short / near-stationary believed trajectory can trivially fit a
billiard within floor -> flag and route to the F1 verdict, don't credit part 2.

Physics (billiard_rollout) is a byte-identical replica of OccludedBouncingEnv._advance_physics
(verified in V-T011); kept here so the scorer is the single source of truth.

Run:
  python position_consistency.py --calibrate          # synthetic GT-floor + forgetting (no model)
  python position_consistency.py --ceiling --dynamics <ckpt>   # readout feasibility on a model
  python position_consistency.py --dynamics <ckpt> --out experiments/EXP-NNN/posmem.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[1]            # .../src
_ROOT = _SRC.parent                                           # repo root
for _p in (_SRC, _SRC / "probe", _SRC / "C_multi_image_auto_encoder", _SRC / "D_dynamics_model"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

IMG = 64
RADIUS = 10
WALL_LO = RADIUS                # 10
WALL_HI = IMG - 1 - RADIUS      # 53

# Tracking-horizon threshold (px): "near GT" while believed-vs-GT < this. Reported only.
TRACK_THRESH_PX = 6.0
# Non-degeneracy: a believed trajectory must span at least this (px) and have at least this many
# detected points, or its billiard fit is untrustworthy (a near-stationary path fits within floor).
MIN_SPAN_PX = 8.0
MIN_FOUND_POINTS = 6


# --------------------------------------------------------------------- physics (env replica)
def billiard_rollout(x0, y0, vx0, vy0, n):
    """EXACT env physics (advance-then-reflect-and-clamp). Returns (n,2) positions at the n
    frames AFTER each advance — matches how OccludedBouncingEnv records per-step state."""
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
    """Constant-speed billiard with fit[0]=(x0,y0), propagated forward at speed S."""
    vx0 = S * np.cos(theta0)
    vy0 = S * np.sin(theta0)
    out = np.empty((n, 2))
    out[0] = (x0, y0)
    if n > 1:
        out[1:] = billiard_rollout(x0, y0, vx0, vy0, n - 1)
    return out


def best_fit_billiard_residual(beliefs, S, n_theta=72):
    """mean_k ||b_k - fit_k|| minimized over (x0,y0,theta0); speed fixed = S.
    Grid over theta0 (init x0,y0 at b_0) + local refine. Returns (residual, params)."""
    from scipy.optimize import minimize
    beliefs = np.asarray(beliefs, dtype=np.float64)
    n = len(beliefs)

    def resid(p):
        x0, y0, th = p
        return float(np.mean(np.linalg.norm(beliefs - billiard_from_params(x0, y0, th, S, n), axis=1)))

    best = (np.inf, None)
    for th0 in np.linspace(0, 2 * np.pi, n_theta, endpoint=False):
        x0g, y0g = beliefs[0]
        r = minimize(resid, x0=[x0g, y0g, th0], method="Nelder-Mead",
                     options=dict(xatol=1e-3, fatol=1e-3, maxiter=2000))
        if r.fun < best[0]:
            best = (float(r.fun), r.x)
    return best


def onset_anchor_error(beliefs, gt_states, n_anchor):
    """Mean over the first n_anchor steps of believed-pos vs GT-pos error, and believed-velocity
    (finite diff) vs GT velocity error. Returns (pos_err, vel_err)."""
    beliefs = np.asarray(beliefs, dtype=np.float64)
    gt_pos = gt_states[:, :2]
    gt_vel = gt_states[:, 2:4]
    na = min(n_anchor, len(beliefs))
    pos_e = float(np.mean(np.linalg.norm(beliefs[:na] - gt_pos[:na], axis=1)))
    bvel = np.diff(beliefs, axis=0)
    vel_e = (float(np.mean(np.linalg.norm(bvel[:na] - gt_vel[:na], axis=1)))
             if na <= len(bvel) and na > 0 else float("nan"))
    return pos_e, vel_e


def tracking_horizon(beliefs, gt_states, thresh=TRACK_THRESH_PX):
    """First step index where believed-vs-GT position error exceeds `thresh` (report-only)."""
    err = np.linalg.norm(np.asarray(beliefs)[:, :2] - gt_states[:, :2], axis=1)
    over = np.nonzero(err > thresh)[0]
    return int(over[0]) if len(over) else len(err)


def is_degenerate(beliefs):
    """True if the believed trajectory is too short / near-stationary to constrain a billiard fit."""
    b = np.asarray(beliefs, dtype=np.float64)
    if len(b) < MIN_FOUND_POINTS:
        return True
    span = float(np.linalg.norm(b.max(axis=0) - b.min(axis=0)))
    return span < MIN_SPAN_PX


def score_belief(beliefs, gt_states, S, n_anchor):
    """Score one believed trajectory. `beliefs` (n,2), `gt_states` (n,>=4) aligned, NaN rows in
    beliefs = ball-not-detected at that step. Returns the metric dict."""
    beliefs = np.asarray(beliefs, dtype=np.float64)
    found = np.isfinite(beliefs).all(axis=1)
    lost_rate = float((~found).mean())
    b_ok = beliefs[found]
    gt_ok = gt_states[found]
    n_occ = len(beliefs)
    degenerate = (found.sum() < MIN_FOUND_POINTS) or is_degenerate(b_ok)

    pos_e, vel_e = onset_anchor_error(beliefs, gt_states, n_anchor)  # onset uses leading steps
    if degenerate:
        resid = float("nan")
    else:
        resid, _ = best_fit_billiard_residual(b_ok, S)
    horizon = tracking_horizon(beliefs[found] if found.all() else beliefs, gt_states, TRACK_THRESH_PX) \
        if found.any() else 0
    return {
        "billiard_residual": resid,
        "onset_pos_err": pos_e,
        "onset_vel_err": vel_e,
        "tracking_horizon": horizon,
        "ball_lost_rate": lost_rate,
        "degenerate": bool(degenerate),
        "n_occ": int(n_occ),
        "residual_trustworthy": bool((not degenerate) and n_occ >= 8),
        "S": float(S),
    }


# --------------------------------------------------------------------- synthetic surrogates
def gt_belief(states):
    return states[:, :2].copy()


def frozen_belief(states):
    return np.repeat(states[0:1, :2], len(states), axis=0)


def shuffled_belief(states, rng):
    b = states[:, :2].copy()
    return b[rng.permutation(len(b))]


def smooth_random_drift(states, rng, S):
    """Smooth-but-random heading walk at env speed S, box-clamped (no reflection) — a plausible
    'lost the object' wander."""
    n = len(states)
    b = np.empty((n, 2)); b[0] = states[0, :2]
    th = rng.uniform(0, 2 * np.pi)
    for t in range(1, n):
        th += rng.normal(0, 0.6)
        b[t] = (np.clip(b[t - 1, 0] + S * np.cos(th), WALL_LO, WALL_HI),
                np.clip(b[t - 1, 1] + S * np.sin(th), WALL_LO, WALL_HI))
    return b


def hallucinated_physical(states, rng, S):
    """Coherent billiard at env speed but a made-up initial state (C2 naive adversary)."""
    return billiard_from_params(rng.uniform(WALL_LO, WALL_HI), rng.uniform(WALL_LO, WALL_HI),
                                rng.uniform(0, 2 * np.pi), S, len(states))


def f2_bounce_desync(states, S):
    """F2: true physics from the GT onset state but nudged so a bounce lands a frame off — a
    still-physical trajectory that diverges from GT. Should PASS."""
    x0, y0, vx0, vy0 = states[0, 0], states[0, 1], states[0, 2], states[0, 3]
    return billiard_rollout(x0 + 1.5, y0, vx0, vy0, len(states))


def _gt_occluded_states(seed, n_occ):
    """Drive the env curtain-down for n_occ steps; return (states (n,5), S)."""
    from data_generators.occluded_bouncing import OccludedBouncingEnv
    env = OccludedBouncingEnv(img_size=IMG, radius=RADIUS).reset(seed=seed)
    states = np.empty((n_occ, 5), dtype=np.float64)
    for t in range(n_occ):
        _, states[t] = env.step(1)
    return states, float(np.hypot(states[0, 2], states[0, 3]))


def calibrate(n_occ=16, n_seeds=40, n_anchor=4, detector_noise_px=0.65, seed_master=0):
    """Two-sided instrument validation (Merlin): GT must score ~floor; forgetting surrogates must
    score HIGH; F2 must pass like GT; naive hallucination must fail the onset anchor. No model."""
    rng = np.random.default_rng(seed_master)
    cats = ["gt", "f2_bounce", "hallucinated", "frozen", "shuffled", "smooth_drift"]
    acc = {c: {"residual": [], "onset_pos": [], "onset_vel": []} for c in cats}
    for i in range(n_seeds):
        states, S = _gt_occluded_states(seed=1000 + i, n_occ=n_occ)
        noisy = lambda b: b + rng.normal(0, detector_noise_px, size=b.shape)
        beliefs = {
            "gt": noisy(gt_belief(states)),
            "f2_bounce": noisy(f2_bounce_desync(states, S)),
            "hallucinated": noisy(hallucinated_physical(states, rng, S)),
            "frozen": noisy(frozen_belief(states)),
            "shuffled": noisy(shuffled_belief(states, rng)),
            "smooth_drift": noisy(smooth_random_drift(states, rng, S)),
        }
        for c, b in beliefs.items():
            r, _ = best_fit_billiard_residual(b, S)
            pe, ve = onset_anchor_error(b, states, n_anchor)
            acc[c]["residual"].append(r); acc[c]["onset_pos"].append(pe); acc[c]["onset_vel"].append(ve)

    def summ(vs):
        return dict(mean=float(np.mean(vs)), p50=float(np.median(vs)),
                    p90=float(np.percentile(vs, 90)), min=float(np.min(vs)))
    return {
        "config": dict(n_occ=n_occ, n_seeds=n_seeds, n_anchor=n_anchor,
                       detector_noise_px=detector_noise_px, seed_master=seed_master),
        "billiard_residual": {c: summ(acc[c]["residual"]) for c in cats},
        "onset_pos_err": {c: summ(acc[c]["onset_pos"]) for c in cats},
        "onset_vel_err": {c: summ(acc[c]["onset_vel"]) for c in cats},
    }


# --------------------------------------------------------------------- model belief readout
def _belief_at_k(tok, dyn, seed, k, P, K, device, use_memory=False):
    """Roll out [P up | k down | 1 up], decode the reveal frame, detect the ball.
    FF7 checkpoints (use_register_memory) MUST use the register-carry rollout (generate_memory),
    else their beyond-window memory is crippled and they read as vanilla. Same signature.
    Returns (found, x, y, gt_xy(2,), gt_vel(2,))."""
    import torch
    from probe_env import make_probe_episode
    from revisit_probe import _prefix_latents, _decode_frame, detect_ball
    ep = make_probe_episode(seed=seed, P=P, k=k, R=1)
    context = _prefix_latents(tok, ep.frames, ep.P, device)
    action_idx = torch.from_numpy(ep.actions.astype(np.int64)).unsqueeze(0).to(device)
    torch.manual_seed(seed)  # shared noise draws across k => consecutive-step belief trajectory
    rollout = dyn.generate_memory if use_memory else dyn.generate
    gen = rollout(context, n_generate=ep.frames.shape[0] - ep.P, K=K, action_idx=action_idx)
    full = torch.cat((context, gen), dim=1)
    found, x, y, _ = detect_ball(_decode_frame(tok, full[:, ep.reveal_index]))
    gt = ep.states[ep.reveal_index]
    return found, x, y, gt[:2].copy(), gt[2:4].copy()


def model_belief_trajectory(tok, dyn, seed, k_max, P, K, device, use_memory=False):
    """Believed (x,y) at each occluded horizon k=1..k_max for one seed. NaN where ball not found.
    Returns (beliefs (k_max,2), gt_states (k_max,4), S)."""
    beliefs = np.full((k_max, 2), np.nan)
    gt_states = np.empty((k_max, 4))
    S = None
    for j, k in enumerate(range(1, k_max + 1)):
        found, x, y, gxy, gvel = _belief_at_k(tok, dyn, seed, k, P, K, device, use_memory)
        if found:
            beliefs[j] = (x, y)
        gt_states[j] = (*gxy, *gvel)
        if S is None:
            S = float(np.hypot(gvel[0], gvel[1]))
    return beliefs, gt_states, S


def run_model(args):
    import torch
    from revisit_probe import load_models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, dyn, dcfg, _ = load_models(args.tokenizer, args.dynamics, args.window_N, device)
    K = dcfg.inference_steps if args.K is None else args.K
    P = args.prefix_P
    use_memory = bool(getattr(dcfg, "use_register_memory", False))  # FF7 -> register-carry rollout
    print(f"[posmem] {'FF7 register-carry (generate_memory)' if use_memory else 'vanilla sliding-window (generate)'} rollout")
    n_anchor = max(2, args.window_N - P - 1)  # strictly in-window early steps (V-T011 fix #4)
    k_max = args.k_max
    n_seeds = 8 if args.dry_run else args.episodes

    # Ceiling / readout-feasibility FIRST (V-T011 fix #5): at a short in-window horizon the model
    # HAS the info — does the counterfactual reveal render a detectable ball near GT?
    ceil_found, ceil_pos = [], []
    for i in range(n_seeds):
        found, x, y, gxy, _ = _belief_at_k(tok, dyn, seed=3000 + i, k=2, P=P, K=K,
                                           device=device, use_memory=use_memory)
        ceil_found.append(found)
        if found:
            ceil_pos.append(float(np.hypot(x - gxy[0], y - gxy[1])))
    # readout_ok gates on DETECTABILITY only: does the counterfactual reveal render a ball we can
    # find? The pos_err at small k is the model's intrinsic short-horizon dynamics quality (compare
    # to the EXP-012 teacher-forced 1-step pos_err, e.g. vanilla 4.66px), NOT a readout fault — so
    # it is reported, not gated. (V-T011 fix #5, corrected: a weak model has high ceiling pos_err
    # by its own dynamics; that is what the metric then measures, not a broken readout.)
    ceiling = {"found_rate": float(np.mean(ceil_found)),
               "pos_err_px_mean": float(np.mean(ceil_pos)) if ceil_pos else float("nan"),
               "pos_err_px_p90": float(np.percentile(ceil_pos, 90)) if ceil_pos else float("nan"),
               "readout_ok": bool(np.mean(ceil_found) >= 0.95)}
    print(f"[ceiling] found_rate={ceiling['found_rate']:.2f} "
          f"pos_err={ceiling['pos_err_px_mean']:.2f}px (model dynamics, cf EXP-012 1-step) "
          f"readout_ok={ceiling['readout_ok']}")

    scores = []
    if not args.ceiling:
        if not ceiling["readout_ok"]:
            print("  !! ceiling found_rate low -- model does not render a detectable ball on the "
                  "counterfactual reveal; readout unreliable (consider state-probe fallback).")
        for i in range(n_seeds):
            beliefs, gt_states, S = model_belief_trajectory(tok, dyn, 4000 + i, k_max, P, K,
                                                            device, use_memory)
            scores.append(score_belief(beliefs, gt_states, S, n_anchor))
            print(f"  seed {i}: resid={scores[-1]['billiard_residual']:.2f} "
                  f"onset_pos={scores[-1]['onset_pos_err']:.2f} "
                  f"track_h={scores[-1]['tracking_horizon']} lost={scores[-1]['ball_lost_rate']:.2f} "
                  f"{'[degenerate]' if scores[-1]['degenerate'] else ''}")

    def agg(key, trust=False):
        vs = [s[key] for s in scores
              if np.isfinite(s[key]) and (s["residual_trustworthy"] or not trust)]
        return float(np.mean(vs)) if vs else float("nan")

    out = {
        "meta": dict(dynamics=str(args.dynamics.name), tokenizer=str(args.tokenizer.name),
                     window_N=args.window_N, prefix_P=P, K=K, k_max=k_max, n_anchor=n_anchor,
                     n_seeds=n_seeds, device=device, dry_run=args.dry_run,
                     rollout=("generate_memory" if use_memory else "generate")),
        "ceiling": ceiling,
        "billiard_residual_mean": agg("billiard_residual", trust=True),
        "onset_pos_err_mean": agg("onset_pos_err"),
        "onset_vel_err_mean": agg("onset_vel_err"),
        "tracking_horizon_mean": agg("tracking_horizon"),
        "ball_lost_rate_mean": agg("ball_lost_rate"),
        "per_seed": scores,
    }
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Position-memory consistency metric (T-011/D-018)")
    p.add_argument("--calibrate", action="store_true",
                   help="synthetic GT-floor + forgetting-surrogate validation (no model)")
    p.add_argument("--ceiling", action="store_true",
                   help="only run the readout-feasibility ceiling on the model")
    p.add_argument("--dry-run", action="store_true", help="few seeds")
    p.add_argument("--episodes", type=int, default=40, help="seeds (full run)")
    p.add_argument("--k-max", type=int, default=16, help="max occlusion horizon swept")
    p.add_argument("--window-N", type=int, default=8)
    p.add_argument("--prefix-P", type=int, default=3)
    p.add_argument("--K", type=int, default=None, help="shortcut steps (default: cfg)")
    p.add_argument("--tokenizer", type=pathlib.Path, default=_ROOT / "trained_autoencoder.pt")
    p.add_argument("--dynamics", type=pathlib.Path, default=_ROOT / "my_dynamics.pt")
    p.add_argument("--out", type=pathlib.Path, default=_SRC / "probe" / "last_posmem.json")
    return p.parse_args()


def main():
    args = parse_args()
    if args.calibrate:
        res = calibrate(n_occ=args.k_max, n_seeds=(6 if args.dry_run else 40),
                        n_anchor=max(2, args.window_N - args.prefix_P))
        br = res["billiard_residual"]; op = res["onset_pos_err"]
        print("[calibrate] billiard residual (mean):",
              {c: round(br[c]["mean"], 2) for c in br})
        print("[calibrate] onset_pos_err (mean):    ",
              {c: round(op[c]["mean"], 2) for c in op})
        gt_floor = br["gt"]["mean"]
        forget = min(br["frozen"]["mean"], br["shuffled"]["mean"], br["smooth_drift"]["mean"])
        sep = forget > 2.5 * gt_floor
        halluc_caught = op["hallucinated"]["min"] > 2.0 * op["gt"]["mean"]
        f2_pass = br["f2_bounce"]["mean"] < 2.5 * gt_floor
        print(f"[calibrate] GT~floor={gt_floor:.2f}  min-forget={forget:.2f}  "
              f"separation={'PASS' if sep else 'FAIL'}  "
              f"halluc-caught={'PASS' if halluc_caught else 'FAIL'}  "
              f"F2-passes={'PASS' if f2_pass else 'FAIL'}")
    else:
        res = run_model(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"[posmem] wrote {args.out}")


if __name__ == "__main__":
    main()
