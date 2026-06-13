"""C2 adversary: can a belief PASS the onset anchor yet be a hallucination?

The anchor scores believed (x,y) AND believed velocity vs GT over the first
n_anchor in-window occluded steps. To 'pass' it, a belief must coincide with the
real ball over those steps. We test two adversary classes:

  A) anchor-matched-then-diverge: equals GT for the first n_anchor steps, then
     switches to a DIFFERENT but physical billiard. Is this 'hallucination' or
     just F2 (butterfly divergence)? -> conceptual question.

  B) speed-mismatched coherent ball: a billiard at a DIFFERENT constant speed S'
     (not env S). Part-2 residual fixes speed = env S, so this should be penalised
     by the residual even though it is 'coherent'. Tests whether fixed-S is doing work.

We also quantify the C2 claim directly: over random hallucinations, how often does
onset_pos_err fall BELOW a candidate pass threshold (i.e. a hallucination that
happens to start near GT)?
"""
from __future__ import annotations
import sys, pathlib, json
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from scorer_probe import (billiard_rollout, billiard_from_params,
                          best_fit_billiard_residual, onset_anchor_error,
                          make_gt_episode, WALL_LO, WALL_HI)


def anchor_matched_diverge(states, n_anchor, rng):
    """GT for first n_anchor steps, then a different physical billiard continuing from
    the GT anchor end-state but with a perturbed velocity direction (still speed S)."""
    n = len(states)
    S = float(np.hypot(states[0, 2], states[0, 3]))
    b = states[:, :2].astype(float).copy()
    # at step n_anchor-1 we are at GT pos; redirect with a random heading at same speed
    x, y = b[n_anchor - 1]
    th = rng.uniform(0, 2 * np.pi)
    cont = billiard_rollout(x, y, S * np.cos(th), S * np.sin(th), n - n_anchor)
    b[n_anchor:] = cont
    return b


def speed_mismatch_ball(states, rng, scale):
    """Coherent billiard from GT onset state but speed scaled by `scale` (!=1)."""
    n = len(states)
    x0, y0 = states[0, :2]
    S = float(np.hypot(states[0, 2], states[0, 3]))
    th = np.arctan2(states[0, 3], states[0, 2])
    return billiard_from_params(x0, y0, th, S * scale, n)


def run(n_occ=16, n_seeds=60, n_anchor=3, noise=0.65, seed_master=7):
    rng = np.random.default_rng(seed_master)
    # how often does a fully-random hallucination land near GT onset by chance?
    rnd_onset = []
    amd_resid, amd_pos, amd_vel = [], [], []
    sm_resid = {0.7: [], 1.3: []}
    for i in range(n_seeds):
        states, S = make_gt_episode(seed=2000 + i, n_occ=n_occ)
        # random hallucination onset pos err
        x0, y0 = rng.uniform(WALL_LO, WALL_HI, 2)
        th = rng.uniform(0, 2 * np.pi)
        hb = billiard_from_params(x0, y0, th, S, n_occ) + rng.normal(0, noise, (n_occ, 2))
        pe, ve = onset_anchor_error(hb, states, n_anchor)
        rnd_onset.append(pe)
        # anchor-matched-then-diverge
        b = anchor_matched_diverge(states, n_anchor, rng) + rng.normal(0, noise, (n_occ, 2))
        r, _ = best_fit_billiard_residual(b, S)
        pe2, ve2 = onset_anchor_error(b, states, n_anchor)
        amd_resid.append(r); amd_pos.append(pe2); amd_vel.append(ve2)
        # speed-mismatched coherent balls
        for sc in (0.7, 1.3):
            bm = speed_mismatch_ball(states, rng, sc) + rng.normal(0, noise, (n_occ, 2))
            rr, _ = best_fit_billiard_residual(bm, S)  # residual uses TRUE env S
            sm_resid[sc].append(rr)

    def s(v): return dict(mean=float(np.mean(v)), p50=float(np.median(v)),
                          p10=float(np.percentile(v, 10)), min=float(np.min(v)))
    out = dict(
        config=dict(n_occ=n_occ, n_seeds=n_seeds, n_anchor=n_anchor, noise=noise),
        random_hallucination_onset_pos_err=s(rnd_onset),
        anchor_matched_diverge=dict(residual=s(amd_resid), onset_pos=s(amd_pos), onset_vel=s(amd_vel)),
        speed_mismatch_residual_at_trueS={str(k): s(v) for k, v in sm_resid.items()},
    )
    print(json.dumps(out, indent=2))
    (pathlib.Path(__file__).parent / "adversary_c2_result.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    run()
