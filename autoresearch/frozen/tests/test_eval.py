"""Gate tests: the comeback eval — oracle == 1.0 exactly, the reference adapters
land where the design says, bookkeeping matches an independent brute-force
reimplementation, aggregation math is what the spec claims, and (post red-team)
the scalar is monotone in genuine retention horizon with no >0.6 shortcut for
bounded-window models."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from autoresearch.frozen.adapters import make_adapter  # noqa: E402
from autoresearch.frozen.env import (  # noqa: E402
    CELL_PX, N_CELLS, OUT_IDX, PITCH_PX, TL_OFFSET, VIEW_PX, apply_action)
from autoresearch.frozen.eval_comeback import (  # noqa: E402
    aggregate, run_episode, run_eval)
from autoresearch.frozen.eval_policies import (  # noqa: E402
    EvalBoxLoop, EvalIdiotWalk, EvalOutAndBack, EvalRetrace)
from autoresearch.frozen.readout import cells_in_view, read_cells  # noqa: E402
from autoresearch.frozen.env import PALETTE  # noqa: E402

SMALL_SUITE = [
    ("oab_mid", lambda: EvalOutAndBack(30, 55)),
    ("box_small", lambda: EvalBoxLoop(12, 25, laps=10)),
    ("idiot_fast", lambda: EvalIdiotWalk(0.92)),
    ("retrace_mid", lambda: EvalRetrace(30, 60)),
]
SMALL = dict(suite=SMALL_SUITE, n_seeds=2, prefix_len=96, imag_len=256,
             min_events=10, privileged=True)


def test_oracle_scores_exactly_one():
    r = run_eval(make_adapter("oracle"), **SMALL)
    assert r["gates_passed"], r["fail_reasons"]
    assert r["composite"] == 1.0 and r["composite_gated"] == 1.0
    assert r["real_anchored"]["score"] == 1.0 and r["consistency"]["score"] == 1.0
    assert r["real_anchored"]["n_events"] >= 50 and r["consistency"]["n_events"] >= 20
    assert r["gates"]["fidelity"]["value"] == 1.0
    assert r["border_drift_px"] == 0.0


def test_perfect_imaginary_liar_scores_zero():
    """The 'consistent liar' (perfect self-consistency of a WRONG world) passes
    the gates but must score ~0: real-anchored is at chance, chance correction
    clamps it to ~0, and consistency can only MULTIPLY real retention. Under the
    old additive scoring this adapter got 0.43 and beat honest short memory —
    the red-team's non-monotonicity finding."""
    r = run_eval(make_adapter("perfect_imaginary"), **SMALL)
    assert r["gates_passed"], r["fail_reasons"]
    assert r["consistency"]["score"] == 1.0
    assert r["composite_gated"] <= 0.05, r["composite_gated"]


def test_constant_color_is_gated_to_zero():
    r = run_eval(make_adapter("constant_color"), **SMALL)
    assert not r["gates_passed"]
    assert r["composite_gated"] == 0.0


def test_noise_cells_fails_fidelity():
    r = run_eval(make_adapter("noise_cells"), **SMALL)
    assert not r["gates"]["fidelity"]["passed"]
    assert r["composite_gated"] == 0.0


def test_copy_last_is_gated_to_zero():
    """A frozen frame under a MOVING registration still yields (garbage) comeback
    reads — the fidelity gate is what kills it. Load-bearing gate, by design."""
    r = run_eval(make_adapter("copy_last"), **SMALL)
    assert r["composite_gated"] == 0.0
    assert not r["gates"]["fidelity"]["passed"]


def test_unprivileged_factories_get_none():
    seen = []

    class Probe:
        def __init__(self, env):
            seen.append(env)
            self.frame = np.zeros((VIEW_PX, VIEW_PX, 3), dtype=np.uint8)

        def begin(self, f, a):
            pass

        def step(self, a):
            return self.frame

    run_episode(lambda env: Probe(env), EvalIdiotWalk(0.9), 1, 2,
                prefix_len=8, imag_len=8, privileged=False)
    assert seen == [None]


# --- bounded-memory monotonicity (the red-team's central exploit, as a fence) --

class BoundedMemoryAdapter:
    """Genuine retention horizon of W frames (belief self-refreshing): remembers
    on-screen cell reads for W steps, correct OUT geometry, persistent random
    colors elsewhere (passes fidelity + entropy, consistency ~1). This is the
    red-team's GeoOutWorld shape: under the OLD scoring W=64 scored 0.62; the
    fixed scoring must keep bounded windows low and be monotone in W."""

    def __init__(self, env, W, seed=7):
        self.W = W
        self.rng = np.random.default_rng(seed)
        self.pos = env.pos          # privileged registration (test-only shortcut;
        self.mem = {}               # red-team verified it's recoverable honestly)
        self.hallu = {}
        self.t = 0

    def _observe(self, frame, pos):
        for key, r in read_cells(frame, pos).items():
            if r.on_screen:
                self.mem[key] = (r.color, self.t)

    def begin(self, frames, actions):
        self._observe(frames[0], self.pos)
        for a in actions[1:]:
            self.t += 1
            self.pos = apply_action(self.pos, int(a), check=True)
            self._observe(frames[self.t], self.pos)

    def _color(self, cell):
        if cell in self.mem and self.t - self.mem[cell][1] <= self.W:
            return self.mem[cell][0]
        if not (0 <= cell[0] < N_CELLS and 0 <= cell[1] < N_CELLS):
            return OUT_IDX          # geometry knowledge, no content memory
        if cell not in self.hallu:
            self.hallu[cell] = int(self.rng.integers(0, 5))
        return self.hallu[cell]

    def step(self, action):
        self.t += 1
        self.pos = apply_action(self.pos, int(action), check=False)
        frame = np.empty((VIEW_PX, VIEW_PX, 3), dtype=np.uint8)
        painted = {}
        for ci, cj, y0, x0, ov_y, ov_x in cells_in_view(self.pos):
            c = self._color((ci, cj))
            painted[(ci, cj)] = c
            frame[y0:y0 + ov_y, x0:x0 + ov_x] = PALETTE[c]
        for key, c in painted.items():   # self-refresh belief with what was shown
            self.mem[key] = (c, self.t)
        return frame


def test_bounded_window_monotone_and_capped():
    """Post-fix semantics: a bounded-W model's score ~= (qualified age bins fully
    covered by W) / (all qualified bins) — proportional, monotone credit; its
    BEYOND-window bins must be ~0 (chance-corrected), not the old 0.2 floor, and
    consistency must not add anything on top (multiplicative). The old additive/
    uncorrected scoring gave W=64 -> 0.62 with 6/6 bins (red-team exploit)."""
    scores, results = {}, {}
    for W in (16, 64, 10**9):
        r = run_eval(lambda env, W=W: BoundedMemoryAdapter(env, W), **SMALL)
        assert r["gates_passed"], (W, r["fail_reasons"])
        scores[W], results[W] = r["composite_gated"], r
    assert scores[10**9] >= 0.99, scores            # full memory ~ oracle
    assert scores[16] + 0.03 < scores[64] < scores[10**9] - 0.05, scores
    assert scores[16] <= 0.35, scores
    # the semantics fence: covered-bin fraction, and dead beyond-window bins
    bins64 = [b for b in results[64]["real_anchored"]["bins"] if b["qualified"]]
    covered = [b for b in bins64 if (b["hi"] is not None and b["hi"] <= 65)]
    beyond = [b for b in bins64 if b["lo"] >= 65]
    assert covered and beyond, bins64               # both regimes must be measured
    for b in covered:
        # within-window: high but NOT 1.0 — young comebacks include cells that
        # expired long ago and were re-imagined (self-refreshed hallucination is
        # young in age but wrong vs GT). ~0.88 observed; fence at 0.75.
        assert b["acc_cc"] >= 0.75, b
    for b in beyond:
        assert b["acc_cc"] <= 0.15, b               # beyond-window: ~0, no floor
    want = len(covered) / len(bins64)
    assert abs(scores[64] - want) <= 0.12, (scores[64], want, bins64)


# --- independent brute-force bookkeeping reference ---------------------------

def brute_force_events(positions, prefix_len):
    """Recompute (cell, t, age, phase) of every comeback event from the position
    trace alone, with a completely different formulation: per-cell boolean
    timelines + run detection."""
    T = len(positions)
    prs = [p[0] for p in positions]
    pcs = [p[1] for p in positions]
    lo_ci = (min(prs) * PITCH_PX + TL_OFFSET) // CELL_PX - 1
    hi_ci = (max(prs) * PITCH_PX + TL_OFFSET + VIEW_PX) // CELL_PX + 1
    lo_cj = (min(pcs) * PITCH_PX + TL_OFFSET) // CELL_PX - 1
    hi_cj = (max(pcs) * PITCH_PX + TL_OFFSET + VIEW_PX) // CELL_PX + 1
    events = []
    for ci in range(lo_ci, hi_ci + 1):
        for cj in range(lo_cj, hi_cj + 1):
            onscreen = np.zeros(T, dtype=bool)
            anyov = np.zeros(T, dtype=bool)
            for t, (pr, pc) in enumerate(positions):
                tly, tlx = PITCH_PX * pr + TL_OFFSET, PITCH_PX * pc + TL_OFFSET
                ov_y = min(VIEW_PX, ci * CELL_PX + CELL_PX - tly) - max(0, ci * CELL_PX - tly)
                ov_x = min(VIEW_PX, cj * CELL_PX + CELL_PX - tlx) - max(0, cj * CELL_PX - tlx)
                anyov[t] = ov_y > 0 and ov_x > 0
                onscreen[t] = ov_y >= 6 and ov_x >= 6
            visits = []
            t = 0
            while t < T:
                if onscreen[t]:
                    s = t
                    while t < T and onscreen[t]:
                        t += 1
                    visits.append((s, t - 1))
                else:
                    t += 1
            for (ps, pe), (ns, ne) in zip(visits[:-1], visits[1:]):
                # v2.1 age: LONGEST contiguous zero-overlap run between visits
                best = run = 0
                for t in range(pe + 1, ns):
                    run = run + 1 if not anyov[t] else 0
                    best = max(best, run)
                if best > 0:
                    events.append(((ci, cj), ns, best,
                                   "imag" if ns >= prefix_len else "prefix"))
    return sorted(events)


def test_tracker_matches_brute_force():
    for seed in (0, 1, 2):
        events, _, _, _, positions = run_episode(
            make_adapter("oracle"), EvalRetrace(25, 50),
            map_seed=1000 + seed, ep_seed=2000 + seed,
            prefix_len=80, imag_len=220, privileged=True)
        got = sorted((tuple(e["cell"]), e["t"], e["age"], e["phase"]) for e in events)
        want = brute_force_events(positions, prefix_len=80)
        assert got == want, (seed, len(got), len(want),
                             [x for x in got if x not in want][:5],
                             [x for x in want if x not in got][:5])
        assert len(got) > 0, "brute-force check ran on an eventless episode"


# --- aggregation math ----------------------------------------------------------

def _ev(prov, age, correct, weight=1.0, ref=0):
    return {"provenance": prov, "age": age, "correct": correct, "weight": weight,
            "ref": ref, "phase": "imag"}


def test_aggregate_math():
    # real bin1 [1,16]: 12 in-map events, 9 correct -> acc .75, cc (.75-.2)/.8 = .6875
    # real bin2 [17,32]: 10 in-map events, 4 correct -> acc .4,  cc (.4-.2)/.8  = .25
    events = [_ev("real", 5, i < 9) for i in range(12)]
    events += [_ev("real", 20, i < 4) for i in range(10)]
    # imag bin1: 10 in-map, 6 correct -> acc .6, cc .5
    events += [_ev("imag", 5, i < 6) for i in range(10)]
    # unqualified stragglers must be ignored (n < min_events)
    events += [_ev("real", 300, True) for _ in range(3)]
    fidelity = [True] * 95 + [False] * 5              # 0.95 >= 0.9 -> pass
    colors = list(np.tile(np.arange(5), 8))           # perfectly uniform -> pass
    r = aggregate(events, fidelity, colors, min_events=10)
    assert r["gates_passed"], r["fail_reasons"]
    want_real = (0.6875 + 0.25) / 2                   # equal-weight over cc bins
    assert abs(r["real_anchored"]["score"] - want_real) < 1e-12
    assert abs(r["consistency"]["score"] - 0.5) < 1e-12
    assert abs(r["composite"] - want_real * (0.7 + 0.3 * 0.5)) < 1e-12

    # border (OUT-referenced) events must NOT move the scored accuracy — they are
    # pure geometry and dominate far bins at any nonzero weight (red-team S3).
    # They surface only as the border_recall diagnostic.
    events_out = [_ev("real", 5, i < 8) for i in range(10)]
    events_out += [_ev("real", 5, True, weight=0.1, ref=OUT_IDX) for _ in range(40)]
    r_out = aggregate(events_out, fidelity, colors, min_events=10)
    b0 = r_out["real_anchored"]["bins"][0]
    assert abs(b0["acc_cc"] - (0.8 - 0.2) / 0.8) < 1e-12   # in-map only: 8/10
    assert b0["n_border"] == 40 and b0["border_recall"] == 1.0

    # chance clamp: at-chance bin scores exactly 0
    events_ch = [_ev("real", 5, i < 2) for i in range(10)]    # acc = .2 = chance
    r_ch = aggregate(events_ch, fidelity, colors, min_events=10)
    assert r_ch["real_anchored"]["bins"][0]["acc_cc"] == 0.0

    # consistency cannot rescue a no-real-signal run (multiplicative form)
    events_liar = [_ev("real", 5, i < 2) for i in range(10)]  # chance -> cc 0
    events_liar += [_ev("imag", 5, True) for _ in range(10)]  # consistency 1.0
    r_liar = aggregate(events_liar, fidelity, colors, min_events=10)
    assert r_liar["composite"] == 0.0

    # prefix-phase events must be excluded
    r2 = aggregate([dict(e, phase="prefix") for e in events], fidelity, colors,
                   min_events=10)
    assert r2["real_anchored"]["score"] is None and r2["composite_gated"] == 0.0
    # fidelity gate
    r3 = aggregate(events, [True] * 80 + [False] * 20, colors, min_events=10)
    assert not r3["gates_passed"] and r3["composite_gated"] == 0.0
    # entropy gate: collapsed colors
    r4 = aggregate(events, fidelity, [1] * 40, min_events=10)
    assert not r4["gates"]["entropy"]["passed"] and r4["composite_gated"] == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("test_eval: ALL PASS")
