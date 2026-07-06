"""Gate tests: the comeback eval — oracle == 1.0 exactly, the reference adapters
land where the design says, bookkeeping matches an independent brute-force
reimplementation, and the aggregation math is what the spec claims."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from autoresearch.frozen.adapters import make_adapter  # noqa: E402
from autoresearch.frozen.env import CELL_PX, PITCH_PX, TL_OFFSET, VIEW_PX  # noqa: E402
from autoresearch.frozen.eval_comeback import (  # noqa: E402
    aggregate, run_episode, run_eval)
from autoresearch.frozen.eval_policies import (  # noqa: E402
    EvalBoxLoop, EvalIdiotWalk, EvalOutAndBack, EvalRetrace)

SMALL_SUITE = [
    ("oab_mid", lambda: EvalOutAndBack(30, 55)),
    ("box_small", lambda: EvalBoxLoop(12, 25, laps=10)),
    ("idiot_fast", lambda: EvalIdiotWalk(0.92)),
    ("retrace_mid", lambda: EvalRetrace(30, 60)),
]
SMALL = dict(suite=SMALL_SUITE, n_seeds=2, prefix_len=96, imag_len=256,
             min_events=10)


def test_oracle_scores_exactly_one():
    r = run_eval(make_adapter("oracle"), **SMALL)
    assert r["gates_passed"], r["fail_reasons"]
    assert r["composite"] == 1.0 and r["composite_gated"] == 1.0
    assert r["real_anchored"]["score"] == 1.0 and r["consistency"]["score"] == 1.0
    assert r["real_anchored"]["n_events"] >= 50 and r["consistency"]["n_events"] >= 20
    assert r["gates"]["fidelity"]["value"] == 1.0
    assert r["border_drift_px"] == 0.0


def test_perfect_imaginary_shows_why_anchoring():
    """The 'consistent liar': perfect self-consistency of a WRONG world. Must pass
    the gates yet score ~chance on the real-anchored component — the reason the
    composite is 0.7 ground-truth-anchored."""
    r = run_eval(make_adapter("perfect_imaginary"), **SMALL)
    assert r["gates_passed"], r["fail_reasons"]
    assert r["consistency"]["score"] == 1.0
    assert 0.02 <= r["real_anchored"]["score"] <= 0.45
    assert 0.25 <= r["composite_gated"] <= 0.62


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


# --- independent brute-force bookkeeping reference ---------------------------

def brute_force_events(positions, prefix_len):
    """Recompute (cell, t, age, phase) of every comeback event from the position
    trace alone, with a completely different formulation: per-cell boolean
    timelines + run detection. Colors/refs are not needed to check WHICH events
    fire (the tracker's geometry bookkeeping is what is under test here)."""
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
            # visits = maximal runs of onscreen
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
                if not anyov[pe + 1:ns].all() and (~anyov[pe + 1:ns]).any():
                    pass  # placeholder to keep logic explicit below
                had_zero_gap = bool((~anyov[pe + 1:ns]).any())
                if had_zero_gap:
                    events.append(((ci, cj), ns, ns - pe,
                                   "imag" if ns >= prefix_len else "prefix"))
    return sorted(events)


def test_tracker_matches_brute_force():
    for seed in (0, 1, 2):
        events, _, _, _, positions = run_episode(
            make_adapter("oracle"), EvalRetrace(25, 50),
            map_seed=1000 + seed, ep_seed=2000 + seed,
            prefix_len=80, imag_len=220)
        got = sorted((tuple(e["cell"]), e["t"], e["age"], e["phase"]) for e in events)
        want = brute_force_events(positions, prefix_len=80)
        assert got == want, (seed, len(got), len(want),
                             [x for x in got if x not in want][:5],
                             [x for x in want if x not in got][:5])
        assert len(got) > 0, "brute-force check ran on an eventless episode"


# --- aggregation math ----------------------------------------------------------

def _ev(prov, age, correct, weight=1.0):
    return {"provenance": prov, "age": age, "correct": correct, "weight": weight,
            "phase": "imag"}


def test_aggregate_math():
    # bin1 [1,16]: 12 events, 9 correct -> 0.75 ; bin2 [17,32]: 10 events 4 correct -> 0.4
    events = [_ev("real", 5, i < 9) for i in range(12)]
    events += [_ev("real", 20, i < 4) for i in range(10)]
    # imag: one qualified bin, weighted: 10 x w1 (6 correct) + 10 x w0.1 (0 correct)
    events += [_ev("imag", 5, i < 6) for i in range(10)]
    events += [_ev("imag", 5, False, weight=0.1) for i in range(10)]
    # unqualified stragglers must be ignored (n < min_events)
    events += [_ev("real", 300, True) for _ in range(3)]
    fidelity = [True] * 95 + [False] * 5              # 0.95 >= 0.9 -> pass
    colors = list(np.tile(np.arange(5), 8))           # perfectly uniform -> pass
    r = aggregate(events, fidelity, colors, min_events=10)
    assert r["gates_passed"], r["fail_reasons"]
    assert abs(r["real_anchored"]["score"] - (0.75 + 0.4) / 2) < 1e-12
    want_imag = (6 * 1.0) / (10 * 1.0 + 10 * 0.1)     # 6/11
    assert abs(r["consistency"]["score"] - want_imag) < 1e-12
    assert abs(r["composite"] - (0.7 * 0.575 + 0.3 * want_imag)) < 1e-12
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
