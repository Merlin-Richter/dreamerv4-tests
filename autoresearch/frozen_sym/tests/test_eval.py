"""Gate tests: the ColorField-SYM comeback eval — oracle == 1.0 exactly, the
reference adapters land where the design says, bookkeeping matches an
independent brute-force reimplementation from position traces, aggregation
math is what the spec claims (OUT exclusion, chance clamp, multiplicative
composite, the v2.2 continuous score fid*(0.2*ent + 0.8*composite)), and the
bounded-memory monotonicity fence holds under 5x-dilated tick ages.
FROZEN-LAYER-sym test; spec: tasks/in-progress/colorfield-sym-frozen-layer.md
(+ the dated v2.2 scoring addendum)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from autoresearch.frozen_sym.adapters import make_adapter, paint_grid  # noqa: E402
from autoresearch.frozen_sym.env import (  # noqa: E402
    BOARD, N_COLORS, OUT_IDX, VIEW_CELLS, VIEW_HALF, apply_action)
from autoresearch.frozen_sym.eval_comeback import (  # noqa: E402
    aggregate, run_episode, run_eval)
from autoresearch.frozen_sym.eval_policies import (  # noqa: E402
    EvalBoxLoop, EvalIdiotWalk, EvalOutAndBack, EvalRetrace)

# Sizes in TICKS (phase-5 dilation: /5 for effective moves). Mirrors the pixel
# SMALL config's move counts: 48 prefix moves, 256 imagination moves.
SMALL_SUITE = [
    ("oab_mid", lambda: EvalOutAndBack(5, 9)),
    ("box_small", lambda: EvalBoxLoop(2, 4, laps=10)),
    ("idiot_fast", lambda: EvalIdiotWalk(0.92)),
    ("retrace_mid", lambda: EvalRetrace(5, 10)),
]
SMALL = dict(suite=SMALL_SUITE, n_seeds=2, prefix_len=240, imag_len=1280,
             min_events=10, privileged=True)


def _score_identity(r):
    """The v2.2 formula must hold exactly on the returned components."""
    comp = r["composite"] if r["composite"] is not None else 0.0
    want = r["fidelity"]["value"] * (0.2 * r["entropy"]["ent"] + 0.8 * comp)
    assert abs(r["score"] - want) < 1e-12, (r["score"], want)


def test_oracle_scores_exactly_one():
    r = run_eval(make_adapter("oracle"), **SMALL)
    assert r["score"] == 1.0 and r["composite"] == 1.0
    assert r["real_anchored"]["score"] == 1.0 and r["consistency"]["score"] == 1.0
    assert r["real_anchored"]["n_events"] >= 50 and r["consistency"]["n_events"] >= 20
    assert r["fidelity"]["value"] == 1.0
    assert r["fidelity"]["move"] == 1.0 and r["fidelity"]["hold"] == 1.0
    assert r["entropy"]["ent"] == 1.0
    assert r["border_drift_cells"] == 0.0
    _score_identity(r)


def test_perfect_imaginary_liar_capped_at_fidelity_floor():
    """The 'consistent liar' (perfect self-consistency of a WRONG world) is a
    coherent scroller with zero real retention: under v2.2 it earns AT MOST
    the 0.2 fidelity floor — real-anchored is at chance (chance correction
    clamps to ~0) and consistency can only MULTIPLY real retention (pixel
    red-team, inherited verbatim). It must never beat a model with genuine
    retention: everything above the floor requires real_cc > 0."""
    r = run_eval(make_adapter("perfect_imaginary"), **SMALL)
    assert r["consistency"]["score"] == 1.0
    assert r["composite"] <= 0.05, r["composite"]
    assert r["score"] <= 0.2 + 0.8 * 0.05, r["score"]
    _score_identity(r)


def test_constant_color_scores_near_zero():
    """Sym-tier nuance: a uniform grid is shift-invariant, so constant_color
    earns FULL fidelity (unlike the pixel tier, where the shift estimator's
    tie-break failed it) — the ENTROPY ramp must be the killer (KL = ln 5 >>
    0.6 -> ent = 0), leaving only 0.8*composite, itself ~0 at chance. The
    collapse must land WELL BELOW the 0.2 honest-scroller floor, or collapse
    would be the loop's cheapest 'improvement'."""
    r = run_eval(make_adapter("constant_color"), **SMALL)
    assert r["fidelity"]["value"] >= 0.99            # shift-invariant: full credit
    assert r["entropy"]["ent"] == 0.0                # the load-bearing guard
    assert r["score"] <= 0.10, r["score"]
    _score_identity(r)


def test_noise_cells_scores_near_zero():
    r = run_eval(make_adapter("noise_cells"), **SMALL)
    assert r["fidelity"]["value"] < 0.10             # both pools fail
    assert r["score"] <= 0.05, r["score"]
    _score_identity(r)


def test_copy_last_pins_at_half_fidelity():
    """A frozen grid under a MOVING registration: every off-phase hold passes
    free (unchanged grid IS the correct off-phase prediction), every phase-0
    move fails — the v2.2 equal-weight split pins fid at ~0.5 (the pooled
    per-tick mean would have said ~0.8, diluted by the 4:1 holds). Garbage
    comeback reads keep composite ~0, so the score sits near 0.5*0.2 = 0.1,
    below the 0.2 coherent-scroller floor."""
    r = run_eval(make_adapter("copy_last"), **SMALL)
    assert r["fidelity"]["hold"] >= 0.99
    assert r["fidelity"]["move"] <= 0.05
    assert 0.45 <= r["fidelity"]["value"] <= 0.55
    assert r["score"] <= 0.15, r["score"]
    _score_identity(r)


def test_unprivileged_factories_get_none():
    seen = []

    class Probe:
        def __init__(self, env):
            seen.append(env)
            self.grid = np.zeros((VIEW_CELLS, VIEW_CELLS), dtype=np.uint8)

        def begin(self, g, a):
            pass

        def step(self, a):
            return self.grid

    run_episode(lambda env: Probe(env), EvalIdiotWalk(0.9), 1, 2,
                prefix_len=10, imag_len=10, privileged=False)
    assert seen == [None]


# --- bounded-memory monotonicity (the pixel red-team's central exploit, as a
# --- fence; W is in TICKS — ages are 5x dilated by the phase rule) ------------

class BoundedMemoryAdapter:
    """Genuine retention horizon of W TICKS (belief self-refreshing): remembers
    viewport cell reads for W ticks, correct OUT geometry, persistent random
    colors elsewhere (passes fidelity + entropy, consistency ~1). The pixel
    red-team's GeoOutWorld shape: the v2.1 scoring must keep bounded windows
    low and be monotone in W (beyond-window age bins are chance by
    construction under max-gap age)."""

    def __init__(self, env, W, seed=7):
        self.W = W
        self.rng = np.random.default_rng(seed)
        self.pos = env.pos          # privileged registration (test-only shortcut;
        self.mem = {}               # pixel red-team verified it's recoverable honestly)
        self.hallu = {}
        self.t = 0

    def _observe(self, grid, pos):
        for i in range(VIEW_CELLS):
            for j in range(VIEW_CELLS):
                key = (pos[0] - VIEW_HALF + i, pos[1] - VIEW_HALF + j)
                self.mem[key] = (int(grid[i, j]), self.t)

    def begin(self, grids, actions):
        self._observe(grids[0], self.pos)
        for t in range(1, len(actions)):
            self.t += 1
            self.pos = apply_action(self.pos, int(actions[t]), check=True)
            self._observe(grids[t], self.pos)

    def _color(self, cell):
        if cell in self.mem and self.t - self.mem[cell][1] <= self.W:
            return self.mem[cell][0]
        if not (0 <= cell[0] < BOARD and 0 <= cell[1] < BOARD):
            return OUT_IDX          # geometry knowledge, no content memory
        if cell not in self.hallu:
            self.hallu[cell] = int(self.rng.integers(0, N_COLORS))
        return self.hallu[cell]

    def step(self, action):
        self.t += 1
        self.pos = apply_action(self.pos, int(action), check=False)
        grid = paint_grid(self.pos, self._color)
        self._observe(grid, self.pos)   # self-refresh belief with what was shown
        return grid


def test_bounded_window_monotone_and_capped():
    """v2.1 semantics under 5x tick dilation: a bounded-W model's score ~= the
    fraction of qualified age bins its window covers — proportional, monotone
    credit; its BEYOND-window bins must be ~0 (chance-corrected), not a 0.2
    floor, and consistency must not add anything on top (multiplicative).
    W=16 ticks = 3.2 effective moves (the relay-training window); W=80 ticks
    = 16 moves; huge = full memory."""
    scores, comps, results = {}, {}, {}
    for W in (16, 80, 10**9):
        r = run_eval(lambda env, W=W: BoundedMemoryAdapter(env, W), **SMALL)
        # honest bounded-memory worlds earn the full fid/ent factors — the
        # composite fences below are then EXACTLY the v2.1 fences, unchanged
        assert r["fidelity"]["value"] >= 0.99 and r["entropy"]["ent"] == 1.0, (W, r["fidelity"], r["entropy"])
        _score_identity(r)
        scores[W], comps[W], results[W] = r["score"], r["composite"], r
    # v2.1 fences on the memory term (verbatim — the red-team's exploit fence):
    assert comps[10**9] >= 0.99, comps              # full memory ~ oracle
    assert comps[16] + 0.03 < comps[80] < comps[10**9] - 0.05, comps
    # W=16 ticks (the relay window) bridges almost nothing: ~0, and W=80's
    # credit ~ its covered-bin fraction (2/6 fully + partials; observed 0.34)
    assert comps[16] <= 0.15, comps
    assert 0.20 <= comps[80] <= 0.55, comps
    # v2.2 fences on the headline: monotone in W, floor+0.8*comp shape, and a
    # bounded window must NOT reach the top (0.2 + 0.8*comp caps it):
    assert scores[16] < scores[80] < scores[10**9], scores
    assert scores[10**9] >= 0.99, scores
    assert scores[16] <= 0.2 + 0.8 * 0.15 + 1e-9, scores
    assert scores[80] <= 0.2 + 0.8 * 0.55 + 1e-9, scores
    # the semantics fence: fully-covered bins high, beyond-window bins dead.
    # NB the [1,16] bin is EXCLUDED from the high fence: young comebacks are
    # dominated by flickers of long-expired, self-refresh-hallucinated path
    # cells (young in age, wrong vs GT). The pollution is heavier than the
    # pixel tier's config (W/episode ratio 16x smaller here: 80/1280 ticks vs
    # 64/256 moves), pinning that bin at ~0.4 instead of ~0.88 — measured
    # behavior of the adapter, not scoring slack; it must still clearly beat
    # the beyond-window bins.
    bins80 = {(b["lo"], b["hi"]): b for b in results[80]["real_anchored"]["bins"]
              if b["qualified"]}
    covered = [b for (lo, hi), b in bins80.items()
               if lo >= 17 and hi is not None and hi <= 65]
    beyond = [b for (lo, hi), b in bins80.items() if lo >= 129]
    assert covered and beyond, bins80               # both regimes must be measured
    for b in covered:
        assert b["acc_cc"] >= 0.50, b               # within-window: high, not 1.0
    for b in beyond:
        assert b["acc_cc"] <= 0.15, b               # beyond-window: ~0, no floor
    young80 = bins80.get((1, 17))
    if young80 is not None:
        assert 0.20 <= young80["acc_cc"] <= 0.65, young80   # polluted but alive
    bins16 = [b for b in results[16]["real_anchored"]["bins"] if b["qualified"]]
    for b in bins16:
        if b["lo"] >= 33:                           # far beyond W=16
            assert b["acc_cc"] <= 0.15, b


# --- independent brute-force bookkeeping reference ---------------------------

def brute_force_events(positions, prefix_len):
    """Recompute (cell, t, age, phase) of every comeback event from the position
    trace alone, with a completely different formulation: per-cell boolean
    in-viewport timelines + run detection. on-screen := chebyshev distance to
    the center <= VIEW_HALF; age := longest contiguous off run between visits
    (== the single gap, since no partial states exist)."""
    T = len(positions)
    cells = set()
    for (pr, pc) in positions:
        for di in range(-VIEW_HALF, VIEW_HALF + 1):
            for dj in range(-VIEW_HALF, VIEW_HALF + 1):
                cells.add((pr + di, pc + dj))
    events = []
    for (ci, cj) in cells:
        onscreen = np.array([abs(ci - pr) <= VIEW_HALF and abs(cj - pc) <= VIEW_HALF
                             for (pr, pc) in positions])
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
            best = run = 0
            for u in range(pe + 1, ns):
                run = run + 1 if not onscreen[u] else 0
                best = max(best, run)
            if best > 0:
                events.append(((ci, cj), ns, best,
                               "imag" if ns >= prefix_len else "prefix"))
    return sorted(events)


def test_tracker_matches_brute_force():
    for seed in (0, 1, 2):
        events, _, _, _, positions = run_episode(
            make_adapter("oracle"), EvalRetrace(5, 10),
            map_seed=1000 + seed, ep_seed=2000 + seed,
            prefix_len=200, imag_len=600, privileged=True)
        got = sorted((tuple(e["cell"]), e["t"], e["age"], e["phase"]) for e in events)
        want = brute_force_events(positions, prefix_len=200)
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
    # fidelity (v2.2): (is_move, ok) pairs — 80 holds all pass, 20 moves 15/20
    # -> fid_hold 1.0, fid_move 0.75, fid = 0.875
    fidelity = ([(False, True)] * 80
                + [(True, True)] * 15 + [(True, False)] * 5)
    colors = list(np.tile(np.arange(5), 8))           # uniform -> KL 0 -> ent 1
    r = aggregate(events, fidelity, colors, min_events=10)
    want_real = (0.6875 + 0.25) / 2                   # equal-weight over cc bins
    assert abs(r["real_anchored"]["score"] - want_real) < 1e-12
    assert abs(r["consistency"]["score"] - 0.5) < 1e-12
    want_comp = want_real * (0.7 + 0.3 * 0.5)
    assert abs(r["composite"] - want_comp) < 1e-12
    assert r["fidelity"]["move"] == 0.75 and r["fidelity"]["hold"] == 1.0
    assert abs(r["fidelity"]["value"] - 0.875) < 1e-12
    assert r["entropy"]["ent"] == 1.0
    assert abs(r["score"] - 0.875 * (0.2 * 1.0 + 0.8 * want_comp)) < 1e-12

    # border (OUT-referenced) events must NOT move the scored accuracy — they
    # are pure geometry (pixel red-team S3). Only the border_recall diagnostic.
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

    # prefix-phase events must be excluded: no qualified real bins -> composite
    # None -> the memory term contributes 0, the fidelity floor remains
    r2 = aggregate([dict(e, phase="prefix") for e in events], fidelity, colors,
                   min_events=10)
    assert r2["real_anchored"]["score"] is None and r2["composite"] is None
    assert "no_qualified_real_bins" in r2["flags"]
    assert abs(r2["score"] - 0.875 * 0.2 * 1.0) < 1e-12
    # fidelity scales the WHOLE score linearly (v2.2: factor, not gate)
    r3 = aggregate(events, [(False, True)] * 80 + [(True, i < 10) for i in range(20)],
                   colors, min_events=10)
    assert abs(r3["fidelity"]["value"] - 0.75) < 1e-12         # (1.0 + 0.5)/2
    assert abs(r3["score"] - 0.75 * (0.2 * 1.0 + 0.8 * want_comp)) < 1e-12
    # entropy ramp endpoint: collapsed colors (KL = ln 5 > 0.6) -> ent 0 -> the
    # fidelity floor vanishes, memory term survives (damped by fid only)
    r4 = aggregate(events, fidelity, [1] * 40, min_events=10)
    assert r4["entropy"]["ent"] == 0.0
    assert abs(r4["score"] - 0.875 * 0.8 * want_comp) < 1e-12
    # entropy: insufficient samples (< 20) must earn ent 0, not 1 by default
    r5 = aggregate(events, fidelity, [0, 1, 2, 3, 4] * 3, min_events=10)
    assert r5["entropy"]["ent"] == 0.0 and "insufficient_imag_colors" in r5["flags"]
    assert abs(r5["score"] - 0.875 * 0.8 * want_comp) < 1e-12
    # entropy ramp midpoint: a distribution with 0.2 < KL < 0.6 must land
    # strictly between 0 and 1, at the documented linear position
    skew = [0] * 24 + [1, 2, 3, 4] * 4                # heavy-0 skew
    r6 = aggregate(events, fidelity, skew, min_events=10)
    kl6 = r6["entropy"]["kl_to_uniform"]
    assert 0.2 < kl6 < 0.6, kl6                       # mid-ramp by construction
    assert abs(r6["entropy"]["ent"] - (0.6 - kl6) / 0.4) < 1e-12
    assert 0.0 < r6["entropy"]["ent"] < 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("test_eval: ALL PASS")
