"""CLAIM 3 — THE SAFETY-CRITICAL CLAIM.

(3a) Closed-loop eval policies (all of EVAL_SUITE) NEVER return a move whose band
     >= BAND_BLOCK (30), for ARBITRARY / adversarial / incoherent band sequences:
       - bands flickering 0<->64 every step
       - all four sides blocked
       - bands appearing mid-pattern
       - random bands in [0,64]
       - negative/absurd band values
     Tens of thousands of steps per policy.
(3b) On REAL frames, band >= 30 on a side  <=>  the outward move on that axis is
     INVALID in the true lattice. -> driving any eval policy with real frames can
     NEVER produce an invalid env action (env.step never raises).
(3c) border_bands cannot mis-measure on real frames. Construct the worst-case map:
     edge cells set to the palette color NEAREST to OUT (green). Check the band is
     still exact everywhere.

Run:  venv/Scripts/python.exe -u experiments/colorfield-geometry-audit/probe_safety.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from autoresearch.frozen import env as E
from autoresearch.frozen import eval_policies as EP
from autoresearch.frozen.readout import border_bands, label_pixels

BB = EP.BAND_BLOCK
KEY = {E.UP: "up", E.DOWN: "down", E.LEFT: "left", E.RIGHT: "right"}


def band_generators():
    """Yield (name, fn(rng, step)->bands) adversarial band streams."""
    def flicker(rng, t):
        v = 64 if (t % 2 == 0) else 0
        return {"up": v, "down": 64 - v, "left": v, "right": 64 - v}
    def all_blocked(rng, t):
        return {"up": 64, "down": 64, "left": 64, "right": 64}
    def all_open(rng, t):
        return {"up": 0, "down": 0, "left": 0, "right": 0}
    def random_bands(rng, t):
        return {k: int(rng.integers(0, 65)) for k in ("up", "down", "left", "right")}
    def near_threshold(rng, t):
        # values clustered at 29/30/31 to probe the boundary
        return {k: int(rng.choice([28, 29, 30, 31, 32])) for k in ("up", "down", "left", "right")}
    def mid_pattern(rng, t):
        # open for a while, then a border SUDDENLY appears on a random side
        if t < 20:
            return {"up": 0, "down": 0, "left": 0, "right": 0}
        side = ("up", "down", "left", "right")[(t // 5) % 4]
        b = {"up": 0, "down": 0, "left": 0, "right": 0}
        b[side] = 64
        return b
    def negative_absurd(rng, t):
        # incoherent: negative and huge values a broken model could yield
        return {k: int(rng.choice([-5, -1, 0, 30, 31, 100, 1000])) for k in ("up", "down", "left", "right")}
    def three_blocked(rng, t):
        # only one escape direction, rotating
        sides = ["up", "down", "left", "right"]
        open_side = sides[t % 4]
        return {s: (0 if s == open_side else 64) for s in sides}
    return [("flicker", flicker), ("all_blocked", all_blocked), ("all_open", all_open),
            ("random", random_bands), ("near_threshold", near_threshold),
            ("mid_pattern", mid_pattern), ("negative_absurd", negative_absurd),
            ("three_blocked", three_blocked)]


def probe_3a():
    STEPS = 6000
    n_seeds = 6
    violations = 0
    total = 0
    for pol_name, factory in EP.EVAL_SUITE:
        for bg_name, bg in band_generators():
            for s in range(n_seeds):
                rng = np.random.default_rng(777 * s + hash((pol_name, bg_name)) % 100000)
                pol = factory()
                pol.reset(rng)
                for t in range(STEPS):
                    bands = bg(rng, t)
                    a = pol.act(bands, rng)
                    total += 1
                    # SAFETY PREDICATE: if a is a MOVE, its own band must be < 30.
                    if a != E.STAY:
                        b = bands[KEY[a]]
                        if b >= BB:
                            violations += 1
                            if violations <= 20:
                                print(f"    VIOLATION {pol_name}/{bg_name} seed{s} t{t}: "
                                      f"returned {E.ACTION_NAMES[a]} into band {b}")
    print(f"[3a] fuzz: {total} (policy,bandstream,step) actions; band>=30 moves returned = {violations}")
    return violations


def probe_3b():
    """On real frames: band>=30 side <=> outward invalid; driving eval policies with
    real frames never yields an invalid env action."""
    rng = np.random.default_rng(3)
    equiv_fail = 0
    LAT = E.LATTICE
    # (i) exhaustive equivalence over all positions on a few maps
    for _ in range(3):
        m = E.sample_map(rng); world = E.build_world(m)
        for pr in range(LAT):
            for pc in range(0, LAT, 2):
                bands = border_bands(E.render(world, (pr, pc)))
                valid = set(E.valid_actions((pr, pc)))
                for a in (E.UP, E.DOWN, E.LEFT, E.RIGHT):
                    blocked = bands[KEY[a]] >= BB
                    invalid = a not in valid
                    if blocked != invalid:
                        equiv_fail += 1
                        if equiv_fail <= 10:
                            print(f"    EQUIV FAIL p=({pr},{pc}) {E.ACTION_NAMES[a]} "
                                  f"band={bands[KEY[a]]} blocked={blocked} invalid={invalid}")
    print(f"[3b-i] band>=30 <=> outward-invalid over positions/maps: {equiv_fail} mismatches")

    # (ii) closed-loop drive with REAL frames through the env: never raises
    step_raises = 0
    invalid_returned = 0
    n_ep = 200
    T = 400
    for pol_name, factory in EP.EVAL_SUITE:
        for s in range(2):
            rng2 = np.random.default_rng(50 + s)
            env = E.ColorFieldEnv()
            frame = env.reset(seed=int(rng2.integers(0, 2**62)))
            pol = factory(); pol.reset(rng2)
            for t in range(T):
                bands = border_bands(frame)
                a = pol.act(bands, rng2)
                if a == E.STAY:
                    frame = E.render(env.world, env.pos)  # stay: same frame
                    continue
                if a not in env.valid_actions():
                    invalid_returned += 1
                try:
                    frame = env.step(a)
                except ValueError:
                    step_raises += 1
                    if step_raises <= 10:
                        print(f"    {pol_name} caused env.step RAISE at {env.pos} action {E.ACTION_NAMES[a]}")
                    break
    print(f"[3b-ii] real-frame closed-loop drive ({len(EP.EVAL_SUITE)}x2 eps x {T}): "
          f"env.step raises = {step_raises}, invalid actions returned = {invalid_returned}")
    return equiv_fail + step_raises + invalid_returned


def probe_3c():
    """Worst-case map for band mis-measurement: color every map cell the palette
    color NEAREST to OUT (green, idx1), so map pixels are as OUT-like as possible.
    On a REAL frame pixels are still EXACT palette values -> band must stay exact."""
    LAT = E.LATTICE
    fails = 0
    # nearest map color to OUT:
    d_out = ((E.PALETTE[:E.N_COLORS].astype(int) - E.PALETTE[E.OUT_IDX].astype(int)) ** 2).sum(1)
    worst_color = int(np.argmin(d_out))
    print(f"[3c] nearest map color to OUT = idx {worst_color} "
          f"(dist {np.sqrt(d_out[worst_color]):.1f}); building all-{worst_color} map")
    m = np.full((E.N_CELLS, E.N_CELLS), worst_color, dtype=np.uint8)
    world = E.build_world(m)
    # also a checkerboard of the two OUT-closest colors along edges
    m2 = E.sample_map(np.random.default_rng(9))
    m2[0, :] = worst_color; m2[-1, :] = worst_color; m2[:, 0] = worst_color; m2[:, -1] = worst_color
    world2 = E.build_world(m2)
    for label, w in [("all-nearest-OUT", world), ("edge-nearest-OUT", world2)]:
        for pr in range(LAT):
            for pc in range(0, LAT, 3):
                frame = E.render(w, (pr, pc))
                bb = border_bands(frame)
                claim = {"up": max(0, 31 - 2 * pr), "down": max(0, 2 * pr - 147),
                         "left": max(0, 31 - 2 * pc), "right": max(0, 2 * pc - 147)}
                if bb != claim:
                    fails += 1
                    if fails <= 8:
                        print(f"    [{label}] band MISMEASURE p=({pr},{pc}) bb={bb} claim={claim}")
                # sanity: no map pixel labeled OUT
        # verify no in-map pixel ever labels as OUT in this worst case
        frame = E.render(w, (40, 40))  # fully interior, all map
        lp = label_pixels(frame)
        if (lp == E.OUT_IDX).any():
            fails += 1
            print(f"    [{label}] interior frame has {int((lp==E.OUT_IDX).sum())} px labeled OUT!")
    print(f"[3c] worst-case-map band exactness: {fails} mismeasurements")
    return fails


def main():
    v3a = probe_3a()
    v3b = probe_3b()
    v3c = probe_3c()
    ok = (v3a == 0 and v3b == 0 and v3c == 0)
    print("\n=== CLAIM 3 VERDICT:", "CONFIRMED" if ok else
          f"REFUTED (3a={v3a} 3b={v3b} 3c={v3c})", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
