"""CLAIM 4 — estimate_shift sign convention: for a view moving with action a, the
estimated (dy,dx) == 2*DELTAS[a] on real frames.

  A. GENERIC textured maps: verify (dy,dx)==2*DELTAS[a] for all 5 actions at a broad
     sweep of positions INCLUDING near/at borders (low-texture OUT regions).
  B. ADVERSARIAL worst cases the task asks for: views dominated by OUT + a single-
     color map region. Construct all-one-color maps and straight-edge geometry that
     make the frame translation-ambiguous -> does estimate_shift misidentify?

Run:  venv/Scripts/python.exe -u experiments/colorfield-geometry-audit/probe_shift.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from autoresearch.frozen import env as E
from autoresearch.frozen.readout import estimate_shift

LAT = E.LATTICE


def expected(a):
    dr, dc = E.DELTAS[a]
    return (2 * dr, 2 * dc)


def apply_valid(pos, a):
    """Apply action if valid, else None (skip: can't move there)."""
    try:
        return E.apply_action(pos, a, check=True)
    except ValueError:
        return None


def probe_A_generic():
    rng = np.random.default_rng(0)
    fails = 0
    total = 0
    detail = []
    for _ in range(4):
        m = E.sample_map(rng)
        world = E.build_world(m)
        # positions: interior stride + a ring near every border (where OUT dominates)
        ps = [(a, b) for a in range(2, LAT - 2, 9) for b in range(2, LAT - 2, 9)]
        ps += [(a, b) for a in (1, 2, 3, 5, 8) for b in range(3, LAT - 3, 13)]
        ps += [(a, b) for a in range(3, LAT - 3, 13) for b in (1, 2, 3, 5, 8)]
        ps += [(a, b) for a in (LAT - 2, LAT - 3, LAT - 6) for b in range(3, LAT - 3, 13)]
        for pos in ps:
            prev = E.render(world, pos)
            for a in (E.UP, E.DOWN, E.LEFT, E.RIGHT, E.STAY):
                npos = apply_valid(pos, a)
                if npos is None:
                    continue
                cur = E.render(world, npos)
                dy, dx, mse = estimate_shift(prev, cur)
                total += 1
                if (dy, dx) != expected(a):
                    fails += 1
                    if len(detail) < 12:
                        detail.append((pos, E.ACTION_NAMES[a], (dy, dx), expected(a), round(mse, 2)))
    print(f"[A] GENERIC maps: {total} (pos,action) shift estimates; {fails} wrong")
    for d in detail:
        print(f"      pos={d[0]} {d[1]} est={d[2]} exp={d[3]} mse={d[4]}")
    return fails, total


def probe_B_adversarial():
    """Construct real frames where estimate_shift provably fails."""
    print("\n[B] ADVERSARIAL worst cases (task-requested):")
    results = []

    # B1: fully-uniform interior view (all-one-color map, deep interior -> no OUT).
    #     Every shift gives MSE 0 -> scan-order tie -> returns (-3,-3) regardless of a.
    m = np.full((E.N_CELLS, E.N_CELLS), 2, dtype=np.uint8)  # all blue
    world = E.build_world(m)
    pos = (45, 45)  # interior: view is 64x64 of a single color, zero OUT
    prev = E.render(world, pos)
    b1_bad = []
    for a in (E.UP, E.DOWN, E.LEFT, E.RIGHT, E.STAY):
        npos = E.apply_action(pos, a, check=True)
        cur = E.render(world, npos)
        est = estimate_shift(prev, cur)[:2]
        ok = est == expected(a)
        if not ok:
            b1_bad.append((E.ACTION_NAMES[a], est, expected(a)))
    uniform = np.unique(prev.reshape(-1, 3), axis=0)
    print(f"  B1 uniform interior (all-blue map, pos={pos}): view is {len(uniform)} distinct color(s). "
          f"wrong estimates = {len(b1_bad)}/5: {b1_bad}")
    results.append(("uniform_interior", len(b1_bad)))

    # B2: straight top band + horizontally-uniform map region -> dx unidentifiable.
    #     pos row small (top band present), col mid (no side band); map all-one-color
    #     so each row is constant across columns -> horizontal shift is a free tie.
    m = np.full((E.N_CELLS, E.N_CELLS), 3, dtype=np.uint8)  # all orange
    world = E.build_world(m)
    pos = (0, 40)  # top band = 31px, no side band, map region single color
    prev = E.render(world, pos)
    b2_bad = []
    for a in (E.LEFT, E.RIGHT, E.STAY, E.DOWN):
        npos = E.apply_action(pos, a, check=True)
        cur = E.render(world, npos)
        est = estimate_shift(prev, cur)[:2]
        ok = est == expected(a)
        if not ok:
            b2_bad.append((E.ACTION_NAMES[a], est, expected(a)))
    print(f"  B2 straight top-band + uniform map (pos={pos}): wrong estimates on "
          f"{{L,R,STAY,DOWN}} = {len(b2_bad)}/4: {b2_bad}")
    results.append(("straight_edge_uniform", len(b2_bad)))

    # B3: DOES a distinct OUT boundary rescue single-color maps at a CORNER?
    #     (the L-shaped OUT edge pins both dy,dx even when the map block is 1 color.)
    m = np.full((E.N_CELLS, E.N_CELLS), 4, dtype=np.uint8)  # all purple
    world = E.build_world(m)
    pos = (2, 2)  # both bands present -> L-shaped boundary
    prev = E.render(world, pos)
    b3_bad = []
    for a in (E.UP, E.DOWN, E.LEFT, E.RIGHT, E.STAY):
        npos = E.apply_action(pos, a, check=True)
        cur = E.render(world, npos)
        est = estimate_shift(prev, cur)[:2]
        if est != expected(a):
            b3_bad.append((E.ACTION_NAMES[a], est, expected(a)))
    print(f"  B3 CORNER single-color (L-shaped OUT edge, pos={pos}): wrong = {len(b3_bad)}/5: {b3_bad}"
          f"   (expected: boundary RESCUES -> 0 wrong)")
    results.append(("corner_singlecolor_rescued", len(b3_bad)))

    # B4: probability a RANDOM real map yields a uniform/ambiguous interior view
    #     (contextualize how 'real' the worst case is).
    rng = np.random.default_rng(1)
    N = 3000
    ambiguous = 0
    for _ in range(N):
        m = E.sample_map(rng)
        world = E.build_world(m)
        pr, pc = int(rng.integers(20, 70)), int(rng.integers(20, 70))
        prev = E.render(world, (pr, pc))
        npos = E.apply_action((pr, pc), E.RIGHT, check=True)
        cur = E.render(world, npos)
        if estimate_shift(prev, cur)[:2] != expected(E.RIGHT):
            ambiguous += 1
    print(f"  B4 random real maps: {ambiguous}/{N} interior RIGHT-moves misidentified "
          f"(rate {ambiguous/N:.4f})")
    results.append(("random_ambiguous", ambiguous))

    return results


def main():
    fA, tA = probe_A_generic()
    resB = probe_B_adversarial()
    b1 = dict(resB)["uniform_interior"]
    b2 = dict(resB)["straight_edge_uniform"]
    b3 = dict(resB)["corner_singlecolor_rescued"]
    print("\n--- interpretation ---")
    print(f"  Generic textured frames: estimate_shift correct ({tA-fA}/{tA}).")
    print(f"  Literal claim 'equals 2*DELTAS[a] on ALL real frames' is "
          f"{'REFUTED' if (b1 or b2) else 'not refuted'} by constructed uniform real maps "
          f"(B1 wrong={b1}, B2 wrong={b2}).")
    print(f"  Corner single-color is RESCUED by the OUT boundary (B3 wrong={b3}).")
    # Verdict: claim CONFIRMED for generic frames, REFUTED as a universal statement.
    verdict = "CONFIRMED-generic / REFUTED-universal" if (fA == 0 and (b1 or b2)) else \
              ("CONFIRMED" if fA == 0 and b1 == 0 and b2 == 0 else "UNEXPECTED")
    print(f"\n=== CLAIM 4 VERDICT: {verdict} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
