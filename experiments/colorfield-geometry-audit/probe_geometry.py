"""CLAIM 1 — Geometry. Independent from-scratch pixel renderer vs env.build_world/render.

We do NOT trust env.py's rendering. We re-derive the world image and view slices
pixel-by-pixel from the stated geometry (15x15 cells, 12px cell, view top-left =
2*p - 31 in map-world coords, OUT everywhere outside [0,180)^2) and compare
bit-for-bit at every lattice position, every corner, and every OUT-band boundary.

Run:  venv/Scripts/python.exe -u experiments/colorfield-geometry-audit/probe_geometry.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from autoresearch.frozen import env as E
from autoresearch.frozen.readout import view_tl, border_bands

PALETTE = E.PALETTE
OUT = PALETTE[E.OUT_IDX]
NC, CP, WP, VP, PITCH, LAT = E.N_CELLS, E.CELL_PX, E.WORLD_PX, E.VIEW_PX, E.PITCH_PX, E.LATTICE


def indep_view(map_arr, pr, pc):
    """From-scratch egocentric 64x64 view. Independent of env.py.
    map-world coord of view pixel (vy,vx) = (2*pr-31+vy, 2*pc-31+vx).
    Inside [0,180)^2 -> the cell color; else OUT."""
    v = np.empty((VP, VP, 3), dtype=np.uint8)
    tly = PITCH * pr + E.TL_OFFSET   # = 2*pr - 31  (independently recomputed below too)
    tlx = PITCH * pc + E.TL_OFFSET
    assert tly == 2 * pr - 31 and tlx == 2 * pc - 31, "TL formula mismatch vs 2p-31"
    for vy in range(VP):
        my = tly + vy
        for vx in range(VP):
            mx = tlx + vx
            if 0 <= my < WP and 0 <= mx < WP:
                v[vy, vx] = PALETTE[map_arr[my // CP, mx // CP]]
            else:
                v[vy, vx] = OUT
    return v


def indep_bands(map_arr, pr, pc):
    """True geometric OUT band per side = count of leading view rows/cols entirely
    outside the map on that side (pure geometry, no color thresholding)."""
    tly, tlx = 2 * pr - 31, 2 * pc - 31
    # a full row is OUT-above iff its map-world row < 0; OUT-below iff >= 180
    up = sum(1 for vy in range(VP) if all(not (0 <= tly + vy < WP) for _ in [0]))  # row fully above/below
    # simpler: leading rows with map-world row <0
    up = 0
    for vy in range(VP):
        if tly + vy < 0:
            up += 1
        else:
            break
    down = 0
    for vy in range(VP - 1, -1, -1):
        if tly + vy >= WP:
            down += 1
        else:
            break
    left = 0
    for vx in range(VP):
        if tlx + vx < 0:
            left += 1
        else:
            break
    right = 0
    for vx in range(VP - 1, -1, -1):
        if tlx + vx >= WP:
            right += 1
        else:
            break
    return {"up": up, "down": down, "left": left, "right": right}


def main():
    rng = np.random.default_rng(0)
    fails = []

    # --- A. full pixel-exact comparison across positions x maps -------------
    n_maps = 6
    maps = [E.sample_map(rng) for _ in range(n_maps)]
    # exhaustive positions on a stride grid + all four corners + every near-edge p
    corner_ps = [(0, 0), (0, LAT - 1), (LAT - 1, 0), (LAT - 1, LAT - 1)]
    edge_band = list(range(0, 17)) + list(range(LAT - 17, LAT))  # where OUT bands live
    stride_ps = [(a, b) for a in range(0, LAT, 7) for b in range(0, LAT, 7)]
    boundary_ps = [(a, b) for a in edge_band for b in edge_band] + \
                  [(a, b) for a in edge_band for b in range(0, LAT, 11)] + \
                  [(a, b) for a in range(0, LAT, 11) for b in edge_band]
    test_ps = corner_ps + stride_ps + boundary_ps
    npix = 0
    for m in maps:
        world = E.build_world(m)
        for (pr, pc) in test_ps:
            got = E.render(world, (pr, pc))
            exp = indep_view(m, pr, pc)
            if not np.array_equal(got, exp):
                fails.append(("pixel", pr, pc, int(np.abs(got.astype(int) - exp.astype(int)).max())))
            npix += 1
    print(f"[A] pixel-exact: {npix} (map,pos) renders compared, {len([f for f in fails if f[0]=='pixel'])} mismatches")

    # --- B. exhaustive ALL 8100 positions on ONE map (bit-exact) ------------
    m = maps[0]
    world = E.build_world(m)
    allfail = 0
    for pr in range(LAT):
        for pc in range(LAT):
            if not np.array_equal(E.render(world, (pr, pc)), indep_view(m, pr, pc)):
                allfail += 1
    print(f"[B] exhaustive {LAT*LAT} positions on map0: {allfail} mismatches")
    if allfail:
        fails.append(("exhaustive", allfail))

    # --- C. band-width formula: geometry, claimed formula, border_bands -----
    band_fail = 0
    formula_fail = 0
    for m in maps:
        world = E.build_world(m)
        for pr in range(LAT):
            for pc in range(0, LAT, 3):
                frame = E.render(world, (pr, pc))
                geo = indep_bands(m, pr, pc)
                bb = border_bands(frame)   # color-threshold measurement on the real frame
                # claimed per-side formula
                claim = {
                    "up": max(0, 31 - 2 * pr),
                    "down": max(0, 2 * pr - 147),
                    "left": max(0, 31 - 2 * pc),
                    "right": max(0, 2 * pc - 147),
                }
                if geo != claim:
                    formula_fail += 1
                    if formula_fail <= 5:
                        print(f"    FORMULA MISMATCH p=({pr},{pc}) geo={geo} claim={claim}")
                if bb != geo:
                    band_fail += 1
                    if band_fail <= 5:
                        print(f"    BORDER_BANDS != geometry p=({pr},{pc}) bb={bb} geo={geo}")
    print(f"[C] band geometry vs claimed formula: {formula_fail} mismatches")
    print(f"[C] border_bands(real frame) vs true geometry: {band_fail} mismatches")
    if formula_fail:
        fails.append(("formula", formula_fail))
    if band_fail:
        fails.append(("border_bands", band_fail))

    # --- D. odd-band + max-band assertions ----------------------------------
    m = maps[0]; world = E.build_world(m)
    seen = set()
    for pr in range(LAT):
        seen.add(border_bands(E.render(world, (pr, 45)))["up"])
    seen.discard(0)
    odd_ok = all(w % 2 == 1 for w in seen) and max(seen) == 31 and min(seen) == 1
    print(f"[D] nonzero up-band widths observed = {sorted(seen)}; all-odd&[1,31]={odd_ok}")
    if not odd_ok:
        fails.append(("odd", sorted(seen)))

    # --- E. world image size / no-truncation at p=89 ------------------------
    size_ok = world.shape == (WP + 2 * E.PAD_PX, WP + 2 * E.PAD_PX, 3)
    edge_view = E.render(world, (LAT - 1, LAT - 1))
    trunc_ok = edge_view.shape == (VP, VP, 3)
    print(f"[E] world shape {world.shape} ok={size_ok}; p=89 view shape {edge_view.shape} ok={trunc_ok}")
    if not (size_ok and trunc_ok):
        fails.append(("size", world.shape, edge_view.shape))

    print("\n=== CLAIM 1 VERDICT:", "CONFIRMED" if not fails else f"REFUTED {fails[:8]}", "===")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
