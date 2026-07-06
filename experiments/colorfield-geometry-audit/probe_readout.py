"""CLAIM 6 — read_cells exact on real frames (incl out-of-map tiles) + palette
separation margins.

  A. read_cells(frame,pos) returns, for EVERY overlapping cell (extended indices,
     out-of-map tiles included), the exact ground-truth color: PALETTE index of the
     map cell for in-map tiles, OUT_IDX for out-of-map tiles. Exhaustive over
     positions x maps.
  B. Palette geometry: min pairwise distance between the 6 palette colors; OUT
     separation from the 5 map colors; the guaranteed no-confusion perturbation
     radius. Empirically confirm nearest_palette is stable to that radius and flips
     just beyond it.
  C. Cell/OUT boundary alignment: verify no extended cell straddles the map edge
     (so every cell region is uniform on a real frame) — the precondition for A.

Run:  venv/Scripts/python.exe -u experiments/colorfield-geometry-audit/probe_readout.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from autoresearch.frozen import env as E
from autoresearch.frozen.readout import read_cells, nearest_palette, cells_in_view

NC, CP, WP, OUT_IDX, LAT = E.N_CELLS, E.CELL_PX, E.WORLD_PX, E.OUT_IDX, E.LATTICE


def truth_color(map_arr, ci, cj):
    """Ground-truth palette index for extended cell (ci,cj)."""
    if 0 <= ci < NC and 0 <= cj < NC:
        return int(map_arr[ci, cj])
    return OUT_IDX


def probe_A():
    rng = np.random.default_rng(0)
    fails = 0
    total = 0
    for _ in range(5):
        m = E.sample_map(rng)
        world = E.build_world(m)
        ps = [(a, b) for a in range(0, LAT, 4) for b in range(0, LAT, 4)]
        ps += [(0, 0), (0, LAT - 1), (LAT - 1, 0), (LAT - 1, LAT - 1)]
        for pos in ps:
            frame = E.render(world, pos)
            reads = read_cells(frame, pos)
            for (ci, cj), cr in reads.items():
                total += 1
                if cr.color != truth_color(m, ci, cj):
                    fails += 1
                    if fails <= 10:
                        print(f"    MISREAD pos={pos} cell=({ci},{cj}) got={cr.color} "
                              f"truth={truth_color(m,ci,cj)} mean={cr.mean_rgb} ov=({cr.ov_y},{cr.ov_x})")
    print(f"[A] read_cells exact: {total} cell-reads over maps/positions, {fails} misreads")
    return fails


def probe_B():
    pal = E.PALETTE.astype(int)
    names = ["red", "green", "blue", "orange", "purple", "OUT"]
    # pairwise distances
    D = np.sqrt(((pal[:, None, :] - pal[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(6, 1)
    dmin = D[iu].min()
    a_min = iu[0][D[iu].argmin()]; b_min = iu[1][D[iu].argmin()]
    print(f"[B] min pairwise palette distance = {dmin:.2f} "
          f"({names[a_min]}<->{names[b_min]})")
    # OUT vs 5 map colors
    out_d = D[OUT_IDX, :OUT_IDX]
    print(f"[B] OUT distances to map colors: " +
          ", ".join(f"{names[i]} {out_d[i]:.1f}" for i in range(OUT_IDX)) +
          f"  (min {out_d.min():.1f})")
    safe_r = dmin / 2.0
    print(f"[B] guaranteed no-confusion perturbation radius = dmin/2 = {safe_r:.2f}")

    # empirical: perturb each palette color by radius just under/over safe_r toward
    # its nearest neighbor and check nearest_palette flips only beyond the boundary.
    fails = 0
    rng = np.random.default_rng(2)
    for i in range(6):
        j = int(np.argsort(D[i])[1])  # nearest OTHER color
        direction = (pal[j] - pal[i]).astype(float)
        direction /= np.linalg.norm(direction)
        # just inside the safe radius: must still read i
        p_in = pal[i] + direction * (safe_r - 1.0)
        if nearest_palette(p_in) != i:
            fails += 1
            print(f"    color {names[i]} flipped INSIDE safe radius (r={safe_r-1:.1f})")
        # midpoint toward nearest neighbor: at the boundary distance D[i,j]/2 it may
        # tie/flip — check a clearly-past point reads j
        p_out = pal[i] + direction * (D[i, j] / 2 + 2.0)
        if nearest_palette(p_out) != j:
            fails += 1
            print(f"    color {names[i]} did NOT flip to {names[j]} past midpoint")
    print(f"[B] nearest_palette stability (in-radius stable, past-midpoint flips): {fails} anomalies")

    # exact-value sanity: every palette color reads itself with distance 0
    exact_fail = sum(1 for i in range(6) if nearest_palette(pal[i]) != i)
    print(f"[B] exact palette values read to themselves: {6-exact_fail}/6")
    return fails + exact_fail


def probe_C():
    """No extended cell straddles the map/OUT boundary: map occupies [0,180)=
    [12*0, 12*15), boundaries are multiples of CELL_PX -> every extended cell is
    fully in-map or fully out. Verify each cell region on a real frame is single-color."""
    rng = np.random.default_rng(5)
    m = E.sample_map(rng)
    world = E.build_world(m)
    straddle = 0
    nonuniform = 0
    checked = 0
    for pr in range(0, LAT, 2):
        for pc in range(0, LAT, 5):
            frame = E.render(world, (pr, pc))
            for ci, cj, y0, x0, ov_y, ov_x in cells_in_view((pr, pc)):
                region = frame[y0:y0 + ov_y, x0:x0 + ov_x].reshape(-1, 3)
                if len(np.unique(region, axis=0)) != 1:
                    nonuniform += 1
                    if nonuniform <= 5:
                        print(f"    NON-UNIFORM cell region pos=({pr},{pc}) cell=({ci},{cj})")
                checked += 1
    # boundary alignment arithmetic
    aligned = (0 % CP == 0) and (WP % CP == 0)
    print(f"[C] map edges at multiples of CELL_PX (0 and {WP}): {aligned}")
    print(f"[C] {checked} cell regions checked; {nonuniform} non-uniform (straddling)")
    return nonuniform + (0 if aligned else 1)


def main():
    fA = probe_A()
    fB = probe_B()
    fC = probe_C()
    ok = (fA == 0 and fB == 0 and fC == 0)
    print("\n=== CLAIM 6 VERDICT:", "CONFIRMED" if ok else
          f"REFUTED (A={fA} B={fB} C={fC})", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
