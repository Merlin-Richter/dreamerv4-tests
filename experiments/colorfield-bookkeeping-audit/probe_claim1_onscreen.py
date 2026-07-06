"""Claim 1: on_screen (overlap>=6px in x AND y, as computed by readout.cells_in_view)
is EXACTLY equivalent to the cell CENTER being inside the view, on this geometry.

Independent method:
- For every lattice position p in [0,89] AND a band of OFF-lattice positions
  (imagination can leave the lattice), compute for every candidate cell index:
    * my_overlap(ci, tl)  -- independent closed-form 1D overlap of cell [ci*12, ci*12+12)
                             with view [tl, tl+64)
    * my_center_in(ci, tl) -- tl <= ci*12+6 < tl+64
  and compare (my_overlap>=6) to my_center_in.  Search for ANY disagreement and
  report the parity of the offset u=ci*12-tl there.
- Cross-check the readout: for a 2D sweep, run read_cells on the true rendered frame
  and assert r.on_screen == (center_in_view in BOTH axes) for every returned cell,
  AND that cells_in_view returns EXACTLY the set of cells with >=1px pixel overlap
  (brute-forced per pixel).  This catches enumeration off-by-ones too.
"""
import numpy as np
from autoresearch.frozen.env import (CELL_PX, VIEW_PX, PITCH_PX, TL_OFFSET, LATTICE,
                                      sample_map, build_world, render)
from autoresearch.frozen.readout import read_cells, cells_in_view, view_tl

def my_overlap_1d(ci, tl):
    a0, a1 = ci * CELL_PX, ci * CELL_PX + CELL_PX      # cell world interval
    b0, b1 = tl, tl + VIEW_PX                           # view world interval
    return max(0, min(a1, b1) - max(a0, b0))

def my_center_in_1d(ci, tl):
    c = ci * CELL_PX + CELL_PX // 2                     # cell center world coord = ci*12+6
    return tl <= c < tl + VIEW_PX

def scan_1d():
    # positions: all on-lattice, plus a generous off-lattice band on both sides
    positions = list(range(-40, LATTICE + 40))
    disagreements = []
    boundary_u_seen = set()      # values of u=ci*12-tl where overlap==6 exactly (the fragile edge)
    parity_of_u = set()
    n_checked = 0
    for p in positions:
        tl = PITCH_PX * p + TL_OFFSET
        # candidate cells: any that could touch the view, plus slack
        lo = (tl - CELL_PX) // CELL_PX - 2
        hi = (tl + VIEW_PX + CELL_PX) // CELL_PX + 2
        for ci in range(lo, hi + 1):
            u = ci * CELL_PX - tl
            parity_of_u.add(u % 2)
            ov = my_overlap_1d(ci, tl)
            onscr = ov >= 6
            cin = my_center_in_1d(ci, tl)
            if onscr != cin:
                disagreements.append((p, ci, tl, u, ov, onscr, cin))
            if ov == 6:
                boundary_u_seen.add(u)
            n_checked += 1
    return disagreements, parity_of_u, boundary_u_seen, n_checked

def crosscheck_readout_2d(n_pos=400, seed=0):
    """On true frames: readout.on_screen must equal center-in-view (both axes), and
    cells_in_view must equal the brute-force >=1px-overlap set."""
    rng = np.random.default_rng(seed)
    mismatches_onscreen = 0
    mismatches_enum = 0
    total_cells = 0
    for _ in range(n_pos):
        m = sample_map(rng)
        world = build_world(m)
        pr = int(rng.integers(0, LATTICE)); pc = int(rng.integers(0, LATTICE))
        frame = render(world, (pr, pc))
        tly, tlx = view_tl((pr, pc))
        reads = read_cells(frame, (pr, pc))
        # (a) on_screen == center-in-view
        for (ci, cj), r in reads.items():
            total_cells += 1
            cin = my_center_in_1d(ci, tly) and my_center_in_1d(cj, tlx)
            if r.on_screen != cin:
                mismatches_onscreen += 1
        # (b) enumeration exactness: brute-force which cells cover >=1 view pixel
        bf = set()
        for vy in range(VIEW_PX):
            wy = tly + vy
            ci = wy // CELL_PX if wy >= 0 else -((-wy - 1) // CELL_PX) - 1  # floor div for neg
            ci = wy // CELL_PX  # python // already floors toward -inf
            for vx in range(VIEW_PX):
                wx = tlx + vx
                cj = wx // CELL_PX
                bf.add((ci, cj))
        enum = set(reads.keys())
        if enum != bf:
            mismatches_enum += 1
    return mismatches_onscreen, mismatches_enum, total_cells

if __name__ == "__main__":
    dis, parity, bnd, n = scan_1d()
    print(f"[1D scan] checked {n} (position,cell) pairs over on- and off-lattice positions")
    print(f"[1D scan] parity set of u=ci*12-tl : {sorted(parity)}  (1 => u always ODD)")
    print(f"[1D scan] values of u where overlap==6 EXACTLY (the fragile edge): {sorted(bnd)}")
    print(f"[1D scan] disagreements (on_screen != center_in_view): {len(dis)}")
    for d in dis[:20]:
        print("   ", d)
    mo, me, tc = crosscheck_readout_2d()
    print(f"[2D readout] on_screen!=center mismatches: {mo} / {tc} cells")
    print(f"[2D readout] cells_in_view enumeration mismatches vs brute pixel set: {me} / 400 frames")
    ok = (len(dis) == 0 and parity == {1} and mo == 0 and me == 0)
    print("CLAIM 1 VERDICT:", "CONFIRMED" if ok else "REFUTED / SUSPECT")
