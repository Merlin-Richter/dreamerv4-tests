"""Claims 2/3/4 boundary stressors that random walks under-cover:
  (A) minimum achievable comeback age & minimum OFF-frames-in-gap under 2px dynamics
      (is a 1-frame *total* gap / age<6 even reachable?), and that single-OFF-frame
      comebacks DO fire (the tightest real comeback),
  (B) the prefix/imagination boundary: an event with re-entry first_on == t must be
      INCLUDED (phase=imag) at prefix_len=t and EXCLUDED (phase=prefix) at
      prefix_len=t+1 -- i.e. the exclusion flips exactly at the boundary,
  (C) an explicit multi-leave single cell (leaves & returns >=3 times) agrees.
"""
import numpy as np
from collections import Counter
from autoresearch.frozen.env import (LATTICE, sample_map, build_world, render, apply_action)
from autoresearch.frozen.eval_comeback import CellTracker
import importlib.util, os
spec = importlib.util.spec_from_file_location(
    "p1", os.path.join(os.path.dirname(__file__), "probe_claim234_comeback.py"))
p1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p1)
ref_events, my_read, compare = p1.ref_events, p1.my_read, p1.ref_events  # noqa

def gap_stats(frames, positions, T):
    """Per comeback: (first_on, prev_last, n_off_in_gap, age)."""
    tl_all = {}
    for t in range(T):
        for cell, r in my_read(frames[t], positions[t]).items():
            tl_all.setdefault(cell, {})[t] = r
    out = []
    leaves = Counter()
    for cell, tl in tl_all.items():
        on_ts = [t for t in range(T) if (t in tl and tl[t]["on"])]
        if not on_ts:
            continue
        visits, cur, prev = [], [], None
        for t in on_ts:
            if prev is not None and t != prev + 1:
                visits.append(cur); cur = []
            cur.append(t); prev = t
        if cur:
            visits.append(cur)
        n_cb = 0
        for i in range(1, len(visits)):
            gap = list(range(visits[i - 1][-1] + 1, visits[i][0]))
            n_off = sum(1 for g in gap if g not in tl)
            if n_off >= 1:
                n_cb += 1
                out.append((cell, visits[i][0], visits[i - 1][-1], n_off,
                            visits[i][0] - visits[i - 1][-1]))
        if n_cb >= 3:
            leaves[cell] = n_cb
    return out, leaves

def run_tracker(frames, positions, prefix_len, m):
    tr = CellTracker(m, prefix_len)
    for t in range(len(frames)):
        tr.observe(t, frames[t], positions[t], is_real=(t < prefix_len))
    tr.finalize()
    return [dict(e, cell=tuple(e["cell"])) for e in tr.events]

if __name__ == "__main__":
    fails = 0
    # ---- gather comebacks from a big pool of random walks (true frames) --------
    all_gaps = []
    multi = {}
    for seed in range(200):
        rng = np.random.default_rng(seed)
        m = sample_map(rng); world = build_world(m)
        start = (int(rng.integers(0, LATTICE)), int(rng.integers(0, LATTICE)))
        T = 200
        pos = tuple(start); positions = [pos]
        for _ in range(T - 1):
            a = int(rng.integers(0, 5))
            pos = apply_action(pos, a, check=False)
            pos = (min(max(pos[0], -15), LATTICE + 14), min(max(pos[1], -15), LATTICE + 14))
            positions.append(pos)
        frames = [render(world, p) if (0 <= p[0] < LATTICE and 0 <= p[1] < LATTICE)
                  else p1._paint(p, lambda c: __import__("autoresearch.frozen.eval_comeback",
                                 fromlist=["gt_color"]).gt_color(m, c)) for p in positions]
        gs, lv = gap_stats(frames, positions, T)
        all_gaps += [(seed, m, world, positions, T) + g for g in gs]
        for c, n in lv.items():
            multi.setdefault((seed,) + c, (n, m, world, positions, T))

    ages = [g[8] for g in all_gaps]      # age is index 5 within g-tuple offset by 4 prefix fields
    # all_gaps entries: (seed,m,world,positions,T, cell, first_on, prev_last, n_off, age)
    ages = [g[9] for g in all_gaps]
    noffs = [g[8] for g in all_gaps]
    print(f"total comebacks pooled: {len(all_gaps)}")
    print(f"  min age            : {min(ages)}   (theory: 6 is the tightest under 2px steps)")
    print(f"  max age            : {max(ages)}")
    print(f"  min OFF frames/gap : {min(noffs)}")
    print(f"  #comebacks w/ EXACTLY 1 OFF frame: {sum(1 for n in noffs if n == 1)}")
    print(f"  #multi-leave cells (>=3 comebacks): {len(multi)}")

    # ---- (B) prefix-boundary flip test on a real comeback ---------------------
    # pick a comeback with first_on comfortably inside and prev_last>0
    cand = next(g for g in all_gaps if g[6] > 20 and g[6] < g[4] - 5)
    seed, m, world, positions, T, cell, first_on, prev_last, n_off, age = cand
    frames = [render(world, p) if (0 <= p[0] < LATTICE and 0 <= p[1] < LATTICE)
              else p1._paint(p, lambda c: __import__("autoresearch.frozen.eval_comeback",
                             fromlist=["gt_color"]).gt_color(m, c)) for p in positions]
    def event_for(cell, first_on, prefix_len):
        evs = run_tracker(frames, positions, prefix_len, m)
        return [e for e in evs if tuple(e["cell"]) == cell and e["t"] == first_on]
    # at prefix_len == first_on: re-entry is the FIRST imagination frame -> INCLUDED (phase imag)
    e_at = event_for(cell, first_on, first_on)
    # at prefix_len == first_on+1: re-entry is the last prefix frame -> EXCLUDED (phase prefix)
    e_after = event_for(cell, first_on, first_on + 1)
    inc_ok = len(e_at) == 1 and e_at[0]["phase"] == "imag"
    exc_ok = len(e_after) == 1 and e_after[0]["phase"] == "prefix"
    print(f"\n[boundary] comeback cell={cell} first_on={first_on} prev_last={prev_last} age={age}")
    print(f"  prefix_len={first_on}   -> phase={e_at[0]['phase'] if e_at else None} (want imag)  {'OK' if inc_ok else 'FAIL'}")
    print(f"  prefix_len={first_on+1} -> phase={e_after[0]['phase'] if e_after else None} (want prefix) {'OK' if exc_ok else 'FAIL'}")
    # cross-check reference agrees for a range of prefix_len around the boundary
    ref_flip_ok = True
    for pl in range(first_on - 2, first_on + 3):
        got = run_tracker(frames, positions, pl, m)
        ref, _ = ref_events(frames, positions, [t < pl for t in range(T)], m, pl)
        ok, msg = p1.compare(got, [dict(e, cell=tuple(e["cell"])) for e in ref])
        if not ok:
            ref_flip_ok = False
            print(f"  [FAIL] ref!=tracker at prefix_len={pl}: {msg}")
    if not (inc_ok and exc_ok and ref_flip_ok):
        fails += 1

    # ---- (C) an explicit multi-leave cell agrees full-trace -------------------
    if multi:
        k = next(iter(multi))
        n, m, world, positions, T = multi[k]
        frames = [render(world, p) if (0 <= p[0] < LATTICE and 0 <= p[1] < LATTICE)
                  else p1._paint(p, lambda c: __import__("autoresearch.frozen.eval_comeback",
                                 fromlist=["gt_color"]).gt_color(m, c)) for p in positions]
        got = run_tracker(frames, positions, 40, m)
        ref, _ = ref_events(frames, positions, [t < 40 for t in range(T)], m, 40)
        ok, msg = p1.compare(got, [dict(e, cell=tuple(e["cell"])) for e in ref])
        cell = k[1:]
        n_ev = sum(1 for e in got if tuple(e["cell"]) == cell)
        print(f"\n[multi-leave] cell={cell} produced {n_ev} comeback events; agree={ok} {msg if not ok else ''}")
        if not ok:
            fails += 1

    print("\nBOUNDARY/COVERAGE VERDICT:", "CONFIRMED" if fails == 0 else "REFUTED / SUSPECT")
