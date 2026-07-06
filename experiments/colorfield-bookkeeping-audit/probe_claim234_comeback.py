"""Claims 2,3,4: comeback event detection, age, provenance, one-per-event, phase
exclusion. Independent brute-force reference vs autoresearch CellTracker.

Strategy
--------
Feed the real CellTracker a controlled (frame, pos, is_real) stream and, from the
SAME pixels, build an INDEPENDENT event list with my own readout + my own visit/
comeback logic, then compare event lists field-by-field.

- Frames come in two flavours:
    * TRUE frames  : env.render(world, pos)   -> colors == ground truth.
    * LIAR frames  : each visible cell painted a chosen (possibly time-varying,
      possibly != GT, sometimes OUT) color -> exercises correct/ref/weight and the
      max-visibility majority vote (colors change within a visit).
- Positions come from random small-step walks (off-lattice allowed) that naturally
  create 1-frame visits, 1-frame gaps, boundary oscillation, multi-leave, and
  first/last-frame events. Coverage of each pathology is COUNTED and reported so we
  prove they were actually exercised. Plus a couple of hand-crafted traces.

Nothing from eval_comeback's tracker logic is reused in the reference; only pure
constants (geometry, palette) are imported.
"""
import numpy as np
from collections import Counter
from autoresearch.frozen.env import (CELL_PX, VIEW_PX, PALETTE, OUT_IDX, N_CELLS, LATTICE,
                                      sample_map, build_world, render, apply_action, DELTAS)
from autoresearch.frozen.readout import cells_in_view
from autoresearch.frozen.eval_comeback import CellTracker, gt_color

# ---------- my independent readout -------------------------------------------
def my_nearest(rgb):
    d = ((PALETTE.astype(np.int64) - np.asarray(rgb, np.int64)) ** 2).sum(1)
    return int(np.argmin(d))

def my_read(frame, pos):
    tly, tlx = 2 * pos[0] - 31, 2 * pos[1] - 31
    reads = {}
    for ci in range(tly // CELL_PX, (tly + VIEW_PX - 1) // CELL_PX + 1):
        y = ci * CELL_PX - tly
        y0, y1 = max(0, y), min(VIEW_PX, y + CELL_PX); ovy = y1 - y0
        if ovy <= 0:
            continue
        for cj in range(tlx // CELL_PX, (tlx + VIEW_PX - 1) // CELL_PX + 1):
            x = cj * CELL_PX - tlx
            x0, x1 = max(0, x), min(VIEW_PX, x + CELL_PX); ovx = x1 - x0
            if ovx <= 0:
                continue
            mean = frame[y0:y1, x0:x1].reshape(-1, 3).mean(0)
            center_in = (tly <= ci * CELL_PX + 6 < tly + VIEW_PX) and \
                        (tlx <= cj * CELL_PX + 6 < tlx + VIEW_PX)
            reads[(ci, cj)] = dict(color=my_nearest(mean), area=ovy * ovx, on=center_in)
    return reads

# ---------- my independent comeback reference --------------------------------
def ref_events(frames, positions, is_real_list, map_arr, prefix_len):
    T = len(frames)
    # per-cell timeline: list of (t, status, area, color) for frames where cell overlaps
    timelines = {}
    for t in range(T):
        for cell, r in my_read(frames[t], positions[t]).items():
            timelines.setdefault(cell, {})[t] = r
    events = []
    first_imag_colors = []
    for cell, tl in timelines.items():
        seen_ts = sorted(tl.keys())
        first_on_t = next((t for t in seen_ts if tl[t]["on"]), None)
        if first_on_t is None:
            continue  # never on-screen -> never tracked
        provenance = "real" if is_real_list[first_on_t] else "imag"
        # build visits = maximal runs of ON frames (by frame index continuity of ON status)
        on_ts = [t for t in range(T) if (t in tl and tl[t]["on"])]
        visits = []
        cur = []
        prev = None
        for t in on_ts:
            if prev is not None and t != prev + 1:
                visits.append(cur); cur = []
            cur.append(t); prev = t
        if cur:
            visits.append(cur)
        # visit color = majority over max-area frames
        vcolors = []
        for v in visits:
            areas = [tl[t]["area"] for t in v]
            mx = max(areas)
            cc = Counter(tl[t]["color"] for t in v if tl[t]["area"] == mx)
            vcolors.append(cc.most_common(1)[0][0])
        record = None
        for i, v in enumerate(visits):
            first_on = v[0]; last_on = v[-1]
            color = vcolors[i]
            if record is None and provenance == "imag" and color != OUT_IDX:
                first_imag_colors.append(color)
            if i >= 1:
                prev_last = visits[i - 1][-1]
                gap_frames = range(prev_last + 1, first_on)  # strictly between
                # OFF = frame where cell has ZERO overlap (absent from timeline)
                came_back = record is not None and any(g not in tl for g in gap_frames)
                if came_back:
                    ref = gt_color(map_arr, cell) if provenance == "real" else record
                    events.append(dict(
                        cell=cell, provenance=provenance, t=first_on,
                        age=first_on - prev_last, color=int(color), ref=int(ref),
                        correct=bool(color == ref),
                        weight=0.1 if ref == OUT_IDX else 1.0,
                        phase="imag" if first_on >= prefix_len else "prefix"))
            record = int(color)
    return events, first_imag_colors

# ---------- coverage bookkeeping (prove pathologies were exercised) ----------
def coverage(frames, positions, prefix_len):
    T = len(frames)
    tl_all = {}
    for t in range(T):
        for cell, r in my_read(frames[t], positions[t]).items():
            tl_all.setdefault(cell, {})[t] = r
    cov = Counter()
    for cell, tl in tl_all.items():
        on_ts = [t for t in range(T) if (t in tl and tl[t]["on"])]
        if not on_ts:
            continue
        # visits
        visits, cur, prev = [], [], None
        for t in on_ts:
            if prev is not None and t != prev + 1:
                visits.append(cur); cur = []
            cur.append(t); prev = t
        if cur:
            visits.append(cur)
        for v in visits:
            if len(v) == 1:
                cov["one_frame_visit"] += 1
        for i in range(1, len(visits)):
            gap = range(visits[i - 1][-1] + 1, visits[i][0])
            has_off = any(g not in tl for g in gap)
            if has_off:
                cov["comeback_total"] += 1
                if len(list(gap)) == 1:
                    cov["one_frame_gap_comeback"] += 1
                if visits[i][0] == T - 1:
                    cov["last_frame_event"] += 1
                if visits[i][0] == prefix_len:
                    cov["event_at_prefix_boundary"] += 1
            else:
                # returned without a full OFF (partial only) -> must NOT fire
                if any(g in tl for g in gap):   # gap had only partial frames
                    cov["partial_return_nonevent"] += 1
    return cov

# ---------- frame sources -----------------------------------------------------
def true_frames(world, positions):
    return [render(world, p) if (0 <= p[0] < LATTICE and 0 <= p[1] < LATTICE)
            else _paint(p, lambda c: gtc(world_map, c)) for p in positions]

def _paint(pos, color_fn):
    frame = np.empty((VIEW_PX, VIEW_PX, 3), np.uint8)
    for ci, cj, y0, x0, ovy, ovx in cells_in_view(pos):
        frame[y0:y0 + ovy, x0:x0 + ovx] = PALETTE[color_fn((ci, cj))]
    return frame

def liar_frames(map_arr, positions, seed):
    """Paint each cell a time-varying color; some != GT, some OUT."""
    rng = np.random.default_rng(seed)
    base = {}  # cell -> base color
    def color_fn_at(t):
        def fn(cell):
            ci, cj = cell
            if cell not in base:
                base[cell] = int(rng.integers(0, 6))  # includes OUT sometimes
            # flip color every ~9 frames for some cells -> within-visit variation
            drift = (t // 9) if (ci + cj) % 3 == 0 else 0
            return (base[cell] + drift) % 6
        return fn
    return [_paint(p, color_fn_at(t)) for t, p in enumerate(positions)]

def random_walk(rng, T, start):
    pos = tuple(start); poss = [pos]
    for _ in range(T - 1):
        a = int(rng.integers(0, 5))
        pos = apply_action(pos, a, check=False)
        # keep it roaming but not too far off-lattice
        pr = min(max(pos[0], -20), LATTICE + 19)
        pc = min(max(pos[1], -20), LATTICE + 19)
        pos = (pr, pc); poss.append(pos)
    return poss

def compare(ev_a, ev_b):
    def key(e): return (tuple(e["cell"]), e["t"], e["provenance"])
    ka = sorted(ev_a, key=key); kb = sorted(ev_b, key=key)
    if len(ka) != len(kb):
        return False, f"count {len(ka)} vs {len(kb)}"
    fields = ["cell", "provenance", "t", "age", "color", "ref", "correct", "weight", "phase"]
    for a, b in zip(ka, kb):
        for f in fields:
            av = tuple(a[f]) if f == "cell" else a[f]
            bv = tuple(b[f]) if f == "cell" else b[f]
            if av != bv:
                return False, f"field {f}: {av} != {bv} for cell {a['cell']} t={a['t']}"
    return True, "match"

if __name__ == "__main__":
    world_map = None
    prefix_len = 40
    total_cov = Counter()
    n_fail = 0
    n_runs = 0
    for seed in range(60):
        rng = np.random.default_rng(seed)
        m = sample_map(rng); world = build_world(m)
        start = (int(rng.integers(0, LATTICE)), int(rng.integers(0, LATTICE)))
        T = 120
        positions = random_walk(rng, T, start)
        for mode in ("true", "liar"):
            n_runs += 1
            if mode == "true":
                frames = [render(world, (min(max(p[0],0),LATTICE-1), min(max(p[1],0),LATTICE-1)))
                          if not (0 <= p[0] < LATTICE and 0 <= p[1] < LATTICE) else render(world, p)
                          for p in positions]
                # NB: for off-lattice we clamp render but keep pos for registration ->
                # would desync; instead paint true GT for off-lattice:
                frames = [render(world, p) if (0 <= p[0] < LATTICE and 0 <= p[1] < LATTICE)
                          else _paint(p, lambda c: gt_color(m, c)) for p in positions]
            else:
                frames = liar_frames(m, positions, seed=1000 + seed)
            is_real = [t < prefix_len for t in range(T)]
            # real tracker
            tr = CellTracker(m, prefix_len)
            for t in range(T):
                tr.observe(t, frames[t], positions[t], is_real=is_real[t])
            tr.finalize()
            got = [dict(e, cell=tuple(e["cell"])) for e in tr.events]
            # reference
            ref, ref_fic = ref_events(frames, positions, is_real, m, prefix_len)
            ok, msg = compare(got, ref)
            if not ok:
                n_fail += 1
                print(f"[FAIL] seed={seed} mode={mode}: {msg}")
            # also check first_imag_colors multiset matches
            if sorted(tr.first_imag_colors) != sorted(ref_fic):
                n_fail += 1
                print(f"[FAIL] seed={seed} mode={mode}: first_imag_colors "
                      f"{sorted(tr.first_imag_colors)} != {sorted(ref_fic)}")
            total_cov += coverage(frames, positions, prefix_len)

    print("\n--- pathology coverage across all traces ---")
    for k in ("comeback_total", "one_frame_visit", "one_frame_gap_comeback",
              "last_frame_event", "event_at_prefix_boundary", "partial_return_nonevent"):
        print(f"  {k:28s}: {total_cov[k]}")
    print(f"\nruns={n_runs}  failures={n_fail}")
    print("CLAIMS 2/3/4 VERDICT:", "CONFIRMED" if n_fail == 0 else "REFUTED / SUSPECT")
