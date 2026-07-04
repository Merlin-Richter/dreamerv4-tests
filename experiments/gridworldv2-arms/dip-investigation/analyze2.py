"""Hypothesis-trajectory test: is the dip belief a trajectory with ONE move deleted?

Reconstructs each rollout's move stream deterministically (same seed -> same env.rng ->
same sample_moves) and simulates belief trajectories:
  * del_m: the true trajectory with occluded move o_m deleted (m=1: first occluded move;
    m=4: the move at the first write slot, abs pos 8 when n_ctx=4; m=k: pure one-behind).
  * Scored on the DISCRIMINATIVE subset where the hypothesis disagrees with the truth
    (q_k != p_k): P(pred==q_k | disagree) vs P(pred==p_k | disagree).
Only valid for hide-mode records (the standard occluded probe).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from envs.gridworldv2 import MOVES, GridWorldV2Env, sample_moves  # noqa: E402

GRID_N = 6


def clamp_step(cell, a):
    dc, dr = MOVES.get(a, (0, 0))
    return (min(max(cell[0] + dc, 0), GRID_N - 1), min(max(cell[1] + dr, 0), GRID_N - 1))


def sim_deleted(p0, occ_moves, m):
    """Trajectory q_0..q_K from p0 applying occ_moves (1-indexed via list) with move(s) m
    deleted. m: int, or a set/tuple of ints, or 0/empty for none."""
    ms = {m} if isinstance(m, int) else set(m)
    q = [p0]
    for j, a in enumerate(occ_moves, start=1):
        q.append(q[-1] if j in ms else clamp_step(q[-1], a))
    return q


def analyze(path: Path, del_ms=(1, 4)):
    d = json.loads(path.read_text())
    recs, meta = d["records"], d["meta"]
    n_ctx, max_k = meta["n_ctx"], meta["max_k"]
    assert not meta.get("no_hide", False), "hypothesis test assumes hide mode"
    print(f"\n==== {path.name} arm={meta['arm']} w={meta['window']} n_ctx={n_ctx} "
          f"mask={meta.get('mask')} n={len(recs)} ====")

    # reconstruct streams + verify the recorded traj matches the resim (driver self-check)
    per_rec = []
    for r in recs:
        env = GridWorldV2Env().reset(r["seed"])
        stream = sample_moves(env.rng, n_ctx + max_k)
        occ = stream[n_ctx:n_ctx + max_k]
        tr = [tuple(x) for x in r["traj"]]
        resim = sim_deleted(tr[0], occ, m=0)  # m=0 deletes nothing
        assert resim == tr, f"stream reconstruction mismatch seed={r['seed']}"
        per_rec.append((r, occ, tr))

    hdr = "  k :   acc |" + "".join(
        f"  del{m}: q==pred  p==pred  n_dis |" for m in del_ms) + "  delk: q==pred n_dis"
    print(hdr)
    for k in range(1, max_k + 1):
        n = ok = 0
        cols = []
        for m in del_ms + ("k",):
            mm = {k} if m == "k" else ({m} if isinstance(m, int) else set(m))
            hit_q = hit_p = n_dis = 0
            for r, occ, tr in per_rec:
                pr = r["preds"].get(str(k)) or r["preds"].get(k)
                if pr is None or max(mm) > k:
                    continue
                pc = tuple(pr[:2])
                q = sim_deleted(tr[0], occ[:k], mm)
                if q[k] == tr[k]:
                    continue  # non-discriminative
                n_dis += 1
                hit_q += pc == q[k]
                hit_p += pc == tr[k]
            cols.append((hit_q, hit_p, n_dis))
        for r, occ, tr in per_rec:
            pr = r["preds"].get(str(k)) or r["preds"].get(k)
            if pr is None:
                continue
            n += 1
            ok += tuple(pr[:2]) == tr[k]
        if not n:
            continue
        line = f" {k:3d}: {ok / n:5.3f} |"
        for (hq, hp, nd) in cols[:-1]:
            if nd:
                line += f"     {hq / nd:5.3f}   {hp / nd:5.3f}   {nd:4d} |"
            else:
                line += "         -       -      0 |"
        hq, hp, nd = cols[-1]
        line += f"    {hq / nd:5.3f} {nd:5d}" if nd else "        -     0"
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--del-ms", type=str, nargs="+", default=["1", "4"],
                    help="each item: an int or comma-list, e.g. 4 or 4,5 (delete both)")
    args = ap.parse_args()
    ms = tuple(int(x) if "," not in x else tuple(int(y) for y in x.split(","))
               for x in args.del_ms)
    for f in args.files:
        analyze(HERE / f, del_ms=ms)
