"""Deeper analysis of probe JSONs (written by run.py).

Per k:
  * acc | p_{k-1} != p_k  (the conditional off-by-one test: a one-behind belief collapses here)
  * exclusive error classification: among pred != p_k, attribute each record to the smallest
    |lag| j-match (pred == p_{k-lag}), else "off-traj" with its Chebyshev distance.
  * lag histogram: fraction of ALL records matching p_{k-lag} for each lag (non-exclusive).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def analyze(path: Path, max_lag: int = 8):
    d = json.loads(path.read_text())
    recs = d["records"]
    meta = d["meta"]
    max_k = meta["max_k"]
    print(f"\n==== {path.name}  arm={meta['arm']} w={meta['window']} n_ctx={meta['n_ctx']} "
          f"hide={not meta.get('no_hide', False)} mask={meta.get('mask')} n={len(recs)} ====")
    print("  k :   acc | acc(move) n_mv | err-> lag1  lag2  lag3+  ahead  p0/oth-traj  off(cheb)")
    for k in range(1, max_k + 1):
        n = ok = n_mv = ok_mv = 0
        cls = Counter()
        off_cheb = []
        for r in recs:
            pr = r["preds"].get(str(k)) or r["preds"].get(k)
            if pr is None:
                continue
            pc = tuple(pr[:2])
            tr = [tuple(x) for x in r["traj"]]
            n += 1
            hit = pc == tr[k]
            ok += hit
            moved = tr[k] != tr[k - 1]
            if moved:
                n_mv += 1
                ok_mv += hit
            if hit:
                continue
            # exclusive: smallest |lag| with a match; positive lag = behind, negative = ahead
            best = None
            for a in range(1, max(k, len(tr) - k) + 1):
                if k - a >= 0 and pc == tr[k - a]:
                    best = a
                    break
                if k + a < len(tr) and pc == tr[k + a]:
                    best = -a
                    break
            if best is None:
                cls["off"] += 1
                off_cheb.append(max(abs(pc[0] - tr[k][0]), abs(pc[1] - tr[k][1])))
            elif best < 0:
                cls["ahead"] += 1
            elif best == 1:
                cls["lag1"] += 1
            elif best == 2:
                cls["lag2"] += 1
            else:
                cls["lag3+"] += 1
        if not n:
            continue
        nerr = max(1, n - ok)
        oc = sum(off_cheb) / len(off_cheb) if off_cheb else 0.0
        print(f" {k:3d}: {ok / n:5.3f} |  {ok_mv / max(1, n_mv):5.3f}   {n_mv:4d} |"
              f"  {cls['lag1'] / nerr:5.2f} {cls['lag2'] / nerr:5.2f} {cls['lag3+'] / nerr:6.2f}"
              f" {cls['ahead'] / nerr:6.2f}   {cls['off'] / nerr:5.2f} ({oc:4.1f})"
              f"   [nerr={n - ok}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    for f in args.files:
        analyze(HERE / f)
