"""OUT-dilution exploit: measure composite AND decompose real-anchored bins into
OUT (weight 0.1) vs in-map (weight 1.0) sub-populations to expose the dilution."""
import sys
import numpy as np
from harness import SMALL, summarize
from autoresearch.frozen.eval_comeback import run_eval, BIN_EDGES
from autoresearch.frozen.env import OUT_IDX
from exploits import GeoOutWorld, CornerHugOut, ShortMemWorld

def fac(cls, **kw):
    return lambda env: cls(env, **kw)

def decompose(result):
    edges = list(BIN_EDGES) + [np.inf]
    ev = [e for e in result["events"] if e["phase"] == "imag" and e["provenance"] == "real"]
    print("  real-anchored bin decomposition (OUT vs in-map):")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [e for e in ev if lo <= e["age"] < hi]
        if not sel:
            continue
        out = [e for e in sel if e["ref"] == OUT_IDX]
        inm = [e for e in sel if e["ref"] != OUT_IDX]
        wsum = sum(e["weight"] for e in sel)
        wacc = sum(e["weight"] * e["correct"] for e in sel) / wsum if wsum else float("nan")
        inm_acc = np.mean([e["correct"] for e in inm]) if inm else float("nan")
        out_acc = np.mean([e["correct"] for e in out]) if out else float("nan")
        hi_s = "inf" if hi == np.inf else int(hi)
        print(f"    age[{lo},{hi_s}): n={len(sel):>4} OUT={len(out):>4}(acc{out_acc:.2f}) "
              f"inmap={len(inm):>4}(acc{inm_acc:.2f})  -> WEIGHTED bin acc={wacc:.3f}")

runs = {
    "ShortMemWorld(W=0) [inv off-map, baseline]": fac(ShortMemWorld, mem_window=0),
    "GeoOutWorld(W=0) [correct OUT, no steer]":   fac(GeoOutWorld, mem_window=0),
    "CornerHugOut(W=0,R=9)":                       fac(CornerHugOut, mem_window=0, box_radius=9),
    "CornerHugOut(W=0,R=6)":                       fac(CornerHugOut, mem_window=0, box_radius=6),
    "CornerHugOut(W=0,R=4)":                       fac(CornerHugOut, mem_window=0, box_radius=4),
}
which = sys.argv[1] if len(sys.argv) > 1 else "all"
for name, factory in runs.items():
    if which != "all" and which not in name:
        continue
    res = run_eval(factory, **SMALL)
    summarize(res, name, show_bins=False)
    decompose(res)
    print()
