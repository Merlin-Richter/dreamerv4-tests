"""FULL frozen-ish config: per-bin detail + OUT decomposition for the key
adapters, to see whether the high-age tail is actually populated/qualified and
whether bounded mid-range memory masquerades as long-range."""
import numpy as np
from harness import summarize
from autoresearch.frozen.eval_comeback import run_eval, BIN_EDGES
from autoresearch.frozen.env import OUT_IDX
from autoresearch.frozen.adapters import make_adapter
from autoresearch.frozen.eval_policies import (
    EvalBoxLoop, EvalIdiotWalk, EvalOutAndBack, EvalRetrace, EvalSweep,
    EvalDwellDart)
from exploits import GeoOutWorld, HonestShortMem, ShortMemWorld

FULL_SUITE = [
    ("oab_short", lambda: EvalOutAndBack(15, 30)),
    ("oab_mid", lambda: EvalOutAndBack(30, 55)),
    ("oab_long", lambda: EvalOutAndBack(55, 85)),
    ("box_small", lambda: EvalBoxLoop(12, 25, laps=10)),
    ("box_big", lambda: EvalBoxLoop(30, 55, laps=6)),
    ("sweep_narrow", lambda: EvalSweep(3, 6)),
    ("sweep_wide", lambda: EvalSweep(8, 14)),
    ("idiot_slow", lambda: EvalIdiotWalk(0.7)),
    ("idiot_fast", lambda: EvalIdiotWalk(0.92)),
    ("retrace_mid", lambda: EvalRetrace(30, 60)),
    ("retrace_long", lambda: EvalRetrace(60, 110)),
    ("dwell_dart", lambda: EvalDwellDart()),
]
FULL = dict(suite=FULL_SUITE, n_seeds=3, prefix_len=192, imag_len=768, min_events=30)

def decompose(result):
    edges = list(BIN_EDGES) + [np.inf]
    ev = [e for e in result["events"] if e["phase"] == "imag" and e["provenance"] == "real"]
    print("  real bin decomposition (OUT weight .1 vs in-map weight 1):", flush=True)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [e for e in ev if lo <= e["age"] < hi]
        if not sel:
            continue
        out = [e for e in sel if e["ref"] == OUT_IDX]
        inm = [e for e in sel if e["ref"] != OUT_IDX]
        wsum = sum(e["weight"] for e in sel)
        wacc = sum(e["weight"] * e["correct"] for e in sel) / wsum if wsum else float("nan")
        inm_acc = np.mean([e["correct"] for e in inm]) if inm else float("nan")
        q = "Q" if len(sel) >= 30 else " "
        hi_s = "inf" if hi == np.inf else int(hi)
        print(f"    [{q}] age[{lo},{hi_s}): n={len(sel):>5} OUT={len(out):>5} inmap={len(inm):>5}"
              f"(inmap_acc{inm_acc:.2f}) -> WEIGHTED={wacc:.3f}", flush=True)

def fac(cls, **kw):
    return lambda env: cls(env, **kw)

runs = [
    ("perfect_imaginary (zero retention)", make_adapter("perfect_imaginary")),
    ("HonestShortMem(W=16) [fair limited]", fac(HonestShortMem, mem_window=16)),
    ("HonestShortMem(W=64)", fac(HonestShortMem, mem_window=64)),
    ("GeoOutWorld(W=16)", fac(GeoOutWorld, mem_window=16)),
    ("GeoOutWorld(W=32)", fac(GeoOutWorld, mem_window=32)),
    ("GeoOutWorld(W=64)", fac(GeoOutWorld, mem_window=64)),
    ("GeoOutWorld(W=128)", fac(GeoOutWorld, mem_window=128)),
    ("GeoOutWorld(W=inf) [full prefix mem]", fac(GeoOutWorld, mem_window=10**9)),
]
for name, factory in runs:
    res = run_eval(factory, **FULL)
    summarize(res, name, show_bins=False)
    decompose(res)
    print(flush=True)
