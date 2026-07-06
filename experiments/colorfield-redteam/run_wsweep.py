"""Exchange-rate probe: composite vs genuine memory window W for GeoOutWorld
(persistent consistency + correct OUT + bounded W-frame retention).  Shows how
much score a BOUNDED (non-long-range) memory buys, and where the high-age bins
that should distinguish true long-range retention actually contribute."""
import sys
import numpy as np
from harness import SMALL, summarize
from autoresearch.frozen.eval_comeback import run_eval
from autoresearch.frozen.eval_policies import (
    EvalBoxLoop, EvalIdiotWalk, EvalOutAndBack, EvalRetrace, EvalSweep,
    EvalDwellDart)
from exploits import GeoOutWorld, HonestShortMem

# a fuller config closer to the frozen defaults (min_events=30)
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

cfg_name = sys.argv[1] if len(sys.argv) > 1 else "small"
CFG = SMALL if cfg_name == "small" else FULL
print(f"### config = {cfg_name}  {CFG if cfg_name=='small' else {k:(v if k!='suite' else len(v)) for k,v in CFG.items()}}")

def fac(cls, **kw):
    return lambda env: cls(env, **kw)

print("\n# GeoOutWorld: composite vs bounded memory window W")
print(f"{'W':>6} {'real':>7} {'consist':>8} {'composite':>10} {'gated':>7} {'#qbins(real)':>12}")
for W in [0, 8, 16, 32, 48, 64, 96, 128, 256, 100000]:
    res = run_eval(fac(GeoOutWorld, mem_window=W), **CFG)
    ra = res["real_anchored"]
    nq = sum(1 for b in ra["bins"] if b["qualified"])
    print(f"{W:>6} {str(round(ra['score'],3)) if ra['score'] is not None else 'None':>7} "
          f"{round(res['consistency']['score'],3):>8} {round(res['composite'],3):>10} "
          f"{round(res['composite_gated'],3):>7} {nq:>12}")

print("\n# HonestShortMem (fair limited baseline) vs W")
print(f"{'W':>6} {'real':>7} {'consist':>8} {'composite':>10} {'gated':>7}")
for W in [0, 8, 16, 32, 64, 128]:
    res = run_eval(fac(HonestShortMem, mem_window=W), **CFG)
    print(f"{W:>6} {round(res['real_anchored']['score'],3):>7} "
          f"{round(res['consistency']['score'],3):>8} {round(res['composite'],3):>10} "
          f"{round(res['composite_gated'],3):>7}")
