"""Shared harness for the ColorField comeback-eval red-team.

Adds the repo root to sys.path, imports the FROZEN eval, and provides a
pretty-printer that surfaces every component / gate / per-bin number that the
composite depends on. NOTHING under autoresearch/ is modified.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from autoresearch.frozen.eval_comeback import run_eval  # noqa: E402
from autoresearch.frozen.eval_policies import (  # noqa: E402
    EvalBoxLoop, EvalIdiotWalk, EvalOutAndBack, EvalRetrace)

# The reduced config the task suggests (also used by test_eval SMALL).
SMALL_SUITE = [
    ("oab_mid", lambda: EvalOutAndBack(30, 55)),
    ("box_small", lambda: EvalBoxLoop(12, 25, laps=10)),
    ("idiot_fast", lambda: EvalIdiotWalk(0.92)),
    ("retrace_mid", lambda: EvalRetrace(30, 60)),
]
SMALL = dict(suite=SMALL_SUITE, n_seeds=2, prefix_len=96, imag_len=256,
             min_events=10)


def summarize(result, name="", show_bins=True):
    lines = []
    cg = result["composite_gated"]
    comp = result["composite"]
    lines.append(f"=== {name} ===")
    lines.append(f"  composite_gated = {cg}   (raw composite = {comp})")
    lines.append(f"  gates_passed = {result['gates_passed']}   "
                 f"fail_reasons = {result['fail_reasons']}")
    ra = result["real_anchored"]
    co = result["consistency"]
    lines.append(f"  real_anchored.score  = {ra['score']}  (n_events={ra['n_events']})")
    lines.append(f"  consistency.score    = {co['score']}  (n_events={co['n_events']})")
    g = result["gates"]
    fid = g["fidelity"]
    lines.append(f"  gate.fidelity = {fid['value']:.4f}  passed={fid['passed']}")
    ent = g["entropy"]
    lines.append(f"  gate.entropy  = kl={ent.get('kl_to_uniform')}  n={ent.get('n')}  "
                 f"passed={ent['passed']}")
    lines.append(f"  border_drift_px = {result['border_drift_px']}")
    lines.append(f"  age_stats = {result['age_stats']}")
    if show_bins:
        for comp_name in ("real_anchored", "consistency"):
            lines.append(f"  -- {comp_name} bins --")
            for b in result[comp_name]["bins"]:
                mark = "Q" if b["qualified"] else " "
                acc = "None" if b["acc"] is None else f"{b['acc']:.3f}"
                lines.append(f"     [{mark}] age[{b['lo']},{b['hi']}) n={b['n']:>4}  acc={acc}")
    lines.append("  -- per_policy weighted_acc --")
    for k, v in result["per_policy"].items():
        lines.append(f"     {k:16s} n={v['n_events']:>4}  wacc={v['weighted_acc']}")
    txt = "\n".join(lines)
    print(txt, flush=True)
    return txt
