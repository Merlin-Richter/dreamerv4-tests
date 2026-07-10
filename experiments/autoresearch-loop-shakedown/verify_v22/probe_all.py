"""Independent verification of the v2.2-sym rewrite of
autoresearch/frozen_sym/eval_comeback.py. Does NOT import the author's tests.

Run: venv/Scripts/python.exe -u experiments/autoresearch-loop-shakedown/verify_v22/probe_all.py
All expected values are computed by hand in this file, never read from the
module under test. HEAD's aggregate (head_eval.py) is loaded side-by-side for
the bit-identical component comparison (claim 2).
"""
import os
import sys
import importlib.util

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from autoresearch.frozen_sym.eval_comeback import (  # noqa: E402
    aggregate, run_episode, run_eval, fidelity_ok)
from autoresearch.frozen_sym.env import (  # noqa: E402
    STAY, OUT_IDX, PHASE_PERIOD, VIEW_CELLS, DELTAS, N_ACTIONS)
from autoresearch.frozen_sym.adapters import make_adapter  # noqa: E402
from autoresearch.frozen_sym.eval_policies import (  # noqa: E402
    EvalOutAndBack, EvalBoxLoop, EvalIdiotWalk, EvalRetrace)

# --- load HEAD's module side-by-side ------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "head_eval", os.path.join(os.path.dirname(__file__), "head_eval.py"))
head = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(head)

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f"  --  {detail}" if detail and not cond else ""))


def ev(prov, age, correct, ref=0, weight=1.0):
    return {"provenance": prov, "age": age, "correct": bool(correct), "ref": ref,
            "weight": weight, "phase": "imag"}


# =============================================================================
# CLAIM 1 — aggregate() implements score = fid*(0.2*ent + 0.8*composite) with
#   fid = (fid_move+fid_hold)/2, ent = clip((0.6-KL)/0.4,0,1) (ent=0 if <20),
#   composite = real_cc*(0.7 + 0.3*consistency_cc), composite->0 in score if None
# =============================================================================
print("\n=== CLAIM 1: aggregate formula (hand-computed) ===")

# Build events. min_events=10.
#  real bin1 [1,16): 10 in-map, 7 correct -> acc .7 -> cc (.7-.2)/.8 = .625
#  real bin2 [17,32): 10 in-map, 5 correct -> acc .5 -> cc (.5-.2)/.8 = .375
#  -> real_score = (.625+.375)/2 = .5
#  imag bin1 [1,16): 10 in-map, 8 correct -> acc .8 -> cc .75  -> imag_score .75
events = []
events += [ev("real", 5, i < 7) for i in range(10)]
events += [ev("real", 20, i < 5) for i in range(10)]
events += [ev("imag", 5, i < 8) for i in range(10)]
# OUT/border events that must be EXCLUDED from scored accuracy (ref==OUT_IDX)
events += [ev("real", 5, True, ref=OUT_IDX, weight=0.1) for _ in range(20)]
# unqualified straggler bin (n < 10) must be ignored
events += [ev("real", 300, True) for _ in range(3)]

# fidelity: 40 holds all True (fid_hold=1.0), 10 moves 6 True (fid_move=0.6)
fidelity = [(False, True)] * 40 + [(True, True)] * 6 + [(True, False)] * 4
# colors: uniform-ish, >=20 samples -> KL ~ 0 -> ent = clip(0.6/0.4,0,1) = 1.0
colors = list(np.tile(np.arange(5), 6))  # 30 samples, exactly uniform

r = aggregate(events, fidelity, colors, min_events=10)

exp_real = 0.5
exp_imag = 0.75
exp_composite = exp_real * (0.7 + 0.3 * exp_imag)   # 0.5 * 0.925 = 0.4625
exp_fid_move = 0.6
exp_fid_hold = 1.0
exp_fid = 0.5 * (exp_fid_move + exp_fid_hold)        # 0.8
exp_ent = 1.0
exp_score = exp_fid * (0.2 * exp_ent + 0.8 * exp_composite)  # 0.8*(0.2+0.37)=0.456

check("C1 real_score = 0.5", abs(r["real_anchored"]["score"] - exp_real) < 1e-12,
      f"got {r['real_anchored']['score']}")
check("C1 imag/consistency_score = 0.75",
      abs(r["consistency"]["score"] - exp_imag) < 1e-12, f"got {r['consistency']['score']}")
check("C1 composite = real*(0.7+0.3*cons)",
      abs(r["composite"] - exp_composite) < 1e-12, f"got {r['composite']} want {exp_composite}")
check("C1 fid_move = 0.6", abs(r["fidelity"]["move"] - exp_fid_move) < 1e-12,
      f"got {r['fidelity']['move']}")
check("C1 fid_hold = 1.0", abs(r["fidelity"]["hold"] - exp_fid_hold) < 1e-12,
      f"got {r['fidelity']['hold']}")
check("C1 fid = (move+hold)/2 = 0.8", abs(r["fidelity"]["value"] - exp_fid) < 1e-12,
      f"got {r['fidelity']['value']}")
check("C1 ent = clip((0.6-KL)/0.4,0,1) = 1.0", r["entropy"]["ent"] == exp_ent,
      f"got {r['entropy']['ent']}, kl={r['entropy']['kl_to_uniform']}")
check("C1 score = fid*(0.2*ent+0.8*composite)",
      abs(r["score"] - exp_score) < 1e-12, f"got {r['score']} want {exp_score}")

# 1b: border/OUT events truly excluded — scored acc uses in-map only
b0 = r["real_anchored"]["bins"][0]  # bin [1,16)
check("C1b bin0 n counts in-map only (=10, not 30)", b0["n"] == 10, f"n={b0['n']}")
check("C1b bin0 n_border=20", b0["n_border"] == 20, f"n_border={b0['n_border']}")
check("C1b bin0 acc_cc uses in-map only (=.625)", abs(b0["acc_cc"] - 0.625) < 1e-12,
      f"acc_cc={b0['acc_cc']}")

# 1c: empty move pool -> fid_move=0
r_nomove = aggregate(events, [(False, True)] * 40, colors, min_events=10)
check("C1c empty move pool -> fid_move=0.0", r_nomove["fidelity"]["move"] == 0.0,
      f"got {r_nomove['fidelity']['move']}")
check("C1c empty move pool -> fid=(0+1)/2=0.5", abs(r_nomove["fidelity"]["value"] - 0.5) < 1e-12,
      f"got {r_nomove['fidelity']['value']}")
# empty hold pool -> fid_hold=0
r_nohold = aggregate(events, [(True, True)] * 10, colors, min_events=10)
check("C1c empty hold pool -> fid_hold=0.0", r_nohold["fidelity"]["hold"] == 0.0,
      f"got {r_nohold['fidelity']['hold']}")

# 1d: <20 colors -> ent = 0 (not default 1)
r_fewc = aggregate(events, fidelity, [0, 1, 2, 3, 4] * 3, min_events=10)  # 15 samples
check("C1d <20 colors -> ent=0.0", r_fewc["entropy"]["ent"] == 0.0,
      f"got {r_fewc['entropy']['ent']}")
check("C1d <20 colors -> score = fid*0.8*composite (no floor)",
      abs(r_fewc["score"] - exp_fid * 0.8 * exp_composite) < 1e-12, f"got {r_fewc['score']}")

# 1e: mid-ramp KL -> ent = (0.6-KL)/0.4 exactly (independent KL computation)
skew = [0] * 24 + [1, 2, 3, 4] * 4   # 40 samples, heavy-0
r_mid = aggregate(events, fidelity, skew, min_events=10)
cnt = np.bincount(skew, minlength=5)[:5].astype(float)
p = cnt / cnt.sum()
kl_indep = float(np.sum([pi * np.log(pi / 0.2) for pi in p if pi > 0]))
ent_indep = float(np.clip((0.60 - kl_indep) / (0.60 - 0.20), 0.0, 1.0))
check("C1e mid-ramp KL matches independent computation",
      abs(r_mid["entropy"]["kl_to_uniform"] - kl_indep) < 1e-12,
      f"got {r_mid['entropy']['kl_to_uniform']} want {kl_indep}")
check("C1e mid-ramp ent = clip((0.6-KL)/0.4,0,1)",
      abs(r_mid["entropy"]["ent"] - ent_indep) < 1e-12,
      f"got {r_mid['entropy']['ent']} want {ent_indep}")
check("C1e mid-ramp ent strictly in (0,1)", 0.0 < r_mid["entropy"]["ent"] < 1.0,
      f"got {r_mid['entropy']['ent']}")

# 1f: real_score None -> composite None -> treated as 0 in score (floor only)
events_nreal = [ev("real", 5, i < 2) for i in range(10)]  # acc .2 = chance -> cc 0
# ^ that's still qualified with real_score=0.0, not None. Force None: no in-map real bins.
events_none = [ev("real", 5, True, ref=OUT_IDX) for _ in range(30)]  # all border -> no in-map
events_none += [ev("imag", 5, True) for _ in range(10)]
r_none = aggregate(events_none, fidelity, colors, min_events=10)
check("C1f no in-map real bins -> real_score None", r_none["real_anchored"]["score"] is None,
      f"got {r_none['real_anchored']['score']}")
check("C1f real None -> composite None", r_none["composite"] is None,
      f"got {r_none['composite']}")
check("C1f composite None -> score = fid*0.2*ent (composite treated 0)",
      abs(r_none["score"] - exp_fid * 0.2 * 1.0) < 1e-12, f"got {r_none['score']}")
check("C1f real None -> flag no_qualified_real_bins",
      "no_qualified_real_bins" in r_none["flags"], f"flags={r_none['flags']}")

# 1g: imag_score None -> composite = real*0.7 (amp floor)
events_noimag = [ev("real", 5, i < 7) for i in range(10)]  # real_score cc .625
r_noimag = aggregate(events_noimag, fidelity, colors, min_events=10)
check("C1g imag None -> composite = real_cc*0.7",
      abs(r_noimag["composite"] - 0.625 * 0.7) < 1e-12, f"got {r_noimag['composite']}")


# =============================================================================
# CLAIM 2 — component (bin) math + composite bit-identical to HEAD aggregate
# =============================================================================
print("\n=== CLAIM 2: real/consistency/composite bit-identical to HEAD ===")


def deep_eq(a, b):
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            return False
        return all(deep_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(deep_eq(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        return a is b
    if isinstance(a, float) or isinstance(b, float):
        return a == b  # bit-identical demanded
    return a == b


# Randomized battery of synthetic event sets; feed HEAD a flat bool fidelity
# list (its old signature) and NEW the paired list — component math ignores
# fidelity entirely, so real/consistency/composite must be bit-identical.
rng = np.random.default_rng(20260710)
n_bitident = 0
for trial in range(200):
    evs = []
    for prov in ("real", "imag"):
        for age in (5, 20, 40, 100, 200, 400):
            k = int(rng.integers(0, 25))
            nc = int(rng.integers(0, k + 1))
            for i in range(k):
                r_out = bool(rng.integers(0, 4) == 0)  # ~25% border
                evs.append(ev(prov, age, i < nc, ref=(OUT_IDX if r_out else 0)))
    rng.shuffle(evs)
    colors_t = [int(x) for x in rng.integers(0, 5, size=int(rng.integers(0, 60)))]
    fid_flat = [bool(x) for x in rng.integers(0, 2, size=50)]
    fid_pairs = [(bool(rng.integers(0, 2)), ok) for ok in fid_flat]
    r_head = head.aggregate(evs, fid_flat, colors_t, min_events=10)
    r_new = aggregate(evs, fid_pairs, colors_t, min_events=10)
    ok_real = deep_eq(r_head["real_anchored"], r_new["real_anchored"])
    ok_cons = deep_eq(r_head["consistency"], r_new["consistency"])
    ok_comp = (r_head["composite"] is None and r_new["composite"] is None) or \
              (r_head["composite"] == r_new["composite"])
    if ok_real and ok_cons and ok_comp:
        n_bitident += 1
    else:
        check(f"C2 trial {trial} bit-identical", False,
              f"real={ok_real} cons={ok_cons} comp={ok_comp} "
              f"head_comp={r_head['composite']} new_comp={r_new['composite']}")
        break
check("C2 real_anchored/consistency/composite bit-identical over 200 random trials",
      n_bitident == 200, f"{n_bitident}/200")


# =============================================================================
# CLAIM 3 — run_episode emits (is_move, ok); is_move=(t%5==0 and a!=STAY);
#           fidelity_ok byte-identical in behavior to HEAD
# =============================================================================
print("\n=== CLAIM 3: run_episode is_move + fidelity_ok vs HEAD ===")

# 3a: differential test fidelity_ok vs HEAD.fidelity_ok on random inputs
mism = 0
for _ in range(5000):
    prev = rng.integers(0, 6, size=(VIEW_CELLS, VIEW_CELLS)).astype(np.uint8)
    # nxt sometimes a shifted copy of prev, sometimes random, sometimes equal
    mode = int(rng.integers(0, 3))
    action = int(rng.integers(0, N_ACTIONS))
    t = int(rng.integers(0, 50))
    if mode == 0:
        nxt = prev.copy()
    elif mode == 1:
        nxt = rng.integers(0, 6, size=(VIEW_CELLS, VIEW_CELLS)).astype(np.uint8)
    else:
        dr, dc = DELTAS[action]
        nxt = np.roll(np.roll(prev, -dr, axis=0), -dc, axis=1).astype(np.uint8)
    if fidelity_ok(prev, nxt, action, t) != head.fidelity_ok(prev, nxt, action, t):
        mism += 1
check("C3a fidelity_ok byte-identical to HEAD over 5000 random inputs", mism == 0,
      f"{mism} mismatches")

# 3b: run a real episode, verify the is_move flag structure.
#   is_move must be True only at imagination ticks where (prefix_len+i)%5==0
#   AND the emitted action was non-STAY. We assert the necessary condition
#   directly (is_move True => phase-0 tick) and that some moves fire.
PL, IL = 200, 600
events, fidelity_pairs, colors3, _berr, positions = run_episode(
    make_adapter("oracle"), EvalRetrace(5, 10), map_seed=1234, ep_seed=1235,
    prefix_len=PL, imag_len=IL, privileged=True)
check("C3b fidelity is a list of 2-tuples of length imag_len",
      len(fidelity_pairs) == IL and all(isinstance(x, tuple) and len(x) == 2 for x in fidelity_pairs),
      f"len={len(fidelity_pairs)}")
bad_phase = [i for i, (is_move, ok) in enumerate(fidelity_pairs)
             if is_move and (PL + i) % PHASE_PERIOD != 0]
check("C3b is_move True => (prefix_len+i)%5==0 (phase-0 tick)", len(bad_phase) == 0,
      f"violations={bad_phase[:5]}")
# every off-phase tick must be a HOLD (is_move False)
off_but_move = [i for i, (is_move, ok) in enumerate(fidelity_pairs)
                if (PL + i) % PHASE_PERIOD != 0 and is_move]
check("C3b off-phase ticks are never is_move", len(off_but_move) == 0,
      f"violations={off_but_move[:5]}")
n_moves = sum(1 for is_move, ok in fidelity_pairs if is_move)
check("C3b some move checks fired", n_moves > 0, f"n_moves={n_moves}")
# oracle => all checks pass
check("C3b oracle: all fidelity checks ok", all(ok for _, ok in fidelity_pairs),
      f"n_fail={sum(1 for _, ok in fidelity_pairs if not ok)}")


# =============================================================================
# CLAIM 4 — end-to-end real eval invariants
# =============================================================================
print("\n=== CLAIM 4: end-to-end real eval (privileged baselines) ===")

SMALL_SUITE = [
    ("oab_mid", lambda: EvalOutAndBack(5, 9)),
    ("box_small", lambda: EvalBoxLoop(2, 4, laps=10)),
    ("idiot_fast", lambda: EvalIdiotWalk(0.92)),
    ("retrace_mid", lambda: EvalRetrace(5, 10)),
]
SMALL = dict(suite=SMALL_SUITE, n_seeds=2, prefix_len=240, imag_len=1280,
             min_events=10, privileged=True)


def score_identity_ok(r):
    comp = r["composite"] if r["composite"] is not None else 0.0
    want = r["fidelity"]["value"] * (0.2 * r["entropy"]["ent"] + 0.8 * comp)
    return abs(r["score"] - want) < 1e-12


# 4a oracle
r_or = run_eval(make_adapter("oracle"), **SMALL)
check("C4a oracle score EXACTLY 1.0", r_or["score"] == 1.0, f"score={r_or['score']!r}")
check("C4a oracle composite==1.0, fid==1.0, ent==1.0",
      r_or["composite"] == 1.0 and r_or["fidelity"]["value"] == 1.0 and r_or["entropy"]["ent"] == 1.0,
      f"comp={r_or['composite']} fid={r_or['fidelity']['value']} ent={r_or['entropy']['ent']}")
check("C4a oracle score identity holds", score_identity_ok(r_or))

# 4b constant_color
r_cc = run_eval(make_adapter("constant_color"), **SMALL)
check("C4b constant_color score <= 0.1", r_cc["score"] <= 0.1, f"score={r_cc['score']}")
check("C4b constant_color score < 0.2 (below honest floor)", r_cc["score"] < 0.2,
      f"score={r_cc['score']}")
check("C4b constant_color mechanism: fid full (>=0.99) but ent==0 is the killer",
      r_cc["fidelity"]["value"] >= 0.99 and r_cc["entropy"]["ent"] == 0.0,
      f"fid={r_cc['fidelity']['value']} ent={r_cc['entropy']['ent']}")

# 4c copy_last
r_cl = run_eval(make_adapter("copy_last"), **SMALL)
check("C4c copy_last fid in [0.45,0.55]",
      0.45 <= r_cl["fidelity"]["value"] <= 0.55, f"fid={r_cl['fidelity']['value']}")
check("C4c copy_last score <= 0.15", r_cl["score"] <= 0.15, f"score={r_cl['score']}")

# 4d monotonicity — analytic + numeric grid.
# Vary ONE of (fid, ent, composite) at a time via synthetic inputs; score must
# be nondecreasing. score = fid*(0.2*ent + 0.8*composite), all coeffs >= 0.
def synth(fid_target_moves, ent_colors, real_correct_frac):
    # real bin: 20 in-map events at correctness fraction -> composite driver
    n = 20
    nc = int(round(real_correct_frac * n))
    evs = [ev("real", 5, i < nc) for i in range(n)]
    # holds all pass; moves pass a fraction to set fid
    m = 20
    mo = int(round(fid_target_moves * m))
    fid = [(False, True)] * 20 + [(True, True)] * mo + [(True, False)] * (m - mo)
    return aggregate(evs, fid, ent_colors, min_events=10)


uniform_colors = list(np.tile(np.arange(5), 8))         # ent = 1
collapse_colors = [0] * 40                               # ent = 0
mid_colors = [0] * 24 + [1, 2, 3, 4] * 4                 # ent in (0,1)

# vary fid, hold ent=1, real_frac=1.0
sc_fid = [synth(f, uniform_colors, 1.0)["score"] for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
check("C4d monotone nondecreasing in fid", all(a <= b + 1e-12 for a, b in zip(sc_fid, sc_fid[1:])),
      f"{sc_fid}")
# vary ent via color dists, hold fid=1 (all moves pass), real_frac=1.0
sc_ent = [synth(1.0, c, 1.0)["score"] for c in (collapse_colors, mid_colors, uniform_colors)]
check("C4d monotone nondecreasing in ent", all(a <= b + 1e-12 for a, b in zip(sc_ent, sc_ent[1:])),
      f"{sc_ent}")
# vary composite via real correctness, hold fid=1, ent=1
sc_comp = [synth(1.0, uniform_colors, fr)["score"] for fr in (0.2, 0.4, 0.6, 0.8, 1.0)]
check("C4d monotone nondecreasing in composite",
      all(a <= b + 1e-12 for a, b in zip(sc_comp, sc_comp[1:])), f"{sc_comp}")
# strict where inputs strictly increase (guard against a degenerate constant score)
check("C4d fid axis strictly increasing (0->1)", sc_fid[0] < sc_fid[-1], f"{sc_fid}")
check("C4d ent axis strictly increasing", sc_ent[0] < sc_ent[-1], f"{sc_ent}")
check("C4d composite axis strictly increasing", sc_comp[0] < sc_comp[-1], f"{sc_comp}")


# =============================================================================
print("\n=== SUMMARY ===")
n_pass = sum(1 for _, ok, _ in RESULTS if ok)
n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
for name, ok, detail in RESULTS:
    if not ok:
        print(f"  FAIL: {name}  [{detail}]")
print(f"\n{n_pass} passed, {n_fail} failed, of {len(RESULTS)} checks")
sys.exit(1 if n_fail else 0)
