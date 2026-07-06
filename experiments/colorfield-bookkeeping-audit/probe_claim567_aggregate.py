"""Claims 5,6,7.
5  aggregate: bins (1,17,33,65,129,257,inf); per-bin WEIGHTED accuracy (0.1 iff ref
   is OUT else 1.0); component = EQUAL-weight mean over bins with n>=min_events;
   composite = 0.7*real+0.3*consistency; gated to 0 if any gate fails. Recompute
   INDEPENDENTLY from result['events'] and compare to the reported numbers.
6  oracle scores exactly 1.0 (run it); equal-weight rule really protects against
   age-distribution shift (synthetic construction) + its caveat.
7  determinism: two run_eval calls, identical args -> identical result.
"""
import json
import numpy as np
from autoresearch.frozen.eval_comeback import run_eval, BIN_EDGES, W_REAL, W_IMAG, OUT_IDX
from autoresearch.frozen.adapters import make_adapter
from autoresearch.frozen.eval_policies import EVAL_SUITE

# ---------- independent re-derivation of the scoring from raw events ----------
def recompute(events, gates, bin_edges=BIN_EDGES, min_events=30):
    edges = list(bin_edges) + [np.inf]
    imag = [e for e in events if e["phase"] == "imag"]
    def comp(prov):
        evs = [e for e in imag if e["provenance"] == prov]
        accs, bininfo = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [e for e in evs if lo <= e["age"] < hi]
            wsum = sum(e["weight"] for e in sel)
            acc = (sum(e["weight"] * (1 if e["correct"] else 0) for e in sel) / wsum) if wsum > 0 else None
            q = len(sel) >= min_events
            bininfo.append((lo, len(sel), acc, q))
            if q:
                accs.append(acc)
        return (float(np.mean(accs)) if accs else None), bininfo
    r, rb = comp("real")
    i, ib = comp("imag")
    composite = (W_REAL * r + W_IMAG * i) if (r is not None and i is not None) else None
    reasons = [k for k, g in gates.items() if not g["passed"]]
    if r is None:
        reasons.append("no_qualified_real_bins")
    if i is None:
        reasons.append("no_qualified_imag_bins")
    gated = composite if (not reasons and composite is not None) else 0.0
    return dict(real=r, imag=i, composite=composite, gated=gated, reasons=reasons,
                real_bins=rb, imag_bins=ib)

def approx(a, b, tol=1e-12):
    if a is None or b is None:
        return a == b
    return abs(a - b) <= tol

# =============================================================================
if __name__ == "__main__":
    fails = 0

    # ---- Claim 6: ORACLE exactness -------------------------------------------
    print("=== Claim 6: oracle ===")
    r = run_eval(make_adapter("oracle"), suite=EVAL_SUITE, n_seeds=4,
                 prefix_len=192, imag_len=512, min_events=10)
    # every POPULATED bin (n>0), both components, must have acc EXACTLY 1.0
    bad = []
    for comp in ("real_anchored", "consistency"):
        for b in r[comp]["bins"]:
            if b["n"] > 0 and b["acc"] != 1.0:
                bad.append((comp, b["lo"], b["n"], b["acc"]))
    print(f"  populated bins with acc!=1.0: {bad}")
    print(f"  real_score={r['real_anchored']['score']} imag_score={r['consistency']['score']}")
    print(f"  gates: fidelity={r['gates']['fidelity']} entropy_passed={r['gates']['entropy']['passed']}")
    print(f"  composite={r['composite']} composite_gated={r['composite_gated']} reasons={r['fail_reasons']}")
    oracle_ok = (not bad and r["composite_gated"] == 1.0 and r["composite"] == 1.0
                 and r["gates_passed"])
    print("  ORACLE == 1.0 EXACTLY:", "CONFIRMED" if oracle_ok else "REFUTED")
    fails += (0 if oracle_ok else 1)

    # ---- Claim 5: independent recompute from result['events'] ----------------
    print("\n=== Claim 5: independent aggregate recompute ===")
    for name in ("oracle", "perfect_imaginary", "noise_cells"):
        res = run_eval(make_adapter(name), suite=EVAL_SUITE, n_seeds=4,
                       prefix_len=192, imag_len=512, min_events=10)
        rec = recompute(res["events"], res["gates"], min_events=10)
        checks = {
            "real": approx(rec["real"], res["real_anchored"]["score"]),
            "imag": approx(rec["imag"], res["consistency"]["score"]),
            "composite": approx(rec["composite"], res["composite"]),
            "gated": approx(rec["gated"], res["composite_gated"]),
            "reasons": sorted(rec["reasons"]) == sorted(res["fail_reasons"]),
        }
        ok = all(checks.values())
        print(f"  [{name}] recompute vs reported: {checks}")
        print(f"     reported real={res['real_anchored']['score']} imag={res['consistency']['score']}"
              f" composite={res['composite']} gated={res['composite_gated']} reasons={res['fail_reasons']}")
        print(f"     mine     real={rec['real']} imag={rec['imag']}"
              f" composite={rec['composite']} gated={rec['gated']} reasons={sorted(rec['reasons'])}")
        fails += (0 if ok else 1)
        if not ok:
            print("     *** MISMATCH ***")

    # ---- Claim 7: determinism ------------------------------------------------
    print("\n=== Claim 7: determinism ===")
    a = run_eval(make_adapter("noise_cells"), suite=EVAL_SUITE[:4], n_seeds=2,
                 prefix_len=96, imag_len=256, min_events=5)
    b = run_eval(make_adapter("noise_cells"), suite=EVAL_SUITE[:4], n_seeds=2,
                 prefix_len=96, imag_len=256, min_events=5)
    ja, jb = json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True)
    det_ok = ja == jb
    print(f"  identical result JSON: {det_ok}  (len {len(ja)})")
    if not det_ok:
        # locate first difference
        for i, (ca, cb) in enumerate(zip(ja, jb)):
            if ca != cb:
                print("   first diff at", i, repr(ja[i-40:i+40]), "vs", repr(jb[i-40:i+40]))
                break
    fails += (0 if det_ok else 1)

    # ---- Claim 6b: equal-weight rule really is population-invariant -----------
    print("\n=== Claim 6b: equal-weight age-standardization ===")
    def synth(bin_counts, bin_acc, min_events=30):
        """events with given per-bin (count, target weighted-accuracy). All weight 1
        (in-map). correct set to hit the target acc for integer counts."""
        edges = list(BIN_EDGES)
        ages = [8, 20, 40, 80, 160, 300]   # one age landing in each of the 6 bins
        evs = []
        for age, n, acc in zip(ages, bin_counts, bin_acc):
            n_correct = round(acc * n)
            for j in range(n):
                evs.append(dict(phase="imag", provenance="real", age=age, weight=1.0,
                                correct=(j < n_correct)))
                evs.append(dict(phase="imag", provenance="imag", age=age, weight=1.0,
                                correct=(j < n_correct)))
        gates = {"fidelity": {"passed": True}, "entropy": {"passed": True}}
        return recompute(evs, gates, min_events=min_events)
    # Distribution A and B: SAME per-bin acc, WILDLY different populations (all >=30)
    accs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    A = synth([200, 200, 200, 200, 200, 200], accs)
    B = synth([500, 40, 300, 35, 400, 33], accs)   # shifted population, all still >=30
    same = approx(A["composite"], B["composite"])
    print(f"  same per-bin acc, different populations (all>=30): "
          f"A.composite={A['composite']:.6f} B.composite={B['composite']:.6f}  invariant={same}")
    # Caveat: if a shift drops a bin BELOW min_events, the qualified SET changes -> score moves
    C = synth([500, 40, 300, 20, 400, 33], accs)    # bin4 now n=20 (*2 provenance=... still per-prov 20<30)
    moved = not approx(A["composite"], C["composite"])
    print(f"  bin dropping below min_events changes qualified set: "
          f"A.composite={A['composite']:.6f} C.composite={C['composite']:.6f}  score_moved={moved}")
    print("  -> equal-weight protects WITHIN a fixed qualified set (as spec claims);")
    print("     qualification itself is still population-dependent (documented caveat).")
    eqw_ok = same and moved
    fails += (0 if eqw_ok else 1)

    print("\n==============================")
    print("CLAIMS 5/6/7 VERDICT:", "CONFIRMED" if fails == 0 else f"ISSUES ({fails})")
