"""Claim 6b (clean): the equal-weight-over-qualified-bins rule makes the component
score invariant to age-DISTRIBUTION shifts that keep the qualified SET fixed, and
DOES move when a shift changes which bins qualify. Prior run's 'invariant=False'
was a test artifact (count 33 cannot represent acc 0.40 exactly). Here bin counts
are chosen so every per-bin accuracy is represented EXACTLY.
"""
import numpy as np
from autoresearch.frozen.eval_comeback import BIN_EDGES, W_REAL, W_IMAG

def recompute(events, min_events=30):
    edges = list(BIN_EDGES) + [np.inf]
    imag = [e for e in events if e["phase"] == "imag"]
    def comp(prov):
        evs = [e for e in imag if e["provenance"] == prov]
        accs, perbin = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [e for e in evs if lo <= e["age"] < hi]
            wsum = sum(e["weight"] for e in sel)
            acc = (sum(e["weight"] * e["correct"] for e in sel) / wsum) if wsum > 0 else None
            q = len(sel) >= min_events
            perbin.append((len(sel), acc, q))
            if q:
                accs.append(acc)
        return (float(np.mean(accs)) if accs else None), perbin
    r, rb = comp("real"); i, ib = comp("imag")
    composite = W_REAL * r + W_IMAG * i if (r is not None and i is not None) else None
    return composite, rb, ib

def synth(bin_counts, bin_acc):
    ages = [8, 20, 40, 80, 160, 300]  # one per bin
    evs = []
    for age, n, acc in zip(ages, bin_counts, bin_acc):
        n_correct = round(acc * n)
        assert abs(n_correct - acc * n) < 1e-9, f"acc {acc} not exact for n={n}"
        for j in range(n):
            for prov in ("real", "imag"):
                evs.append(dict(phase="imag", provenance=prov, age=age,
                                weight=1.0, correct=(j < n_correct)))
    return evs

accs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
# A and B: identical per-bin acc (counts chosen so each acc is exact), all bins >=30
A_counts = [200, 200, 200, 200, 200, 200]
B_counts = [500, 40, 300, 40, 400, 30]   # 500*.9,40*.8,300*.7,40*.6,400*.5,30*.4 all integer
cA, rbA, ibA = recompute(synth(A_counts, accs))
cB, rbB, ibB = recompute(synth(B_counts, accs))
perbin_equal = all(abs(a[1] - b[1]) < 1e-12 for a, b in zip(rbA, rbB))
print(f"per-bin acc identical A vs B: {perbin_equal}")
print(f"A.composite={cA:.12f}  B.composite={cB:.12f}  invariant={abs(cA-cB)<1e-12}")

# C: shift drops bin-4 (age 80) below min_events=30 -> qualified set shrinks -> moves
C_counts = [500, 40, 300, 20, 400, 30]
cC, rbC, ibC = recompute(synth(C_counts, accs))
print(f"C drops a bin below min_events -> composite={cC:.12f}  moved={abs(cA-cC)>1e-9}")
print("VERDICT:", "CONFIRMED" if (perbin_equal and abs(cA-cB) < 1e-12 and abs(cA-cC) > 1e-9)
      else "SUSPECT")
