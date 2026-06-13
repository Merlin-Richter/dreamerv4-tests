"""EXP-013 follow-up: the COHERENCE evaluation proper (Merlin's point — the headline should be
self-consistency, not GT-distance). For each model, sweep the believed trajectory and compute the
best-fit constant-speed billiard residual over a GROWING window belief[1..k], for k=2..KMAX.

residual(k) low  => the belief through k blind steps traces a physical ball at env speed
                    (coherent memory), REGARDLESS of whether it matches the exact GT path.
coherence_horizon = largest k whose residual stays below THR (a coherent-memory cutoff).

Reference bands (from calibration / V-T011): GT floor ~0.77; forgetting surrogates 4.9-10.8.
"""
from __future__ import annotations
import sys, json, pathlib
import numpy as np
sys.path.insert(0, 'src'); sys.path.insert(0, 'src/probe')
import torch
from probe.position_consistency import model_belief_trajectory, best_fit_billiard_residual
from probe.revisit_probe import load_models

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
KMAX, NS, THR = 12, 20, 2.0      # THR ~2.5x the 0.77 GT floor = "still a coherent ball"
MODELS = [('vanilla_s0', 'experiments/EXP-012/vanilla_s0.pt', False),
          ('ff7_k1', 'experiments/EXP-010/k1/ff7_k1_s0.pt', True),
          ('ff7_k3', 'experiments/EXP-010/k3/ff7_k3_s0.pt', True)]

out = {}
for name, ck, um in MODELS:
    tok, dyn, dcfg, _ = load_models(pathlib.Path('trained_autoencoder.pt'), pathlib.Path(ck), 8, DEV)
    K = dcfg.inference_steps
    resid_by_k = np.full((NS, KMAX + 1), np.nan)   # resid_by_k[:,k] = residual over belief[1..k]
    for si in range(NS):
        b, g, S = model_belief_trajectory(tok, dyn, 4000 + si, KMAX, 3, K, DEV, use_memory=um)
        for k in range(2, KMAX + 1):
            bb = b[:k]
            ok = np.isfinite(bb).all(axis=1)
            if ok.sum() >= 2:
                r, _ = best_fit_billiard_residual(bb[ok], S)
                resid_by_k[si, k] = r
    mean_resid = np.nanmean(resid_by_k, axis=0)
    # coherence horizon: largest k (>=2) with mean residual still below THR
    coh = 0
    for k in range(2, KMAX + 1):
        if mean_resid[k] < THR:
            coh = k
        else:
            break
    out[name] = dict(resid_by_k=[round(float(mean_resid[k]), 2) for k in range(2, KMAX + 1)],
                     k_index=list(range(2, KMAX + 1)), coherence_horizon=coh)
    print(f"{name:10s} coh_resid(k=2..12):", ' '.join(f'{mean_resid[k]:4.1f}' for k in range(2, KMAX + 1)),
          f" | coherence_horizon(<{THR})={coh}")

out['_ref'] = dict(gt_floor=0.77, forgetting_band=[4.9, 10.8], THR=THR)
json.dump(out, open('experiments/EXP-013/coherence_by_k.json', 'w'), indent=2)
print("wrote experiments/EXP-013/coherence_by_k.json")
