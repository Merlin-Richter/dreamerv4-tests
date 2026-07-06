"""CLAIM 5 — Procedural dataset exactness + determinism.

  A. render_episode(map,start,actions) is bit-identical to stepping the env frame
     by frame (the procedural storage is lossless).
  B. Regenerating the dataset with the SAME seed reproduces identical sidecar arrays
     (maps/starts/actions/policy_ids/ep_seeds).
  C. A DIFFERENT seed produces different data (sanity: determinism isn't vacuous).

Run:  venv/Scripts/python.exe -u experiments/colorfield-geometry-audit/probe_dataset.py
"""
import sys, os, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from autoresearch.frozen import env as E
from autoresearch.frozen import datagen as DG


def load(td):
    return {k: np.load(os.path.join(td, k + ".npy")) for k in
            ("maps", "starts", "actions", "policy_ids", "ep_seeds")}


def main():
    fails = []

    # --- A. render_episode == env stepping --------------------------------
    with tempfile.TemporaryDirectory() as td:
        DG.generate(td, n_episodes=60, T=300, seed=11, verbose=False)
        d = load(td)
    mism = 0
    checked = 0
    for i in range(len(d["maps"])):
        m = d["maps"][i]; start = tuple(int(v) for v in d["starts"][i]); acts = d["actions"][i]
        proc = E.render_episode(m, start, acts)         # procedural renderer
        # independent env stepping:
        env = E.ColorFieldEnv()
        f0 = env.reset(seed=0, map_arr=m, start=start)
        env_frames = np.empty_like(proc)
        env_frames[0] = f0
        assert acts[0] == E.STAY
        for t in range(1, len(acts)):
            env_frames[t] = env.step(int(acts[t]))
        if not np.array_equal(proc, env_frames):
            mism += 1
            if mism <= 3:
                nbad = int((proc != env_frames).any(axis=(1, 2, 3)).sum())
                print(f"    ep{i}: {nbad} frames differ between render_episode and env.step")
        checked += 1
    print(f"[A] render_episode vs env.step: {checked} episodes, {mism} mismatched")
    if mism: fails.append(("render_vs_step", mism))

    # --- B. regeneration determinism (same seed) --------------------------
    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
        DG.generate(t1, n_episodes=120, T=256, seed=123, verbose=False)
        DG.generate(t2, n_episodes=120, T=256, seed=123, verbose=False)
        a, b = load(t1), load(t2)
    det_fail = 0
    for k in a:
        if not np.array_equal(a[k], b[k]):
            det_fail += 1
            print(f"    regeneration DIFFERS on {k}: {int((a[k]!=b[k]).sum())} elements")
    print(f"[B] same-seed regeneration identical: {'YES' if det_fail==0 else 'NO'} "
          f"({len(a)-det_fail}/{len(a)} arrays match)")
    if det_fail: fails.append(("determinism", det_fail))

    # --- C. different seed => different data (non-vacuous) -----------------
    with tempfile.TemporaryDirectory() as t3:
        DG.generate(t3, n_episodes=120, T=256, seed=999, verbose=False)
        c = load(t3)
    differs = not np.array_equal(a["maps"], c["maps"])
    print(f"[C] different-seed produces different maps: {differs}")
    if not differs: fails.append(("vacuous_determinism",))

    # --- D. actions[:,0]==STAY convention ---------------------------------
    conv_ok = bool((a["actions"][:, 0] == E.STAY).all())
    print(f"[D] actions[:,0]==STAY convention holds: {conv_ok}")
    if not conv_ok: fails.append(("stay_convention",))

    print("\n=== CLAIM 5 VERDICT:", "CONFIRMED" if not fails else f"REFUTED {fails}", "===")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
