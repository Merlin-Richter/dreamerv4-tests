"""CLAIM 2 — Invalid-action semantics + valid_actions exactness + no datagen policy
ever emits an invalid action (and the dataset therefore never contains one).

Independent checks:
  A. valid_actions() exactly equals a from-scratch bounds truth-table at EVERY
     lattice position (all 8100), esp. corners/edges/interior.
  B. env.step() raises on every outward-at-edge action, and mutates nothing on raise.
  C. Fuzz ALL 8 datagen policies over many seeds x long horizons; assert every
     proposed action is in valid_actions(pos) at the true position it is applied.
     (Our own checker, NOT relying on rollout_policy's assert.)
  D. Re-derive: generate a small dataset and scan every stored action against the
     path-integral position -> no outward-at-edge action present.

Run:  venv/Scripts/python.exe -u experiments/colorfield-geometry-audit/probe_actions.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from autoresearch.frozen import env as E
from autoresearch.frozen import policies as P

LAT = E.LATTICE
UP, DOWN, LEFT, RIGHT, STAY = E.UP, E.DOWN, E.LEFT, E.RIGHT, E.STAY


def truth_valid(pr, pc):
    """Independent truth table for the valid action set on [0,LAT)^2."""
    s = {STAY}
    if pr > 0: s.add(UP)
    if pr < LAT - 1: s.add(DOWN)
    if pc > 0: s.add(LEFT)
    if pc < LAT - 1: s.add(RIGHT)
    return s


def main():
    fails = []

    # --- A. valid_actions exhaustive --------------------------------------
    va_fail = 0
    for pr in range(LAT):
        for pc in range(LAT):
            got = set(E.valid_actions((pr, pc)))
            if got != truth_valid(pr, pc):
                va_fail += 1
                if va_fail <= 8:
                    print(f"    valid_actions mismatch ({pr},{pc}) got={got} exp={truth_valid(pr,pc)}")
    print(f"[A] valid_actions over all {LAT*LAT} positions: {va_fail} mismatches")
    if va_fail: fails.append(("valid_actions", va_fail))

    # --- B. step raises on outward-at-edge + no mutation on raise ----------
    raise_fail = 0
    mutate_fail = 0
    edge_cases = [((0, 5), UP), ((LAT - 1, 5), DOWN), ((5, 0), LEFT), ((5, LAT - 1), RIGHT),
                  ((0, 0), UP), ((0, 0), LEFT), ((LAT - 1, LAT - 1), DOWN), ((LAT - 1, LAT - 1), RIGHT)]
    env = E.ColorFieldEnv()
    for pos, a in edge_cases:
        env.reset(seed=1, map_arr=E.sample_map(np.random.default_rng(1)), start=pos)
        before = env.pos
        raised = False
        try:
            env.step(a)
        except ValueError:
            raised = True
        if not raised:
            raise_fail += 1
            print(f"    step did NOT raise for invalid {E.ACTION_NAMES[a]} at {pos}")
        if env.pos != before:
            mutate_fail += 1
            print(f"    step mutated pos on raise: {before}->{env.pos}")
    # positive control: a valid action does NOT raise
    valid_ok = True
    env.reset(seed=1, start=(5, 5))
    try:
        env.step(UP)
    except ValueError:
        valid_ok = False
    print(f"[B] outward-at-edge raises: {8-raise_fail}/8; no-mutation-on-raise: {8-mutate_fail}/8; valid-move-ok={valid_ok}")
    if raise_fail or mutate_fail or not valid_ok:
        fails.append(("step_raise", raise_fail, mutate_fail, valid_ok))

    # --- C. fuzz all 8 policies: every proposed action valid ---------------
    T = 4000
    n_seeds = 40
    total_actions = 0
    invalid_emitted = 0
    per_policy = {}
    for pid, (name, cls) in enumerate(P.POLICY_REGISTRY):
        cnt = 0
        for s in range(n_seeds):
            rng = np.random.default_rng(10_000 + pid * 131 + s)
            env = E.ColorFieldEnv()
            env.reset(seed=int(rng.integers(0, 2**62)))
            pol = cls()
            pol.reset(rng, env.pos)
            for t in range(1, T):
                a = int(pol.act(env.pos, rng))
                # OUR independent validity check at the CURRENT true position:
                if a not in truth_valid(*env.pos):
                    invalid_emitted += 1
                    if invalid_emitted <= 10:
                        print(f"    {name} emitted INVALID {E.ACTION_NAMES[a]} at {env.pos} (seed {s}, t {t})")
                    # still advance if we can, else stop this episode
                    if a in E.valid_actions(env.pos):
                        env.step(a)
                    else:
                        break
                else:
                    env.step(a)
                cnt += 1
                total_actions += 1
        per_policy[name] = cnt
    print(f"[C] fuzzed 8 policies x {n_seeds} seeds x ~{T} steps = {total_actions} actions; "
          f"invalid emitted = {invalid_emitted}")
    if invalid_emitted: fails.append(("policy_invalid", invalid_emitted))

    # --- D. generated dataset never contains an outward-at-edge action -----
    import tempfile
    from autoresearch.frozen import datagen as DG
    with tempfile.TemporaryDirectory() as td:
        DG.generate(td, n_episodes=200, T=512, seed=7, verbose=False)
        maps = np.load(os.path.join(td, "maps.npy"))
        starts = np.load(os.path.join(td, "starts.npy"))
        actions = np.load(os.path.join(td, "actions.npy"))
    ds_invalid = 0
    for i in range(len(maps)):
        pos = tuple(int(v) for v in starts[i])
        if actions[i, 0] != STAY:
            ds_invalid += 1  # convention violation
        for t in range(1, actions.shape[1]):
            a = int(actions[i, t])
            if a not in truth_valid(*pos):
                ds_invalid += 1
                if ds_invalid <= 5:
                    print(f"    dataset ep{i} t{t} invalid {E.ACTION_NAMES[a]} at {pos}")
                break
            pos = E.apply_action(pos, a, check=False)
    print(f"[D] scanned 200-ep dataset (T=512) via path-integral: {ds_invalid} invalid/convention violations")
    if ds_invalid: fails.append(("dataset_invalid", ds_invalid))

    print("\n=== CLAIM 2 VERDICT:", "CONFIRMED" if not fails else f"REFUTED {fails[:8]}", "===")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
