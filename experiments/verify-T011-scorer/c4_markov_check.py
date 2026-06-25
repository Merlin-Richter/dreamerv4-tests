"""C4: confirm the curtain action is ABSOLUTE/MARKOV in the env (a 1-frame reveal is
in-distribution and needs no history). We verify directly from env behaviour:

  - Rendering at frame t depends ONLY on action[t] and the current physics state,
    not on the action history. We test: run two episodes with IDENTICAL seeds (=>
    identical physics) but DIFFERENT curtain histories that arrive at the same frame;
    at any frame where both set action=0, the rendered frame must be byte-identical.
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from envs.occluded_bouncing import OccludedBouncingEnv

IMG, R = 64, 10

def rollout(seed, actions):
    env = OccludedBouncingEnv(IMG, R).reset(seed=seed)
    frames, states = [], []
    for a in actions:
        f, s = env.step(int(a))
        frames.append(f); states.append(s)
    return np.stack(frames), np.stack(states)

def main():
    n = 12
    # Two different curtain histories, same seed -> same physics trajectory.
    a1 = np.array([0,1,1,1,0,1,0,1,1,0,0,0])
    a2 = np.array([0,0,1,0,0,0,0,1,0,0,1,0])
    f1, s1 = rollout(123, a1)
    f2, s2 = rollout(123, a2)
    # physics identical regardless of action history?
    pos_id = np.allclose(s1[:, :4], s2[:, :4])
    print("physics state identical across action histories:", pos_id)
    # at every frame where BOTH chose up (0), rendered frame must match byte-for-byte
    both_up = (a1 == 0) & (a2 == 0)
    mism = 0
    for t in np.nonzero(both_up)[0]:
        if not np.array_equal(f1[t], f2[t]):
            mism += 1
    print(f"frames with both-up={both_up.sum()}, render mismatches={mism} "
          f"-> reveal render is Markov in action:", mism == 0)
    # A 1-frame reveal injected mid-occlusion equals the standalone reveal at that physics state
    # (curtain-up render = bg + ball at current pos, independent of prior curtain frames).
    assert pos_id and mism == 0, "curtain action is NOT absolute/Markov!"
    print("C4 env-mechanics: PASS (curtain action absolute/Markov; 1-frame reveal in-distribution)")

if __name__ == "__main__":
    main()
