"""Instrument ShortMemWorld: for every real comeback event, was the painted color
from the W-window TRUE-memory branch or the persistent-invented branch? If a
16-step model is 'correct' at age 100, either (a) my gap!=eval age, or (b) inv
happens to match GT. This settles which."""
import harness  # noqa: F401  (sets sys.path)
import numpy as np
from autoresearch.frozen.eval_comeback import run_episode
from autoresearch.frozen.eval_policies import EvalBoxLoop, EvalRetrace
from exploits import ShortMemWorld

holder = {}
class Probe(ShortMemWorld):
    def __init__(self, env):
        super().__init__(env, seed=0, mem_window=16)
        self.log = {}   # (t, cell) -> ("true"/"inv", color)
        holder["a"] = self
    def color_of(self, cell):
        ls = self.last_seen.get(cell)
        if cell in self.true_color and ls is not None and (self.t - ls) <= self.W:
            c = self.true_color[cell]; branch = "true"
        else:
            c = self._inv(cell); branch = "inv"
        self.log[(self.t, cell)] = (branch, c)
        return c

for pol, mseed in [(EvalBoxLoop(12, 25, laps=10), 400), (EvalRetrace(30, 60), 500)]:
    events, fid, colors, berr, positions = run_episode(
        lambda env: Probe(env), pol, map_seed=mseed, ep_seed=mseed + 1,
        prefix_len=96, imag_len=256)
    a = holder["a"]
    real = [e for e in events if e["provenance"] == "real" and e["phase"] == "imag"]
    from collections import Counter
    by_branch_age = Counter()
    correct_by_branch = Counter(); total_by_branch = Counter()
    for e in real:
        key = (e["t"], tuple(e["cell"]))
        br = a.log.get(key, ("MISSING", None))[0]
        agebin = "<=16" if e["age"] <= 16 else ("17-32" if e["age"] <= 32 else ("33-64" if e["age"] <= 64 else ">64"))
        by_branch_age[(br, agebin)] += 1
        total_by_branch[br] += 1
        correct_by_branch[br] += int(e["correct"])
    print(f"--- policy={type(pol).__name__} map={mseed}  #real_imag_events={len(real)} ---")
    print("  branch x agebin counts:", dict(by_branch_age))
    for br in total_by_branch:
        print(f"  branch={br}: n={total_by_branch[br]} acc={correct_by_branch[br]/max(1,total_by_branch[br]):.3f}")
    # how many real events have age>16 yet were painted 'true'?
    anom = [(e["age"]) for e in real if e["age"] > 16 and a.log.get((e["t"], tuple(e["cell"])), ("?",))[0] == "true"]
    print(f"  real events age>16 painted TRUE (should be 0 for a real 16-window): {len(anom)}  sample ages {sorted(set(anom))[:10]}")
