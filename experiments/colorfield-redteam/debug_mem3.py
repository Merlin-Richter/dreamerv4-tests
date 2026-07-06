import harness  # noqa: F401
import numpy as np
from collections import Counter
from autoresearch.frozen.eval_comeback import run_episode, gt_color
from autoresearch.frozen.eval_policies import EvalBoxLoop, EvalRetrace
from exploits import ShortMemWorld

holder = {}
class Probe(ShortMemWorld):
    def __init__(self, env):
        super().__init__(env, seed=0, mem_window=16)
        self.entry_branch = {}   # (t_enter, cell) -> "remember"/"forget"
        holder["a"] = self
    def _enter_visit(self, cell):
        lo = self.last_on.get(cell)
        rem = lo is not None and (self.t - lo) <= self.W and cell in self.belief
        self.entry_branch[(self.t, cell)] = "remember" if rem else "forget"
        return super()._enter_visit(cell)

for pol, mseed in [(EvalBoxLoop(12,25,laps=10),400),(EvalRetrace(30,60),500)]:
    events, *_ = run_episode(lambda env: Probe(env), pol, map_seed=mseed,
                             ep_seed=mseed+1, prefix_len=96, imag_len=256)
    a = holder["a"]
    real = [e for e in events if e["provenance"]=="real" and e["phase"]=="imag"]
    # correlate event correctness with age bucket
    buck = Counter(); corr = Counter()
    for e in real:
        b = "<=16" if e["age"]<=16 else ("17-32" if e["age"]<=32 else ("33-64" if e["age"]<=64 else ">64"))
        buck[b]+=1; corr[b]+=int(e["correct"])
    print(f"--- {type(pol).__name__} map={mseed} #real={len(real)} ---")
    for b in ["<=16","17-32","33-64",">64"]:
        if buck[b]: print(f"   age {b:>6}: n={buck[b]:>4} acc={corr[b]/buck[b]:.3f}")
