import harness  # noqa: F401
import numpy as np
from autoresearch.frozen.eval_comeback import run_episode, gt_color
from autoresearch.frozen.eval_policies import EvalBoxLoop
from autoresearch.frozen.env import ColorFieldEnv, OUT_IDX
from exploits import ShortMemWorld

holder = {}
class Probe(ShortMemWorld):
    def __init__(self, env):
        super().__init__(env, seed=0, mem_window=16)
        holder["a"] = self; holder["map"] = env.map.copy()

pol = EvalBoxLoop(12, 25, laps=10)
events, *_ = run_episode(lambda env: Probe(env), pol, map_seed=400, ep_seed=401,
                         prefix_len=96, imag_len=256)
a = holder["a"]; mp = holder["map"]
real = [e for e in events if e["provenance"] == "real" and e["phase"] == "imag"]
print("map color histogram:", np.bincount(mp.ravel(), minlength=6))
print("#distinct inv colors assigned:", np.bincount([v for v in a.inv.values()], minlength=6))
print(f"{'cell':>12} {'age':>4} {'read':>4} {'ref':>4} {'inv[cell]':>9} {'gt':>3} {'corr':>4}")
for e in real[:20]:
    cell = tuple(e["cell"])
    inv = a.inv.get(cell, "-")
    gt = gt_color(mp, cell)
    print(f"{str(cell):>12} {e['age']:>4} {e['color']:>4} {e['ref']:>4} {str(inv):>9} {gt:>3} {str(e['correct']):>4}")
# summary: fraction of real events where inv[cell]==gt
match = sum(1 for e in real if a.inv.get(tuple(e['cell'])) == gt_color(mp, tuple(e['cell'])))
print(f"real events with inv[cell]==gt: {match}/{len(real)}")
