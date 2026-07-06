"""Definitive confirmation at the TRUE frozen defaults (run_eval defaults:
suite=EVAL_SUITE, n_seeds=8, prefix_len=192, imag_len=768, min_events=30).
Headline adapters only."""
from harness import summarize
from autoresearch.frozen.eval_comeback import run_eval
from autoresearch.frozen.adapters import make_adapter
from exploits import GeoOutWorld, HonestShortMem

def fac(cls, **kw):
    return lambda env: cls(env, **kw)

runs = [
    ("oracle (ceiling)", make_adapter("oracle")),
    ("perfect_imaginary (ZERO retention)", make_adapter("perfect_imaginary")),
    ("HonestShortMem(W=16) [fair limited]", fac(HonestShortMem, mem_window=16)),
    ("GeoOutWorld(W=32) [bounded 32-frame]", fac(GeoOutWorld, mem_window=32)),
    ("GeoOutWorld(W=48) [bounded 48-frame]", fac(GeoOutWorld, mem_window=48)),
    ("GeoOutWorld(W=64) [bounded 64-frame]", fac(GeoOutWorld, mem_window=64)),
]
for name, factory in runs:
    res = run_eval(factory)   # TRUE frozen defaults
    summarize(res, name, show_bins=True)
    print(flush=True)
