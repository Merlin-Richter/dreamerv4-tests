"""src/evals — evaluation toolbox with a common Eval interface.

Layout (one folder per proper eval):
  base.py                  the Eval ABC + EvalConfig/EvalResult + REGISTRY + load()
  probe_env.py             shared FROZEN probe-episode builder
  revisit/                 FROZEN revisit-consistency spine (probe.py @ 5503e75) + RevisitEval
  position_consistency/    FROZEN position-consistency spine
  motion/                  working motion curves + MotionEval
  rollout_view/            working GT-vs-rollout strip / A-B headline renderer

Kept intentionally light at import time (no torch/model imports) to avoid import cycles — call
`discover()` to populate REGISTRY with the concrete evals.
"""
from .base import (  # noqa: F401
    Eval, EvalConfig, EvalResult, REGISTRY, MIDRUN, register, load,
)


def discover() -> dict:
    """Import the adapter subpackages so they register into REGISTRY. Call before reading REGISTRY.

    (position_consistency / rollout_view expose their primitives but no Eval adapter yet.)
    """
    from . import revisit, motion  # noqa: F401  — registers RevisitEval, MotionEval
    return REGISTRY
