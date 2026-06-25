"""src/envs — steppable world-model / RL environments behind a shared interface.

Each environment subclasses `BaseEnv` (reset/step). Evals run against any env; data
generation (src/datagen/) drives them to write datasets. See `base.py` for the contract.
"""
from .base import BaseEnv  # noqa: F401
