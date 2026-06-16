"""Eval interface — the common contract every eval implements.

Two capabilities on one interface (this is the "score vs visual" distinction Merlin asked for,
expressed as methods rather than a directory fork):

  * ``score(tok, dyn, cfg)`` -> dict[str, float]   REQUIRED, cheap. Scalar metrics suitable as a
    mid-run training signal (loggable to W&B) and as a worker verification signal.
  * ``report(tok, dyn, cfg, out_dir)`` -> EvalResult   OPTIONAL, rich. Renders the chart / rollout
    strip / HTML view for human review. Default wraps ``score`` so a score-only eval still conforms.

Evals are registered into ``REGISTRY`` (by importing their subpackage) and declare which environments
they can run on via ``compatible_envs`` — so a color/position eval is never silently pointed at the
curtain-less bouncing env. ``cfg`` carries the knobs every eval actually needs (K, window_N, tok_win,
episode grid, horizon) — NOT a magic "budget" string. ``load()`` constructs models identically for the
CLI and a training loop.

Frozen-spine note: ``frozen=True`` evals (revisit, position_consistency) wrap the FROZEN measurement
logic in src/evals/{revisit,position_consistency}/ — any change to that logic is a logged decision
(GOAL.md §8), because it silently redefines every prior result.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalConfig:
    """Knobs an eval needs. Defaults match the frozen-probe geometry (EXP-009/010/011)."""

    K: int = 4                      # diffusion / shortcut inference steps
    window_N: int = 8               # sliding context window at inference
    prefix_P: int = 3               # visible prefix frames
    tok_win: int | None = None      # tokenizer temporal window (<= model window); None = from load()
    episodes: int = 8               # episodes per condition (small = cheap mid-run; large = full report)
    horizon: int = 24               # rollout horizon for motion-style curves
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    scores: dict[str, float]                       # scalar metrics (mid-run loggable)
    artifacts: dict[str, Path] = field(default_factory=dict)   # png / html / json paths
    meta: dict[str, Any] = field(default_factory=dict)         # provenance: checkpoint, commit, env, n_eps


class Eval(ABC):
    name: str = "eval"
    frozen: bool = False
    compatible_envs: tuple[str, ...] = ()          # env names / capabilities this eval requires

    @abstractmethod
    def score(self, tok, dyn, cfg: EvalConfig, *, device) -> dict[str, float]:
        """Cheap scalar metrics. REQUIRED."""
        raise NotImplementedError

    def report(self, tok, dyn, cfg: EvalConfig, out_dir: Path, *, device) -> EvalResult:
        """Rich artifacts. OPTIONAL — defaults to wrapping ``score``."""
        return EvalResult(scores=self.score(tok, dyn, cfg, device=device))


# --- registry ---------------------------------------------------------------
REGISTRY: dict[str, Eval] = {}
MIDRUN: list[str] = []  # names of the cheap evals safe to run during training


def register(eval_obj: Eval, *, midrun: bool = False) -> Eval:
    REGISTRY[eval_obj.name] = eval_obj
    if midrun and eval_obj.name not in MIDRUN:
        MIDRUN.append(eval_obj.name)
    return eval_obj


def load(checkpoint, tokenizer, *, window_N: int = 8, device: str = "cuda"):
    """Construct (tok, dyn, dcfg, tok_win) the same way for the CLI and a training loop.

    Lazy import of the frozen loader to avoid a base<->revisit import cycle. Returns the tokenizer
    window too (``tok_win``), which ``_encode_window`` requires.
    """
    from evals.revisit.probe import load_models  # lazy
    return load_models(str(tokenizer), str(checkpoint), window_N, device)
