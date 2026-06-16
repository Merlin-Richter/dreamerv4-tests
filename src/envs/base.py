"""BaseEnv — the shared interface for this project's world-model / RL environments.

A new environment is a `BaseEnv` subclass: implement `reset(seed)` and `step(action)`.
Once it conforms, every eval in `src/evals/` can run against it and `src/datagen/` can drive
it to write a dataset. The project expects many memory environments (see IDEAS.md), which
is the whole reason this interface exists rather than bespoke per-env scripts.

CHANNEL-ORDER CONTRACT (measurement-validity invariant — do NOT break):
    Envs render with cv2, so frames are physically BGR. The dataset stores them in this
    same native order and the rest of the pipeline treats the channel axis opaquely
    (RGB<->BGR conversion happens ONLY for on-screen display). Keep one order end-to-end;
    evals compare decoded-frame colors to the env's hidden color WITHOUT any swap.

PRIVILEGED-STATE CONTRACT (IDEAS.md, non-negotiable):
    The model/training sees only the rendered frame (+ action). The env's hidden state
    (`hidden_state()`, `.color`) is exposed for *measurement* — evals may read it to SCORE
    recall — and is NEVER fed to the model as an input.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEnv(ABC):
    """Single-episode, seeded, deterministic environment.

    Subclasses set `n_actions` (0 = unconditioned) and `img_size`, and expose any hidden
    quantities evals need to score (e.g. `.color`). The per-step `state` vector's width and
    semantics are PER-ENV (they are not uniform across envs — e.g. the occluded env appends
    a curtain flag), so the interface types it as a plain `np.ndarray` and each subclass
    documents its layout.
    """

    n_actions: int = 0
    img_size: int = 64

    @abstractmethod
    def reset(self, seed: int) -> "BaseEnv":
        """Seed and initialize the episode. Deterministic given `seed`. Returns self."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Advance one frame.

        Returns ``(frame, state)`` where ``frame`` is ``(H, W, 3) uint8`` in the env's
        native (BGR) channel order and ``state`` is a per-env ``np.ndarray`` (see subclass).
        `action` is ignored by unconditioned envs (`n_actions == 0`).
        """
        raise NotImplementedError

    def hidden_state(self) -> np.ndarray:
        """MEASUREMENT-ONLY view of the sim's current hidden state, for evals to score
        recall. NEVER a model input. Per-env layout; subclasses override. Color, when the
        env has one, is exposed separately as ``.color``.
        """
        raise NotImplementedError
