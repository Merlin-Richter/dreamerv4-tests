"""Closed-loop eval policies for the ColorField comeback eval. FROZEN LAYER.

Why closed-loop (design round 2, 2026-07-06): outward-at-border is an INVALID
action that never occurs in training data. During the imagination phase there is
no true lattice — only the model's imagined world. A fixed action script that hits
an imagined-early border would feed the model an out-of-distribution input
(pushing into a border), i.e. undefined behavior. So eval policies act from the
BORDER BANDS of the last (real or imagined) frame plus their own action history,
and NEVER push in a direction whose band >= BAND_BLOCK.

On real frames the band at the true lattice edge is exactly 31px, and 29px one
step in — BAND_BLOCK = 30 therefore blocks outward pushes exactly at the edge and
nowhere else. On imagined frames it blocks wherever the model painted a border.

Each policy: reset(rng) then act(bands, rng) -> action with bands from
readout.border_bands(). All are structured to create comeback events (leave cells
fully off-screen, then return) at a spread of ages.
"""

import numpy as np

from .env import DOWN, LEFT, OPPOSITE, RIGHT, STAY, UP
from .readout import VIEW_PX  # noqa: F401  (documentation import)

BAND_BLOCK = 30  # px; band >= BAND_BLOCK on a side => that direction is forbidden
MOVES = (UP, DOWN, LEFT, RIGHT)
_BAND_KEY = {UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right"}


def allowed_moves(bands):
    return [a for a in MOVES if bands[_BAND_KEY[a]] < BAND_BLOCK]


class ClosedLoopPolicy:
    """Base: pattern proposes; the band rule disposes. Every taken action is
    recorded in self.history (the path-integral registration uses the same list
    at the eval level)."""

    def reset(self, rng):
        self.history = []
        self._reset(rng)

    def act(self, bands, rng):
        allowed = allowed_moves(bands)
        a = self._propose(allowed, rng)
        if a != STAY and a not in allowed:  # pattern bug guard — never push a border
            a = int(rng.choice(allowed)) if allowed else STAY
        self.history.append(a)
        return a

    # subclass API ------------------------------------------------------------
    def _reset(self, rng):
        raise NotImplementedError

    def _propose(self, allowed, rng):
        """Return an action; MUST be in `allowed` or STAY."""
        raise NotImplementedError


class EvalOutAndBack(ClosedLoopPolicy):
    """Straight out A steps, straight back; new direction/amplitude each cycle.
    If blocked mid-out-leg, turn around early (return matches executed length)."""

    def __init__(self, amp_lo=15, amp_hi=70):
        self.amp_lo, self.amp_hi = amp_lo, amp_hi

    def _reset(self, rng):
        self.dir = None
        self.out_left = 0
        self.back_left = 0
        self.out_done = 0

    def _propose(self, allowed, rng):
        if self.back_left > 0:
            self.back_left -= 1
            back = OPPOSITE[self.dir]
            return back if back in allowed else (int(rng.choice(allowed)) if allowed else STAY)
        if self.out_left > 0 and self.dir in allowed:
            self.out_left -= 1
            self.out_done += 1
            if self.out_left == 0:
                self.back_left = self.out_done
            return self.dir
        if self.out_left > 0:                      # blocked mid-leg: turn around
            self.out_left = 0
            self.back_left = self.out_done
            return self._propose(allowed, rng)
        if not allowed:
            return STAY
        self.dir = int(rng.choice(allowed))
        self.out_left = int(rng.integers(self.amp_lo, self.amp_hi + 1))
        self.out_done = 0
        return self._propose(allowed, rng)


class EvalBoxLoop(ClosedLoopPolicy):
    """Rectangle circuit R w, D h, L w, U h, repeated; blocked => next leg early."""

    def __init__(self, lo=15, hi=50, laps=8):
        self.lo, self.hi, self.laps = lo, hi, laps

    def _reset(self, rng):
        w = int(rng.integers(self.lo, self.hi + 1))
        h = int(rng.integers(self.lo, self.hi + 1))
        self.legs = [(RIGHT, w), (DOWN, h), (LEFT, w), (UP, h)] * self.laps
        self.leg_left = 0
        self.leg_dir = STAY

    def _propose(self, allowed, rng):
        for _ in range(64):                        # bounded: all-blocked => STAY
            while self.leg_left == 0:
                if not self.legs:
                    self._reset(rng)
                self.leg_dir, self.leg_left = self.legs.pop(0)
            if self.leg_dir in allowed:
                self.leg_left -= 1
                return self.leg_dir
            self.leg_left = 0                      # blocked: skip to next leg
        return STAY


class EvalSweep(ClosedLoopPolicy):
    """Lawnmower against imagined borders: run until blocked, shift, run back."""

    def __init__(self, lane_lo=4, lane_hi=12):
        self.lane_lo, self.lane_hi = lane_lo, lane_hi

    def _reset(self, rng):
        self.run_dir = int(rng.choice(MOVES))
        self.shift_dir = int(rng.choice([d for d in MOVES if d not in (self.run_dir, OPPOSITE[self.run_dir])]))
        self.lane = int(rng.integers(self.lane_lo, self.lane_hi + 1))
        self.shift_left = 0

    def _propose(self, allowed, rng):
        if self.shift_left > 0:
            self.shift_left -= 1
            if self.shift_dir in allowed:
                return self.shift_dir
            self.shift_dir = OPPOSITE[self.shift_dir]
            return self.shift_dir if self.shift_dir in allowed else STAY
        if self.run_dir in allowed:
            return self.run_dir
        self.run_dir = OPPOSITE[self.run_dir]
        self.shift_left = self.lane
        return self._propose(allowed, rng)


class EvalIdiotWalk(ClosedLoopPolicy):
    """Momentum random walk (Merlin: 'walk around like an idiot' stays valid —
    comebacks happen by stumbling into old cells)."""

    def __init__(self, p=0.85):
        self.p = p

    def _reset(self, rng):
        self.last = None

    def _propose(self, allowed, rng):
        if not allowed:
            return STAY
        if self.last in allowed and rng.random() < self.p:
            return self.last
        self.last = int(rng.choice(allowed))
        return self.last


class EvalRetrace(ClosedLoopPolicy):
    """Wander out for A steps (momentum walk), then retrace the exact reverse
    path — guaranteed high-age comebacks along the whole outbound path."""

    def __init__(self, amp_lo=30, amp_hi=90):
        self.amp_lo, self.amp_hi = amp_lo, amp_hi

    def _reset(self, rng):
        self.out_left = int(rng.integers(self.amp_lo, self.amp_hi + 1))
        self.retrace = []
        self.last = None

    def _propose(self, allowed, rng):
        if self.out_left > 0:
            if not allowed:
                return STAY
            if self.last not in allowed or rng.random() > 0.8:
                self.last = int(rng.choice(allowed))
            self.out_left -= 1
            self.retrace.append(OPPOSITE[self.last])
            return self.last
        if self.retrace:
            back = self.retrace.pop()
            if back in allowed or back == STAY:
                return back
            return int(rng.choice(allowed)) if allowed else STAY
        self._reset(rng)
        return self._propose(allowed, rng)


class EvalDwellDart(ClosedLoopPolicy):
    """Dwell (stay-heavy) then dash in a straight-ish line, dwell, dash back-ish.
    Produces comebacks with a stationary-gap age profile the walkers don't."""

    def __init__(self, dash_lo=25, dash_hi=60):
        self.dash_lo, self.dash_hi = dash_lo, dash_hi

    def _reset(self, rng):
        self.mode = "dwell"
        self.left = int(rng.integers(15, 50))
        self.dir = None

    def _propose(self, allowed, rng):
        self.left -= 1
        if self.mode == "dwell":
            if self.left <= 0:
                self.mode = "dash"
                self.left = int(rng.integers(self.dash_lo, self.dash_hi + 1))
                prefer = OPPOSITE[self.dir] if self.dir is not None else None
                self.dir = prefer if prefer in allowed else (int(rng.choice(allowed)) if allowed else STAY)
            return STAY
        if self.left <= 0:
            self.mode = "dwell"
            self.left = int(rng.integers(15, 50))
            return STAY
        if self.dir in allowed:
            return self.dir
        self.dir = int(rng.choice(allowed)) if allowed else STAY
        return self.dir if self.dir != STAY else STAY


# The eval suite: (name, factory). Ages spread from ~15 (short out-and-back) to
# hundreds (long retraces / box laps). 12 entries x n_seeds episodes.
EVAL_SUITE = [
    ("oab_short", lambda: EvalOutAndBack(15, 30)),
    ("oab_mid", lambda: EvalOutAndBack(30, 55)),
    ("oab_long", lambda: EvalOutAndBack(55, 85)),
    ("box_small", lambda: EvalBoxLoop(12, 25, laps=10)),
    ("box_big", lambda: EvalBoxLoop(30, 55, laps=6)),
    ("sweep_narrow", lambda: EvalSweep(3, 6)),
    ("sweep_wide", lambda: EvalSweep(8, 14)),
    ("idiot_slow", lambda: EvalIdiotWalk(0.7)),
    ("idiot_fast", lambda: EvalIdiotWalk(0.92)),
    ("retrace_mid", lambda: EvalRetrace(30, 60)),
    ("retrace_long", lambda: EvalRetrace(60, 110)),
    ("dwell_dart", lambda: EvalDwellDart()),
]
