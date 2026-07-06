"""Datagen behaviour-policy zoo for ColorField. FROZEN LAYER (see env.py header).

Diversity is the point (Merlin, 2026-07-06): 8 genuinely different behaviours, all
emitting ONLY valid actions (outward-at-edge cannot be tried). Datagen policies are
privileged — they see the true position; that is fine (they only shape the data
distribution, they are never a model input).

Each policy: reset(rng, start) then act(pos, rng) -> valid action int.
"""

import numpy as np

from .env import (DOWN, LATTICE, LEFT, OPPOSITE, RIGHT, STAY, UP,
                  apply_action, valid_actions)

MOVES = (UP, DOWN, LEFT, RIGHT)


def _valid_moves(pos):
    return [a for a in valid_actions(pos) if a != STAY]


def _random_pos(rng):
    return (int(rng.integers(0, LATTICE)), int(rng.integers(0, LATTICE)))


def _step_toward(pos, goal, rng):
    """Merlin's P1 kernel: move along axis r with p = d_r / (d_r + d_c), toward goal.
    Returns STAY if already there."""
    dr = goal[0] - pos[0]
    dc = goal[1] - pos[1]
    adr, adc = abs(dr), abs(dc)
    if adr + adc == 0:
        return STAY
    if rng.random() < adr / (adr + adc):
        return DOWN if dr > 0 else UP
    return RIGHT if dc > 0 else LEFT


class GoalSeek:
    """P1 (Merlin's): random goal on the lattice; probabilistic axis choice toward
    it; eps-fraction uniform-random valid moves; resample goal on arrival."""

    def __init__(self, eps=None):
        self.eps = eps

    def reset(self, rng, start):
        if self.eps is None:
            self.eps = float(rng.uniform(0.05, 0.3))
        self.goal = _random_pos(rng)

    def act(self, pos, rng):
        if pos == self.goal:
            self.goal = _random_pos(rng)
        if rng.random() < self.eps:
            return int(rng.choice(_valid_moves(pos)))
        a = _step_toward(pos, self.goal, rng)
        if a == STAY:
            a = int(rng.choice(_valid_moves(pos)))
        return a


class MomentumWalk:
    """P2: repeat the last move w.p. p, else a uniform valid move."""

    def __init__(self, p=None):
        self.p = p

    def reset(self, rng, start):
        if self.p is None:
            self.p = float(rng.uniform(0.6, 0.95))
        self.last = None

    def act(self, pos, rng):
        vm = _valid_moves(pos)
        if self.last in vm and rng.random() < self.p:
            return self.last
        self.last = int(rng.choice(vm))
        return self.last


class Lawnmower:
    """P3: boustrophedon sweep — run to one edge, shift by `lane`, run back."""

    def reset(self, rng, start):
        self.horizontal = bool(rng.integers(0, 2))
        self.lane = int(rng.integers(2, 9))
        self.run_dir = RIGHT if self.horizontal else DOWN
        if rng.integers(0, 2):
            self.run_dir = OPPOSITE[self.run_dir]
        self.shift_dir = DOWN if self.horizontal else RIGHT
        if rng.integers(0, 2):
            self.shift_dir = OPPOSITE[self.shift_dir]
        self.shift_left = 0

    def act(self, pos, rng):
        vm = _valid_moves(pos)
        if self.shift_left > 0:
            if self.shift_dir in vm:
                self.shift_left -= 1
                return self.shift_dir
            self.shift_dir = OPPOSITE[self.shift_dir]  # hit the far edge: bounce
            self.shift_left -= 1
            return self.shift_dir
        if self.run_dir in vm:
            return self.run_dir
        self.run_dir = OPPOSITE[self.run_dir]          # end of the run: turn around
        self.shift_left = self.lane
        return self.act(pos, rng)


class BoxPatrol:
    """P4: walk to a random rectangle, lap its perimeter (>= 2 laps), new rect."""

    def reset(self, rng, start):
        self._new_box(rng)

    def _new_box(self, rng):
        r0 = int(rng.integers(0, LATTICE - 15))
        c0 = int(rng.integers(0, LATTICE - 15))
        h = int(rng.integers(10, min(50, LATTICE - r0)))
        w = int(rng.integers(10, min(50, LATTICE - c0)))
        corners = [(r0, c0), (r0, c0 + w - 1), (r0 + h - 1, c0 + w - 1), (r0 + h - 1, c0)]
        self.waypoints = corners * 2 + corners[:1]     # 2 laps, close the loop
        self.wp = 0
        self.laps_rng_next = rng

    def act(self, pos, rng):
        while self.wp < len(self.waypoints) and pos == self.waypoints[self.wp]:
            self.wp += 1
        if self.wp >= len(self.waypoints):
            self._new_box(rng)
            return self.act(pos, rng)
        a = _step_toward(pos, self.waypoints[self.wp], rng)
        return a if a != STAY else int(rng.choice(_valid_moves(pos)))


class DwellAndDart:
    """P5: linger near an anchor (stay/jitter), then dart to a far point."""

    def reset(self, rng, start):
        self.anchor = start
        self.mode = "dwell"
        self.left = int(rng.integers(20, 80))
        self.goal = None

    def act(self, pos, rng):
        if self.mode == "dwell":
            self.left -= 1
            if self.left <= 0:
                self.mode = "dart"
                while True:
                    g = _random_pos(rng)
                    if abs(g[0] - pos[0]) + abs(g[1] - pos[1]) >= 40:
                        self.goal = g
                        break
            if abs(pos[0] - self.anchor[0]) + abs(pos[1] - self.anchor[1]) > 3:
                a = _step_toward(pos, self.anchor, rng)
                return a if a != STAY else STAY
            return STAY if rng.random() < 0.6 else int(rng.choice(_valid_moves(pos)))
        # dart
        if pos == self.goal:
            self.anchor = pos
            self.mode = "dwell"
            self.left = int(rng.integers(20, 80))
            return self.act(pos, rng)
        a = _step_toward(pos, self.goal, rng)
        return a if a != STAY else int(rng.choice(_valid_moves(pos)))


class BorderHug:
    """P6: go to the nearest corner, then lap the lattice border (max OUT-band
    exposure)."""

    def reset(self, rng, start):
        m = LATTICE - 1
        corners = [(0, 0), (0, m), (m, m), (m, 0)]
        near = min(range(4), key=lambda i: abs(corners[i][0] - start[0]) + abs(corners[i][1] - start[1]))
        order = corners[near:] + corners[:near]
        self.waypoints = order * 3 + order[:1]
        self.wp = 0

    def act(self, pos, rng):
        while self.wp < len(self.waypoints) and pos == self.waypoints[self.wp]:
            self.wp += 1
        if self.wp >= len(self.waypoints):
            self.reset(rng, pos)
        a = _step_toward(pos, self.waypoints[self.wp], rng)
        return a if a != STAY else int(rng.choice(_valid_moves(pos)))


class UniformRandom:
    """P7: uniform over valid moves (incl. stay w.p. 1/5)."""

    def reset(self, rng, start):
        pass

    def act(self, pos, rng):
        return int(rng.choice(valid_actions(pos)))


class OutAndBack:
    """P8: pick a valid direction + amplitude (10..60, clipped to room), go out,
    come straight back; repeat with fresh parameters."""

    def reset(self, rng, start):
        self.leg = []          # remaining actions of the current out+back pattern

    def act(self, pos, rng):
        if not self.leg:
            d = int(rng.choice(_valid_moves(pos)))
            room = {UP: pos[0], DOWN: LATTICE - 1 - pos[0],
                    LEFT: pos[1], RIGHT: LATTICE - 1 - pos[1]}[d]
            amp = int(min(int(rng.integers(10, 61)), room))
            if amp == 0:
                return STAY
            self.leg = [d] * amp + [OPPOSITE[d]] * amp
        return self.leg.pop(0)


POLICY_REGISTRY = [
    ("goal_seek", GoalSeek),
    ("momentum", MomentumWalk),
    ("lawnmower", Lawnmower),
    ("box_patrol", BoxPatrol),
    ("dwell_dart", DwellAndDart),
    ("border_hug", BorderHug),
    ("uniform", UniformRandom),
    ("out_and_back", OutAndBack),
]


def rollout_policy(policy, env, T, rng):
    """Run `policy` in `env` (already reset) for T frames total; returns actions
    (T,) uint8 with actions[0] = STAY (the dataset convention). Raises if the
    policy ever emits an invalid action — that is a bug, not a recoverable event."""
    actions = np.empty(T, dtype=np.uint8)
    actions[0] = STAY
    policy.reset(rng, env.pos)
    for t in range(1, T):
        a = int(policy.act(env.pos, rng))
        if a not in env.valid_actions():
            raise AssertionError(f"policy emitted invalid action {a} at {env.pos}")
        env.step(a)
        actions[t] = a
    return actions
