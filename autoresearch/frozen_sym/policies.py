"""Datagen behaviour-policy zoo for ColorField-SYM. FROZEN-LAYER-sym (see
env.py header; spec: tasks/in-progress/colorfield-sym-frozen-layer.md).

The pixel tier's 8-policy zoo ported 1:1 to the 15x15 center lattice: diversity
is the point — 8 genuinely different behaviours, all emitting ONLY valid
actions (outward-at-edge cannot be tried). Policies are PHASE-FREE: they are
consulted only when producing a phase-0 tick — the rollout loop forces STAY on
off-phase ticks — so every policy step is one EFFECTIVE MOVE. Amplitude/lane
parameters are the pixel-tier ranges scaled to cell units (~/6 vs the
90-lattice): out-and-back amp 2..10, box sides 2..8, lawnmower lane 1..2,
dwell 4..16 effective moves. Datagen policies are privileged — they see the
true position; that is fine (they only shape the data distribution, they are
never a model input).

Each policy: reset(rng, start) then act(pos, rng) -> valid SPATIAL action int.
"""

import numpy as np

from .env import (BOARD, DOWN, LEFT, OPPOSITE, PHASE_PERIOD, RIGHT, STAY, UP,
                  spatial_valid_actions)

MOVES = (UP, DOWN, LEFT, RIGHT)


def _valid_moves(pos):
    return [a for a in spatial_valid_actions(pos) if a != STAY]


def _random_pos(rng):
    return (int(rng.integers(0, BOARD)), int(rng.integers(0, BOARD)))


def _step_toward(pos, goal, rng):
    """Merlin's P1 kernel: move along axis r with p = d_r / (d_r + d_c), toward
    goal. Returns STAY if already there."""
    dr = goal[0] - pos[0]
    dc = goal[1] - pos[1]
    adr, adc = abs(dr), abs(dc)
    if adr + adc == 0:
        return STAY
    if rng.random() < adr / (adr + adc):
        return DOWN if dr > 0 else UP
    return RIGHT if dc > 0 else LEFT


class GoalSeek:
    """P1 (Merlin's): random goal on the board; probabilistic axis choice toward
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
    """P3: boustrophedon sweep — run to one edge, shift by `lane` (1..2 cells),
    run back."""

    def reset(self, rng, start):
        self.horizontal = bool(rng.integers(0, 2))
        self.lane = int(rng.integers(1, 3))
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
    """P4: walk to a random rectangle (sides 2..8), lap its perimeter (>= 2
    laps), new rect."""

    def reset(self, rng, start):
        self._new_box(rng)

    def _new_box(self, rng):
        r0 = int(rng.integers(0, BOARD - 2))
        c0 = int(rng.integers(0, BOARD - 2))
        h = int(rng.integers(2, min(9, BOARD - r0)))
        w = int(rng.integers(2, min(9, BOARD - c0)))
        corners = [(r0, c0), (r0, c0 + w - 1), (r0 + h - 1, c0 + w - 1), (r0 + h - 1, c0)]
        self.waypoints = corners * 2 + corners[:1]     # 2 laps, close the loop
        self.wp = 0

    def act(self, pos, rng):
        while self.wp < len(self.waypoints) and pos == self.waypoints[self.wp]:
            self.wp += 1
        if self.wp >= len(self.waypoints):
            self._new_box(rng)
            return self.act(pos, rng)
        a = _step_toward(pos, self.waypoints[self.wp], rng)
        return a if a != STAY else int(rng.choice(_valid_moves(pos)))


class DwellAndDart:
    """P5: linger near an anchor (stay/jitter) for 4..16 effective moves, then
    dart to a far point (manhattan >= 7)."""

    def reset(self, rng, start):
        self.anchor = start
        self.mode = "dwell"
        self.left = int(rng.integers(4, 17))
        self.goal = None

    def act(self, pos, rng):
        if self.mode == "dwell":
            self.left -= 1
            if self.left <= 0:
                self.mode = "dart"
                while True:
                    g = _random_pos(rng)
                    if abs(g[0] - pos[0]) + abs(g[1] - pos[1]) >= 7:
                        self.goal = g
                        break
            if abs(pos[0] - self.anchor[0]) + abs(pos[1] - self.anchor[1]) > 1:
                a = _step_toward(pos, self.anchor, rng)
                return a if a != STAY else STAY
            return STAY if rng.random() < 0.6 else int(rng.choice(_valid_moves(pos)))
        # dart
        if pos == self.goal:
            self.anchor = pos
            self.mode = "dwell"
            self.left = int(rng.integers(4, 17))
            return self.act(pos, rng)
        a = _step_toward(pos, self.goal, rng)
        return a if a != STAY else int(rng.choice(_valid_moves(pos)))


class BorderHug:
    """P6: go to the nearest corner, then lap the board border (max OUT-band
    exposure)."""

    def reset(self, rng, start):
        m = BOARD - 1
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
    """P7: uniform over valid phase-0 actions (incl. stay w.p. 1/5)."""

    def reset(self, rng, start):
        pass

    def act(self, pos, rng):
        return int(rng.choice(spatial_valid_actions(pos)))


class OutAndBack:
    """P8: pick a valid direction + amplitude (2..10, clipped to room), go out,
    come straight back; repeat with fresh parameters."""

    def reset(self, rng, start):
        self.leg = []          # remaining moves of the current out+back pattern

    def act(self, pos, rng):
        if not self.leg:
            d = int(rng.choice(_valid_moves(pos)))
            room = {UP: pos[0], DOWN: BOARD - 1 - pos[0],
                    LEFT: pos[1], RIGHT: BOARD - 1 - pos[1]}[d]
            amp = int(min(int(rng.integers(2, 11)), room))
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
    """Run `policy` in `env` (already reset) for T TICKS total; returns actions
    (T,) uint8 with actions[0] = STAY (the dataset convention). The policy is
    consulted only when producing a phase-0 tick (t % 5 == 0); off-phase ticks
    are FORCED STAY by this loop. Raises if the policy ever emits an invalid
    action — that is a bug, not a recoverable event."""
    actions = np.empty(T, dtype=np.uint8)
    actions[0] = STAY
    policy.reset(rng, env.pos)
    for t in range(1, T):
        if t % PHASE_PERIOD == 0:
            a = int(policy.act(env.pos, rng))
            if a not in env.valid_actions():
                raise AssertionError(f"policy emitted invalid action {a} at {env.pos}")
        else:
            a = STAY
        env.step(a)
        actions[t] = a
    return actions
