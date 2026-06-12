"""Controlled probe-episode generator for the revisit-consistency suite (T-002 / D-011).

Drives ``OccludedBouncingEnv`` directly to build episodes with a *fixed, parametrized*
structure that the saved ``occluded.npy`` cannot give us (its occlusion runs are kept
short on purpose):

    [ P visible frames (curtain UP) ] [ k occluded frames (curtain DOWN) ] [ R reveal (UP) ]

The ball keeps bouncing behind the curtain, so to predict the reveal frame a model must
have retained the hidden ball color + position+velocity across the occluded stretch —
information absent from every occluded frame. For H2 we sweep k from below to above the
model's inference window N: k<N keeps the evidence in context (in-context attention),
k>=N pushes it out (the real memory test).

Ground truth is exact and needs NO detection on the GT side: we own the env, so we read
``env.color`` (ball color) and the per-frame state ``[x, y, vx, vy, curtain]`` directly.

CHANNEL-ORDER CONTRACT (measurement-validity, do not break):
    The env renders with cv2 -> arrays are physically BGR. The dataset stores them in
    this same native order and the rest of the pipeline (encode/decode/train) treats the
    channel axis opaquely (it converts RGB<->BGR only for on-screen display). So frames
    here are in the dataset-native order and ``ball_color`` is in that SAME order. The
    probe pipeline must compare decoded-frame colors to ``ball_color`` WITHOUT any
    RGB<->BGR swap. Keep one order end-to-end.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[1]  # .../src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_generators.occluded_bouncing import OccludedBouncingEnv  # noqa: E402

# Action convention (matches occluded_bouncing): 0 = curtain UP (revealed), 1 = DOWN.
ACTION_UP = 0
ACTION_DOWN = 1


@dataclass
class ProbeEpisode:
    """One probe trial. Frames are dataset-native channel order (see module docstring)."""

    frames: np.ndarray      # (T, H, W, 3) uint8, T = P + k + R
    actions: np.ndarray     # (T,) uint8, 0=up/1=down
    states: np.ndarray      # (T, 5) float32: [x, y, vx, vy, curtain]
    ball_color: np.ndarray  # (3,) uint8, dataset-native order (BGR), per-episode constant
    P: int                  # visible-prefix length
    k: int                  # occlusion length
    R: int                  # reveal length (frames after occlusion, curtain up)
    seed: int

    @property
    def reveal_index(self) -> int:
        """Index of the first reveal frame (the frame we measure recall at)."""
        return self.P + self.k

    @property
    def gt_xy(self) -> np.ndarray:
        """Ground-truth ball (x, y) at the (first) reveal frame."""
        return self.states[self.reveal_index, :2].copy()


def make_probe_episode(
    seed: int,
    P: int = 3,
    k: int = 8,
    R: int = 1,
    img_size: int = 64,
    radius: int = 10,
) -> ProbeEpisode:
    """Generate one structured probe episode. Physics is seeded -> fully deterministic."""
    if P < 2:
        raise ValueError("P must be >= 2 so velocity is observable in the prefix.")
    if k < 0 or R < 1:
        raise ValueError("Need k >= 0 and R >= 1.")

    env = OccludedBouncingEnv(img_size=img_size, radius=radius).reset(seed=seed)
    ball_color = np.array(env.color, dtype=np.uint8)  # native (BGR) order

    actions = np.array([ACTION_UP] * P + [ACTION_DOWN] * k + [ACTION_UP] * R, dtype=np.uint8)
    T = len(actions)
    frames = np.empty((T, img_size, img_size, 3), dtype=np.uint8)
    states = np.empty((T, 5), dtype=np.float32)
    for t, a in enumerate(actions):
        frames[t], states[t] = env.step(int(a))

    return ProbeEpisode(
        frames=frames, actions=actions, states=states, ball_color=ball_color,
        P=P, k=k, R=R, seed=int(seed),
    )


def make_probe_batch(
    k: int,
    n_seeds: int,
    seed0: int = 0,
    P: int = 3,
    R: int = 1,
    img_size: int = 64,
    radius: int = 10,
) -> list[ProbeEpisode]:
    """A batch of episodes at a fixed occlusion length k, one per seed."""
    return [
        make_probe_episode(seed=seed0 + i, P=P, k=k, R=R, img_size=img_size, radius=radius)
        for i in range(n_seeds)
    ]


# Control variants -----------------------------------------------------------------

def make_no_occlusion_episode(seed: int, total_after_prefix: int, **kw) -> ProbeEpisode:
    """Drift control: same length, curtain UP throughout (k=0 conceptually, R long).

    Lets us measure ordinary autoregressive drift over the same horizon so recall loss
    can be attributed to occlusion rather than generic rollout degradation.
    """
    return make_probe_episode(seed=seed, k=0, R=total_after_prefix, **kw)


if __name__ == "__main__":
    # Self-test: structure, determinism, and that GT color/position are available.
    P, k, R = 3, 8, 1
    ep = make_probe_episode(seed=7, P=P, k=k, R=R)
    assert ep.frames.shape == (P + k + R, 64, 64, 3), ep.frames.shape
    assert ep.actions.tolist() == [0, 0, 0] + [1] * k + [0], ep.actions.tolist()
    assert ep.reveal_index == P + k == 11
    assert ep.ball_color.shape == (3,) and ep.ball_color.dtype == np.uint8

    ep2 = make_probe_episode(seed=7, P=P, k=k, R=R)
    assert np.array_equal(ep.frames, ep2.frames), "non-deterministic for fixed seed"
    assert np.array_equal(ep.ball_color, ep2.ball_color)

    # Different seeds -> different ball colors (sanity).
    colors = np.stack([make_probe_episode(seed=s, P=P, k=k).ball_color for s in range(8)])
    assert len(np.unique(colors, axis=0)) > 1, "ball color not varying across seeds"

    # Ball should be visible in the prefix (a bright blob exists) and the curtain frames
    # should be ~uniform (low spatial variance) — a cheap occlusion sanity check.
    prefix_var = ep.frames[:P].reshape(P, -1).var(axis=1).mean()
    curtain_var = ep.frames[P:P + k].reshape(k, -1).var(axis=1).mean()
    assert curtain_var < prefix_var, (curtain_var, prefix_var)

    print("probe_env self-test OK:",
          f"T={ep.frames.shape[0]} reveal_idx={ep.reveal_index} "
          f"ball_color(native/BGR)={ep.ball_color.tolist()} "
          f"GT_xy@reveal={ep.gt_xy.round(2).tolist()} "
          f"prefix_var={prefix_var:.0f} curtain_var={curtain_var:.0f}")
