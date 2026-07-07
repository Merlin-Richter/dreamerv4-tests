"""ColorField-SYM dataset generation — PROCEDURAL storage. FROZEN-LAYER-sym
(see env.py header; spec: tasks/in-progress/colorfield-sym-frozen-layer.md).

Grids are NOT materialized: an episode is fully determined by (map, start,
actions) under the phase-5 rule, so we store only those sidecars — exactly the
pixel tier's layout — and render (T,5,5) symbolic grids on demand
(env.render_episode — pure slicing, trivially cheap).

Layout under <out_dir>/:
    maps.npy        (N, 15, 15) uint8     iid cell colors (same palette ids)
    starts.npy      (N, 2)      int16     board start positions (c in [0,14]^2)
    actions.npy     (N, T)      uint8     per-TICK actions; actions[i, 0] == STAY,
                                          off-phase ticks (t % 5 != 0) are STAY
    policy_ids.npy  (N,)        uint8     index into POLICY_REGISTRY
    ep_seeds.npy    (N,)        int64     per-episode rng seed (reproducibility)
    meta.json                              geometry/palette/version/split info
"""

import argparse
import json
import os

import numpy as np

from .env import (BOARD, ColorFieldSymEnv, N_COLORS, PALETTE, PHASE_PERIOD,
                  STAY, VIEW_CELLS, render_episode)
from .policies import POLICY_REGISTRY, rollout_policy

VERSION = "colorfield-sym-v1"


def generate(out_dir, n_episodes, T=1024, seed=0, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    maps = np.empty((n_episodes, BOARD, BOARD), dtype=np.uint8)
    starts = np.empty((n_episodes, 2), dtype=np.int16)
    actions = np.empty((n_episodes, T), dtype=np.uint8)
    policy_ids = np.empty(n_episodes, dtype=np.uint8)
    ep_seeds = np.empty(n_episodes, dtype=np.int64)

    ss = np.random.SeedSequence(seed)
    env = ColorFieldSymEnv()
    for i, child in enumerate(ss.spawn(n_episodes)):
        rng = np.random.default_rng(child)
        ep_seed = int(rng.integers(0, 2**62))
        ep_rng = np.random.default_rng(ep_seed)
        pid = int(ep_rng.integers(0, len(POLICY_REGISTRY)))
        env.reset(seed=int(ep_rng.integers(0, 2**62)))
        maps[i] = env.map
        starts[i] = env.pos
        policy_ids[i] = pid
        ep_seeds[i] = ep_seed
        policy = POLICY_REGISTRY[pid][1]()
        actions[i] = rollout_policy(policy, env, T, ep_rng)
        if verbose and (i + 1) % 500 == 0:
            print(f"[datagen-sym] {i + 1}/{n_episodes}", flush=True)

    np.save(os.path.join(out_dir, "maps.npy"), maps)
    np.save(os.path.join(out_dir, "starts.npy"), starts)
    np.save(os.path.join(out_dir, "actions.npy"), actions)
    np.save(os.path.join(out_dir, "policy_ids.npy"), policy_ids)
    np.save(os.path.join(out_dir, "ep_seeds.npy"), ep_seeds)
    meta = {
        "version": VERSION, "n_episodes": n_episodes, "T": T, "seed": seed,
        "geometry": {"n_cells": BOARD, "view_cells": VIEW_CELLS,
                     "phase_period": PHASE_PERIOD, "n_colors": N_COLORS},
        "palette_rgb": PALETTE.tolist(),
        "policies": [name for name, _ in POLICY_REGISTRY],
        "actions_convention": ("actions[:,0]==STAY; grid[t] results from actions[t]; "
                               "moves apply only at ticks t % 5 == 0, off-phase "
                               "ticks are forced STAY; phase[t] = t % 5"),
        "storage": "procedural — render grids via env.render_episode(map, start, actions)",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


class ColorFieldSymDataset:
    """Loader for procedural datasets: episode(i) renders grids on demand."""

    def __init__(self, path):
        self.maps = np.load(os.path.join(path, "maps.npy"))
        self.starts = np.load(os.path.join(path, "starts.npy"))
        self.actions = np.load(os.path.join(path, "actions.npy"))
        self.policy_ids = np.load(os.path.join(path, "policy_ids.npy"))
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)

    def __len__(self):
        return len(self.maps)

    def episode(self, i):
        """grids (T,5,5) uint8 + actions (T,) — rendered on the fly. Phases are
        implicit: phase[t] = t % 5."""
        grids = render_episode(self.maps[i], tuple(self.starts[i]), self.actions[i])
        return grids, self.actions[i]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-episodes", type=int, required=True)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    meta = generate(args.out, args.n_episodes, T=args.T, seed=args.seed)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
