"""GridWorldV2 gates: env semantics + v1-readout compatibility + recall instrument self-test.

Covers (spec: specs/envs/gridworldv2.md + specs/evals/gridworldv2/recall.md, both DRAFT):
  * determinism: (seed, actions) -> identical frames/states;
  * movement semantics: clamping at every wall, stay, toggle ticks never move the square,
    curtain latch persists through movement ticks;
  * the v1 closed-form readout is EXACT on v2 frames (position + colors + is_occluded);
  * datagen: all 7 actions appear; states are (T,3); action[t] describes frame t;
  * recall(v2) instrument: oracle position_acc == color_acc == 1.0 at every k (ceiling
    self-test), copy_last <= oracle, model curve present (tiny untrained models, CPU).

Run:  python -u src/tests/test_gridworldv2.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datagen.generate_gridworldv2 import generate_episode, make_action_schedule  # noqa: E402
from envs.gridworld import GRID_N  # noqa: E402
from envs.gridworldv2 import (A_DOWN, A_HIDE, A_LEFT, A_REVEAL, A_RIGHT, A_STAY, A_UP,  # noqa: E402
                              GridWorldV2Env)
from evals.gridworld.readout import read_square  # noqa: E402
from evals.gridworldv2.recall import recall  # noqa: E402
from models.dynamics_model import DynamicsModel, DynamicsModelConfig  # noqa: E402
from models.tokenizer import AutoEncoder, AutoEncoderConfig  # noqa: E402


def test_determinism():
    rng = np.random.default_rng(0)
    acts = rng.integers(0, 7, size=60)
    runs = []
    for _ in range(2):
        env = GridWorldV2Env().reset(seed=11)
        runs.append([env.step(int(a)) for a in acts])
    for (f1, s1), (f2, s2) in zip(*runs):
        assert np.array_equal(f1, f2) and np.array_equal(s1, s2)
    print("[ok] determinism: (seed, actions) -> identical frames/states")


def test_movement_semantics():
    env = GridWorldV2Env().reset(seed=3)
    for a, axis, target in ((A_LEFT, 0, 0), (A_UP, 1, 0), (A_RIGHT, 0, GRID_N - 1),
                            (A_DOWN, 1, GRID_N - 1)):
        for _ in range(GRID_N + 3):                       # drive past the wall: must clamp
            _, s = env.step(a)
        assert int(s[axis]) == target, f"clamp failed for action {a}"
    pos = (env.col, env.row)
    _, s = env.step(A_STAY)
    assert (env.col, env.row) == pos, "stay moved the square"
    _, s = env.step(A_HIDE)                               # toggle ticks: latch + no movement
    assert (env.col, env.row) == pos and int(s[2]) == 1
    _, s = env.step(A_UP)                                 # movement does not change the latch
    assert int(s[2]) == 1, "movement tick changed the curtain latch"
    _, s = env.step(A_REVEAL)
    assert (env.col, env.row) == (pos[0], pos[1] - 1) and int(s[2]) == 0
    print("[ok] movement semantics: clamping, stay, latch, toggle-ticks-don't-move")


def test_readout_exact_on_v2_frames():
    for sd in range(30):
        fr, ac, st, col = generate_episode(n_frames=60, seed=sd)
        for t in range(len(fr)):
            occ = read_square(fr[t])["is_occluded"]
            assert occ == bool(st[t, 2]), f"is_occluded mismatch sd{sd} t{t}"
            if occ:
                continue
            rd = read_square(fr[t])
            assert (rd["col"], rd["row"]) == (int(st[t, 0]), int(st[t, 1])), f"pos sd{sd} t{t}"
            assert rd["color_idx"] == int(col[1]) and rd["bg_idx"] == int(col[0])
    print("[ok] v1 readout exact on v2 frames (pos+colors+is_occluded), 30 episodes")


def test_datagen_schedule():
    seen = set()
    for sd in range(20):
        rng = np.random.default_rng(sd)
        acts = make_action_schedule(rng, 200)
        assert len(acts) == 200
        seen |= set(int(a) for a in acts)
        fr, ac, st, _ = generate_episode(n_frames=50, seed=sd)
        assert st.shape == (50, 3) and ac.dtype == np.uint8
    assert seen == set(range(7)), f"not all 7 actions appear across episodes: {sorted(seen)}"
    print("[ok] datagen: all 7 actions appear; states (T,3); schedule length exact")


def test_recall_oracle_ceiling_and_baselines():
    torch.manual_seed(0)
    tok = AutoEncoder(AutoEncoderConfig(embedding_dim=64, depth=4, n_heads=4, mlp_ratio=2.0,
          patch_size=8, n_latents=4, bottleneck_dim=16, max_temporal_length=8)).eval()
    mdl = DynamicsModel(DynamicsModelConfig(embedding_dim=64, depth=9, n_heads=8, mlp_ratio=2.0,
          n_latents=4, bottleneck_dim=16, max_temporal_length=8, max_sampling_steps=16,
          inference_steps=4, n_actions=7, n_memory=2, ff9_k=2)).eval()
    out = recall(mdl, tok, n_ctx=4, max_k=8, n_rollouts=4, K=2, device="cpu")
    assert set(out.keys()) == {"model", "copy_last", "oracle", "chance"}
    assert all(v == 1.0 for v in out["oracle"]["position_acc"].values()), out["oracle"]["position_acc"]
    assert all(v == 1.0 for v in out["oracle"]["color_acc"].values())
    for k, v in out["copy_last"]["position_acc"].items():
        assert v <= out["oracle"]["position_acc"][k] + 1e-9
    assert out["model"]["position_acc"], "model curve must be populated"
    print("[ok] recall(v2): oracle ceiling==1.0 (pos+color), copy_last<=oracle, model curve present")


if __name__ == "__main__":
    test_determinism()
    test_movement_semantics()
    test_readout_exact_on_v2_frames()
    test_datagen_schedule()
    test_recall_oracle_ceiling_and_baselines()
    print("\nALL GRIDWORLDV2 TESTS PASSED")
