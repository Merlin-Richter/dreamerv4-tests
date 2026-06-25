"""GridWorld recall eval is a SOUND instrument (spec: evals/gridworld/{readout,recall}.md).

The discrete env's payoff is a closed-form, provable readout:
  * on TRUE frames the readout recovers the exact (col,row,color,bg)        -> instrument is exact;
  * is_occluded keys on the black grid lines (present revealed / absent occluded);
  * recall()'s ORACLE source scores position_acc == color_acc == 1.0 at every k  -> ceiling self-test;
  * COPY-LAST (no-memory) never beats oracle                                -> baseline is meaningful;
  * chance_levels are the analytic floors (1/36 position, 1/4 colour).

Run:  python src/tests/test_gridworld_eval.py   (CPU only; tiny untrained tokenizer+model).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datagen.generate_gridworld import generate_episode  # noqa: E402
from envs.gridworld import GRID_N, PALETTE  # noqa: E402
from evals.gridworld.readout import read_square  # noqa: E402
from evals.gridworld.recall import chance_levels, recall  # noqa: E402
from models.dynamics_model import DynamicsModel, DynamicsModelConfig  # noqa: E402
from models.tokenizer import AutoEncoder, AutoEncoderConfig  # noqa: E402


def test_readout_exact_on_true_frames():
    for sd in range(50):
        fr, ac, st, col = generate_episode(n_frames=30, seed=sd)
        for t in range(len(fr)):
            if ac[t] == 1:
                continue  # occluded frame: nothing to localize
            rd = read_square(fr[t])
            assert (rd["col"], rd["row"]) == (int(st[t, 0]), int(st[t, 1])), f"pos sd{sd} t{t}"
            assert rd["color_idx"] == int(col[1]), f"square color sd{sd} t{t}"
            assert rd["bg_idx"] == int(col[0]), f"bg sd{sd} t{t}"
    print("[ok] readout exact on true frames (pos+color+bg), 50 episodes")


def test_is_occluded_black_pixel_check():
    """Revealed frames have black grid lines (not occluded); the curtain frame is flat gray (occluded)."""
    for sd in range(30):
        fr, ac, st, col = generate_episode(n_frames=40, seed=sd)
        for t in range(len(fr)):
            occ = read_square(fr[t])["is_occluded"]
            assert occ == bool(ac[t]), f"is_occluded {occ} != curtain {ac[t]} (sd{sd} t{t})"
    print("[ok] is_occluded matches the curtain on revealed/occluded frames, 30 episodes")


def test_chance_levels():
    ch = chance_levels()
    assert abs(ch["position_acc"] - 1.0 / (GRID_N * GRID_N)) < 1e-12
    assert abs(ch["color_acc"] - 1.0 / len(PALETTE)) < 1e-12
    assert 0.0 < ch["position_score"] < 0.2  # graded credit averaged over all cells is small
    print(f"[ok] chance levels: pos_acc={ch['position_acc']:.4f} "
          f"pos_score={ch['position_score']:.4f} color={ch['color_acc']:.2f}")


def test_recall_oracle_ceiling_and_baselines():
    torch.manual_seed(0)
    tok = AutoEncoder(AutoEncoderConfig(embedding_dim=64, depth=4, n_heads=4, mlp_ratio=2.0,
          patch_size=8, n_latents=4, bottleneck_dim=16, max_temporal_length=8)).eval()
    mdl = DynamicsModel(DynamicsModelConfig(embedding_dim=64, depth=8, n_heads=8, mlp_ratio=2.0,
          n_latents=4, bottleneck_dim=16, max_temporal_length=8, max_sampling_steps=16,
          inference_steps=4, n_actions=2, n_memory=2, ff9_k=2)).eval()
    out = recall(mdl, tok, n_ctx=4, max_k=8, n_rollouts=4, K=2, device="cpu")
    assert set(out.keys()) == {"model", "copy_last", "oracle", "chance"}
    # ceiling self-test: oracle reads the TRUE revealed frame -> exact at every k
    assert all(v == 1.0 for v in out["oracle"]["position_acc"].values()), out["oracle"]["position_acc"]
    assert all(v == 1.0 for v in out["oracle"]["color_acc"].values())
    # copy_last never beats the oracle ceiling
    for k, v in out["copy_last"]["position_acc"].items():
        assert v <= out["oracle"]["position_acc"][k] + 1e-9
    assert out["model"]["position_acc"], "model curve must be populated"
    print("[ok] recall: oracle ceiling==1.0 (pos+color), copy_last<=oracle, model curve present")


if __name__ == "__main__":
    test_readout_exact_on_true_frames()
    test_is_occluded_black_pixel_check()
    test_chance_levels()
    test_recall_oracle_ceiling_and_baselines()
    print("\nALL GRIDWORLD EVAL TESTS PASSED")
