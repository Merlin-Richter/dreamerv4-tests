"""ENV-DIRECT GridWorld recall eval for FF9 ROLLOUT-TRAINED models (D-048).

Same env-direct protocol, k-grid, n_ctx and scorer as EXP-028/recall_env.py (so curves overlay the
existing recall_env_{vanilla,ff9}.json baselines directly), with one addition: --inference relay =
the UPDATING-memory carry that C1 trains (`generate_updating_memory`: a persistent memory token
re-extracted and carried each step while the latent window is window-2). The plain sliding-window
("windowed") and frozen-snapshot ("snapshot") inferences are kept for the same model so we can show
the relay's contribution in isolation.

Writes experiments/EXP-030/recall_env_<tag>.json. KS extended past 32 (to 44) so the DEEP model's
reach is visible; the <=32 points overlay the EXP-028 baselines.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import torch  # noqa: E402

from evals.gridworld.adapter import (  # noqa: E402
    dynamics_rollout_frames, gen_recall_episode, load_dynamics, load_tokenizer,
)
from evals.gridworld.recall import (  # noqa: E402
    aggregate, chance_levels, copylast_frames, oracle_frames, score_episode,
)

# EXP-028 grid (<=32 overlays the baselines) + deeper points for the rollout models.
KS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 44]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=str(ROOT / "checkpoints/gridworld/tokenizer.pt"))
    ap.add_argument("--dynamics", required=True)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--n-ctx", type=int, default=8)
    ap.add_argument("--n-per-k", type=int, default=64)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-k", type=int, default=None, help="cap KS (e.g. 32 to match baselines).")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, _ = load_tokenizer(args.tokenizer, device)
    model, cfg = load_dynamics(args.dynamics, device)
    ks = [k for k in KS if args.max_k is None or k <= args.max_k]
    print(f"[{args.tag}] env-direct recall: n_ctx={args.n_ctx} N/k={args.n_per_k} "
          f"n_memory={getattr(cfg,'n_memory',0)} ks={ks}")

    recs = {"model": [], "control": [], "oracle": [], "copy_last": []}
    for k in ks:
        for i in range(args.n_per_k):
            f, st, co, cu = gen_recall_episode(seed=70000 + k * 1000 + i, n_ctx=args.n_ctx, k=k)
            recs["model"] += score_episode(
                dynamics_rollout_frames(model, tok, f, cu, device), st, co, cu)
            recs["control"] += score_episode(
                dynamics_rollout_frames(model, tok, f, cu, device, control_curtain_up=True), st, co, cu)
            recs["oracle"] += score_episode(oracle_frames(st, co, cu), st, co, cu)
            recs["copy_last"] += score_episode(copylast_frames(st, co, cu), st, co, cu)
        print(f"  k={k} done")

    res = {"tag": args.tag, "n_ctx": args.n_ctx,
           "n_per_k": args.n_per_k, "chance": chance_levels(),
           **{s: aggregate(r) for s, r in recs.items()}}
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"recall_env_{args.tag}.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {out_dir / f'recall_env_{args.tag}.json'}")

    ch = res["chance"]
    for m in ("position_acc", "color_acc"):
        print(f"\n== {m} (chance={ch[m]:.3f}) ==")
        for k in sorted(res["model"][m], key=int):
            r = {s: res[s][m].get(k, float("nan")) for s in ("model", "control", "copy_last", "oracle")}
            print(f"  k={int(k):>3}: model={r['model']:.3f} ctrl={r['control']:.3f} "
                  f"copy={r['copy_last']:.3f} oracle={r['oracle']:.3f}")


if __name__ == "__main__":
    main()
