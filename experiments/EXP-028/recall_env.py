"""ENV-DIRECT GridWorld recall eval (Merlin: evals drive the env, not the val set).

For each target occlusion length k, generate N controlled episodes from GridWorldEnv (n_ctx revealed
context -> exactly k occluded -> reveal), roll the model through the occlusion, and score the reveal
frame through the FROZEN recall core (D-045). Balanced N per k, no dataset schedule artifacts, no
periodicity confound. Frame sources: model rollout, matched-horizon control (curtain up), oracle,
copy-last. Writes results.json (+ headline.png if matplotlib present). Same scorer as EXP-027 so the
env-direct vanilla re-run is A/B-matched with FF9. cv2/torch only for compute; plot optional.
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

KS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=str(ROOT / "checkpoints/gridworld/tokenizer.pt"))
    ap.add_argument("--dynamics", default=str(ROOT / "checkpoints/gridworld/dynamics_vanilla.pt"))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--n-ctx", type=int, default=8)
    ap.add_argument("--n-per-k", type=int, default=64)
    ap.add_argument("--tag", default="vanilla")
    ap.add_argument("--inference", default="auto", choices=["auto", "windowed", "memory"],
                    help="windowed=base dynamics (dead-reckon); memory=FF9 snapshot carry; auto=by config")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, _ = load_tokenizer(args.tokenizer, device)
    model, cfg = load_dynamics(args.dynamics, device)
    print(f"[{args.tag}] env-direct recall: n_ctx={args.n_ctx} N/k={args.n_per_k} inference={args.inference} "
          f"use_full_state_memory={getattr(cfg,'use_full_state_memory',False)}")

    recs = {"model": [], "control": [], "oracle": [], "copy_last": []}
    for k in KS:
        for i in range(args.n_per_k):
            f, st, co, cu = gen_recall_episode(seed=70000 + k * 1000 + i, n_ctx=args.n_ctx, k=k)
            recs["model"] += score_episode(
                dynamics_rollout_frames(model, tok, f, cu, device, inference=args.inference), st, co, cu)
            recs["control"] += score_episode(
                dynamics_rollout_frames(model, tok, f, cu, device, control_curtain_up=True,
                                        inference=args.inference), st, co, cu)
            recs["oracle"] += score_episode(oracle_frames(st, co, cu), st, co, cu)
            recs["copy_last"] += score_episode(copylast_frames(st, co, cu), st, co, cu)
        print(f"  k={k} done ({args.n_per_k} eps)")

    res = {"tag": args.tag, "n_ctx": args.n_ctx, "n_per_k": args.n_per_k, "chance": chance_levels(),
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
