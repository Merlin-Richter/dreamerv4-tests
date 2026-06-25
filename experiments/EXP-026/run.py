"""EXP-026 — GridWorld tokenizer-roundtrip recall CEILING.

D-044 tripwire: before training any GridWorld dynamics model, check the frozen tokenizer's latent can
actually represent the square's cell+colour at reveal frames. If encode->decode of the TRUE frames
can't be read back accurately, no dynamics model on this latent ever will.

Scores tokenizer-roundtrip frames through the FROZEN recall core (D-045) and compares per-k to the
oracle ceiling, copy-last (no-memory), and analytic chance. Headline = position_score / position_acc /
color_acc / bg_acc vs occlusion length k (judged PER-K per D-045 periodicity ruling).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evals.gridworld.adapter import load_tokenizer, tokenizer_roundtrip  # noqa: E402
from evals.gridworld.recall import (  # noqa: E402
    aggregate, chance_levels, copylast_frames, oracle_frames, score_episode,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=str(ROOT / "gridworld.npy"))
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/gridworld/tokenizer.pt"))
    ap.add_argument("--n_episodes", type=int, default=200)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results.json"))
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stem = args.frames[:-4] if args.frames.endswith(".npy") else args.frames
    frames = np.load(args.frames, mmap_mode="r")
    states = np.load(stem + "_states.npy", mmap_mode="r")
    colors = np.load(stem + "_colors.npy", mmap_mode="r")
    actions = np.load(stem + "_actions.npy", mmap_mode="r")  # == curtain channel
    n = min(args.n_episodes, len(frames))
    print(f"EXP-026 ceiling on {n} episodes (device={device})")

    model, L = load_tokenizer(args.checkpoint, device)
    print(f"frozen tokenizer loaded; window L={L}")

    tok_recs, oracle_recs, copy_recs = [], [], []
    for i in range(n):
        st, co, cu = np.asarray(states[i]), np.asarray(colors[i]), np.asarray(actions[i])
        recon = tokenizer_roundtrip(model, np.asarray(frames[i]), L, device)
        tok_recs += score_episode(recon, st, co, cu)
        oracle_recs += score_episode(oracle_frames(st, co, cu), st, co, cu)
        copy_recs += score_episode(copylast_frames(st, co, cu), st, co, cu)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n} episodes")

    res = {
        "n_episodes": n, "window_L": L,
        "chance": chance_levels(),
        "tokenizer_roundtrip": aggregate(tok_recs),
        "oracle": aggregate(oracle_recs),
        "copy_last": aggregate(copy_recs),
    }
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")

    # quick console view: position/colour vs k, tokenizer vs oracle vs copy-last
    ch = res["chance"]
    for metric in ("position_score", "position_acc", "color_acc", "bg_acc"):
        print(f"\n== {metric}  (chance={ch[metric]:.3f}) ==")
        ks = sorted(res["tokenizer_roundtrip"][metric], key=int)
        for k in ks:
            tk = res["tokenizer_roundtrip"][metric][k]
            orc = res["oracle"][metric].get(k, float("nan"))
            cl = res["copy_last"][metric].get(k, float("nan"))
            nk = res["tokenizer_roundtrip"]["n_by_k"][k]
            print(f"  k={int(k):>3} (n={nk:>4}): tok={tk:.3f}  oracle={orc:.3f}  copy-last={cl:.3f}")


if __name__ == "__main__":
    main()
