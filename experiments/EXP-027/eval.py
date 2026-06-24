"""EXP-027 recall eval — vanilla GridWorld dynamics baseline (D-046).

Scores the trained dynamics model's reveal-frame recall vs occlusion length k on the 150 HELD-OUT val
episodes (deterministic seed-0 split, == train_dynamics), through the FROZEN recall core (D-045).
Frame sources: model rollout (faithful, recall_design.md), matched-horizon control (curtain held UP),
oracle (ceiling), copy-last (no-memory). Writes results.json + headline.png to --out-dir.
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
    dynamics_rollout_frames, load_dynamics, load_tokenizer,
)
from evals.gridworld.recall import (  # noqa: E402
    aggregate, chance_levels, copylast_frames, oracle_frames, score_episode,
)


def val_indices(n_total: int, val_fraction: float = 0.05) -> np.ndarray:
    """Reproduce train_dynamics' deterministic split (seed-0 generator, val = first n_val of perm)."""
    n_val = min(max(1, int(round(n_total * val_fraction))), n_total - 1)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n_total, generator=g).numpy()
    return perm[:n_val]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=str(ROOT / "gridworld.npy"))
    ap.add_argument("--tokenizer", default=str(ROOT / "checkpoints/gridworld/tokenizer.pt"))
    ap.add_argument("--dynamics", default=str(ROOT / "checkpoints/gridworld/dynamics_vanilla.pt"))
    ap.add_argument("--n_val", type=int, default=0, help="limit val episodes (0 = all 150)")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stem = args.frames[:-4] if args.frames.endswith(".npy") else args.frames
    frames = np.load(args.frames, mmap_mode="r")
    states = np.load(stem + "_states.npy", mmap_mode="r")
    colors = np.load(stem + "_colors.npy", mmap_mode="r")
    actions = np.load(stem + "_actions.npy", mmap_mode="r")  # == curtain

    val_idx = val_indices(len(frames))
    if args.n_val:
        val_idx = val_idx[:args.n_val]
    print(f"EXP-027 recall eval: {len(val_idx)} held-out val episodes (device={device})")

    tok, _ = load_tokenizer(args.tokenizer, device)
    model, cfg = load_dynamics(args.dynamics, device)
    print(f"dynamics: n_actions={cfg.n_actions} max_T={cfg.max_temporal_length} K={cfg.inference_steps}")

    recs = {"model": [], "control": [], "oracle": [], "copy_last": []}
    for n, i in enumerate(val_idx):
        f, st, co, cu = (np.asarray(frames[i]), np.asarray(states[i]),
                         np.asarray(colors[i]), np.asarray(actions[i]))
        recs["model"] += score_episode(dynamics_rollout_frames(model, tok, f, cu, device), st, co, cu)
        recs["control"] += score_episode(
            dynamics_rollout_frames(model, tok, f, cu, device, control_curtain_up=True), st, co, cu)
        recs["oracle"] += score_episode(oracle_frames(st, co, cu), st, co, cu)
        recs["copy_last"] += score_episode(copylast_frames(st, co, cu), st, co, cu)
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(val_idx)} episodes")

    res = {"n_val": len(val_idx), "chance": chance_levels(),
           **{src: aggregate(r) for src, r in recs.items()}}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {out_dir / 'results.json'}")

    ch = res["chance"]
    for metric in ("position_score", "position_acc", "color_acc"):
        print(f"\n== {metric} (chance={ch[metric]:.3f}) ==")
        for k in sorted(res["model"][metric], key=int):
            row = {s: res[s][metric].get(k, float("nan")) for s in
                   ("model", "control", "copy_last", "oracle")}
            nk = res["model"]["n_by_k"][k]
            print(f"  k={int(k):>3} (n={nk:>4}): model={row['model']:.3f} ctrl={row['control']:.3f} "
                  f"copy={row['copy_last']:.3f} oracle={row['oracle']:.3f}")

    _plot(res, out_dir / "headline.png")


def _plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ch = res["chance"]
    metrics = [("position_score", "graded position"), ("position_acc", "exact position (1/36)"),
               ("color_acc", "ball colour (1/4)"), ("bg_acc", "bg colour (1/4)")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    styles = {"oracle": dict(c="k", ls="--"), "model": dict(c="tab:blue", marker="o"),
              "control": dict(c="tab:green", marker="^", ls=":"), "copy_last": dict(c="tab:red", marker="x")}
    for ax, (m, title) in zip(axes.ravel(), metrics):
        for src, stl in styles.items():
            d = res[src].get(m, {})
            ks = sorted(d, key=int)
            if ks:
                ax.plot([int(k) for k in ks], [d[k] for k in ks], label=src, **stl)
        if m in ch:
            ax.axhline(ch[m], c="gray", ls=":", lw=1, label="chance")
        ax.axvline(res["model"].get("_window", 15), c="purple", ls="-", lw=0.6, alpha=0.4)
        ax.set_title(title); ax.set_xlabel("occlusion length k"); ax.set_ylabel(m)
        ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle(f"EXP-027 vanilla GridWorld dynamics recall vs k  (n={res['n_val']} held-out val eps)")
    fig.tight_layout(); fig.savefig(path, dpi=110)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
