"""Overlay Memory-Maze rollout-error curves across models (spec: specs/evals/memmaze/plot_rollout_error.md).

Reads one or more rollout-error result JSONs (written by `rollout_error.py`'s CLI) and renders ONE
figure: mean decoded pixel MSE vs rollout horizon, one curve per model, with the tokenizer-floor and
copy-last references from the FIRST series. Eval each checkpoint once to JSON, then overlay freely
without re-running inference — the quantitative twin of the recall compare plot.

  python -u src/evals/memmaze/plot_rollout_error.py \
    --series "vanilla|outputs/rollout_error/rollout_error_dynamics_vanilla.json|tab:red" \
    --series "mem2mem|outputs/rollout_error/rollout_error_dynamics_mem2mem.json|tab:green" \
    --out outputs/rollout_error/compare.png

Each --series is "label|path|color" ('|' separated so matplotlib colours like 'tab:red' work; color
optional). Paths are relative to the current working directory.

COMPARABILITY IS ENFORCED: series whose evaluation settings differ from the first (protocol, tokenizer,
frames, or the exact scored sample set) are NOT silently overlaid as a fair comparison — they are drawn
dashed with a "(≠)" label and a printed warning, or rejected outright under --strict.

This is a LOCAL post-hoc analysis tool: matplotlib (unlike sheets.py, cv2-only for the cluster venv).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The comparability signature: two results are a fair comparison iff these all match. episodes/starts
# pin the EXACT scored samples; tokenizer sha + frames + protocol pin how each frame was scored.
CAVEAT = ("Exact pixel MSE can penalize a visually-plausible rollout that diverges slightly from the "
          "recorded trajectory:\nshort-horizon reconstruction comparison, not a full world-model measure.")


def _parse_series(spec: str):
    """'label|path|color' -> (label, dict, color|None). color optional."""
    parts = spec.split("|")
    if len(parts) == 2:
        (label, path), color = parts, None
    elif len(parts) == 3:
        label, path, color = parts
    else:
        raise SystemExit(f"--series must be 'label|path' or 'label|path|color', got: {spec!r}")
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"series JSON not found: {p} (paths are relative to the current directory)")
    return label, json.loads(p.read_text()), (color or None)


def _signature(j: dict) -> tuple:
    """Everything that must match for a fair comparison: protocol, tokenizer identity, frames file,
    and the exact scored sample set."""
    proto = j.get("protocol", {})
    smp = j.get("samples", {})
    return (
        proto.get("n_prefill"), proto.get("n_gen"), proto.get("K"), proto.get("window"),
        proto.get("encode_window"), proto.get("metric"),
        Path(j.get("frames", "")).name, j.get("tokenizer_sha256"),
        tuple(smp.get("episodes", [])), tuple(smp.get("starts", [])),
    )


def _describe_mismatch(ref: dict, j: dict) -> str:
    """Human-readable list of which comparability fields differ from the reference."""
    labels = ["n_prefill", "n_gen", "K", "window", "encode_window", "metric", "frames", "tokenizer",
              "episodes", "starts"]
    diffs = [name for name, a, b in zip(labels, _signature(ref), _signature(j)) if a != b]
    return ", ".join(diffs) if diffs else "(none)"


def plot_rollout_error(series, out_path, *, strict=False, logy=False, title=None):
    """Render the overlay figure. series: list of (label, result_dict, color|None). The first series
    supplies the shared references and the comparability reference signature."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = series[0][1]
    ref_sig = _signature(ref)
    proto = ref.get("protocol", {})
    horizons = ref.get("horizons") or list(range(1, (proto.get("n_gen") or 0) + 1))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, j, color in series:
        compatible = _signature(j) == ref_sig
        if not compatible:
            diffs = _describe_mismatch(ref, j)
            msg = f"series {label!r} is NOT directly comparable to {series[0][0]!r} - differs in: {diffs}"
            if strict:
                raise SystemExit(f"ERROR (--strict): {msg}")
            print(f"!! WARNING: {msg}", file=sys.stderr)
        xs = j.get("horizons") or list(range(1, len(j.get("mse", [])) + 1))
        ys = j.get("mse", [])
        lbl = label if compatible else f"{label} (≠)"
        ax.plot(xs, ys, label=lbl, marker="o", ms=3, c=color,
                ls="-" if compatible else "--", alpha=1.0 if compatible else 0.75)

    # references from the FIRST series only (model-independent): reconstruction ceiling + static ref.
    if "tokenizer_floor" in ref:
        ax.plot(horizons, ref["tokenizer_floor"], c="k", ls="--", lw=1, label="tokenizer floor (ceiling)")
    if "copy_last" in ref:
        ax.plot(horizons, ref["copy_last"], c="gray", ls=":", marker="x", ms=3,
                label="copy-last (static, no dynamics)")

    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("rollout horizon  (generated frame after prefill)")
    ax.set_ylabel(f"mean pixel MSE  ({proto.get('metric', 'pixel_mse_01')}, vs ground truth)")
    ax.set_xlim(left=min(horizons) - 0.5 if horizons else 0.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    if title is None:
        smp = ref.get("samples", {})
        title = (f"Memory-Maze rollout error — {proto.get('n_prefill', '?')}-frame streamed prefill "
                 f"(window {proto.get('window', '?')}) → {proto.get('n_gen', '?')}-frame scored rollout, "
                 f"K={proto.get('K', '?')}\n{smp.get('n_samples', '?')} held-out samples, "
                 f"{len(series)} model(s)")
    ax.set_title(title, fontsize=10)
    fig.text(0.5, -0.02, CAVEAT, ha="center", va="top", fontsize=7, color="dimgray")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"wrote {out_path}  ({len(series)} series)")


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description="Overlay Memory-Maze rollout-error curves from result JSONs.")
    ap.add_argument("--series", action="append", required=True,
                    help="'label|path|color' (repeatable; color optional, e.g. tab:green).")
    ap.add_argument("--out", type=Path, default=root / "outputs" / "rollout_error" / "compare.png")
    ap.add_argument("--strict", action="store_true",
                    help="Error out on any series not directly comparable to the first (default: flag).")
    ap.add_argument("--logy", action="store_true", help="Log-scale the y-axis (spreads the floor out).")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    series = [_parse_series(s) for s in args.series]
    plot_rollout_error(series, args.out, strict=args.strict, logy=args.logy, title=args.title)


if __name__ == "__main__":
    main()
