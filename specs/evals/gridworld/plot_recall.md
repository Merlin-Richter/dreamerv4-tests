# plot_recall.py — overlay GridWorld recall curves across runs into one comparison figure.

The picture for `recall.py`'s numbers. Reads one or more recall-result JSONs (written by `recall.py`'s
CLI) and renders a 2x2 figure: per-k MODEL curves for every series overlaid, with `copy_last` / `oracle`
references and the `chance` floor taken from the FIRST series, and a vertical marker at the latent-window
edge. Eval each checkpoint once → JSON; then plot/compare any set of runs without re-evaling. This is the
EXP-027/028/030 hand-rolled compare plot made first-class.

LOCAL post-hoc analysis tool: uses matplotlib (unlike `sheets.py`, cv2-only for the cluster venv).
Plotting is not part of training — run it locally on pulled-back JSONs.

## Interface
- `plot_recall(series, out_path, *, window=None, title=None) -> None` — `series` is a list of
  `(label, result_dict, color|None)`; renders the figure to `out_path`. `window` (total frames) defaults
  to the first series' `meta.window`; the vertical marker is drawn at `k=window-1`. `title` is
  auto-derived from the first series' meta if omitted.
- `__main__` CLI: `--series "label|path|color"` (repeatable; `'|'`-separated so matplotlib colours like
  `tab:red` parse; color optional), `--out` (default `outputs/recall/compare.png`), `--window`, `--title`.
  Series paths resolve relative to the current working directory.
- Helpers (internal): `_curve` ({k:v}→sorted xs,ys), `_parse_series` (spec string → loaded series).

## Behavior
- Metrics are auto-detected from the first series' `model` block, drawn in the order
  `position_acc, position_score, color_acc, bg_acc` — only those present appear. So it works unchanged
  with the current 3-metric `recall` (3 panels, the 4th hidden) and with any future metric (e.g. bg_acc).
- Each series contributes ONE model curve per metric (its own colour); `copy_last`/`oracle`/`chance` come
  from the FIRST series only (shared baselines/ceiling — they are run-independent references).
- Panel grid: 1 metric → 1x1, 2 → 1x2, 3 or 4 → 2x2 (unused panels hidden). y in [-0.02, 1.05].
- Consumes the JSON schema `recall.recall()` returns (`model`/`copy_last`/`oracle`/`chance`, each metric a
  `{k: value}` map) plus the optional `meta` block the CLI adds. Tolerates extra keys.

## Invariants
- Purely a renderer: NO model loading, NO eval, NO recomputation — it only reads JSONs and draws. (Running
  the eval and producing the JSON is `recall.py`'s job; comparing several is this tool's job.)
- References are drawn from the first series so the baseline/ceiling/chance are unambiguous and not
  double-plotted per series.
