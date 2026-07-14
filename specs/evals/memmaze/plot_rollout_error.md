# plot_rollout_error.py — overlay Memory-Maze rollout-error curves from result JSONs into one comparison figure.

The comparison view for the quantitative memmaze rollout-error eval (`evals/memmaze/rollout_error.py`),
the quantitative twin of `evals/gridworld/plot_recall.py`. Reads one or more saved result JSONs and
renders ONE figure: mean decoded pixel MSE (y) vs rollout horizon 1..n_gen (x), one clearly-labelled
curve per model, so vanilla, memory-token, archive, and future variants overlay in the same axes. Eval
each checkpoint once to JSON, then plot/compare freely without re-running inference.

A LOCAL post-hoc analysis tool: matplotlib (Agg), unlike the cv2-only sheets. Plotting is not part of
training or eval — run it on the JSONs you have.

## Interface
- `plot_rollout_error(series, out_path, *, strict=False, logy=False, title=None)` — render the overlay.
  `series` is a list of `(label, result_dict, color|None)`; the FIRST series supplies the shared
  reference curves and the comparability reference signature. `strict` errors on any incomparable
  series (default: flag it). `logy` log-scales the y-axis (spreads the near-zero floor out).
- `_parse_series("label|path|color") -> (label, dict, color|None)` — `'|'`-separated so matplotlib
  colours like `tab:red` work; color optional; path relative to the CWD; missing file → `SystemExit`.
- `_signature(result) -> tuple` — the comparability key: `(n_prefill, n_gen, K, window, encode_window,
  metric, frames basename, tokenizer_sha256, episodes, starts)`. Two results are a fair comparison iff
  their signatures are equal. `_describe_mismatch(ref, j)` names the differing fields for the message.
- `__main__` CLI: `--series` (repeatable, required), `--out` (default
  `outputs/rollout_error/compare.png`), `--strict`, `--logy`, `--title`.

## Behavior
- x-axis = rollout horizon (generated frame after prefill), from the result's `horizons` (1..n_gen);
  y-axis = mean pixel MSE (`protocol.metric`, vs raw ground truth). One solid, coloured, marked curve
  per series from its `mse` array.
- References from the FIRST series only (model-independent): `tokenizer_floor` (black dashed, the
  reconstruction ceiling) and `copy_last` (gray dotted, the static no-dynamics reference).
- The title records the protocol (n_prefill / window / n_gen / K / n_samples / #models); a footnote
  carries the eval's central caveat (exact pixel MSE penalizes visually-plausible rollouts that
  diverge from the recorded trajectory — a short-horizon reconstruction comparison, not a full
  world-model measure).

## Invariants
- Comparability is ENFORCED, never silent: a series whose `_signature` differs from the first is drawn
  dashed with a `(≠)` label suffix AND a stderr warning naming the differing fields; under `--strict`
  it is rejected with a `SystemExit`. Incompatible runs cannot be presented as a fair comparison.
- Pure post-hoc: reads only the saved JSONs, never loads a model or tokenizer and never re-runs
  inference — any compatible set of saved results can be plotted together.
- The figure is self-describing: metric, protocol, and the divergence caveat are legible from the
  image without consulting the eval code.
