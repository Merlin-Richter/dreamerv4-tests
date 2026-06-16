# src/eval — evaluation & rollout toolbox (working, NON-frozen)

Shared, reusable evaluation code. This is the mutable toolbox that experiments import instead of
re-pasting eval logic into each `experiments/EXP-NNN/` script.

**Relationship to `src/probe/`:** `src/probe/` is the **FROZEN** revisit/position-consistency probe
*spine* (frozen @ commit 5503e75; any change there is a logged decision — see `GOAL.md` §8, because
it silently redefines every prior result). `src/eval/` is the *non-frozen* layer on top: it calls the
frozen probe's primitives (`load_models`, `_encode_window`, `_decode_frame`, `detect_ball`,
`make_probe_episode`) and adds higher-level curves/drivers that we evolve freely.

## Modules
- `motion.py` — motion-prediction curves on the probe env (curtain-up):
  - `open_loop_curve` / `teacher_forced_curve` — per-horizon position error (autoregressive vs GT-context).
  - `tau_context_sweep` — TF 1-step error vs `context_signal` (the train/inference noise-level mismatch).
  - `open_loop_displacement` — predicted inter-frame displacement (copy-last collapse monitor).
  - `cross_chance_h` — first horizon a curve reaches chance.
  - Also re-exports `load_models` and the `N, P = 8, 3` probe geometry.

Originally lived in `experiments/EXP-018/probe_multistep.py` (+ `EXP-020/ab_eval.py`); extracted in D-028.

## Usage
```python
import sys; sys.path.insert(0, "src")          # so `eval` is importable as a package
from eval.motion import open_loop_curve, teacher_forced_curve, load_models, N, P
tok, dyn, dcfg, _ = load_models("checkpoints/occluded/tokenizer.pt", ckpt, N, device)
ol = open_loop_curve(tok, dyn, episodes, device, dcfg.inference_steps, H=24)
```
`motion.py` self-bootstraps the frozen-probe / tokenizer / dynamics import paths, so callers only
need `src` on `sys.path`.
