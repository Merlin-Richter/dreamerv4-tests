# Recall A/B — FF9 memory vs vanilla (first read)

**Provenance:** models from `R-gridworld-retrain` (SHA `0a0e070`), `checkpoints/gridworld/`. Eval =
`src/evals/gridworld/recall.py` (unchanged). Local RTX 4070 (CUDA). Params: n_ctx=4, max_k=20,
n_rollouts=64, K=4. Driver: `experiments/recall-ab/run.py`; raw curves: `results.json`.

**PROVISIONAL** on the recall k↔tick alignment (off-by-one absolute-k convention atop recall.py, awaiting
Merlin's sign-off). The convention is applied identically to model and baselines, so the
**model-vs-baseline comparison below is convention-robust**; only the absolute k labels are pending.

## Result
- **Instrument self-test:** oracle position_acc = 1.000 at every k → readout exact. ✓
- **Color (STATIC hidden attribute): memory works.** FF9 color_acc = **1.000 at every k** incl. k=20
  (far past the max_temporal_length=16 window); vanilla **decays 0.56 → 0.22** as the context evicts.
  The FF9 memory token carries the static color past the window; the no-memory model forgets it. This
  reproduces the pre-rebuild verdict ("FF9 memory retained STATIC hidden state past the window").
- **Position (DYNAMIC hidden state under physics): NOT retained by either model.** FF9 position_acc
  ~0.03–0.09, vanilla ~0.0–0.17 — both near chance (1/36=0.028) and **below copy_last**; mean
  position_score edge over copy_last is −0.22 (FF9), −0.24 (vanilla). FF9 does **not** beat vanilla on
  position. Tracking the moving square from memory (motion + wall reflections, not just storing a value)
  is unsolved here.

## Caveats / things to weigh (not silently acted on)
- **Env periodicity confound:** copy_last position_acc/color_acc spike to **1.000 at k=10 and k=20** —
  the deterministic bounce has period ~10, so the true square periodically returns to its last-seen
  cell, inflating the freeze baseline (and making "position recall" partly exploitable by periodicity).
  Worth considering for the eval/env design (e.g. randomize phase, or score off-period k).
- **Under-trained / small:** 50 epochs, 7.75M params, 1000 episodes. The position null result may be
  capacity/'training-time, not fundamental — a longer / bigger run is the obvious next probe before
  concluding "position memory is hard".
- n_rollouts=64 → position_acc granularity 1/64; curves are noisy at the low values.

## Suggested next steps (for Merlin)
1. Sign off (or adjust) the k↔tick alignment so the absolute axis is trusted.
2. Decide whether the position null is worth chasing with more train time / capacity, or whether the
   color win is the result of interest for now.
3. Consider the periodicity confound in the recall env before position numbers are over-interpreted.
