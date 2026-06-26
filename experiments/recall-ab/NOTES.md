# Recall A/B — FF9 memory vs vanilla

## r2 (2026-06-26) — RETRAIN WORKS: position recall goes from null to near-perfect to k~12
Models: `R-gridworld-retrain2` (5x data + fixed LR schedule, SHA 1688818). `results_r2.json`.
n_ctx=4, max_k=20, n_rollouts=64, K=4. Same eval as r1 (still provisional on k-alignment;
comparison convention-robust).

| k                  | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 20 |
|--------------------|----|----|----|----|----|----|----|----|----|
| oracle pos_acc     |1.00|1.00|1.00|1.00|1.00|1.00|1.00|1.00|1.00|
| **FF9 pos_acc**    |1.00|1.00|0.95|1.00|0.97|0.70|0.28|0.19|0.14|
| vanilla pos_acc    |0.14|0.05|0.09|0.16|0.05|0.03|0.05|0.03|0.02|
| copy_last pos_acc  |0.17|0.16|0.09|0.14|1.00*|0.17|0.16|0.09|1.00*|

(* copy_last periodicity spikes — env bounce period ~10.)

- **FF9 now tracks hidden POSITION** to k≈10-12 (near-perfect), decaying after. Mean position_score
  edge over copy_last flipped **−0.22 (r1) → +0.35 (r2)**. r1's position null was under-training, not
  a fundamental limit — 5x data + warmup→flat→late-cosine LR fixed it.
- vanilla stays at chance (no memory). Color: FF9 1.00 all k; vanilla decays to ~chance by k=20.
- **Residual = the long-horizon tail (k≳14)**, where the window has fully evicted and only the
  memory→memory relay carries state. This is exactly what the mem→mem run (job 410376) targets; A/B
  with `dynamics_mem2mem.pt` pending.

---

# Recall A/B — FF9 memory vs vanilla (r1 first read, under-trained)

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
