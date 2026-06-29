# Fair no-FF9 ablation (re-test "is FF9 necessary?" without the 411270 confounds)

## Why
The `mem2mem-rollout-noff9` run (job 411270) concluded "FF9 is NECESSARY — without it recall = chance."
Merlin flagged this as conceptually off: the rollout's 50% full-noise mode reconstructs the new half from
carried memory ALONE, so the new-half flow loss *should* force memory to encode hidden state even without
the explicit FF9 term — provided the relay gradient flows back through the memory chain. Investigation:

1. **The relay gradient DOES flow** (not a severed-gradient bug). `experiments/mem2mem/test_autograd.py`
   passes on current code (frame-0 |grad| 6.5e-3 relay-on, 0.0 detached). A training-scale probe at the
   EXACT clean-re-run loss (`use_ff9=False, bootstrap=False, d_min`, real DynamicsModelConfig, W=16, 6
   slides) gives init-only-frame |grad| **0.499 relay-on / 0.0 detached** — the noise-mode loss trains
   memory construction, strongly. `experiments/mem2mem-rollout-noff9-fair/probe_relay_grad.py`.
2. **411270 was CONFOUNDED** exactly like the discredited bootstrap run (411221): bootstrap ON + curriculum
   ON, UNSTABLE (val 0.006→0.022 at curriculum full-unlock), only 36 epochs. Its own NOTES self-flag:
   "the run was also unstable, so 'FF9 needed' is partly entangled with 'no-FF9 needs different HPs'."

So "FF9 necessary" rests on a confounded run. This task re-tests it cleanly.

## What "done" means
One clean ferranti run = the rollout-only WINNER config minus FF9 (the only change vs the 0.99-recall
winner is FF9 on→off), so the result isolates FF9:
- `--no-bootstrap` (d_min only, uniform τ — the stable winner sampler; NO curriculum, NO bootstrap)
- `--no-ff9` (memory trained ONLY by the rollout flow loss; the 50% full-noise mode is the memory signal)
- `--mem2mem-frac 1.0`, 50 epochs, bs64, clip-len64, n-memory4, ff9 3 (architecture), same data/tokenizer.

Eval: recall @ window=8, max_k=64 (K=4, +K=2/1), overlay vs: winner (`dynamics_mem2mem_rollout.pt`, with
FF9), the old confounded no-FF9 (411270), and vanilla/copy_last. position_acc mean / tail (k≥14).

Decision:
- If recall is near-ceiling (≈ winner) → the noise-mode relay flow loss ALONE trains memory; **FF9 is NOT
  necessary**, the 411270 negative was the confounds (Merlin vindicated).
- If recall is still chance → FF9 is genuinely load-bearing DESPITE the relay gradient flowing; the
  noise-mode signal is too weak/slow/long-horizon to learn the carry from scratch (a real, interesting
  result — FF9 acts as a dense short-horizon scaffold). Follow-ups then: higher noise fraction, longer
  training, larger n_memory.
- If partial → quantify the gap; FF9 helps but isn't strictly required.

## Status
- [2026-06-29] in-progress. Relay gradient verified healthy WITHOUT FF9 (probe above); also found the relay
  gradient EXPLODES backward at init (~2-3x/hop, 88 @W=4) → motivated a normalizer arm. TWO parallel runs
  on ferranti:
  - **Arm 1** (no normalizer): job **412506** @ SHA `8f54d09` (--no-bootstrap --no-ff9, 50ep).
  - **Arm 2** (+ per-hop relay grad-norm `--relay-grad-clip 0.05`): job **412510** @ SHA `e266bea`.
  Eval pending on completion → 4-way recall vs winner (with FF9) + old 411270.
