# EXP-027 — vanilla GridWorld dynamics baseline (recall)

Decision D-046. Sacred baseline: unmodified DreamerV4-style dynamics (no ff7/ff9/multistep) on the
frozen GridWorld tokenizer. Training job 409479 (val diffusion 0.0146→0.00139, stable). Recall eval
job 409559 on the 150 held-out val episodes (deterministic seed-0 split) through the FROZEN recall core
(D-045); rollout protocol audited by critical-claim-verifier (V-EXP027, SUPPORTED). Frame sources:
model open-loop rollout through occlusion, matched-horizon control (curtain held UP), oracle, copy-last.

## Reconciliation
Expected (D-046): position recalled within the 16-frame window, cliffing to ~copy-last/chance past it
(no memory carrier); colour retained within window, likely lost past it.

Observed (n-weighted; chance: pos_acc 0.028, pos_score 0.086, colour 0.25):
```
                       in-window (k<=14)        past-window (k>=16)
position_acc   model      0.573                    0.015
               copy-last  0.118                    0.191   (inflated by k≡9 mod10 spikes)
               control    0.718                    0.676
colour_acc     model      0.999                    0.265   (≈ chance)
               copy-last  1.000                    1.000
               control    0.996                    1.000
oracle = 1.000 everywhere (instrument valid).
```
Per-k position_acc decays smoothly in-window: k1 0.69 → k5 0.43 → k8 0.22 → k14 0.11, then 0.00 past k15.

Surprise: none (textbook).
Hypothesis impact: establishes the H-gridworld no-memory FLOOR. Vanilla has memory exactly up to its
temporal window and nothing beyond — the canonical DreamerV4 limitation we set out to study.

Three things the data nails:
1. **Real within-window memory+reasoning on position.** Model 0.573 exact-cell vs copy-last 0.118 (5×
   chance-adjusted) — it genuinely retains the last-seen square AND dead-reckons the bounce through up
   to ~14 hidden steps. Imperfect (0.57, not 1.0) because integrating motion blind is hard.
2. **Hard memory cliff at the window edge (k=15).** Past the window, position → 0.015 (BELOW chance and
   below copy-last 0.191): once the last-observed frame scrolls out of the 16-frame window the model has
   nothing to integrate and actively hallucinates. Even the STATIC colour collapses to chance (0.265)
   past the window while copy-last trivially holds it at 1.0 — the cleanest "no memory" signature (per
   Merlin's framing: colour = static-retention memory, which vanilla also loses past the window).
3. **The cliff is MEMORY LOSS, not weak dynamics (matched-horizon control).** With the curtain held UP,
   the model tracks position at ~0.68–0.72 FLAT at every k incl. past the window. So it CAN propagate
   motion arbitrarily far given observations; it fails under occlusion only because it cannot retain the
   hidden state past the window. This is exactly the memory deficit, cleanly isolated.

Tripwires (D-046): (a) "vanilla ≈oracle past window off-period → env too easy" → NOT triggered (0.015 ≪
1.0). (b) "fails within window / in the clear → tokenizer/budget problem" → NOT triggered (in-window
0.57, control 0.72 → dynamics learned). Clean baseline, no confound. Next: memory methods must push the
position curve RIGHT (extend the horizon past k=15) and UP.

## Caveats / loose ends
- High-k tail (k>22) has 1–2 events/bin → noisy; headline x-capped at 24, full data in results.json.
- The cluster job "FAILED" only on a final `import matplotlib` (not in the cluster venv); results.json
  was already written. Fixed: _plot wrapped non-fatal (eval.py); plotting is done locally.
- Checkpoint archived: experiments/EXP-027/dynamics_vanilla.pt (staged via the sheet job 409595 cp;
  gitignored like all .pt, kept locally). Still on ferranti at checkpoints/gridworld/dynamics_vanilla.pt.
- run.sh convention fix still pending: training run.sh should save the checkpoint INTO the run dir
  (runs/<name>/) so pull_results reaches it without the eval/sheet-job staging workaround.

## Files
- headline.png (open first): position graded/exact + ball/bg colour vs k; model vs control vs copy-last
  vs oracle; purple line = window edge (k=15).
- sheet_normal.png: free-run rollout (curtain up), GT top / rollout bottom, columns = timesteps.
- sheet_occlusion.png: rollout through occlusion — TOP true underlying square, BOTTOM model belief
  (curtain-up peek each step), orange = blind columns. The qualitative memory view.
- results.json: full per-k curves + SE + n_by_k for all 4 sources.
- dynamics_vanilla.pt: the trained baseline (gitignored, local archive).
