# EXP-028 — FF9 v2 memory method on GridWorld (vs vanilla)

Decision D-047. FF9 v2 (full-state memory token, n_memory=4, ff9_k=3), budget-matched to EXP-027
vanilla. Train job 409625 (val 0.148→0.048 monotone). Recall: ENV-DIRECT A/B — controlled episodes
from GridWorldEnv (n_ctx=8 revealed → exactly k occluded → reveal), N=64 per k, k∈{1..32}, frozen
scorer (D-045).

## CORRECTION (supersedes the first read of this experiment)
The first eval ran FF9 through `generate_full_state_memory` (a FROZEN-snapshot special case: writes one
memory snapshot from the prefix, then predicts every frame from a 2-frame [noise|new] window — NO
attention to recent frames, NO dead-reckoning). That gave k=1 position = 0.00 and period-10 spikes
(snapshot only right when the square cycles back). Merlin flagged it. The INTENDED FF9 inference is the
NORMAL autoregressive rollout: per-frame memory tokens are created each step and carried across the
sliding window via position-wise temporal attention, exactly like the frame latents. Both generate and
generate_cached were dispatching to the snapshot path for n_memory>0 models; added generate_cached(...,
plain=True) to run the normal rollout, and the adapter now uses it. The numbers below are the corrected
(plain normal-rollout) inference. The frozen-snapshot path is retained only as a labeled reference.

## Reconciliation (corrected)
Expected (D-047): FF9 holds STATIC hidden state past the window; dynamic position cliffs.
ACTUAL: FF9 is a strong DYNAMIC-memory win — it tracks POSITION through occlusion far past where vanilla
fails, AND holds colour/bg better past window. The "dynamic position needs op-3" prediction is largely
REFUTED at this scale: the carried memory tokens DO integrate motion through the blind window.

Observed (n-weighted; chance pos 0.028, colour 0.25):
```
                 in-window (k<=14)     past-window (k>=16)
position_acc  vanilla   0.519              0.052
              FF9       0.942              0.195
colour_acc    vanilla   1.000              0.297   (cliffs to chance past window)
              FF9       1.000              0.477
bg_acc        vanilla   1.000              0.260
              FF9       1.000              0.615
```
Per-k position_acc: vanilla 0.81→0.33@k8→0.09@k10→~chance by k14. FF9 1.00 through k8, 0.89@k10,
0.73@k14, 0.44@k16, 0.33@k18, decaying to ~chance by k≈28. KEY: FF9's position curve is a SMOOTH DECAY,
NOT the period-10 spikes the frozen snapshot gave — i.e. genuine motion integration that degrades with
horizon, not a static snapshot that's only periodically right.

Findings:
1. DYNAMIC position memory: FF9 ≫ vanilla everywhere. In-window 0.94 vs 0.52 (vanilla already decays
   inside its own window as motion compounds; FF9's memory stabilises it). Past window FF9 0.44@k16 and
   stays > vanilla out to ~k24 before both reach chance. So memory tokens carry position/velocity through
   the blind run and keep integrating — a real dynamic-memory effect.
2. STATIC colour/bg: FF9 also better past window (colour 0.48 vs 0.30, bg 0.62 vs 0.26 weighted) but it
   DECAYS, not flat-at-1.0. (Vanilla cliffs to chance at the window edge.) So even static retention is
   imperfect at this budget — FF9 helps but doesn't perfectly pin it. NB copy-last holds colour=1.0
   trivially (static); the comparison of interest is FF9 vs the no-memory MODEL (vanilla), which FF9 wins.
3. No periodicity artifact in FF9 (smooth curves) → the readout reflects real per-step belief.

Surprise: MILD-HIGH and favourable — predicted static-only; got dynamic position tracking too. The
earlier "frozen snapshot, position needs op-3" read was an inference artifact, not the method.
Hypothesis impact: H-gridworld memory method retains BOTH static and dynamic hidden state well past the
no-memory cliff (supported, stronger than predicted). op-3 may still help at long horizon (FF9 decays to
chance by ~k28) but is not required for basic dynamic retention.
Tripwires (D-047): "colour cliffs like vanilla" → not triggered (FF9 holds better). "position retained
past window" → TRIGGERED as the favourable surprise (logged; this is the result, not a bug — verified by
the smooth non-periodic decay + k=1=1.0 sanity).

## Loose ends
- FF9 rollout SHEETS not yet regenerated with the corrected (plain) inference — the cluster sheets used
  the snapshot path. Redo make_sheets to call generate_cached(plain=True) for memory models.
- 2nd seed would firm the A/B. Long-horizon (k>24) decay to chance is where op-3 / relay could extend.
- Consider an independent verifier pass on the corrected inference before it becomes a headline claim.

## Files
- compare.png (open first): vanilla vs FF9 recall vs k (position graded/exact + ball/bg colour) +
  copy-last/oracle. FF9 position smooth-decays from 1.0; vanilla cliffs. FF9 colour/bg decay but beat
  vanilla past window.
- recall_env_vanilla.json / recall_env_ff9.json: full per-k curves + SE + n_by_k (corrected inference).
