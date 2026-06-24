# EXP-028 — FF9 v2 memory method on GridWorld (vs vanilla)

Decision D-047. FF9 v2 (full-state memory token, n_memory=4, ff9_k=3), budget-matched to the EXP-027
vanilla. Train job 409625 (val 0.148→0.048 monotone; FF9 objective, not comparable to vanilla's
diffusion loss). Recall: ENV-DIRECT A/B (job 409661, recall_env.py) — controlled episodes from
GridWorldEnv (n_ctx=8 revealed → exactly k occluded → reveal), N=64 per k, k∈{1..32}, frozen scorer
(D-045). FF9 evaluated with the memory-aware inference (adapter dispatches generate→
generate_full_state_memory); vanilla with generate_cached. (First env-direct eval, per Merlin —
no dataset, balanced k, no periodicity confound in the references.)

## Reconciliation
Expected (D-047): FF9 holds STATIC hidden state (colour) past the window where vanilla cliffs;
dynamic POSITION still cliffs (frozen snapshot can't integrate motion → needs op-3).

Observed (n-weighted; chance pos 0.028, colour 0.25):
```
                 in-window (k<=14)     past-window (k>=18)
position_acc  vanilla   0.523              0.050
              FF9       0.178              0.328*  (*entirely period-10 spikes, see below)
colour_acc    vanilla   1.000              0.247   (cliffs to chance)
              FF9       1.000              1.000   (FLAT — retained)
bg_acc        vanilla   1.000              0.256
              FF9       1.000              1.000
```

**1. FF9 solves STATIC memory cleanly.** Colour and bg recall stay at **1.0 flat at every k out to 32**
(6×6 of grid), where vanilla cliffs to chance (~0.25) the moment k exceeds the 16-frame window. The
memory token works: a static attribute written once is held indefinitely past the window. Decisive win
over vanilla; matches occluded EXP-017. (NB colour can't beat copy-last, which is also 1.0 — colour is
static; the win is over vanilla, the no-memory *model*.)

**2. FF9 does NOT retain dynamic POSITION — and the failure mode is a clean frozen-snapshot signature.**
FF9 position_acc is ~0 off-period but hits **exactly 1.000 (64/64) at k=10 and k=20**. The 6×6 bounce
period is 10, so the square returns to its write-time cell every 10 steps — a memory snapshot frozen at
context-time is correct ONLY at k≡0 (mod 10). So FF9's position belief is a static snapshot, not an
integrated trajectory. This is the dynamic-state gap, exactly as predicted; it needs a memory that
UPDATES across steps (op-3 / sequential relay), not a written-once snapshot.

**3. Cost: FF9's memory-inference underperforms vanilla on position IN-window (0.178 vs 0.523).** Because
it leans on the static snapshot instead of dead-reckoning the motion vanilla does inside its window. So
FF9 as-is is a static-memory specialist that trades away dynamic tracking.

Surprise: none — textbook confirmation of D-047 / EXP-017 on the clean GridWorld bench. The period-10
position spikes are a satisfying, unambiguous diagnostic of frozen-snapshot memory.
Hypothesis impact: H-gridworld — memory method retains STATIC hidden state past the window (supported);
dynamic position requires the next method (op-3). Both D-047 tripwires clear (colour held → memory
works; position not retained → no surprise win; the spikes are the period artifact, not real tracking).

## Caveat / loose end (sheets NOT faithful for FF9)
The sheets from this job (sheet_occlusion/normal.png) used make_sheets' free_rollout/occlusion_belief,
which call generate_cached → for FF9 this BYPASSES the memory token (vanilla path). So those FF9 sheets
are NOT the faithful memory inference (the quantitative A/B IS faithful — it dispatches correctly).
A faithful FF9 belief sheet needs the memory-carry rollout primitives (full_state_rollout_init/step),
not a per-step generate (which would re-write memory each step). Follow-up; not presented.

## Files
- compare.png (open first): vanilla vs FF9 recall vs k (position graded/exact + ball/bg colour) with
  copy-last/oracle/chance. FF9 colour/bg flat at 1.0; vanilla cliffs at k=16. Position: vanilla
  in-window hump then cliff; FF9 low with period-10 spikes.
- recall_env_vanilla.json / recall_env_ff9.json: full per-k curves + SE + n_by_k.
