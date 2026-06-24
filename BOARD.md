# BOARD.md — task board

Updated: 2026-06-18 (cleanup pass: completed/historical entries moved to `BOARD-archive.md`).

> Live task state only: in-progress / next / awaiting-Merlin / parked / blocked.
> Completed, superseded, and dropped work lives in `BOARD-archive.md` (grep there for history).

## AWAITING MERLIN — EXP-026 ceiling cleared (ESC-018)
- Eval CORE FROZEN (D-045, ESC-016 resolved). Tokenizer FROZEN (D-044, ESC-017 resolved).
- **EXP-026 tokenizer-roundtrip recall == oracle == 1.0 at every k** → latent not the bottleneck.
  Present-then-stop; not starting dynamics until Merlin weighs in.

## Next (eval adapter + dynamics)
- **Wire the dynamics-rollout frame source** into `src/evals/gridworld/adapter.py` (rollout→decode→score)
  + matched-horizon open-rollout control. Needs a trained dynamics model.

## Next
- **Grad-clip fix for train_dynamics.py** (unclipped at lines 466–468; tokenizer+LM already clip at
  max_norm=1.0). Apply before the dynamics cluster run. (Open question to Merlin from this session.)
- **Vanilla GridWorld dynamics on the cluster** (record a decision first). Frozen tokenizer; train_dynamics
  perf-fixed (D-041) but RE-PROFILE batch (latent-space compute profile ≠ tokenizer bs64).

## Awaiting Merlin
- **ESC-016 Q1 — GridWorld eval sign-off / freeze.** Refined eval BUILT (D-040): graded position_score
  (exact=1/adj=0.25/→0@d3) + exact acc (chance 1/36) + ball/bg 4-way colour + per-k counts/SE; validated
  (oracle 1.0, random≈chance, copy-last decays). **Periodicity finding:** copy-last spikes to 1.0 at
  k≡9 (mod 10) (6×6 bounce period 10) → judge per-k; periodic W&B eval should use off-grid k {3,6,12,16}.
  Pending: freeze + where to wire the during-training eval. (Q2 compute=cluster, answered.)

## Parked (pre-GridWorld-pivot; resume only if Merlin redirects)
> Detail in `BOARD-archive.md` + ORIENT.md "Parked". Occluded-env models under `checkpoints/occluded/`.
- **Occluded-line H3 — DYNAMIC-state relay (op-3).** FF7 colour SUPPORTED (EXP-010); FF9 v2 static-colour
  SUPPORTED beyond window (EXP-017); dynamic POSITION still unsolved. Next method = sequential stop-grad
  register-relay (IDEAS.md). **Blocked on ESC-014** (relay gradient design; V-T014 refuted pure detach →
  cheap tbptt-k sweep recommended before the Mode B build).
- **C1 / motion (exposure-bias / open-loop compounding).** EXP-021 ckpt (~ep10) trained but un-evaluated;
  EXP-018/020/022 established compounding diagnosis + DAgger C1 loss + inference-trust (context_signal) lever.
- **Position-memory consistency metric** — BUILT (`src/probe/position_consistency.py`, EXP-013) but of
  uncertain strength; Merlin chose NOT to freeze it (ESC-009). Revisit only if a position method needs a yardstick.

## Blocked
*(none)*
