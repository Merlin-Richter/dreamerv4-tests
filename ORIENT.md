# ORIENT.md

Rewritten: 2026-06-16 (Merlin live-steered a PIVOT to a new discrete env; D-032/033/034).

## What we are doing and why
**PIVOT (Merlin, 2026-06-16, D-032):** stop the C1/motion thread and build a new, "more solid"
memory env. The fluid bouncing/occluded env was too high-entropy (arbitrary sub-pixel positions,
arbitrary continuous colors) — measurement was fuzzy (the fluid env couldn't cleanly score POSITION;
GOAL demoted it to a drift-confounded non-metric). The new **GridWorldEnv** is discrete so recall is
crisp: 8x8 cell position + 4-way color, read out closed-form and exactly.

## GridWorld env (built this session — `src/envs/gridworld.py`, datagen, evals)
- 64x64, 8x8 grid (1px outer border + 8x6px cells + 7x2px lines = 64), solid bg from {red,green,
  blue,pink}, square a DIFFERENT one of the four, fills a cell's 6x6 interior. Moves 1 cell/tick in
  8 directions, reflects off walls (corner = both flip). Same curtain occlusion (action 0=revealed,
  1=occluded gray); physics runs behind the curtain. `.color/.bg_color` measurement-only.
- **Curtain schedule (Merlin spec):** per block 90% one random action / 5% 8-revealed run /
  5% 8-occluded run.
- **Dataset GENERATED:** `gridworld.npy` 3000x200 (6.9GB) + _actions/_states/_colors. occ frac 0.50.
- Gate tests green: `test_gridworld.py` (env), `test_gridworld_eval.py` (eval instrument).

## Eval design (D-033) — PENDING MERLIN SIGN-OFF (the "vital decision")
`src/evals/gridworld/` {readout.py, recall.py}. HEADLINE = **position recall acc vs occlusion k**
(exact cell) + color (4-way); diagnostics = reflection split, readout margin; refs = oracle(=1.0),
copy-last(no-memory), chance. Validated: oracle=1.0, copy-last 0.08@k1->0, random~1/64. Position is
promoted to headline because it's the only attribute that CHANGES under occlusion (true dynamic
memory). NOT frozen yet — escalated for Merlin's review before locking + wiring the model adapter.

## Checkpoints reorganized (D-034): `checkpoints/<env>/`
occluded/{tokenizer.pt, dynamics_vanilla.pt}, bouncing/{dynamics.pt, tokenizer.pt}, gridworld/ (WIP).
All live refs + frozen-spine default PATHS repointed (logic byte-unchanged). Fixed a NameError in
train_dynamics default tokenizer arg.

## IN FLIGHT — training (Merlin: "start a vanilla model smoke test, 10 epochs, on the new data")
Vanilla DYNAMICS needs a frozen tokenizer; gridworld has none -> training a gridworld TOKENIZER
first (hard prereq). **RUNNING (harness bg):** tokenizer 10ep **bs16 LPIPS(vgg) + W&B** fresh on a
**300-ep SMOKE SUBSET** (`gridworld_smoke.npy`) -> `checkpoints/gridworld/tokenizer.pt`
(log `experiments/_gridworld_tok_smoke.log`; W&B run gridworld-tok-smoke-s0 in transformer-C-tokenizer).
~4.6 s/it, ~2.5h. (Corrected from a first MSE/no-W&B launch — Merlin flagged; LPIPS is the proven
recipe.) **bs dropped 32->16:** tokenizer alone fills ~7.9/8GB on the 4070, so bs32+LPIPS OOMs ->
another data point that the REAL LPIPS run wants the cluster. NEXT (on completion): reconstruction
view -> vanilla dynamics 10ep smoke -> present.
**MUST use `venv/Scripts/python.exe` for training** (bash default python is torch+cpu -> segfault;
see HOWTO/use_venv_python_for_training.md).
**COMPUTE FINDING:** 4070 ~9 s/it for this tokenizer (GPU-bound) -> full-data 10ep ≈ 25h; heavy
training is a CLUSTER job (EXP-006 ran on galvani). Asked Merlin (ESC-016) whether the real gridworld
pipeline goes to the cluster vs local-reduced. Eval design (D-033) also pending his sign-off (ESC-016).

## Current worries
1. D-032 tripwire: discrete env might be TOO easy (model trivially memorizes 8x8 past the window) —
   the position-vs-k curve is also the instrument that detects this; watch at first baseline.
2. Eval not yet blessed by Merlin — don't freeze/over-build the model adapter until he signs off.
3. Tokenizer smoke quality gates whether the dynamics smoke is meaningful (gridworld is visually
   simple, so MSE should be fine, but verify reconstruction before reading dynamics results).

## Parked (the pre-pivot thread — resume if Merlin redirects back)
- C1/motion (EXP-021 checkpoint ~ep10), exposure-bias/open-loop compounding. ESC-014 op-3 relay.
- Occluded-line H3 (FF7 color SUPPORTED; position open). All occluded models under checkpoints/occluded/.
