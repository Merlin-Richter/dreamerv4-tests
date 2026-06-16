# T-019 — Repo reorg plan (PROPOSED, needs Merlin approval + verifier pass)

Status: **PROPOSED, not executed.** D-028 did the safe subset (extract reused evals → `src/eval/`,
remove scratch, write `REPO_MAP.md`). This is the ready-to-execute plan for the deeper move, **rewritten
2026-06-16** on Merlin's steer to (1) reframe probes/evals into a first-class **`src/evals/` with one
folder per eval** (modular, each "perfect for its use case": cheap score evals vs rich visual evals), and
(2) **split environments from data generation** (envs = steppable RL sims behind a shared interface; data
generation = a separate dataset-writing layer that drives them).

The move is high-fanout and touches the FROZEN probe spine, so it stays gated on Merlin. A
critical-claim-verifier pass (2026-06-16) returned **SOUND-WITH-FIXES**; its 5 blocking fixes are folded in
below and summarized in §8. Nothing here is executed until Merlin approves.

---

## 0. Motivation (why now, why this shape)
Merlin: *"we will have a lot of different environments and probes and evals."* The current layout does not
scale to that:
- `src/{A,B,C,D}_*` mix **model + training** per pipeline stage (concern tangled with stage).
- `src/data_generators/` conflates **the simulator** (`OccludedBouncingEnv`, the bouncing sim) with
  **dataset writing** (the `__main__` CLIs that dump `.npy`) and with **interactive viewers** (`--play`).
- Evaluation is split across `src/probe/` (frozen spine) and `src/eval/` (working) as **flat modules** with
  no common interface — there is no uniform "run eval X on checkpoint Y", no mid-run score hook, and adding
  the Nth eval means re-deriving the harness each time.

Target principles:
1. **One Env interface** so a new environment is a subclass, not a new bespoke script — and so every eval
   can run against any env.
2. **One Eval interface** with two capabilities — `score()` (cheap scalars, usable as a mid-run/W&B signal
   and as the worker verification signal) and `report()` (rich charts/rollouts/HTML for human review). The
   "score vs visual" distinction Merlin wants is a *capability on a common interface*, not a directory fork:
   an eval implements whichever it can serve well; the spine implements both.
3. **Generation ≠ simulation ≠ inspection.** Envs step; data-gen writes datasets; interactive viewers are
   their own home.
4. **Provenance is sacred.** The frozen spine's numbers must stay bit-identical across the move (§4).

---

## 1. Proposed target layout
```
src/
  models/        dynamics_model.py            (← D_dynamics_model)
                 tokenizer.py                 (← C_multi_image_auto_encoder/video_auto_encoder.py)
                 single_image_ae.py           (← B_single_image_auto_encoder/video_auto_encoder.py)
                 lm.py                        (← A_LM/model.py)

  training/      train_dynamics.py            (← D_dynamics_model/train_dynamics_model.py)
                 train_tokenizer.py           (← C_.../train_autoencoder_bouncing.py)
                 train_single_image_ae.py     (← B_.../train_autoencoder_bouncing.py)
                 train_lm.py                  (← A_LM/train.py)

  envs/          base.py                      (NEW: BaseEnv ABC / interface — see §2)
                 occluded_bouncing.py         (← data_generators/, ENV CLASS ONLY: OccludedBouncingEnv)
                 bouncing.py                  (← data_generators/bouncing_objects.py, ENV CLASS ONLY)
                 README.md                    (the interface contract + how to add an env)

  data/          generate_occluded.py         (← occluded_bouncing.py __main__ / generate_dataset)
                 generate_bouncing.py         (← bouncing_objects.py __main__)
                 loaders.py                   (the real ChunkClipDataset, lifted out of train_dynamics.py)
                 example_read.py              (← data_generators/load_data.py, demo only)

  evals/         __init__.py                  (NEW: registry name→Eval + MIDRUN set — see §3)
                 base.py                      (NEW: Eval ABC, EvalResult, load() helper)
                 _shared/                     (shared primitives: load_models, encode/decode window,
                                               ball detector, probe-episode builders)
                 revisit/                     (FROZEN spine: color/position revisit-consistency)
                 position_consistency/        (FROZEN spine)
                 motion/                       (working: open-loop / teacher-forced curves; score + chart)
                 rollout_view/                 (working: GT-top / rollout-bottom strips, HTML demo)
                 README.md

  interactive/   play_dynamics.py             (← D_.../play_dynamics_checkpoint.py)
                 play_env.py                  (← occluded_bouncing.py --play env viewer)
                 latent_explorer/             (← src/test/latent_explorer/)

  tests/         test_*.py                    (← D_dynamics_model/test_*.py gate tests)
  wlog.py
data/ (root)     *.npy        (optional dataset move; touches train-script path defaults — defer, low value)
checkpoints/     shared *.pt  (optional; same caveat — defer)
```
(`src/probe/` is retired into `src/evals/{revisit,position_consistency}/` — but see §4 for the
freeze-preserving options; this is the single riskiest sub-decision.)

---

## 2. `src/envs/` — the Env interface
Both current sims become `BaseEnv` subclasses. The interface is deliberately minimal and gym-flavored but
tailored to deterministic, hidden-state sims used as world-model data sources:

```python
class BaseEnv(ABC):
    n_actions: int          # 0 = unconditioned (bouncing); 2 = occluded curtain up/down
    img_size: int
    def reset(self, seed: int) -> "BaseEnv": ...        # seeded → fully deterministic
    def step(self, action: int) -> tuple[np.ndarray, np.ndarray]:  # (frame HxWx3 uint8, state)
        ...
    # MEASUREMENT-ONLY privileged accessor (the IDEAS.md "eval exception"): evals may read
    # hidden state to SCORE recall; it is NEVER a model input. Named to make misuse obvious.
    def hidden_state(self) -> np.ndarray: ...           # per-env semantics (see below); color via .color
```
- `OccludedBouncingEnv` already matches this shape (`reset(seed)`, `step(action)->(frame,state)`, `.color`)
  — subclassing is mostly declaring the ABC, no logic change. (occluded_bouncing.py:129)
- `bouncing.py` (the DVD sim) **has no env class today** — physics lives inline in `generate_episode`
  (bouncing_objects.py:94), `n_actions=0`, no `step`/`reset`. Extracting a `BouncingEnv(BaseEnv)` is a **real
  (not just mechanical) refactor** and it is the only env exercising the unconditioned path — budget for it.
- **State width is NOT uniform** (verifier finding): occluded state is 5-dim `[x,y,vx,vy,curtain]`, bouncing is
  4-dim `[x,y,vx,vy]`. The base must NOT fix a 5-vector — `step`/`hidden_state` return `np.ndarray` with
  **per-env semantics documented in each subclass** (or a dict). The 5-vector is occluded-specific, not the
  contract.
- **Channel-order contract (BGR end-to-end) is part of the interface** and must be restated in `envs/README.md`
  + `BaseEnv` docstring — it is a measurement-validity invariant (probe_env.py docstring), easy to silently
  break in a move.

Why this matters: the probe episode-builder and every future memory env now share one contract, so a new env
drops in and all existing evals run against it unchanged.

---

## 3. `src/evals/` — one folder per eval, common interface
```python
@dataclass
class EvalResult:
    scores: dict[str, float]        # scalar metrics — mid-run loggable, worker verification signal
    artifacts: dict[str, Path]      # png / html / json paths — for human review
    meta: dict                      # provenance: checkpoint, commit, env, n_eps, eval-frozen?

class Eval(ABC):
    name: str
    frozen: bool = False            # spine evals = True; any logic change is a logged decision (§8 protocol)
    compatible_envs: list[str]      # which envs / capabilities this eval needs (see below) — REQUIRED
    def score(self, tok, dyn, cfg, *, device) -> dict[str, float]: ...   # REQUIRED, cheap
    def report(self, tok, dyn, cfg, out_dir, *, device) -> EvalResult: ... # OPTIONAL, rich
```
- **Pass a config object, not a `budget` string (verifier finding).** Every existing eval needs more than
  live-model handles: `run_condition(...)` needs `K` (inference steps), `tok_win`, episode set
  (revisit_probe.py:209); `open_loop_curve(...)` needs `K`, `H`, episodes (motion.py:44). So `cfg` carries
  `K / window_N / tok_win / episode grid / n_occ / horizon`; the cheap-vs-full distinction is a field on `cfg`
  (e.g. episode count), not a magic string. **`load()` MUST return `tok_win`** — `load(checkpoint, tokenizer)
  -> (tok, dyn, dcfg, tok_win)` — because `_encode_window` requires it (window ≤ tokenizer window); the
  earlier `(tok, dyn, cfg)` return dropped it and would force the adapter to reconstruct it from globals (the
  contortion §8 warns about).
- **Env-capability tagging (verifier finding).** A registry of evals with no notion of which envs they can run
  on produces silent nonsense when a color/position eval is pointed at the curtain-less bouncing env. `Eval`
  declares `compatible_envs` (or required capabilities, e.g. `needs_occlusion`, `needs_hidden_color`); the
  runner refuses an incompatible (eval, env) pair. Cheap now, painful to retrofit.
- **Score evals** (mid-run): implement `score` — a small episode budget, scalars only. `train_dynamics.py`
  can call `evals.MIDRUN` every N steps and `wlog` them. This is a concrete new capability the move unlocks.
- **Visual evals**: implement `report` — render the chart/rollout/HTML the experiment currently hand-rolls.
  Default `report` just wraps `score` so a score-only eval still satisfies the interface (verifier confirmed
  this split is sound, not over-engineered — `rollout_view` is genuinely score-less).
- **Registry** (`evals/__init__.py`): `REGISTRY: dict[str, Eval]`, a `MIDRUN` list (the cheap set), and the
  module `load(...) -> (tok, dyn, dcfg, tok_win)` above so CLI and training loop construct models identically.
  CLI: `python -u -m evals <name> --checkpoint ... [--report --out experiments/EXP-NNN/]`.
- **Folder granularity:** a folder per *proper* eval (its own `eval.py` + `README.md` stating use-case,
  budget, and which `scores` keys are headline). Shared primitives live in `_shared/`; we do **not** force a
  folder around a 20-line helper.

Concrete initial set (all already exist as code, just re-homed behind the interface):
| Folder | From | score() headline | report() artifact | frozen |
|--------|------|------------------|-------------------|--------|
| `revisit/` | `src/probe/revisit_probe.py` | color ΔRGB @ n_occ{12,16,24} | `sheet.png` GT/pred sheet | **yes** |
| `position_consistency/` | `src/probe/position_consistency.py` | pos recall vs drift control | sheet | **yes** |
| `motion/` | `src/eval/motion.py` | cross_chance_h, pos_err@h16, displacement | open-loop/TF curve PNG | no |
| `rollout_view/` | `src/eval/ab_view.py` | — (visual only) | GT-top/rollout-bottom strip | no |

---

## 4. The frozen-spine sub-decision (highest risk — needs an explicit call)
`src/probe/` is frozen @ **5503e75**; any change "silently redefines every prior result" (protocol §8). The
reorg wants it inside `src/evals/`, but the spine is referenced by ~10 done-experiment scripts via
`sys.path.insert(.../src/probe)` + `from probe_env import ...`, and its episode-builder imports the env that
itself is moving. Three coherent options, in increasing structural purity / increasing risk:

- **(A) Wrap, don't move (RECOMMENDED).** Leave `src/probe/` physically frozen and untouched. `src/evals/`
  becomes the unified API layer; `evals/revisit/` and `evals/position_consistency/` are thin `Eval` adapters
  that import the frozen probe. Achieves the unified eval surface + mid-run hooks at near-**zero risk** to the
  sacred numbers and zero churn on done-experiment imports. Cost: the spine isn't physically "in the evals
  folder," a mild aesthetic miss vs Merlin's framing.
  - **CORRECTION (verifier): option A is NOT automatically bit-identical.** The frozen spine imports the env
    that DOES move: `probe_env.py:39` and `position_consistency.py:222` do
    `from data_generators.occluded_bouncing import OccludedBouncingEnv`. When Phase 1 moves that env to
    `src/envs/`, the untouched spine fails to *import* (hard break, not numeric drift) unless a working
    `data_generators.occluded_bouncing` alias survives. So under option A the alias is **MANDATORY, not
    optional**, and the spine byte-diff gate must run at the **end of Phase 1** (when the dep actually moves),
    not Phase 3. The same alias also covers `verify-T011-scorer/{scorer_probe,c4_markov_check}.py`.
- **(B) Move under a gate.** `git mv` the spine into `evals/{revisit,position_consistency}/`, import-path
  edits ONLY, then prove bit-identical: re-run `revisit_probe` and diff `results.json` byte-for-byte against
  the committed `last_results.json`; keep a `FROZEN @ 5503e75 (relocated D-NNN)` marker. Leave thin
  re-export shims at `src/probe/` so done-experiment scripts don't break. Clean home, but the spine now has a
  moved-import surface and shims add clutter.
- **(C) Move and update all callers.** As (B) but rewrite every done-experiment import instead of shims.
  Cleanest end state, largest blast radius, most chance of a silently-broken old experiment.

Recommendation: **(A) for the spine, full move for everything else.** Decouples the high-value low-risk part
(unified interface) from the low-value high-risk part (physically relocating frozen files). Flag for Merlin;
let the verifier challenge.

---

## 5. Safe migration recipe (per moved module)
1. `git mv` to the new home; rename only where it removes ambiguity (the two `video_auto_encoder.py` →
   `tokenizer.py` / `single_image_ae.py`).
2. Grep + update every importer's `sys.path` insert and `from X import`. Patterns: `from dynamics_model`,
   `from video_auto_encoder`, `from train_dynamics_model`, `from probe_env`, **`from probe.` (package-style —
   e.g. EXP-013 `coherence_eval.py:16`, EXP-015 `perf_rollout.py:22` do `from probe.revisit_probe import`; a
   bare-import grep MISSES these — verifier finding)**, `from data_generators...`, `from eval.motion`. Do this
   per-module, not all at once.
3. Run ALL gate tests (`test_kv_cache / test_stream_cache / test_ff7_smoke / test_ff9_smoke /
   test_multistep_smoke`) — must stay green (CPU OK).
4. CPU-smoke one representative experiment driver per touched module (e.g. `probe_multistep`, an A/B eval,
   an EXP-017 eval) — must reproduce prior numbers.
5. **Spine byte-diff gate (ALL options, run at the END OF PHASE 1):** re-run `revisit_probe` and byte-diff
   `results.json` vs committed `last_results.json` baseline. This is required even under option A, because
   Phase 1 moves the env the spine imports (see §4 correction) — the gate belongs where the dependency moves,
   not deferred to Phase 3.
6. Update `CLAUDE.md` Key-Files table + `REPO_MAP.md` + the new `envs/` & `evals/` READMEs in the SAME commit.
7. Commit per module (small, revertible). Never one big-bang commit.

Env-move note (re §4 correction): the frozen probe's episode-builder drives `OccludedBouncingEnv`, imported as
`data_generators.occluded_bouncing`. Under option A the spine is untouched, so a working
`data_generators.occluded_bouncing` alias (a thin re-export module left at the old path, pointing at
`envs/occluded_bouncing.py`) is **MANDATORY** — without it the untouched spine fails to import. This is a
load-bearing recipe line, not an optional subtlety. Step 5's byte-diff confirms it.

---

## 6. Suggested execution order (staged, each independently revertible)
- **Phase 0 — evals scaffolding, zero file moves (lowest risk, highest immediate value).** Add `evals/base.py`,
  `evals/__init__.py` registry, and adapters wrapping the EXISTING `src/probe` + `src/eval`. Wire one mid-run
  `score()` call into `train_dynamics.py` behind a flag. Delivers the unified surface + mid-run evals before
  any risky move.
- **Phase 1 — env/data split.** Extract `BaseEnv`; move env classes → `envs/`, dataset CLIs → `data/`,
  viewers → `interactive/`. **Leave the mandatory `data_generators.occluded_bouncing` alias (§4/§5) and run
  the spine byte-diff gate here** — Phase 1 is where the spine's dependency actually moves, so this is the
  phase that can break the frozen spine. Gate tests + probe byte-diff before commit.
- **Phase 2 — models/training/tests/interactive split** (the original T-019 core). Gate tests + driver smokes.
- **Phase 3 — fold evals into folders** (`motion/`, `rollout_view/`, and the spine per §4 choice). Re-run
  smokes.

Phase 0 is safe to do unsupervised; Phases 1–3 are the parts that wait for Merlin.

---

## 7. Decisions for Merlin
1. **Frozen spine (§4): A / B / C?** Recommend **A (wrap, don't move)**.
2. **Datasets/checkpoints into `data/` + `checkpoints/`?** Recommend leave at root for now (touches every
   train-script path default; lowest value/risk).
3. **Mid-run evals in training now or later?** Recommend Phase 0 adds the hook but leaves it off by default
   until we pick the cheap eval budget that doesn't slow training.
4. **Back-compat shims for moved modules** (so done-experiment scripts keep running) vs update-all-callers?
   Recommend shims for the spine (A makes this moot) and update-all-callers for the live model/training code.

---

## 8. Verifier verdict (critical-claim-verifier, 2026-06-16) — SOUND-WITH-FIXES
Design + option-A recommendation judged correct; "safe to execute *as written*" judged FALSE until 5 blocking
fixes (all folded into the sections above):
1. **Mandatory env alias + spine gate at Phase 1** — option A is NOT auto bit-identical; the untouched spine
   imports `data_generators.occluded_bouncing` which moves (§4 correction, §5 note, §6 Phase 1).
2. **`probe.*` package-import pattern added to the recipe grep** — EXP-013/EXP-015 use it; bare grep misses it (§5.2).
3. **`Eval` takes a `cfg` object, not a `budget` string; `load()` returns `tok_win`** — every eval needs
   `K/tok_win/window_N/episodes` (§3).
4. **Variable-width env state** — bouncing is 4-dim, occluded 5-dim; base must not fix a 5-vector (§2).
5. **Env-capability tagging on evals** (`compatible_envs`) — stop pointing a color/position eval at the
   curtain-less env (§3).

Verifier also confirmed (not changed): registry + folder-per-eval is justified not ceremony; the
`score`/`report` split is sound (no awkward no-ops, `rollout_view` genuinely score-less); `OccludedBouncingEnv`
fits `BaseEnv` cleanly; the `BouncingEnv` extraction is real work, not mechanical. Full report retained in
session log (agent ab8c628a3509f2d22). No code was run by the verifier (no EXPERIMENTS.md entry).

**Remaining open question for Merlin** (design, not a flaw): is "wrap don't move" (option A) the right
spine call, or does he want the spine physically inside `src/evals/` (option B, gated)? See §7.1.
```
