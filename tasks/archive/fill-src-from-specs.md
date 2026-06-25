# Fill src/ minimally from the specs (clean rebuild)

Build the new minimal `src/` so every file matches its spec in `specs/`. **The specs are authoritative**
(Merlin owns them); `src_old/` (gitignored snapshot of the pre-rebuild code) is the reference for
clean-copy and for inspiration. Goal: a clean, maintainable, ~5× smaller codebase that does exactly what
the specs say — nothing more. Reversible via git as you go; verify imports after each file.

## Keep-list + how to build each (method per file)

COPY = lift from `src_old` unchanged. COPY+ = copy then make the small noted change. TRIM = copy then
delete the dead parts. WRITE = write from scratch from the spec, taking inspiration from `src_old`.

| file | spec | method | notes |
|---|---|---|---|
| `wlog.py` | `wlog.md` | COPY | clean already |
| `envs/base.py` | `envs/base.md` | COPY | clean |
| `envs/gridworld.py` | `envs/gridworld.md` | COPY+ | `reset(seed=None)` default-random (was 42). Env otherwise unchanged; `is_occluded` logic now lives in readout, not here. |
| `models/tokenizer.py` | `models/tokenizer.md` | COPY | clean (the only non-bloated model file) |
| `datagen/generate_gridworld.py` | `datagen/generate_gridworld.md` | COPY+ | default out path under `data/`; the cv2 viewers are optional — trim if it helps minimality |
| `evals/gridworld/readout.py` | `evals/gridworld/readout.md` | COPY+ | **`is_occluded` = the black-pixel check** (occluded ⇔ NO pixel has R<25 AND G<25 AND B<25, because revealed frames have black grid lines). Replaces the old `top < 8.0`. |
| `evals/gridworld/recall.py` | `evals/gridworld/recall.md` | WRITE | The NEW merged env-based single-rollout scorer `(model,tokenizer)->curves`. Absorbs the old `adapter.py` + `recall.py`. Inspiration: `src_old/evals/gridworld/recall.py` (scoring helpers, oracle/copylast/chance) + `adapter.py` (episode gen, model roll). Implement the one-long-occluded-rollout, branch a reveal at each k, discard, continue (scores all k in one rollout). |
| `models/dynamics_model.py` | `models/dynamics_model.md` | WRITE | The big one. Lift the transformer blocks (space/time attention, RoPE, QK-norm, soft-cap, SwiGLU) and the shortcut-forcing loss from `src_old` — those are correct. The **carrying KV-cached inference is NEW per the spec** (read-old/write-new memory relay, the 5th-pass cache commit at near-clean+memory, RoPE by absolute rollout index) — write it from the spec, do not copy `src_old`'s `generate*`. Keep: shortcut forcing, FF9 sufficiency loss, base scratch registers, memory tokens. DROP: FF7 (`_ff7_loss`, `generate_memory`, register-as-memory), multistep, ff9_rollout, snapshot inference (`generate_full_state_memory`), streaming cache. |
| `training/train_dynamics.py` | `training/train_dynamics.md` | TRIM | Keep vanilla/FF9 training on the frozen tokenizer (encode per batch, ChunkClipDataset, shortcut+ff9 loss, AdamW, grad-clip 1.0, W&B, checkpoint, `--test-checkpoint`). DROP the dead flags (`--ff7 --multistep --ff9-rollout* --rollout-clip-len`) and the deep-clip encode-chunking. |
| `training/train_tokenizer.py` | `training/train_tokenizer.md` | TRIM | 870 lines → lean. Keep the core AE training + the **load-bearing stability** (AdamW beta2≈0.95, grad-spike skip, best-checkpoint by recon, per-step grad-norm logging, MAE dropout, optional LPIPS, recon strips). Drop the rest of the logging cruft. |
| `interactive/play_dynamics.py` | `interactive/play_dynamics.md` | TRIM/WRITE | cv2 single-frame viewer + `model.generate` (carrying). DROP the FF7/snapshot dispatch branches. |

Also: **minimal tests** matching the new code — gridworld env/readout/recall sanity (incl. the
`oracle position_acc==1.0` self-test), dynamics `forward`/`loss`/carrying-`generate`. Drop tests for
deleted features (FF7/stream/multistep/ff9_rollout/ff9_smoke).

## Delete (not in src/)
`envs/occluded_bouncing.py`, `datagen/generate_occluded.py`, `evals/{revisit,position_consistency,motion,
rollout_view}/`, `evals/probe_env.py`, `evals/base.py` (one eval → no registry needed), and the dead
tests above. (single_image_ae + lm already moved to `archive/`; bouncing already deleted.)

## Requirements
- **Clean and maintainable** — minimal, readable, no dead code/flags, no commented-out blocks. Match the
  spec's interface + behavior + invariants exactly; honor Merlin's spec edits (called out above).
- Build leaf-first (base/wlog → env → tokenizer/dynamics → readout/recall → datagen → train/play), and
  check imports + run the tests as you go. Data lives in `data/`. Run python with `-u`.
- For numerically-delicate code (dynamics forward + loss), sanity-check against `src_old` (same output on
  a fixed input) where the design is unchanged.

## Final step (required)
After the build, **spawn the `critical-claim-verifier` agent** to check the code matches the specs —
especially the WRITE files (`dynamics_model.py`, `recall.py`) and the carrying-inference. Frame it
neutrally, e.g. "Does `src/models/dynamics_model.py` faithfully implement `specs/models/dynamics_model.md`
(all of §1–§5 + invariants), and does `src/evals/gridworld/recall.py` implement its spec's single-rollout
scoring?" Read its verdict and fix what it finds before calling this done.

## Done when
`src/` = the keep-list only, all imports work, the minimal tests pass (incl. oracle self-test == 1.0),
the verifier has run on the WRITE files and its findings are addressed.

---

## RESULT (2026-06-25, branch `rebuild/src-from-specs`)
DONE. `src/` is now exactly the keep-list (verified `find src`). Built leaf-first:
- COPY (unchanged, already matched specs): `wlog.py`, `envs/base.py`, `models/tokenizer.py`.
- COPY+: `envs/gridworld.py` (`reset(seed=None)`), `datagen/generate_gridworld.py` (default out
  `data/`), `evals/gridworld/readout.py` (`is_occluded` = black-pixel check, no all-channel-dark pixel).
- WRITE: `models/dynamics_model.py` (lifted blocks+shortcut loss+FF9 sufficiency, NEW carrying
  KV-cached rollout — read-old/write-new memory relay, 5th-pass commit at near-clean+written-memory,
  RoPE-by-absolute-index, sliding-window eviction; `forward` bit-identical to src_old for n_memory=0).
  `evals/gridworld/recall.py` (merged single-rollout env-based scorer, absorbs old adapter.py).
- TRIM: `train_dynamics.py` (610→480), `train_tokenizer.py` (870→582, all stability kept),
  `play_dynamics.py` (372→209).
- Tests: `test_gridworld` (kept), `test_gridworld_eval` (rewritten to new recall API), `test_dynamics`
  (new). All green incl. oracle self-test position_acc/color_acc == 1.0.
- Verifier (critical-claim-verifier): **FAITHFUL, no BUG/DEVIATES** on both WRITE files; probes in
  `experiments/verify-rebuild-dynamics-recall/`, logged in EXPERIMENTS.md. Addressed its one defensive
  caveat (added a window-length assert in `loss`).

**FLAGGED FOR MERLIN (one decision I had to make):** the recall `k`↔tick alignment in a single
branching rollout. I made the reveal at occlusion-length `k` the read-only SIBLING of the k-th occluded
tick (branched before that tick commits; one env.step per k), so the belief at `k` reflects memory
carried through context + the first k-1 occluded frames. It's the only self-consistent single-rollout
reading and is applied identically to model + baselines, but it sets the absolute k-axis convention —
documented at the top of `recall.py`. If you intended k to mean "reveal the frame AFTER k occluded
frames", that's a one-line shift. NOT YET MERGED to master; needs your review.
