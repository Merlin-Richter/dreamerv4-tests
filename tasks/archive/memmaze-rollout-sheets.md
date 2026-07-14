# Qualitative rollout sheets for the memmaze dynamics arms (in src, spec-backed)

Requested by Merlin 2026-07-04, right after vanilla (415103) finished: "produce rollout sheets like
src/evals/gridworld/sheets.py — do that in src too."

## Design (memmaze analogue of the gridworld sheets)
- The gridworld OCCLUSION sheet does not map (no controlled env / curtain action in memmaze). The
  memmaze qualitative instrument is the **rollout sheet**: TOP = ground-truth frames, BOTTOM =
  context reconstructions then an **action-conditioned free-run** (true action sequence from the
  dataset), on **held-out episodes** (reproduce train_dynamics.py's val split: randperm seed 0,
  val_fraction 0.05).
- Lives at `src/evals/memmaze/sheets.py` + spec `specs/evals/memmaze/sheets.md`; reuses the
  drawing/decode/checkpoint-loading layer of `evals/gridworld/sheets.py` (one source of truth).
- Frames (35.7GB) are cluster-only → real sheets render as a small ferranti job; PNGs pulled back.
  Local smoke test with synthetic memmaze-shaped data + the real checkpoints first.

## Done means
- Spec + src file exist, consistent, smoke-tested locally.
- `checkpoints/memmaze/dynamics_vanilla.pt` pulled + load-verified locally.
- Vanilla sheet(s) rendered on real memmaze val episodes (in-window and past-window n_gen), pulled,
  eyeballed, findings + provenance in `experiments/memmaze-dynamics/NOTES.md`.
- Rerun on the mem2mem arm when 415104 lands (that rerun can close under the mem2mem task).

**RESULT (2026-07-04):** DONE — spec+src landed @ 306e147 (smoke-tested local: synthetic + real val12 slice); vanilla sheets rendered on ferranti (job 415142 @ 306e147, rc=0), pulled to `experiments/memmaze-dynamics/sheets_vanilla/_sheet_rollout_{in,past}_window.png`, eyeballed (findings in NOTES.md); val12 slice in `data/` enables local iteration. Mem2mem rerun = one `make_sheets.sh` job when 415104 lands.
