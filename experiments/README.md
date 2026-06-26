# experiments/ — where research ideas are tried (NOT spec-backed)

This is the lab. Code here is **speculative and disposable**: it is exempt from the one-spec-per-file
rule that governs `src/`. The whole point is to try many ideas — new losses, training flows, probes —
**without touching the spec-backed `src/` files**, so the 19 ideas that lose never bloat
`src/models/dynamics_model.py`. (That bloat is exactly what the ~5× `src/` rebuild deleted: FF7,
multistep, ff9-rollout, snapshot inference, … — all "tried in the model file and never cleaned out.")

> **The rule:** `src/` = what's *proven and kept* (spec-backed). `experiments/` = what's *being tried*
> (no spec, dies with the experiment). An idea touches `src/` and `specs/` **only when it graduates.**

## One dir per experiment

`experiments/EXP-NNN/` (or a descriptive run name). Self-contained, like the existing ones:
- `NOTES.md` — hypothesis, what you ran, result, verdict. **This is the durable record** — it survives
  even when the code is deleted, so a loser still leaves a "we tried X, it did Y" trail.
- `run.sh` / `eval.py` / probe scripts — import `src/` and analyze. (See EXP-027/030 for the pattern.)
- `model.py` / `loss.py` — **only if** the idea needs variant model/loss code (see below).

## How to try a new loss / training-flow idea (the seam)

Don't edit `src/models/dynamics_model.py`. Subclass it in your experiment dir and override the one
surface you're changing — the canonical model already exposes the needed primitives (`forward(...,
return_memory=True)`, `sample_tau_d`, `_tau_value`/`_d_value`, `_noise_to_ctx`):

```python
# experiments/EXP-034/model.py        (NOT spec-backed; dies with the experiment)
from models.dynamics_model import DynamicsModel

class DynamicsModelEXP034(DynamicsModel):
    def loss(self, z1, action_idx=None, return_parts=False):
        # assemble the new objective from the exposed primitives / self(...) calls
        ...
```

Train it through the canonical trainer — no trainer fork needed:

```bash
python -u src/training/train_dynamics.py \
    --model-module experiments/EXP-034/model.py:DynamicsModelEXP034 \
    --frames data/gridworld.npy --tokenizer checkpoints/gridworld/tokenizer.pt \
    --checkpoint checkpoints/gridworld/exp034.pt --ff9 3 --n-memory 4
```

The subclass takes the same `DynamicsModelConfig` and inherits `generate()`, so **checkpoints and the
recall eval keep working unchanged** — a loss-only experiment changes training, not inference.

Pick the home by what the idea actually touches:

| Idea touches… | Where it lives |
|---|---|
| **Loss math only** | `experiments/EXP-NNN/model.py` — subclass, override `loss()`. No `src/` edit. |
| **Training flow** (schedule, DAgger, rollout-training) | experiment-local train script (or a hook). It's the *trainer's* concern, not the model. |
| **Architecture** (new token types / attention) | subclass overriding the relevant method. If you're copy-pasting a big chunk of the model just to vary one spot, that's the signal to add a *small* seam to the canonical model — at the varied point only, not preemptively. |

## The decision metric

Win/lose is decided by the **recall curve** (`evals/gridworld/recall.py`) vs the relevant baseline
(FF9 / vanilla / oracle / copy_last) — **not** by reconstruction or training loss. State the bar in
`NOTES.md` before running.

## Graduation (the only time `src/` + `specs/` change)

- **Loser** → leave `NOTES.md` as the record; delete/keep the code, but it never entered `src/`.
- **Winner** → fold it into `src/`, write/update the spec in `specs/`, run the `critical-claim-verifier`
  agent on the changed file, then delete the experiment scaffolding. Spec **and** code land in the same
  atomic change (so master never carries a spec that lies about the code — see `tasks/README.md`).

## A backlog task for "try idea X" should name

- **Hypothesis / why** — the gradient pressure X creates that the current objective doesn't.
- **Where** — `experiments/EXP-NNN/`, which primitive/seam it subclasses.
- **Decision metric** — the recall comparison that graduates or kills it.
- **No `src/` or `specs/` changes** unless it wins.
