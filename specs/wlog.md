# wlog.py — tiny shared Weights & Biases logger for the training scripts.

A no-op unless `--wandb` is passed (so runs need neither wandb installed nor any code change). Metrics
only — checkpoints stay local.

## Interface
- `add_args(parser)` — adds `--wandb`, `--wandb-project/-entity/-name/-tags`.
- `init(args, config=None, project=None) -> run|None` — start a run iff `args.wandb`.
- `log(metrics: dict, step=None)` — log scalars (no-op if disabled).
- `finish()` — close the run (no-op if disabled).

## Behavior
- `init` builds `wandb.config` from the model config dataclass (`asdict`) merged with the argparse
  hyperparameters, so a run page fully describes the launch. Lazy `import wandb` (only when enabled).
- Entity/project resolve in order: CLI flag → `init(project=)` default → `$WANDB_ENTITY/$WANDB_PROJECT`.
  No entity hardcoded.

## Invariants
- Disabled (`--wandb` absent) ⇒ every function is a no-op; never import wandb. Module-level singleton run.
