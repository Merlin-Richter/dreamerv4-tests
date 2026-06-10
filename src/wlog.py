"""
Lightweight Weights & Biases logging shared by the training scripts.

Design goals:
  * **No-op unless ``--wandb`` is passed**, so existing ``python train_*.py`` runs are
    byte-for-byte unchanged and don't need wandb installed.
  * One ``wandb.config`` that merges the model's config dataclass (``asdict``) with the
    argparse hyperparameters, so a run page fully describes how it was launched.
  * Metrics only -- checkpoints stay local. Add artifact upload later if you want hosted ckpts.

Usage in a train script:

    import wlog
    ...
    parser = argparse.ArgumentParser()
    wlog.add_args(parser)            # adds --wandb, --wandb-project, ...
    args = parser.parse_args()
    ...
    wlog.init(args, cfg, project="transformer-C-tokenizer")   # cfg = your config dataclass
    ...
    wlog.log({"train/mse": train_mse, "val/mse": val_mse}, step=epoch)
    ...
    wlog.finish()

Entity/project resolve from (in order) the CLI flag, the ``project=`` default passed to
``init``, then ``$WANDB_ENTITY`` / ``$WANDB_PROJECT``. No entity is hardcoded.
"""
from __future__ import annotations

import os
from argparse import ArgumentParser
from dataclasses import asdict, is_dataclass
from pathlib import Path

_run = None


def add_args(parser: ArgumentParser) -> None:
    """Add the wandb CLI flags to an existing argparse parser."""
    g = parser.add_argument_group("wandb")
    g.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    g.add_argument("--wandb-project", default=None,
                   help="W&B project (default: the script's per-model project, or $WANDB_PROJECT).")
    g.add_argument("--wandb-entity", default=None,
                   help="W&B entity/team (default: $WANDB_ENTITY).")
    g.add_argument("--wandb-name", default=None, help="Run name (default: W&B auto-generates).")
    g.add_argument("--wandb-tags", default=None, help="Comma-separated run tags.")


def _jsonable(v):
    """Make argparse values safe for wandb.config (Path -> str)."""
    return str(v) if isinstance(v, Path) else v


def init(args, config=None, project: str | None = None):
    """Start a run if ``args.wandb`` is set; otherwise return None (logging stays a no-op).

    ``config`` may be a dataclass instance (e.g. AutoEncoderConfig) or a plain dict; its
    fields plus the argparse hyperparameters become ``wandb.config``.
    """
    global _run
    if not getattr(args, "wandb", False):
        return None

    import wandb  # lazy import: only needed when --wandb is actually passed

    cfg: dict = {}
    if config is not None:
        cfg.update(asdict(config) if is_dataclass(config) else dict(config))
    for k, v in vars(args).items():
        if k.startswith("wandb"):
            continue
        cfg.setdefault(k, _jsonable(v))

    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()] if args.wandb_tags else None

    _run = wandb.init(
        entity=args.wandb_entity or os.environ.get("WANDB_ENTITY"),
        project=(args.wandb_project or project or os.environ.get("WANDB_PROJECT") or "transformer"),
        name=args.wandb_name,
        tags=tags,
        config=cfg,
    )
    return _run


def log(metrics: dict, step: int | None = None) -> None:
    """Log a dict of scalar metrics. No-op when wandb is disabled."""
    if _run is None:
        return
    _run.log(metrics, step=step)


def finish() -> None:
    """Close the run and flush. No-op when wandb is disabled."""
    global _run
    if _run is not None:
        _run.finish()
        _run = None
