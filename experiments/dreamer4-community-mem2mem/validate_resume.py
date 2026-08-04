#!/usr/bin/env python3
"""Exact checkpoint/resume gate for model, optimizer, RNG, and data position."""
from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def equal_nested(left, right, path="root"):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            equal_nested(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            equal_nested(a, b, f"{path}[{index}]")
    else:
        assert left == right, f"{path}: {left!r} != {right!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dreamer4", type=Path, required=True)
    args = ap.parse_args()
    source = args.dreamer4.resolve() / "dreamer4"
    sys.path.insert(0, str(source))
    import model as d4
    rollout = load_module("resume_rollout", HERE / "rollout.py")
    trainer = load_module("resume_trainer_helpers", HERE / "train_mem2mem.py")

    kwargs = dict(
        d_model=24, d_bottleneck=4, d_spatial=8, n_spatial=3,
        n_register=2, n_agent=1, n_heads=4, depth=2, k_max=8,
        n_memory=2, dropout=0.0, mlp_ratio=2.0, time_every=1,
        space_mode="wm_agent_isolated", scale_pos_embeds=False,
    )

    def new_run():
        trainer.seed_everything(0)
        network = d4.Dynamics(**kwargs)
        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4, weight_decay=1e-2)
        generator = torch.Generator().manual_seed(1234)
        return network, optimizer, generator

    def batch(step):
        generator = torch.Generator().manual_seed(90_000 + step)
        z = torch.randn(2, 8, 3, 8, generator=generator)
        actions = torch.randn(2, 8, 16, generator=generator).clamp(-1, 1)
        masks = torch.zeros_like(actions); masks[..., :6] = 1
        return z, actions, masks

    def train_step(network, optimizer, generator, step):
        z, actions, masks = batch(step)
        optimizer.zero_grad(set_to_none=True)
        result = rollout.mem2mem_rollout(
            network, z, actions, masks,
            window=4, clip_length=8, tbptt_frames=8, k_max=8,
            B_self=0, step=step, bootstrap_start=5_000,
            generator=generator, force_mode="latent",
            backward_fn=lambda loss: loss.backward(), detach_boundaries=True,
        )
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()
        return result.mean_loss

    full_model, full_opt, full_gen = new_run()
    full_losses = [train_step(full_model, full_opt, full_gen, step) for step in range(2)]

    split_model, split_opt, split_gen = new_run()
    split_losses = [train_step(split_model, split_opt, split_gen, 0)]
    active_seconds = 12.5
    checkpoint = Path(tempfile.mkdtemp(prefix="d4resume_")) / "resume.pt"
    trainer.atomic_save({
        "step": 1,
        "active_seconds": active_seconds,
        "dynamics": split_model.state_dict(),
        "opt": split_opt.state_dict(),
        "rng": trainer.rng_state(split_gen),
        "resolved_config": {"window": 4, "clip_length": 8, "tbptt_frames": 8},
        "data_position": {"counter_batch_step": 1, "seed": 0},
    }, checkpoint)

    resumed_model, resumed_opt, resumed_gen = new_run()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    resumed_model.load_state_dict(payload["dynamics"], strict=True)
    resumed_opt.load_state_dict(payload["opt"])
    trainer.restore_rng(payload["rng"], resumed_gen)
    assert payload["step"] == payload["data_position"]["counter_batch_step"] == 1
    assert payload["active_seconds"] == active_seconds
    split_losses.append(train_step(resumed_model, resumed_opt, resumed_gen, payload["step"]))

    equal_nested(full_model.state_dict(), resumed_model.state_dict(), "model")
    equal_nested(full_opt.state_dict(), resumed_opt.state_dict(), "optimizer")
    assert torch.equal(full_gen.get_state(), resumed_gen.get_state())
    assert full_losses == split_losses
    checkpoint.unlink()
    checkpoint.parent.rmdir()
    print(
        f"RESUME DETERMINISM PASSED step=2 active_seconds={active_seconds} "
        f"losses={full_losses} temporary_checkpoint_cleaned=true"
    )


if __name__ == "__main__":
    main()
