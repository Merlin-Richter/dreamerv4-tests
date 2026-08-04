#!/usr/bin/env python3
"""Cluster-free correctness gates for the community optional-memory model."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import subprocess
import sys
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXPECTED_VANILLA_SHA256 = "7b077938fec776c74e62201ab79194a7a06e10e54856c69d47b65dda6367d674"


def load_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_reference_module(root: Path):
    source = subprocess.check_output(
        ["git", "-C", str(root), "show", "HEAD:dreamer4/model.py"], text=True
    )
    module = types.ModuleType("d4_reference_model")
    module.__file__ = f"{root}@HEAD:dreamer4/model.py"
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def small_kwargs():
    return dict(
        d_model=32,
        d_bottleneck=4,
        d_spatial=8,
        n_spatial=3,
        n_register=2,
        n_agent=1,
        n_heads=4,
        depth=2,
        k_max=8,
        dropout=0.0,
        mlp_ratio=2.0,
        time_every=1,
        space_mode="wm_agent_isolated",
        scale_pos_embeds=False,
    )


def assert_state_equal(a, b, *, exclude=()):
    sa = {k: v for k, v in a.state_dict().items() if k not in exclude}
    sb = {k: v for k, v in b.state_dict().items() if k not in exclude}
    if sa.keys() != sb.keys():
        raise AssertionError(f"state keys differ: {sa.keys() ^ sb.keys()}")
    for key in sa:
        if not torch.equal(sa[key], sb[key]):
            raise AssertionError(f"shared initialization differs at {key}")


def model_inputs(model, B=2, T=6):
    z = torch.randn(B, T, model.n_spatial, model.d_spatial)
    actions = torch.randn(B, T, 16).clamp(-1, 1)
    mask = torch.zeros_like(actions)
    mask[..., :6] = 1
    step = torch.full((B, T), 3, dtype=torch.long)
    signal = torch.randint(0, 8, (B, T))
    return z, actions, mask, step, signal


def build_production(module, cfg, *, n_memory=0):
    kwargs = dict(
        d_model=int(cfg.get("d_model_dyn", 512)),
        d_bottleneck=32,
        d_spatial=64,
        n_spatial=8,
        n_register=int(cfg.get("n_register", 4)),
        n_agent=int(cfg.get("n_agent", 1)),
        n_heads=int(cfg.get("n_heads", 4)),
        depth=int(cfg.get("dyn_depth", 8)),
        k_max=int(cfg.get("k_max", 8)),
        dropout=float(cfg.get("dropout", 0.0)),
        mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
        time_every=int(cfg.get("time_every", 1)),
        space_mode=str(cfg.get("space_mode", "wm_agent_isolated")),
        scale_pos_embeds=bool(cfg.get("scale_pos_embeds", False)),
    )
    if "n_memory" in inspect.signature(module.Dynamics).parameters:
        kwargs["n_memory"] = n_memory
    return module.Dynamics(**kwargs).eval()


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dreamer4", type=Path, required=True)
    ap.add_argument("--vanilla-checkpoint", type=Path, default=None)
    ap.add_argument("--memory-smoke-checkpoint", type=Path, default=None,
                    help="Write a temporary n_memory=8 loader/player smoke checkpoint.")
    args = ap.parse_args()

    root = args.dreamer4.resolve()
    patched = load_file_module("d4_memory_model", root / "dreamer4" / "model.py")
    reference = load_reference_module(root)
    rollout = load_file_module("d4_mem2mem_rollout", HERE / "rollout.py")
    sampler_mod = load_file_module("d4_mem2mem_sampler", HERE / "sampler.py")

    # Gate 1a: n_memory=0 has exactly the vanilla state and forward path.
    torch.manual_seed(0)
    ref = reference.Dynamics(**small_kwargs()).eval()
    torch.manual_seed(0)
    vanilla = patched.Dynamics(**small_kwargs(), n_memory=0).eval()
    assert_state_equal(ref, vanilla)
    z, actions, mask, step_idx, signal_idx = model_inputs(vanilla)
    with torch.no_grad():
        y_ref = ref(actions, step_idx, signal_idx, z, act_mask=mask)[0]
        y_new = vanilla(actions, step_idx, signal_idx, z, act_mask=mask)[0]
    assert torch.equal(y_ref, y_new)
    print("[gate 1a] vanilla construction/forward parity max_abs=0")

    # Gate 2: adding memory never advances or perturbs shared initialization.
    torch.manual_seed(0)
    memory = patched.Dynamics(**small_kwargs(), n_memory=8).eval()
    assert_state_equal(vanilla, memory, exclude=("memory_tokens",))
    assert set(memory.state_dict()) - set(vanilla.state_dict()) == {"memory_tokens"}
    print("[gate 2] seed-0 shared initialization bit-identical; only memory_tokens is new")

    # Give the zero-initialized prediction head a deterministic probe weight.
    with torch.no_grad():
        probe_gen = torch.Generator().manual_seed(77)
        memory.flow_x_head.weight.normal_(std=0.03, generator=probe_gen)
        memory.flow_x_head.bias.normal_(std=0.01, generator=probe_gen)

    # Gate 3: shape plus exact future causality for predictions and written memory.
    z, actions, mask, step_idx, signal_idx = model_inputs(memory, B=2, T=6)
    memory_in = memory.blank_memory(2, 6, device=z.device, dtype=z.dtype).clone()
    with torch.no_grad():
        y1, _, m1 = memory(
            actions, step_idx, signal_idx, z, act_mask=mask,
            memory_in=memory_in, return_memory=True,
        )
        z2, a2, m2 = z.clone(), actions.clone(), memory_in.clone()
        z2[:, 4:] += 10 * torch.randn_like(z2[:, 4:])
        a2[:, 4:] = torch.randn_like(a2[:, 4:])
        m2[:, 4:] += 10 * torch.randn_like(m2[:, 4:])
        y2, _, mout2 = memory(
            a2, step_idx, signal_idx, z2, act_mask=mask,
            memory_in=m2, return_memory=True,
        )
    assert y1.shape == z.shape and m1.shape == (2, 6, 8, 32)
    assert torch.equal(y1[:, :4], y2[:, :4])
    assert torch.equal(m1[:, :4], mout2[:, :4])
    print("[gate 3] shapes correct; future perturbation changes earlier outputs by max_abs=0")

    # Gate 4: a later loss depends on graph-attached initial written memory, and a
    # deliberate detach before the first scored slide removes that path exactly.
    torch.manual_seed(123)
    z_long = torch.randn(2, 12, 3, 8)
    a_long = torch.randn(2, 12, 16).clamp(-1, 1)
    mask_long = torch.zeros_like(a_long); mask_long[..., :6] = 1
    gen = torch.Generator().manual_seed(999)
    result = rollout.mem2mem_rollout(
        memory, z_long, a_long, mask_long, window=4, clip_length=12,
        tbptt_frames=8, k_max=8, B_self=0, step=0,
        generator=gen, force_mode="memory", detach_boundaries=False,
    )
    relay_grad = torch.autograd.grad(result.loss, result.initial_memory, retain_graph=False)[0]
    relay_norm = float(relay_grad.norm())
    assert relay_norm > 0

    gen = torch.Generator().manual_seed(999)
    detached = rollout.mem2mem_rollout(
        memory, z_long, a_long, mask_long, window=4, clip_length=12,
        tbptt_frames=8, k_max=8, B_self=0, step=0,
        generator=gen, force_mode="memory", detach_boundaries=False,
        detach_before_slide=0,
    )
    detached_grad = torch.autograd.grad(
        detached.loss, detached.initial_memory, allow_unused=True
    )[0]
    detached_norm = 0.0 if detached_grad is None else float(detached_grad.norm())
    assert detached_norm == 0.0
    print(f"[gate 4] relay grad={relay_norm:.6e}; deliberate-detach grad={detached_norm:.1f}")

    # Gates 5/6: written memory is read and rewritten; action, register, and memory
    # channels all have independently measurable influence under matched inputs.
    with torch.no_grad():
        base_y, _, base_m = memory(
            actions, step_idx, signal_idx, z, act_mask=mask,
            memory_in=memory_in, return_memory=True,
        )
        perm = memory_in.flip(1).clone()
        perm[:, 0] += 0.5
        mem_y, _, mem_m = memory(
            actions, step_idx, signal_idx, z, act_mask=mask,
            memory_in=perm, return_memory=True,
        )
        action_y = memory(
            torch.zeros_like(actions), step_idx, signal_idx, z, act_mask=mask,
            memory_in=memory_in,
        )[0]
        registers = memory.register_tokens.detach().clone()
        memory.register_tokens.zero_()
        register_y = memory(
            actions, step_idx, signal_idx, z, act_mask=mask,
            memory_in=memory_in,
        )[0]
        memory.register_tokens.copy_(registers)
    effects = {
        "memory_prediction": float((base_y - mem_y).abs().max()),
        "memory_write": float((base_m - mem_m).abs().max()),
        "actions": float((base_y - action_y).abs().max()),
        "registers": float((base_y - register_y).abs().max()),
    }
    assert min(effects.values()) > 0
    print(f"[gates 5/6] distinct-channel matched-input effects: {effects}")

    # Gate 7: all post-initialization targets are unique, and one-segment
    # callback backward matches a monolithic backward exactly.
    assert result.scored_ranges == ((4, 6), (6, 8), (8, 10), (10, 12))
    scored = [i for a, b in result.scored_ranges for i in range(a, b)]
    assert scored == list(range(4, 12)) and len(scored) == len(set(scored))

    left = patched.Dynamics(**small_kwargs(), n_memory=2)
    right = patched.Dynamics(**small_kwargs(), n_memory=2)
    right.load_state_dict(left.state_dict())
    for candidate in (left, right):
        with torch.no_grad():
            candidate.flow_x_head.weight.normal_(std=0.02, generator=torch.Generator().manual_seed(5))
    short_z = torch.randn(2, 8, 3, 8)
    short_a = torch.randn(2, 8, 16)
    short_mask = torch.ones_like(short_a)
    mono = rollout.mem2mem_rollout(
        left, short_z, short_a, short_mask, window=4, clip_length=8,
        tbptt_frames=8, k_max=8, B_self=0, step=0,
        generator=torch.Generator().manual_seed(44), force_mode="latent",
        detach_boundaries=False,
    )
    mono.loss.backward()
    segmented = rollout.mem2mem_rollout(
        right, short_z, short_a, short_mask, window=4, clip_length=8,
        tbptt_frames=8, k_max=8, B_self=0, step=0,
        generator=torch.Generator().manual_seed(44), force_mode="latent",
        backward_fn=lambda value: value.backward(), detach_boundaries=True,
    )
    grad_diff = 0.0
    for lp, rp in zip(left.parameters(), right.parameters()):
        if lp.grad is None and rp.grad is None:
            continue
        grad_diff = max(grad_diff, float((lp.grad - rp.grad).abs().max()))
    assert grad_diff == 0.0
    print(f"[gate 7] unique loss accounting; segmented-vs-monolithic grad max_abs={grad_diff}")

    # Gate 10 (local sampler half): generate beyond window eviction, prove old
    # written memory affects the next latent, and prove commit noise cannot shift
    # the target denoising noise stream relative to vanilla.
    sampler = sampler_mod.CarryingSampler(
        memory,
        k_max=8,
        K=4,
        context_window=3,
        sample_generator=torch.Generator().manual_seed(321),
        commit_generator=torch.Generator().manual_seed(654),
    )
    context = z[:1, :3]
    context_actions = actions[:1, :3]
    context_masks = mask[:1, :3]
    sampler.initialize(context, context_actions, context_masks)
    for index in range(5):
        generated = sampler.sample_next(actions[0, index], mask[0, index])
        assert torch.isfinite(generated.latent).all()
        assert generated.written_memory is not None
    sample_state = sampler.sample_generator.get_state()
    normal = sampler.sample_next(actions[0, 0], mask[0, 0], commit=False).latent
    sampler.sample_generator.set_state(sample_state)
    blank_override = memory.blank_memory(
        1, 3, device=context.device, dtype=context.dtype
    )
    blanked = sampler.sample_next(
        actions[0, 0], mask[0, 0], memory_override=blank_override, commit=False
    ).latent
    eviction_effect = float((normal - blanked).abs().max())
    assert eviction_effect > 0

    vanilla_from_memory = patched.Dynamics(**small_kwargs(), n_memory=0).eval()
    vanilla_from_memory.load_state_dict(
        {k: v for k, v in memory.state_dict().items() if k != "memory_tokens"}, strict=True
    )
    vanilla_sampler = sampler_mod.CarryingSampler(
        vanilla_from_memory,
        k_max=8,
        K=4,
        context_window=3,
        sample_generator=torch.Generator().manual_seed(777),
        commit_generator=torch.Generator().manual_seed(888),
    )
    memory_sampler = sampler_mod.CarryingSampler(
        memory,
        k_max=8,
        K=4,
        context_window=3,
        sample_generator=torch.Generator().manual_seed(777),
        commit_generator=torch.Generator().manual_seed(888),
    )
    vanilla_sampler.initialize(context, context_actions, context_masks)
    memory_sampler.initialize(context, context_actions, context_masks)
    vanilla_sampler.sample_next(actions[0, 0], mask[0, 0])
    memory_sampler.sample_next(actions[0, 0], mask[0, 0])
    assert torch.equal(
        vanilla_sampler.sample_generator.get_state(), memory_sampler.sample_generator.get_state()
    )
    print(
        f"[gate 10a] finite beyond eviction; blank-after-eviction effect={eviction_effect:.6e}; "
        "vanilla/memory target-noise generator states match"
    )

    if args.vanilla_checkpoint is not None:
        checkpoint = args.vanilla_checkpoint.resolve()
        actual_sha = sha256(checkpoint)
        assert actual_sha == EXPECTED_VANILLA_SHA256, actual_sha
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = dict(payload["args"])
        ref_prod = build_production(reference, cfg)
        new_prod = build_production(patched, cfg)
        ref_prod.load_state_dict(payload["dynamics"], strict=True)
        new_prod.load_state_dict(payload["dynamics"], strict=True)
        torch.manual_seed(808)
        z_prod = torch.randn(1, 4, 8, 64)
        a_prod = torch.randn(1, 4, 16).clamp(-1, 1)
        m_prod = torch.zeros_like(a_prod); m_prod[..., :6] = 1
        s_prod = torch.full((1, 4), 3, dtype=torch.long)
        g_prod = torch.randint(0, 8, (1, 4))
        with torch.no_grad():
            ref_out = ref_prod(a_prod, s_prod, g_prod, z_prod, act_mask=m_prod)[0]
            new_out = new_prod(a_prod, s_prod, g_prod, z_prod, act_mask=m_prod)[0]
        max_abs = float((ref_out - new_out).abs().max())
        assert max_abs == 0.0
        print(
            f"[gate 1b] approved vanilla checkpoint strict-load parity max_abs={max_abs:.1f} "
            f"sha256={actual_sha}"
        )
        if args.memory_smoke_checkpoint is not None:
            smoke_model = build_production(patched, cfg, n_memory=8)
            incompatible = smoke_model.load_state_dict(payload["dynamics"], strict=False)
            assert incompatible.missing_keys == ["memory_tokens"]
            assert incompatible.unexpected_keys == []
            smoke_payload = dict(payload)
            smoke_payload["args"] = dict(cfg, n_memory=8)
            smoke_payload["dynamics"] = smoke_model.state_dict()
            args.memory_smoke_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(smoke_payload, args.memory_smoke_checkpoint)
            print(f"[gate 10b prep] memory player smoke checkpoint -> {args.memory_smoke_checkpoint}")

    print("MODEL/ROLLOUT VALIDATION PASSED")


if __name__ == "__main__":
    main()
