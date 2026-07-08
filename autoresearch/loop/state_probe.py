"""Carried-state byte probe for the autoresearch loop (NOT agent-editable).

Replaces the hard window pin. The rule: attend to / remember ANYTHING, but the state a
model carries across rollout steps must fit a fixed byte budget. This prices memory the
way reality does (state = VRAM + bandwidth), while leaving eviction-exempt token banks,
compression schemes, sparse deep-past attention etc. fully legal.

Method: build the eval adapter, begin() on a real 192-tick prefix (saturates any sliding
window), then step it; sweep the adapter's object graph for tensors/ndarrays (model
params/buffers excluded — weights are priced by the training budget, not state) at step 8
and step 64. Reports:
  state_bytes            max carried bytes observed
  state_growth_per_step  (bytes@64 - bytes@8)/56 — >0 means unbounded state: FAIL
  state_check            PASS iff state_bytes <= STATE_BUDGET_BYTES and growth ~0

Honesty note: an object-graph sweep can be evaded by stashing tensors in globals. The
backstop is the same as everything else in this harness: kept diffs get human review,
and the per-age-bin curve makes buffer-shaped gains visually obvious.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from autoresearch.editable.adapter_sym import make_adapter            # noqa: E402
from autoresearch.editable.train_sym import load_split, render_grid   # noqa: E402

# Budget: 2x the seed recipe's measured carried state at W=16 (dense W=64 ~ 4x -> illegal;
# headroom for real mechanisms). Re-baseline deliberately, never silently.
STATE_BUDGET_BYTES = 518400  # 1.5x the seed recipe's measured 345600 B @ f1544f3 (W=16 dense)
GROWTH_TOL_BYTES = 1024  # per-step growth above this = unbounded state


def sweep_bytes(root, skip_ids):
    seen, todo, total = set(), [root], 0
    while todo:
        o = todo.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, torch.Tensor):
            if id(o) not in skip_ids:
                total += o.numel() * o.element_size()
            continue
        if isinstance(o, np.ndarray):
            total += o.nbytes
            continue
        if isinstance(o, torch.nn.Module):
            skip = {id(p) for p in o.parameters()} | {id(b) for b in o.buffers()}
            skip_ids |= skip
            todo.extend(v for v in o.__dict__.values())
            continue
        if isinstance(o, dict):
            todo.extend(o.keys()); todo.extend(o.values())
        elif isinstance(o, (list, tuple, set, frozenset)):
            todo.extend(o)
        elif hasattr(o, "__dict__"):
            todo.extend(o.__dict__.values())
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val", default="data/colorfield_sym_val")
    ap.add_argument("--budget", type=int, default=STATE_BUDGET_BYTES,
                    help="bytes; 0 = measure-only (prints, no gate)")
    args = ap.parse_args()

    maps, positions, actions = load_split(Path(args.val))
    actions = actions.numpy() if hasattr(actions, "numpy") else np.asarray(actions)
    m, pos, act = maps[0], positions[0], actions[0]
    grids = np.stack([render_grid(m, pos[t]) for t in range(192)])

    factory = make_adapter(args.checkpoint)
    adapter = factory(None)
    adapter.begin(grids, act[:192].astype(np.int64))

    def measure():
        return sweep_bytes(adapter, set())

    b8 = b64 = 0
    for i in range(64):
        adapter.step(int(act[192 + i]))
        if i == 7:
            b8 = measure()
        if i == 63:
            b64 = measure()
    growth = (b64 - b8) / 56.0
    state_bytes = max(b8, b64)

    print(f"state_bytes:      {state_bytes}")
    print(f"state_growth_bps: {growth:.1f}")
    if args.budget > 0:
        ok = state_bytes <= args.budget and growth <= GROWTH_TOL_BYTES
        print(f"state_budget:     {args.budget}")
        print(f"state_check:      {'PASS' if ok else 'FAIL'}")
    else:
        print("state_check:      MEASURE_ONLY")


if __name__ == "__main__":
    main()
