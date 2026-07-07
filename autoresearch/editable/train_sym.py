"""Budgeted from-scratch trainer for the autoresearch loop. EDITABLE LAYER —
SYMBOLIC tier (ColorField-SYM; no tokenizer anywhere).

Trains DynamicsModel (autoresearch/editable/model.py — UNCHANGED, shared with the
pixel tier) FROM SCRATCH on the procedural ColorField-SYM sidecars, with the
mem2mem sliding-rollout objective (autoresearch/editable/rollout.py — also
shared) — the GridWorld WINNER recipe by default: rollout-only mem2mem
(mem2mem_frac=1.0), bootstrap OFF (d_min-only sampling), FF9 OFF.

The one-hot viewport rows ARE the latents (spec:
tasks/in-progress/colorfield-sym-frozen-layer.md, "Model port"): each frame's
(5,5) grid of cell ids 0..5 becomes 5 row tokens of 35 dims — 5 cells x 6
one-hot (=30) + the frame's phase (t % 5) one-hot (5 dims) appended to EVERY
row. The x-prediction target includes the phase dims (trivially predictable,
harmless). Encoding lives in adapter_sym.encode_latents — ONE codec for train
and eval, so they can never drift. Latents are built on the fly per clip (no
cache: procedural render + one-hot is trivially cheap).

================================================================================
PINNED CONFIG CONTRACT — the driver PROBES these; violating them zeroes the score:

  * max_temporal_length W = 16      <<< THE WINDOW PIN. DO NOT RAISE. >>>
      W is in TICKS; under the sym tier's phase-5 dilation that is only 3.2
      effective moves — every age bin beyond W=16 must come from a CARRIED
      memory mechanism (the driver's window probe enforces bit-identity when a
      frame > W back is perturbed; a violation scores 0 and is flagged).
  * n_actions = 5                   ColorField: up/down/left/right/stay.
  * n_latents = 5, bottleneck_dim = 35   THE CODEC (one token per viewport row;
      5 cells x 6 one-hot + 5 phase dims). No tokenizer to match — the codec
      is exact by construction.
================================================================================

Data: procedural sidecars under --data / --val (maps.npy, starts.npy,
actions.npy, ... — autoresearch/frozen_sym/datagen.py). Grids are rendered per
clip from precomputed per-tick centers (vectorized path integral, cross-checked
against the frozen env.positions_from); clips of --clip-len are sliced at
RANDOM offsets. Phase one-hots use the ABSOLUTE episode tick (offset + i).

Budget: --budget-s is the wall-clock budget in seconds, measured from process
start and checked after EVERY optimizer step; on expiry the model is saved and a
final "BUDGET_STOP step=N elapsed=S" line is printed. --epochs is a secondary cap
("EPOCHS_DONE ..."). NOTE: the LR schedule (warmup -> flat -> cosine over the
last 20%) is laid out over len(train_loader) * --epochs steps (or --sched-steps
if given) — size --epochs / --sched-steps so the budget lands near the cosine
tail, otherwise training dies mid-flat (or worse, mid-warmup).

Checkpoint payload (the adapter and driver reload from this — keep the keys):
  {"model_state_dict": ..., "config": asdict(DynamicsModelConfig)}

Run (repo root; STRICTLY CPU when the GPU is owned by another run —
CUDA_VISIBLE_DEVICES=-1):
  venv/Scripts/python.exe -u autoresearch/editable/train_sym.py \
    --data data/colorfield_sym --val data/colorfield_sym_val \
    --checkpoint autoresearch/runs/<tag>/dynamics_sym.pt --budget-s 600
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

try:  # package import (the driver's path)
    from . import adapter_sym as codec  # THE shared one-hot codec (encode_latents + dims)
    from .model import DynamicsModel, DynamicsModelConfig
    from .rollout import mem2mem_rollout_loss
    from ..frozen_sym.datagen import ColorFieldSymDataset
    from ..frozen_sym.env import (BOARD, DELTAS, PHASE_PERIOD, STAY,
                                  positions_from, render_grid)
except ImportError:  # run as a script: python autoresearch/editable/train_sym.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from autoresearch.editable import adapter_sym as codec
    from autoresearch.editable.model import DynamicsModel, DynamicsModelConfig
    from autoresearch.editable.rollout import mem2mem_rollout_loss
    from autoresearch.frozen_sym.datagen import ColorFieldSymDataset
    from autoresearch.frozen_sym.env import (BOARD, DELTAS, PHASE_PERIOD, STAY,
                                             positions_from, render_grid)

encode_latents = codec.encode_latents

# --- THE PINS (see module docstring; the driver verifies these) ---------------
W_PIN = 16            # max_temporal_length — THE WINDOW PIN, do not raise
N_ACTIONS = 5         # ColorField action space (shared with the pixel tier)
N_LATENTS = 5         # one token per viewport ROW
BOTTLENECK_DIM = 35   # 5 cells x 6 one-hot (=30) + 5 phase one-hot dims per row
# The pins must equal the shared codec's geometry (adapter_sym derives it from the
# frozen_sym env) — train-time and eval-time encodings can never drift apart.
assert (N_LATENTS, BOTTLENECK_DIM) == (codec.N_LATENTS, codec.BOTTLENECK_DIM), \
    (N_LATENTS, BOTTLENECK_DIM, codec.N_LATENTS, codec.BOTTLENECK_DIM)


# ------------------------------------------------------------------ data
def load_split(data_dir: Path):
    """(maps (N,15,15) uint8, per-tick centers (N,T,2) int64, long actions (N,T))
    for one procedural dataset dir. Centers are the vectorized path integral of
    the action sidecar under the phase-5 rule; the dataset conventions
    (actions[:,0]==STAY, off-phase STAY, on-board centers) are asserted, and
    episode 0 is cross-checked against the frozen env.positions_from."""
    ds = ColorFieldSymDataset(data_dir)
    actions_np = ds.actions
    n, t = actions_np.shape
    assert int(actions_np.max()) < N_ACTIONS, f"action id >= {N_ACTIONS} in {data_dir}"
    assert (actions_np[:, 0] == STAY).all(), f"actions[:,0] != STAY in {data_dir}"
    off_phase = np.arange(t) % PHASE_PERIOD != 0
    assert (actions_np[:, off_phase] == STAY).all(), \
        f"off-phase non-STAY action in {data_dir} (phase-5 convention violated)"

    deltas = np.zeros((N_ACTIONS, 2), dtype=np.int64)
    for a, d in DELTAS.items():
        deltas[a] = d
    steps = deltas[actions_np]                       # (N, T, 2); STAY rows are (0,0)
    pos = ds.starts[:, None, :].astype(np.int64) + np.cumsum(steps, axis=1)
    assert np.array_equal(pos[0], positions_from(tuple(ds.starts[0]), actions_np[0])), \
        "vectorized path integral drifted from the frozen env.positions_from"
    assert pos.min() >= 0 and pos.max() < BOARD, f"off-board center in {data_dir}"
    return ds.maps, pos, torch.from_numpy(actions_np.astype(np.int64))


class SymClipDataset(Dataset):
    """Fixed-length clips of one-hot latents, rendered ON THE FLY from the
    procedural sidecars (the symbolic analogue of the pixel tier's
    RandomClipDataset over the latent cache — same offset semantics).

    random_offsets=True (train): each access draws a fresh uniform start offset.
    Offsets use torch's RNG (properly re-seeded per DataLoader worker; numpy's
    state would fork identically into workers). One epoch = T // clip_len clips
    per episode (coverage-equivalent to fixed chunking).
    random_offsets=False (val): deterministic chunking at offsets j * clip_len.
    Grids come from the frozen env's render_grid at precomputed centers; phase
    one-hots use the ABSOLUTE episode tick (offset + i)."""

    def __init__(self, maps: np.ndarray, positions: np.ndarray, actions: torch.Tensor,
                 clip_len: int, random_offsets: bool = True) -> None:
        self.maps = maps            # (N, 15, 15) uint8
        self.positions = positions  # (N, T, 2) int64
        self.actions = actions      # (N, T) long
        self.clip_len = int(clip_len)
        n, t = actions.shape
        assert t >= self.clip_len, f"episode length {t} < clip_len {self.clip_len}"
        self.clips_per_ep = max(1, t // self.clip_len)
        self.max_start = t - self.clip_len
        self.random_offsets = random_offsets

    def __len__(self) -> int:
        return self.actions.shape[0] * self.clips_per_ep

    def __getitem__(self, idx: int):
        ep, j = divmod(idx, self.clips_per_ep)
        if self.random_offsets and self.max_start > 0:
            s = int(torch.randint(0, self.max_start + 1, (1,)).item())
        else:
            s = j * self.clip_len
        m = self.maps[ep]
        grids = np.empty((self.clip_len, 5, 5), dtype=np.uint8)
        for i in range(self.clip_len):
            grids[i] = render_grid(m, self.positions[ep, s + i])
        z = torch.from_numpy(encode_latents(grids, np.arange(s, s + self.clip_len)))
        a = self.actions[ep, s:s + self.clip_len].clone()
        return z, a


# ------------------------------------------------------------------ recipe helpers
def step_curriculum(p: float, n_d: int, warmup: float, add_every: float) -> int:
    """Number of FINEST step sizes unlocked at training fraction p in [0,1). 1 (only d_min)
    for the warmup, then +1 every ``add_every``, capped at n_d. Finest-first so a coarse
    step's bootstrap target (a one-finer step) is always already trained. Only used when
    --bootstrap is on (the winner default is bootstrap OFF => d_min only)."""
    if p < warmup:
        return 1
    return min(n_d, 2 + int((p - warmup) / add_every))


def valid_n_ctx(N: int, clip_len: int) -> list:
    """Powers of two in [4, N] that fit at least one slide of the clip (>= 1.5*n_ctx frames)."""
    out, w = [], 4
    while w <= N:
        if w + w // 2 <= clip_len:
            out.append(w)
        w *= 2
    return out or [min(4, N)]


# ------------------------------------------------------------------ main
def main():
    t0 = time.perf_counter()  # the budget clock includes ALL setup (realistic driver cost)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/colorfield_sym"))
    p.add_argument("--val", type=Path, default=Path("data/colorfield_sym_val"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--budget-s", type=float, required=True,
                   help="Wall-clock budget in seconds (from process start; checked per step).")
    p.add_argument("--epochs", type=int, default=50, help="Secondary cap; also the LR-schedule "
                   "horizon unless --sched-steps is given (size it to the budget).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip-len", type=int, default=64, help="Long-clip length fed to the rollout.")
    # model dims (defaults = the ~7.5M GridWorld dynamics sizing)
    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=9, help="Multiple of 3 ([spatial,temporal,spatial]).")
    p.add_argument("--n-heads", type=int, default=16)
    p.add_argument("--gqa-groups", type=int, default=1)
    p.add_argument("--n-registers", type=int, default=4)
    p.add_argument("--n-memory", type=int, default=4, help="0 => vanilla no-memory reference arm "
                   "(mem2mem is then disabled; pure windowed shortcut-forcing loss).")
    # objective (defaults = the rollout-only mem2mem WINNER)
    p.add_argument("--mem2mem-frac", type=float, default=1.0, help="P(batch uses mem2mem vs normal).")
    p.add_argument("--ff9", type=int, default=0, metavar="K",
                   help="FF9 sufficiency lookahead; 0 = OFF (winner; no-FF9 proved sufficient).")
    p.add_argument("--bootstrap", action="store_true",
                   help="Enable shortcut bootstrap distillation + the d curriculum. Default OFF "
                        "(winner): d_min-only sampling, pure x-prediction flow loss.")
    p.add_argument("--no-curriculum", action="store_true",
                   help="With --bootstrap: sample all d steps from step 0 (no finest-first ramp).")
    p.add_argument("--curr-warmup", type=float, default=0.15)
    p.add_argument("--curr-add-every", type=float, default=0.025)
    p.add_argument("--ff9-norm-flow", action="store_true",
                   help="Normalize FF9 by the pure d_min flow magnitude (bootstrap-invariant weight).")
    p.add_argument("--relay-grad-clip", type=float, default=None, metavar="C",
                   help="Per-hop relay gradient normalizer (see rollout.py). None = OFF.")
    p.add_argument("--tbptt-frames", type=int, default=None,
                   help="Detach the memory relay past this many frames (default 2*W).")
    p.add_argument("--max-frames", type=int, default=None, help="Cap rollout length per clip.")
    p.add_argument("--snapshot-at", type=lambda s: {int(x) for x in s.split(",")},
                   default=None,
                   help="Comma-separated step counts at which to save side checkpoints "
                        "(<ckpt>_stepN.pt) — for steps-vs-quality calibration curves.")
    p.add_argument("--fixed-n-ctx", action="store_true",
                   help="Always slide at n_ctx = W_PIN instead of sampling {4,8,16}: fewer, "
                        "fatter, less-serialized forwards and matches the eval's full-window "
                        "distribution; costs relay-hop diversity (untested recipe ingredient).")
    p.add_argument("--sched-steps", type=int, default=None,
                   help="LR-schedule horizon in optimizer steps (default len(loader)*epochs).")
    p.add_argument("--val-batches", type=int, default=32,
                   help="Cap per-epoch val batches (keeps val cheap under the budget).")
    p.add_argument("--num-workers", type=int, default=0,
                   help="0 (default): per-clip render + one-hot is tiny numpy work; workers "
                        "only add Windows-spawn overhead per epoch under a wall-clock budget.")
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    gen = torch.Generator(device=device).manual_seed(args.seed)

    maps, positions, actions = load_split(args.data)
    val_maps, val_positions, val_actions = load_split(args.val)
    clip_len = max(args.clip_len, W_PIN)

    cfg = DynamicsModelConfig(
        bottleneck_dim=BOTTLENECK_DIM, n_latents=N_LATENTS,
        max_temporal_length=W_PIN, n_actions=N_ACTIONS,
        embedding_dim=args.embedding_dim, depth=args.depth, n_heads=args.n_heads,
        gqa_groups=args.gqa_groups, n_registers=args.n_registers,
        n_memory=args.n_memory, ff9_k=args.ff9)
    assert cfg.max_temporal_length == W_PIN, "the window pin is not negotiable"
    model = DynamicsModel(cfg).to(device)

    mem2mem_frac = args.mem2mem_frac
    if cfg.n_memory == 0 and mem2mem_frac > 0:
        print("[train-sym] n_memory=0 -> vanilla reference arm: forcing mem2mem_frac=0.0")
        mem2mem_frac = 0.0
    use_ff9 = args.ff9 > 0

    train_ds = SymClipDataset(maps, positions, actions, clip_len, random_offsets=True)
    val_ds = SymClipDataset(val_maps, val_positions, val_actions, clip_len,
                            random_offsets=False)
    lk = dict(num_workers=args.num_workers, pin_memory=use_amp)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=len(train_ds) >= args.batch_size, **lk)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **lk)
    assert len(train_loader) > 0, "no training batches (dataset smaller than batch size?)"

    nparams = sum(q.numel() for q in model.parameters())
    ncts = valid_n_ctx(W_PIN, clip_len)
    print(f"device={device} params={nparams / 1e6:.2f}M W={W_PIN} (PINNED) n_actions={N_ACTIONS} "
          f"codec=onehot({N_LATENTS}x{BOTTLENECK_DIM}) train_eps={maps.shape[0]} "
          f"val_eps={val_maps.shape[0]} clip_len={clip_len} n_ctx choices={ncts} "
          f"mem2mem_frac={mem2mem_frac} bootstrap={args.bootstrap} use_ff9={use_ff9} "
          f"ff9_k={args.ff9} n_memory={cfg.n_memory} budget_s={args.budget_s}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.sched_steps or max(1, len(train_loader) * args.epochs)
    if args.sched_steps:
        # Budget-sized run: the recipe's 200-step warmup floor would eat a short
        # budget whole (an ~85-step run never leaves warmup) — 10% capped at 200.
        warmup = max(10, min(200, int(0.1 * total_steps)))
    else:
        warmup = max(200, int(0.05 * total_steps))     # recipe path, unchanged
    decay_start = int(0.8 * total_steps)
    emr = 1e-6 / args.lr

    def lr_lambda(s):
        if s < warmup:
            return (s + 1) / warmup
        if s < decay_start:
            return 1.0
        q = (s - decay_start) / max(1, total_steps - decay_start)
        return emr + (1 - emr) * 0.5 * (1 + np.cos(np.pi * q))

    sched = LambdaLR(opt, lr_lambda)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    def save():
        torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg)},
                   args.checkpoint)

    gstep = 0
    budget_hit = False
    last_unlocked = 1
    for epoch in range(args.epochs):
        model.train()
        agg = {"normal": 0.0, "mem2mem": 0.0, "flow": 0.0, "ff9": 0.0, "n_m": 0, "n_n": 0}
        for batch in train_loader:
            z1, acts = batch[0].to(device), batch[1].to(device)
            if not args.bootstrap:
                n_unlocked = 1                      # winner: d_min only, pure flow
            elif args.no_curriculum:
                n_unlocked = None
            else:
                n_unlocked = step_curriculum(gstep / total_steps, model.n_d,
                                             args.curr_warmup, args.curr_add_every)
            last_unlocked = n_unlocked if n_unlocked is not None else model.n_d
            use_m2m = torch.rand(1, generator=gen, device=device).item() < mem2mem_frac
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                if use_m2m:
                    if args.fixed_n_ctx:
                        W = W_PIN   # fixed full-window slides: fewest, fattest forwards
                    else:
                        W = ncts[torch.randint(len(ncts), (1,), generator=gen, device=device).item()]
                    loss, parts = mem2mem_rollout_loss(
                        model, z1, acts, n_ctx=W, device=device, gen=gen,
                        tbptt_frames=args.tbptt_frames, max_frames=args.max_frames,
                        bootstrap=args.bootstrap, n_d_unlocked=n_unlocked,
                        use_ff9=use_ff9, ff9_norm_flow=args.ff9_norm_flow,
                        relay_grad_clip=args.relay_grad_clip)
                    agg["mem2mem"] += float(loss.detach()); agg["n_m"] += 1
                    agg["flow"] += parts["flow"]; agg["ff9"] += parts["ff9"]
                else:
                    off = int(torch.randint(0, clip_len - W_PIN + 1, (1,),
                                            generator=gen, device=device))
                    loss = model.loss(z1[:, off:off + W_PIN], acts[:, off:off + W_PIN])
                    agg["normal"] += float(loss.detach()); agg["n_n"] += 1
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); gstep += 1
            if args.snapshot_at and gstep in args.snapshot_at:
                sp = args.checkpoint.with_name(args.checkpoint.stem + f"_step{gstep}.pt")
                torch.save({"model_state_dict": model.state_dict(),
                            "config": asdict(cfg)}, sp)
                print(f"[snapshot] step {gstep} -> {sp} "
                      f"(elapsed {time.perf_counter() - t0:.0f}s)", flush=True)
            if time.perf_counter() - t0 >= args.budget_s:   # THE budget check (per step)
                budget_hit = True
                break

        # --- light val: normal shortcut-forcing loss on a fixed window (skip if expired) ---
        vloss = float("nan")
        if not budget_hit:
            model.eval()
            vsum, nb = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    if nb >= args.val_batches:
                        break
                    z1, acts = batch[0].to(device), batch[1].to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                        vsum += float(model.loss(z1[:, :W_PIN], acts[:, :W_PIN]))
                    nb += 1
            vloss = vsum / max(1, nb)
        nm, nn = max(1, agg["n_m"]), max(1, agg["n_n"])
        elapsed = time.perf_counter() - t0
        print(f"Epoch {epoch + 1}/{args.epochs} | steps {gstep} | elapsed {elapsed:.1f}s | "
              f"val(normal) {vloss:.5f} | train mem2mem {agg['mem2mem'] / nm:.5f} "
              f"(flow {agg['flow'] / nm:.4f} ff9 {agg['ff9'] / nm:.4f}) "
              f"| train normal {agg['normal'] / nn:.5f} | d_unlocked {last_unlocked}/{model.n_d} "
              f"| lr {opt.param_groups[0]['lr']:.2e}", flush=True)
        save()
        if budget_hit:
            break

    elapsed = time.perf_counter() - t0
    save()
    if budget_hit:
        print(f"BUDGET_STOP step={gstep} elapsed={elapsed:.1f}", flush=True)
    else:
        print(f"EPOCHS_DONE step={gstep} elapsed={elapsed:.1f}", flush=True)
    print(f"saved -> {args.checkpoint}", flush=True)


if __name__ == "__main__":
    main()
