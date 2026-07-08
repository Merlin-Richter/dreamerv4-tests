"""Budgeted from-scratch trainer for the autoresearch loop. EDITABLE LAYER.

Trains DynamicsModel (autoresearch/editable/model.py) FROM SCRATCH on the ColorField
latent cache + action sidecars, with the mem2mem sliding-rollout objective
(autoresearch/editable/rollout.py) — the GridWorld WINNER recipe by default:
rollout-only mem2mem (mem2mem_frac=1.0), bootstrap OFF (d_min-only sampling),
FF9 OFF (the fair no-FF9 ablation proved it unnecessary on GridWorld).

================================================================================
PINNED CONFIG CONTRACT — the driver PROBES these; violating them zeroes the score:

  * max_temporal_length W = 16      <<< THE WINDOW PIN. DO NOT RAISE. >>>
      The comeback eval gives a window-W model ~ the fraction of age bins <= W,
      so the cheapest score move would be "grow the window" — the opposite of the
      memory-token research question. The driver's window probe (perturb a frame
      > W back; the committed prediction must be bit-identical) enforces this;
      a violation scores 0 and is flagged. Every age bin beyond W=16 must come
      from a CARRIED memory mechanism.
  * n_actions = 5                   ColorField: up/down/left/right/stay.
  * n_latents = 4, bottleneck_dim = 64   must match the frozen tokenizer.
================================================================================

Data: the driver-built latent cache <data>/latents-<tokhash12>.npy — fp16,
(N, T=1024, n_latents=4, bottleneck=64) — located via the sha256 of --tokenizer
(the tokenizer itself is NEVER loaded here). Clips of --clip-len are sliced at
RANDOM offsets (window-invariance of the cache was verified by the driver probe).

Budget: --budget-s is the wall-clock budget in seconds, measured from process
start and checked after EVERY optimizer step; on expiry the model is saved and a
final "BUDGET_STOP step=N elapsed=S" line is printed. --epochs is a secondary cap
("EPOCHS_DONE ..."). NOTE: the LR schedule (warmup -> flat -> cosine over the
last 20%) is laid out over len(train_loader) * --epochs steps (or --sched-steps
if given) — size --epochs / --sched-steps so the budget lands near the cosine
tail, otherwise training dies mid-flat (or worse, mid-warmup).

Checkpoint payload (the adapter and driver reload from this — keep the keys):
  {"model_state_dict": ..., "config": asdict(DynamicsModelConfig)}

Run (repo root):
  venv/Scripts/python.exe -u autoresearch/editable/train.py \
    --data data/colorfield --val data/colorfield_val \
    --tokenizer checkpoints/colorfield/tokenizer.pt \
    --checkpoint autoresearch/runs/<tag>/dynamics.pt --budget-s 600
"""
from __future__ import annotations

import argparse
import hashlib
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
    from .model import DynamicsModel, DynamicsModelConfig
    from .rollout import mem2mem_rollout_loss
except ImportError:  # run as a script: python autoresearch/editable/train.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from autoresearch.editable.model import DynamicsModel, DynamicsModelConfig
    from autoresearch.editable.rollout import mem2mem_rollout_loss

# --- THE PINS (see module docstring; the driver verifies these) ---------------
W_PIN = 16            # max_temporal_length — THE WINDOW PIN, do not raise
N_ACTIONS = 5         # ColorField action space
N_LATENTS = 4         # must match the frozen tokenizer
BOTTLENECK_DIM = 64   # must match the frozen tokenizer


# ------------------------------------------------------------------ data
def _file_sha256_12(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def locate_cache(data_dir: Path, tokenizer_path: Path) -> Path:
    """The driver-built latent cache for this (data dir, tokenizer) combo. The tokenizer
    checkpoint is used ONLY to compute the cache filename hash — it is never loaded."""
    key = _file_sha256_12(tokenizer_path)
    p = Path(data_dir) / f"latents-{key}.npy"
    if not p.is_file():
        raise FileNotFoundError(
            f"latent cache not found: {p} — build it first: "
            f"venv/Scripts/python.exe -u -m autoresearch.driver.latent_cache")
    return p


def load_split(data_dir: Path, tokenizer_path: Path):
    """(mmapped fp16 latents (N,T,L,D), long actions (N,T)) for one dataset dir."""
    cache = locate_cache(data_dir, tokenizer_path)
    lat = np.load(cache, mmap_mode="r")
    assert lat.ndim == 4 and lat.shape[2] == N_LATENTS and lat.shape[3] == BOTTLENECK_DIM, \
        f"cache {cache} shape {lat.shape} != (N, T, {N_LATENTS}, {BOTTLENECK_DIM})"
    actions_np = np.load(Path(data_dir) / "actions.npy")
    assert actions_np.shape == lat.shape[:2], (actions_np.shape, lat.shape)
    assert int(actions_np.max()) < N_ACTIONS, f"action id >= {N_ACTIONS} in {data_dir}"
    return lat, torch.from_numpy(actions_np.astype(np.int64)), cache


class RandomClipDataset(Dataset):
    """Fixed-length clips from the mmapped latent cache with matching action slices.

    random_offsets=True (train): each access draws a fresh uniform start offset — sound
    because the driver's window-invariance probe verified arbitrary-offset slicing of the
    chunk-encoded cache. Offsets use torch's RNG (properly re-seeded per DataLoader worker;
    numpy's state would fork identically into workers). One epoch = T // clip_len clips per
    episode (coverage-equivalent to fixed chunking).
    random_offsets=False (val): deterministic chunking at offsets j * clip_len.
    """

    def __init__(self, latents: np.ndarray, actions: torch.Tensor, clip_len: int,
                 random_offsets: bool = True) -> None:
        self.latents = latents      # (N, T, L, D) fp16 mmap
        self.actions = actions      # (N, T) long
        self.clip_len = int(clip_len)
        n, t = latents.shape[:2]
        assert t >= self.clip_len, f"episode length {t} < clip_len {self.clip_len}"
        self.clips_per_ep = max(1, t // self.clip_len)
        self.max_start = t - self.clip_len
        self.random_offsets = random_offsets

    def __len__(self) -> int:
        return self.latents.shape[0] * self.clips_per_ep

    def __getitem__(self, idx: int):
        ep, j = divmod(idx, self.clips_per_ep)
        if self.random_offsets and self.max_start > 0:
            s = int(torch.randint(0, self.max_start + 1, (1,)).item())
        else:
            s = j * self.clip_len
        z = torch.from_numpy(np.asarray(self.latents[ep, s:s + self.clip_len]).astype(np.float32))
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
    p.add_argument("--data", type=Path, default=Path("data/colorfield"))
    p.add_argument("--val", type=Path, default=Path("data/colorfield_val"))
    p.add_argument("--tokenizer", type=Path, default=Path("checkpoints/colorfield/tokenizer.pt"),
                   help="Used ONLY to locate the latent cache by content hash (never loaded).")
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
    p.add_argument("--tau0-anchor", type=float, default=0.0, metavar="P",
                   help="Per-frame P of forcing (tau=0, finest d, GT flow) on the clean mode's "
                        "new half (Arm-D sustained anchor; trains visible-context next-frame "
                        "prediction, which the sampled grid under-weights ~1/K_max). 0 = off.")
    p.add_argument("--tbptt-frames", type=int, default=None,
                   help="Detach the memory relay past this many frames (default 2*W).")
    p.add_argument("--max-frames", type=int, default=None, help="Cap rollout length per clip.")
    p.add_argument("--snapshot-at", type=lambda s: {int(x) for x in s.split(",")},
                   default=None,
                   help="Comma-separated step counts at which to save side checkpoints "
                        "(<ckpt>_stepN.pt) — for steps-vs-quality calibration curves.")
    p.add_argument("--fixed-n-ctx", action="store_true",
                   help="Always slide at n_ctx = W_PIN instead of sampling {4,8,16}: fewer, "
                        "fatter, less-serialized forwards (GPU util + bounded VRAM) and matches "
                        "the eval's full-window distribution; costs relay-hop diversity "
                        "(untested recipe ingredient).")
    p.add_argument("--sched-steps", type=int, default=None,
                   help="LR-schedule horizon in optimizer steps (default len(loader)*epochs).")
    p.add_argument("--val-batches", type=int, default=32,
                   help="Cap per-epoch val batches (keeps val cheap under the budget).")
    p.add_argument("--num-workers", type=int, default=0,
                   help="0 (default): latent clips are tiny mmap slices; workers only add "
                        "Windows-spawn overhead per epoch under a wall-clock budget.")
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    gen = torch.Generator(device=device).manual_seed(args.seed)

    lat, actions, cache = load_split(args.data, args.tokenizer)
    val_lat, val_actions, _ = load_split(args.val, args.tokenizer)
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
        print("[train] n_memory=0 -> vanilla reference arm: forcing mem2mem_frac=0.0")
        mem2mem_frac = 0.0
    use_ff9 = args.ff9 > 0

    train_ds = RandomClipDataset(lat, actions, clip_len, random_offsets=True)
    val_ds = RandomClipDataset(val_lat, val_actions, clip_len, random_offsets=False)
    lk = dict(num_workers=args.num_workers, pin_memory=use_amp)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=len(train_ds) >= args.batch_size, **lk)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **lk)
    assert len(train_loader) > 0, "no training batches (dataset smaller than batch size?)"

    nparams = sum(q.numel() for q in model.parameters())
    ncts = valid_n_ctx(W_PIN, clip_len)
    print(f"device={device} params={nparams / 1e6:.2f}M W={W_PIN} (PINNED) n_actions={N_ACTIONS} "
          f"cache={cache.name} train_eps={lat.shape[0]} val_eps={val_lat.shape[0]} "
          f"clip_len={clip_len} n_ctx choices={ncts} mem2mem_frac={mem2mem_frac} "
          f"bootstrap={args.bootstrap} use_ff9={use_ff9} ff9_k={args.ff9} "
          f"n_memory={cfg.n_memory} budget_s={args.budget_s}", flush=True)

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
                                    # (bounded VRAM: 7 slide graphs at clip 64, vs 31 at n_ctx=4)
                    else:
                        W = ncts[torch.randint(len(ncts), (1,), generator=gen, device=device).item()]
                    loss, parts = mem2mem_rollout_loss(
                        model, z1, acts, n_ctx=W, device=device, gen=gen,
                        tbptt_frames=args.tbptt_frames, max_frames=args.max_frames,
                        bootstrap=args.bootstrap, n_d_unlocked=n_unlocked,
                        use_ff9=use_ff9, ff9_norm_flow=args.ff9_norm_flow,
                        relay_grad_clip=args.relay_grad_clip, tau0_anchor=args.tau0_anchor)
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
                p = args.checkpoint.with_name(args.checkpoint.stem + f"_step{gstep}.pt")
                torch.save({"model_state_dict": model.state_dict(),
                            "config": asdict(cfg)}, p)
                print(f"[snapshot] step {gstep} -> {p} "
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
