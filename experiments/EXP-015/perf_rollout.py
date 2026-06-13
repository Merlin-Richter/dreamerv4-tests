"""EXP-015 (D-022): basic perf tool for the rollout KV cache (T-012).

Cached (`generate_streaming`, cross-frame KV eviction cache) vs no-cache (`generate_windowed`, the
matched uncached twin — same semantics, so the delta is purely the cache). GPU, real batch dimension
(training-relevant), swept over context-window sizes N. Reports rollout-step throughput, peak GPU
memory, and a profiler breakdown of where time goes.

  python experiments/EXP-015/perf_rollout.py            # default sweep (run with venv python for CUDA)
  python experiments/EXP-015/perf_rollout.py --batch 64 --windows 8,16,32 --budget 8

NOTE on "time waiting for memory": exact HBM memory-stall % needs Nsight Compute. As a practical proxy
we split torch.profiler CUDA op-time into compute (matmul/SDPA) vs memory-bound (elementwise/cat/norm)
and report the cache's per-step `cat` cost. Labeled as a proxy, not a hardware counter.
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "probe"))
from probe.revisit_probe import load_models  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
COMPUTE_HINTS = ("mm", "bmm", "matmul", "linear", "addmm", "scaled_dot_product", "sdpa", "attention", "gemm")
MEM_HINTS = ("cat", "copy", "mul", "add", "sub", "div", "rsqrt", "sqrt", "norm", "softmax",
             "to_", "_to_copy", "clamp", "tanh", "sigmoid", "silu", "fill", "slice", "index", "elementwise")


def _tag(name: str) -> str:
    n = name.lower()
    if any(h in n for h in COMPUTE_HINTS):
        return "compute"
    if any(h in n for h in MEM_HINTS):
        return "memory"
    return "other"


def _make_inputs(dyn, dcfg, B, T_ctx, n_gen, device):
    L, D = dcfg.n_latents, dcfg.bottleneck_dim
    context = torch.randn(B, T_ctx, L, D, device=device)
    n_act = getattr(dcfg, "n_actions", 0)
    action_idx = (torch.randint(0, n_act, (B, T_ctx + n_gen), device=device) if n_act else None)
    return context, action_idx


def _run(dyn, method, context, n_gen, K, action_idx):
    fn = dyn.generate_streaming if method == "cached" else dyn.generate_windowed
    return fn(context, n_gen, K=K, action_idx=action_idx)


@torch.no_grad()
def bench_config(dyn, dcfg, method, N, B, K, T_ctx, budget_s, device):
    """Steady-state per-step rollout cost. Measured from a PRE-FILLED window (context = N-1 frames),
    so every generated step is steady-state from the first — no window-fill transient to amortise, so
    a few steps suffice (cheap). One short warmup (kernels) + a tiny calibrate + one budget-capped
    measured pass that ALSO yields peak memory (no separate memory pass)."""
    dyn.config.max_temporal_length = N                       # context window (W = N-1)
    T_full = max(N - 1, 1)                                    # full window -> no fill transient

    # warmup: warm kernels/autotuner/allocator at this (B, window) shape (4 iters — 2 was too few,
    # left fast configs under-warmed and ~30% slow); window already full from context
    ctx, act = _make_inputs(dyn, dcfg, B, T_full, 4, device)
    _run(dyn, method, ctx, 4, K, act); torch.cuda.synchronize()

    # short calibrate (3 steady steps) -> ms/step, used only to size the measured pass
    ctx, act = _make_inputs(dyn, dcfg, B, T_full, 3, device)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    _run(dyn, method, ctx, 3, K, act); torch.cuda.synchronize()
    ms_step = (time.perf_counter() - t0) * 1e3 / 3
    # many steps when cheap (accurate), few when each step is slow but stable (bounded). Floor 8
    # keeps fast-config noise low; the VRAM guard (not under-sampling) is what bounds the slow tail.
    n_meas = int(min(64, max(8, budget_s * 1000.0 / max(ms_step, 1e-3))))

    # measured pass: throughput + peak memory in one short steady-state rollout
    ctx, act = _make_inputs(dyn, dcfg, B, T_full, n_meas, device)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = _run(dyn, method, ctx, n_meas, K, act); torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_alloc = torch.cuda.max_memory_allocated() / 2**20
    peak_resv = torch.cuda.max_memory_reserved() / 2**20
    del out, ctx, act; torch.cuda.empty_cache()

    steps_s = n_meas / elapsed
    return dict(method=method, N=N, B=B, steps=n_meas, secs=round(elapsed, 3),
                steps_per_s=round(steps_s, 1), frames_per_s=round(steps_s * B, 1),
                ms_per_step=round(elapsed * 1e3 / n_meas, 3),
                peak_alloc_MB=round(peak_alloc, 1), peak_reserved_MB=round(peak_resv, 1))


@torch.no_grad()
def profile_config(dyn, dcfg, method, N, B, K, T_ctx, device, n_steps=20):
    from torch.profiler import profile, ProfilerActivity
    dyn.config.max_temporal_length = N
    ctx, act = _make_inputs(dyn, dcfg, B, T_ctx, max(N + 8, 16), device)
    _run(dyn, method, ctx, max(N + 8, 16), K, act); torch.cuda.synchronize()  # warm
    ctx, act = _make_inputs(dyn, dcfg, B, T_ctx, n_steps, device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        _run(dyn, method, ctx, n_steps, K, act); torch.cuda.synchronize()
    rows, split = [], {"compute": 0.0, "memory": 0.0, "other": 0.0}
    for e in prof.key_averages():
        raw = getattr(e, "self_device_time_total", None)
        if raw is None:
            raw = getattr(e, "self_cuda_time_total", 0)
        cu = raw / 1e3  # us -> ms
        if cu <= 0:
            continue
        split[_tag(e.key)] += cu
        rows.append((e.key, round(cu, 2)))
    rows.sort(key=lambda r: -r[1])
    tot = sum(split.values()) or 1.0
    return dict(method=method, N=N, B=B, n_steps=n_steps,
                top_ops=rows[:8],
                cuda_ms_total=round(tot, 2),
                split_pct={k: round(100 * v / tot, 1) for k, v in split.items()})


def _is_oom(err: Exception) -> bool:
    return isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in str(err).lower()


@torch.no_grad()
def batch_sweep(dyn, dcfg, N, batches, K, T_ctx, budget_s, device, total_MB):
    """Push batch up toward each method's OWN memory ceiling at fixed N; record where each OOMs.

    Returns (results, ceilings). `results` is per (method,batch) bench dicts (successful only);
    `ceilings` is the max-fitting batch + its throughput per method."""
    results, ceilings = [], {}
    vram_cap = 0.92 * total_MB                                # stay in real VRAM; beyond this WDDM
    #                                                          spills to system RAM (ms -> minutes)
    hdr = (f"{'method':8s} {'B':>6s} {'steps/s':>9s} {'frames/s':>11s} {'ms/step':>9s} "
           f"{'peakMB':>8s} {'resvMB':>8s} {'%mem':>6s}")
    print(hdr); print("-" * len(hdr))
    for method in ("cached", "windowed"):
        last_ok = None
        for B in batches:
            # predictive memory guard: extrapolate reserved-MB linearly from the last successful
            # config; if the NEXT batch would cross VRAM, stop escalating BEFORE launching it (a
            # past-VRAM config sysmem-thrashes for minutes instead of cleanly OOM-ing).
            if last_ok is not None:
                pred = last_ok["peak_reserved_MB"] * B / last_ok["B"]
                if pred > vram_cap:
                    print(f"{method:8s} {B:>6d} {'SKIP':>9s}  (predicted ~{pred:.0f}MB > {vram_cap:.0f}MB "
                          f"VRAM cap; ceiling = B={last_ok['B']})")
                    break
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                r = bench_config(dyn, dcfg, method, N, B, K, T_ctx, budget_s, device)
            except Exception as e:                       # noqa: BLE001 — want OOM + anything else
                torch.cuda.empty_cache()
                if _is_oom(e):
                    print(f"{method:8s} {B:>6d} {'OOM':>9s}  (ceiling = B={last_ok['B'] if last_ok else None})")
                    break
                raise
            r["pct_mem"] = round(100 * r["peak_reserved_MB"] / total_MB, 1)
            # if a config slipped past the predictor into sysmem fallback, record it but stop here
            if r["peak_reserved_MB"] > total_MB:
                print(f"{r['method']:8s} {r['B']:>6d} {r['steps_per_s']:>9.1f} {r['frames_per_s']:>11.1f} "
                      f"{r['ms_per_step']:>9.3f} {r['peak_alloc_MB']:>8.1f} {r['peak_reserved_MB']:>8.1f} "
                      f"{r['pct_mem']:>5.1f}%  <- OVER VRAM (sysmem fallback; excluded from ceiling)")
                r["oversubscribed"] = True
                results.append(r)
                break
            results.append(r); last_ok = r
            print(f"{r['method']:8s} {r['B']:>6d} {r['steps_per_s']:>9.1f} {r['frames_per_s']:>11.1f} "
                  f"{r['ms_per_step']:>9.3f} {r['peak_alloc_MB']:>8.1f} {r['peak_reserved_MB']:>8.1f} "
                  f"{r['pct_mem']:>5.1f}%")
            torch.cuda.empty_cache()
        if last_ok:
            ceilings[method] = last_ok
        print()
    return results, ceilings


def run_batch_sweep(dyn, dcfg, dcfg_meta, N, batches, K, T_ctx, budget_s, device, outdir, total_MB):
    results, ceilings = batch_sweep(dyn, dcfg, N, batches, K, T_ctx, budget_s, device, total_MB)

    # speedup vs batch on the SHARED batches (both methods fit)
    by = {(r["method"], r["B"]): r for r in results}
    cached_Bs = sorted(b for (m, b) in by if m == "cached")
    wind_Bs = sorted(b for (m, b) in by if m == "windowed")
    shared = sorted(set(cached_Bs) & set(wind_Bs))
    print("speedup (cached steps/s ÷ windowed steps/s) vs batch:")
    speedups = {}
    for B in shared:
        c, w = by[("cached", B)]["steps_per_s"], by[("windowed", B)]["steps_per_s"]
        speedups[B] = round(c / w, 2)
        print(f"  B={B:>5d}: {speedups[B]}x  ({c:.1f} vs {w:.1f} steps/s)")

    print("\nmax-fitting batch per method (its own memory ceiling):")
    for method in ("cached", "windowed"):
        c = ceilings.get(method)
        if c:
            print(f"  [{method}] B={c['B']}  {c['steps_per_s']:.1f} steps/s  {c['frames_per_s']:.1f} "
                  f"frames/s  ({c['peak_reserved_MB']:.0f} MB resv, {c['pct_mem']}% of {total_MB:.0f})")
    if "cached" in ceilings and "windowed" in ceilings:
        cc, cw = ceilings["cached"], ceilings["windowed"]
        print(f"\n  cached fits B={cc['B']} vs windowed B={cw['B']}  "
              f"(batch headroom {cc['B']/max(cw['B'],1):.2f}x); "
              f"peak frames/s {cc['frames_per_s']:.0f} (cached) vs {cw['frames_per_s']:.0f} (windowed) "
              f"= {cc['frames_per_s']/max(cw['frames_per_s'],1e-9):.2f}x end-to-end throughput")

    payload = dict(meta=dict(**dcfg_meta, mode="batch_sweep", N=N, batches=batches, total_MB=total_MB),
                   results=results, ceilings=ceilings, speedups=speedups)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {outdir/'results.json'}")
    _plot_batch_sweep(payload, outdir, total_MB)


def _plot_batch_sweep(payload, outdir, total_MB):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(plot skipped: {e})"); return
    by = {(r["method"], r["B"]): r for r in payload["results"]}
    N = payload["meta"]["N"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for method, mk in (("cached", "o-"), ("windowed", "s--")):
        Bs = sorted(b for (m, b) in by if m == method)
        if not Bs:
            continue
        ax[0].plot(Bs, [by[(method, b)]["frames_per_s"] for b in Bs], mk, label=method)
        ax[1].plot(Bs, [by[(method, b)]["steps_per_s"] for b in Bs], mk, label=method)
        ax[2].plot(Bs, [by[(method, b)]["peak_reserved_MB"] for b in Bs], mk, label=method)
        bmax = max(Bs)                                   # mark each method's ceiling
        ax[0].annotate(f"max B={bmax}", (bmax, by[(method, bmax)]["frames_per_s"]),
                       fontsize=8, ha="right", va="bottom")
    spk = payload["speedups"]
    if spk:
        ax2 = ax[1].twinx()
        sb = sorted(spk)
        ax2.plot(sb, [spk[b] for b in sb], "^:", color="green", alpha=.6, label="speedup")
        ax2.set_ylabel("cached/windowed speedup", color="green"); ax2.tick_params(labelcolor="green")
    ax[2].axhline(total_MB, color="r", lw=1, ls=":", label=f"GPU limit {total_MB:.0f}MB")
    ax[0].set(title=f"end-to-end throughput (N={N})", xlabel="batch size", ylabel="frames/s")
    ax[1].set(title="rollout-step rate (N={})".format(N), xlabel="batch size", ylabel="steps/s")
    ax[2].set(title="peak GPU memory (reserved)", xlabel="batch size", ylabel="MB reserved")
    for a in ax:
        a.legend(fontsize=8); a.grid(alpha=.3); a.set_xscale("log", base=2)
    fig.suptitle(f"EXP-016 batch-limit parallelism sweep @ N={N} — {payload['meta']['gpu']} "
                 f"(cached=generate_streaming vs windowed=generate_windowed)")
    fig.tight_layout()
    fig.savefig(outdir / "perf_batch.png", dpi=110)
    print(f"wrote {outdir/'perf_batch.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--windows", type=str, default="8,16,32,64")
    ap.add_argument("--budget", type=float, default=8.0, help="seconds per config (<=60)")
    ap.add_argument("--tctx", type=int, default=4)
    ap.add_argument("--ckpt", type=str, default="experiments/EXP-012/vanilla_s0.pt")
    ap.add_argument("--tokenizer", type=str, default="trained_autoencoder.pt")
    ap.add_argument("--prof-window", type=int, default=32)
    ap.add_argument("--outdir", type=str, default=None, help="output dir (default: this script's dir)")
    # batch-sweep mode (EXP-016): fix N, push batch toward each method's OWN memory ceiling
    ap.add_argument("--batch-sweep", action="store_true", help="run batch-limit sweep instead of N-sweep")
    ap.add_argument("--batches", type=str, default="32,64,128,256,512,1024,2048,4096",
                    help="ascending batch sizes to try (batch-sweep mode)")
    ap.add_argument("--sweep-window", type=int, default=16, help="fixed N for batch-sweep mode")
    args = ap.parse_args()
    assert args.budget <= 60, "budget capped at 60s/config"
    device = "cuda"
    assert torch.cuda.is_available(), "run with venv python for CUDA"
    Ns = [int(x) for x in args.windows.split(",")]
    B, K = args.batch, None
    outdir = pathlib.Path(args.outdir).resolve() if args.outdir else HERE

    init_N = args.sweep_window if args.batch_sweep else Ns[0]
    tok, dyn, dcfg, _ = load_models(ROOT / args.tokenizer, ROOT / args.ckpt, init_N, device)
    K = dcfg.inference_steps
    total_MB = torch.cuda.get_device_properties(0).total_memory / 2**20
    meta = dict(gpu=torch.cuda.get_device_name(0), batch=B, T_ctx=args.tctx, K=K,
                embedding_dim=dcfg.embedding_dim, depth=dcfg.depth, n_heads=dcfg.n_heads,
                n_latents=dcfg.n_latents, bottleneck_dim=dcfg.bottleneck_dim,
                n_actions=getattr(dcfg, "n_actions", 0), budget_s=args.budget, windows=Ns)
    print(f"GPU {meta['gpu']} ({total_MB:.0f}MB) | model E{dcfg.embedding_dim} d{dcfg.depth} "
          f"h{dcfg.n_heads} K={K} n_act={meta['n_actions']} | T_ctx={args.tctx} "
          f"budget={args.budget}s/config\n")

    if args.batch_sweep:
        batches = [int(x) for x in args.batches.split(",")]
        N = args.sweep_window
        dmeta = dict(gpu=meta["gpu"], T_ctx=args.tctx, K=K, embedding_dim=dcfg.embedding_dim,
                     depth=dcfg.depth, n_heads=dcfg.n_heads, n_latents=dcfg.n_latents,
                     bottleneck_dim=dcfg.bottleneck_dim, n_actions=meta["n_actions"],
                     budget_s=args.budget)
        print(f"BATCH-SWEEP @ N={N}, batches={batches}\n")
        run_batch_sweep(dyn, dcfg, dmeta, N, batches, K, args.tctx, args.budget, device, outdir, total_MB)
        return

    results = []
    hdr = f"{'method':8s} {'N':>4s} {'steps':>6s} {'steps/s':>9s} {'frames/s':>10s} {'ms/step':>9s} {'peakMB':>8s} {'resvMB':>8s}"
    print(hdr); print("-" * len(hdr))
    for N in Ns:
        for method in ("cached", "windowed"):
            r = bench_config(dyn, dcfg, method, N, B, K, args.tctx, args.budget, device)
            results.append(r)
            print(f"{r['method']:8s} {r['N']:>4d} {r['steps']:>6d} {r['steps_per_s']:>9.1f} "
                  f"{r['frames_per_s']:>10.1f} {r['ms_per_step']:>9.3f} {r['peak_alloc_MB']:>8.1f} "
                  f"{r['peak_reserved_MB']:>8.1f}")
            torch.cuda.empty_cache()

    # speedup summary
    print("\nspeedup (cached steps/s ÷ windowed steps/s) by N:")
    by = {(r["method"], r["N"]): r for r in results}
    speedups = {}
    for N in Ns:
        c, w = by[("cached", N)]["steps_per_s"], by[("windowed", N)]["steps_per_s"]
        speedups[N] = round(c / w, 2)
        print(f"  N={N:>3d}: {speedups[N]}x  ({c:.0f} vs {w:.0f} steps/s)")

    # profiler: where time goes (one config per method)
    pN = args.prof_window if args.prof_window in Ns else Ns[len(Ns) // 2]
    profs = []
    print(f"\nprofiler @ N={pN}, B={B} (CUDA self-time, ms; compute=matmul/SDPA, memory=elementwise/cat/norm — a PROXY, not HBM stalls):")
    for method in ("cached", "windowed"):
        p = profile_config(dyn, dcfg, method, pN, B, K, args.tctx, device)
        profs.append(p)
        print(f"  [{method}] total {p['cuda_ms_total']}ms  split {p['split_pct']}")
        for name, ms in p["top_ops"][:6]:
            print(f"      {ms:8.2f}ms  {name}")

    payload = dict(meta=meta, results=results, speedups=speedups, profiles=profs)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {outdir/'results.json'}")
    _plot(meta, results, speedups, outdir)


def _plot(meta, results, speedups, outdir=HERE):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(plot skipped: {e})"); return
    Ns = meta["windows"]
    by = {(r["method"], r["N"]): r for r in results}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for method, mk in (("cached", "o-"), ("windowed", "s--")):
        ax[0].plot(Ns, [by[(method, N)]["steps_per_s"] for N in Ns], mk, label=method)
        ax[1].plot(Ns, [by[(method, N)]["ms_per_step"] for N in Ns], mk, label=method)
        ax[2].plot(Ns, [by[(method, N)]["peak_alloc_MB"] for N in Ns], mk, label=method)
    ax[0].set(title=f"rollout throughput (B={meta['batch']})", xlabel="context window N", ylabel="steps/s"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].set(title="latency per step", xlabel="context window N", ylabel="ms/step"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].set(title="peak GPU memory", xlabel="context window N", ylabel="MB allocated"); ax[2].legend(); ax[2].grid(alpha=.3)
    for a in ax:
        a.axhline(0, color="k", lw=.5)
    fig.suptitle(f"EXP-015 rollout KV cache perf — {meta['gpu']}  (cached=generate_streaming vs windowed=generate_windowed)")
    fig.tight_layout()
    fig.savefig(outdir / "perf.png", dpi=110)
    print(f"wrote {outdir/'perf.png'}")


if __name__ == "__main__":
    main()
