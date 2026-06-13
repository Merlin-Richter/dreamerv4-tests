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
    dyn.config.max_temporal_length = N                       # context window (W = N-1)
    # warmup: fill the window + warm kernels
    n_warm = max(N + 8, 16)
    ctx, act = _make_inputs(dyn, dcfg, B, T_ctx, n_warm, device)
    _run(dyn, method, ctx, n_warm, K, act); torch.cuda.synchronize()

    # calibrate ms/step on a short run, then size the measured run to ~budget
    n_cal = max(N + 8, 24)
    ctx, act = _make_inputs(dyn, dcfg, B, T_ctx, n_cal, device)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    _run(dyn, method, ctx, n_cal, K, act); torch.cuda.synchronize()
    ms_step = (time.perf_counter() - t0) * 1e3 / n_cal
    n_meas = int(min(5000, max(50, budget_s * 1000.0 / max(ms_step, 1e-3))))

    # throughput: one long rollout (no re-prefill bias), budget-bounded
    ctx, act = _make_inputs(dyn, dcfg, B, T_ctx, n_meas, device)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = _run(dyn, method, ctx, n_meas, K, act); torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    del out

    # memory: short steady-state rollout (small output history) -> working set incl. the cache
    n_mem = max(2 * N, 32)
    ctx, act = _make_inputs(dyn, dcfg, B, T_ctx, n_mem, device)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    out = _run(dyn, method, ctx, n_mem, K, act); torch.cuda.synchronize()
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--windows", type=str, default="8,16,32,64")
    ap.add_argument("--budget", type=float, default=8.0, help="seconds per config (<=60)")
    ap.add_argument("--tctx", type=int, default=4)
    ap.add_argument("--ckpt", type=str, default="experiments/EXP-012/vanilla_s0.pt")
    ap.add_argument("--tokenizer", type=str, default="trained_autoencoder.pt")
    ap.add_argument("--prof-window", type=int, default=32)
    args = ap.parse_args()
    assert args.budget <= 60, "budget capped at 60s/config"
    device = "cuda"
    assert torch.cuda.is_available(), "run with venv python for CUDA"
    Ns = [int(x) for x in args.windows.split(",")]
    B, K = args.batch, None

    tok, dyn, dcfg, _ = load_models(ROOT / args.tokenizer, ROOT / args.ckpt, Ns[0], device)
    K = dcfg.inference_steps
    meta = dict(gpu=torch.cuda.get_device_name(0), batch=B, T_ctx=args.tctx, K=K,
                embedding_dim=dcfg.embedding_dim, depth=dcfg.depth, n_heads=dcfg.n_heads,
                n_latents=dcfg.n_latents, bottleneck_dim=dcfg.bottleneck_dim,
                n_actions=getattr(dcfg, "n_actions", 0), budget_s=args.budget, windows=Ns)
    print(f"GPU {meta['gpu']} | model E{dcfg.embedding_dim} d{dcfg.depth} h{dcfg.n_heads} "
          f"K={K} n_act={meta['n_actions']} | B={B} T_ctx={args.tctx} budget={args.budget}s/config\n")

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
    (HERE / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {HERE/'results.json'}")
    _plot(meta, results, speedups)


def _plot(meta, results, speedups):
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
    fig.savefig(HERE / "perf.png", dpi=110)
    print(f"wrote {HERE/'perf.png'}")


if __name__ == "__main__":
    main()
