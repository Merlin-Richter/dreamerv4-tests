"""REDUCED frozen_sym comeback eval for the autoresearch loop (NOT agent-editable).

Fixed reduced config — identical for every experiment row, so scores are comparable
within a loop run (absolute numbers are NOT comparable to the sealed full-suite config):
6 policies spanning all families x 2 seeds = 12 episodes, min_events_per_bin=10.
The frozen scorer itself (frozen_sym.eval_comeback) is untouched and hash-checked.

Prints grep-able `key: value` lines; writes the full result JSON next to the ckpt.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from autoresearch.editable.adapter_sym import make_adapter            # noqa: E402
from autoresearch.frozen_sym.eval_comeback import run_eval            # noqa: E402
from autoresearch.frozen_sym.eval_policies import EVAL_SUITE          # noqa: E402

REDUCED_POLICIES = ["oab_mid", "box_small", "sweep_wide", "idiot_fast",
                    "retrace_mid", "dwell_dart"]
N_SEEDS = 2
MIN_EVENTS = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    suite = [(n, f) for n, f in EVAL_SUITE if n in REDUCED_POLICIES]
    assert len(suite) == len(REDUCED_POLICIES), "policy name drift vs frozen suite"

    factory = make_adapter(args.checkpoint)
    t0 = time.perf_counter()
    r = run_eval(factory, suite=suite, n_seeds=N_SEEDS, min_events=MIN_EVENTS,
                 privileged=False)
    dt = time.perf_counter() - t0

    out = Path(args.checkpoint).with_suffix(".eval_reduced.json")
    with open(out, "w") as f:
        json.dump({k: v for k, v in r.items() if k != "events"}, f, indent=1)

    real = r["real_anchored"]
    cons = r["consistency"]
    print(f"score_gated:      {r['composite_gated']:.6f}")
    print(f"composite_raw:    {r['composite'] if r['composite'] is not None else 0.0:.6f}")
    print(f"real_cc:          {real['score'] if real['score'] is not None else 0.0:.6f}")
    print(f"consistency_cc:   {cons['score'] if cons['score'] is not None else 0.0:.6f}")
    print(f"fidelity:         {r['gates']['fidelity']['value']:.4f} "
          f"({'PASS' if r['gates']['fidelity']['passed'] else 'FAIL'})")
    ent = r["gates"].get("entropy", {})
    print(f"entropy_kl:       {ent.get('kl_to_uniform')}")
    print(f"gates_passed:     {r['gates_passed']}  fail={r['fail_reasons']} flags={r['flags']}")
    # frozen scorer returns bins as a LIST of dicts (lo/hi/acc_cc/qualified/...);
    # show the chance-corrected acc the score is built from, "!" = unqualified
    # (n < min_events, excluded from the score), "-" = no in-map events.
    bins = {f"[{b['lo']},{b['hi'] if b['hi'] is not None else 'inf'})":
            (f"{b['acc_cc']:.3f}" if b["acc_cc"] is not None else "-")
            + ("" if b["qualified"] else "!")
            for b in (real.get("bins") or [])}
    print(f"real_bins:        {bins}")
    print(f"eval_seconds:     {dt:.1f}")
    print(f"eval_json:        {out}")


if __name__ == "__main__":
    main()
