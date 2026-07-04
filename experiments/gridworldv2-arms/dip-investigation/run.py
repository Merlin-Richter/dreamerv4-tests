"""Generic probe runner. Examples:
  venv/Scripts/python.exe -u experiments/gridworldv2-arms/dip-investigation/run.py \
      --arm D --n-rollouts 256 --max-k 16 --window 8 --out d_w8.json
  ... --no-hide --teacher-forced          (driver-validation probe)
  ... --mask no_mem_read                  (branch-only memory-read ablation, arm D)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import print_table, run_probe, summarize

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "D"], required=True)
    ap.add_argument("--n-rollouts", type=int, default=256)
    ap.add_argument("--n-ctx", type=int, default=4)
    ap.add_argument("--max-k", type=int, default=16)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-hide", action="store_true")
    ap.add_argument("--teacher-forced", action="store_true")
    ap.add_argument("--mask", default="normal", choices=["normal", "no_mem_read", "mem_only"])
    ap.add_argument("--spoof", default=None, choices=[None, "at_write", "after_write"])
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    hide = not args.no_hide
    recs = run_probe(args.arm, n_rollouts=args.n_rollouts, n_ctx=args.n_ctx, max_k=args.max_k,
                     window=args.window, K=args.K, batch_size=args.batch_size, hide=hide,
                     teacher_forced=args.teacher_forced, branch_mask_mode=args.mask,
                     seed0=args.seed0, spoof=args.spoof)
    summ = summarize(recs, args.max_k, args.n_ctx, hide=hide)
    tag = (f"{args.arm} w{args.window} n_ctx={args.n_ctx} hide={hide} "
           f"tf={args.teacher_forced} mask={args.mask} spoof={args.spoof} n={args.n_rollouts}")
    print_table(tag, summ)
    if args.out:
        payload = {"meta": vars(args), "summary": {str(k): v for k, v in summ.items()},
                   "records": recs}
        (HERE / args.out).write_text(json.dumps(payload))
        print(f"wrote {HERE / args.out}")


if __name__ == "__main__":
    main()
