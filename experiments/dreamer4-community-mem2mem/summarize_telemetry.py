#!/usr/bin/env python3
"""Summarize nvidia-smi samples inside the audited whole training-loop clock."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path


def number(value):
    return float(value.strip().split()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-csv", type=Path, nargs="+", required=True)
    ap.add_argument("--training-ledger", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    intervals = []
    for line in args.training_ledger.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            start = float(row["wall_start_epoch_s"])
            end = float(row["wall_end_epoch_s"])
            assert end >= start, row
            intervals.append((start, end))
    assert intervals, "training ledger is empty"
    intervals.sort()
    merged_intervals = []
    for start, end in intervals:
        if merged_intervals and start <= merged_intervals[-1][1] + 1e-6:
            merged_intervals[-1] = (merged_intervals[-1][0], max(end, merged_intervals[-1][1]))
        else:
            merged_intervals.append((start, end))

    samples = []
    for gpu_csv in args.gpu_csv:
        with gpu_csv.open(newline="") as handle:
            reader = csv.DictReader(handle, skipinitialspace=True)
            for row in reader:
                stamp = row[next(key for key in row if key.strip() == "timestamp")].strip()
                parsed = None
                for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
                    try:
                        parsed = datetime.strptime(stamp, fmt).timestamp()
                        break
                    except ValueError:
                        pass
                if parsed is None or not any(
                    start <= parsed <= end for start, end in merged_intervals
                ):
                    continue
                util_key = next(key for key in row if key.strip().startswith("utilization.gpu"))
                used_key = next(key for key in row if key.strip().startswith("memory.used"))
                total_key = next(key for key in row if key.strip().startswith("memory.total"))
                samples.append((parsed, number(row[util_key]), number(row[used_key]), number(row[total_key])))
    assert samples, "no GPU samples overlap training intervals"
    samples.sort()
    cadence = statistics.median(
        b[0] - a[0] for a, b in zip(samples, samples[1:]) if 0 < b[0] - a[0] < 60
    ) if len(samples) > 1 else 10.0

    longest_low = current_start = previous = None
    for stamp, util, _, _ in samples:
        if util < 90:
            if current_start is None or previous is None or stamp - previous > cadence * 2.5:
                current_start = stamp
            previous = stamp
            longest_low = max(longest_low or 0.0, stamp - current_start + cadence)
        else:
            current_start = previous = None

    utils = [sample[1] for sample in samples]
    used = [sample[2] for sample in samples]
    report = {
        "ledger_rows": len(intervals),
        "training_sessions": len(merged_intervals),
        "training_samples": len(samples),
        "sample_cadence_s": cadence,
        "mean_utilization_percent": statistics.fmean(utils),
        "p05_utilization_percent": sorted(utils)[max(0, int(0.05 * len(utils)) - 1)],
        "longest_below_90_percent_s": longest_low or 0.0,
        "mean_hbm_used_mib": statistics.fmean(used),
        "max_hbm_used_mib": max(used),
        "hbm_total_mib": max(sample[3] for sample in samples),
    }
    report["target_95_percent_met"] = report["mean_utilization_percent"] >= 95.0
    report["hard_health_gate_met"] = (
        report["mean_utilization_percent"] >= 90.0
        and report["longest_below_90_percent_s"] <= 300.0
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["hard_health_gate_met"]:
        raise SystemExit("GPU UTILIZATION HEALTH GATE FAILED")
    print("GPU UTILIZATION HEALTH GATE PASSED")


if __name__ == "__main__":
    main()
