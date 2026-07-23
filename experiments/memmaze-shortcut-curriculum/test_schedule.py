"""CPU gate for the Memory Maze K=4 continuation's exact unlock times."""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "experiments" / "mem2mem" / "train_mem2mem.py"
SPEC = importlib.util.spec_from_file_location("train_mem2mem", TRAINER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_unlock_boundaries():
    before_and_at = [
        (0.000, 1),
        (0.999, 1),
        (1.000, 2),
        (2.249, 2),
        (2.250, 3),
        (3.499, 3),
        (3.500, 4),
        (4.749, 4),
        (4.750, 5),
        (5.999, 5),
        (6.000, 6),
        (12.00, 6),
    ]
    got = [
        MODULE.wallclock_curriculum(
            hour, warmup_hours=1.0, full_hours=6.0, max_unlocked=6)
        for hour, _ in before_and_at
    ]
    expected = [count for _, count in before_and_at]
    assert got == expected, list(zip(before_and_at, got))
    print("[ok] K=4 curriculum unlocks at 1.00, 2.25, 3.50, 4.75, and 6.00 active hours")


if __name__ == "__main__":
    test_unlock_boundaries()
