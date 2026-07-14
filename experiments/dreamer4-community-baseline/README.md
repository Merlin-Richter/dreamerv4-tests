# dreamer4-community-baseline — prep artifacts

Integration glue for training the **independent community Dreamer 4**
([nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4)) on **Memory Maze**, as an
independent vanilla baseline (see `tasks/in-progress/memmaze-community-dreamer4-baseline.md` — the
authoritative runbook). This dir is experimental integration code: it does NOT follow `src/`'s
one-spec-per-file discipline. The community repo itself is NOT vendored here — it is cloned fresh on
the cluster (pinned commit in the task file).

- **`memmaze_to_dreamer4.py`** — converts raw Memory-Maze `.npz` trajectories into the community repo's
  two paired trees: frame shards (`shards/<task>/<task>_shard*.pt` = `{"frames": (S,3,H,W) uint8}`) and a
  per-frame demo file (`demos/<task>.pt` = `{episode, action(one-hot→16-dim), reward}`). Discrete 6-action
  Memory Maze → one-hot in the first 6 of the repo's hardcoded 16 action dims. No time-shift (raw MM
  `action[t]` already = "action that produced frame t", which is exactly the repo's convention).
- **`validate_integration.py`** — cluster-free regression test: synthesizes MM-format `.npz`, runs the
  converter, then loads the output with the REAL community `ShardedFrameDataset` + `WMDataset` and runs a
  64×64 `Encoder/Decoder/Dynamics` forward. Run: `python validate_integration.py --dreamer4 /path/to/dreamer4`.
  Confirmed PASSING locally on 2026-07-14 against upstream commit `b8abafbf`.

Everything here was built and validated WITHOUT the cluster. The cluster work that remains (download,
convert, train, eval) is spelled out step-by-step in the task file.
