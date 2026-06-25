---
name: evals-use-env-directly
description: Generate eval data from the env directly (controlled scenarios), not from the fixed dataset/val set
metadata:
  type: feedback
---

Evals should construct their data by driving the ENV directly to get exactly the scenario wanted
(controlled occlusion length, balanced bins, no schedule artifacts) — not by scoring events found in a
fixed recorded dataset / val set.

Why: Merlin (2026-06-24), re the GridWorld occlusion sheet AND the recall headline: "you should not
use the validation set here. it should use the env directly" and "how it should be like all evals."
The dataset's curtain schedule produces an uneven k-distribution (few events at large k → noisy SE) and
a periodicity confound (copy-last spikes at k≡9 mod 10 on the 6×6 env). Env-direct generation removes
both and lets each k get equal, controlled coverage.
How to apply: for GridWorld memory evals (recall curves, sheets), generate fresh episodes from
GridWorldEnv with the exact curtain schedule the eval needs (e.g. n revealed context → fixed-length k
occlusion), per target k. The FROZEN scoring core (read_square/score_episode/aggregate + oracle/
copy-last baselines, D-045) is unchanged — only the data SOURCE changes, so this is a new eval driver,
not a change to the frozen scorer. When switching an existing eval's data source, RE-RUN prior models
(e.g. the vanilla baseline) under the new driver so A/B comparisons stay matched (§8 baselines sacred).
NOTE: EXP-027's headline.png recall used the val SET (eval.py over held-out gridworld.npy episodes) —
convert to env-direct going forward. Relates to [[gridworld-metric-semantics]].
