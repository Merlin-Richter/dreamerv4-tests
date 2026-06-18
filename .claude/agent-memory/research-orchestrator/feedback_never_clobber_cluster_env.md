---
name: never-clobber-cluster-env
description: scripts/cluster.env is Merlin's gitignored secret config — never overwrite/delete it in tests
metadata:
  type: feedback
---

`scripts/cluster.env` holds Merlin's filled-in cluster connection config (hostnames, partition,
W&B key). It is **gitignored**, so it is NOT recoverable from git if deleted.

**Rule:** never `cp`/`rm`/overwrite `scripts/cluster.env` in tests. For offline `submit_job --dry-run`
or guard tests that need a config, either (a) back it up first (`cp scripts/cluster.env /tmp/ce.bak`)
and restore after, or (b) write the dummy to a different path. I once did `cp cluster.env.example
cluster.env; ...; rm cluster.env` for a dry-run and destroyed Merlin's real config mid-session;
had to reconstruct it from values seen earlier and re-verify with cluster_health.

**Why:** it cost a recovery detour and risked wrong cluster params on a live run.
**How to apply:** before any command that writes/removes a file under `scripts/`, check it isn't the
live `cluster.env`. See [[cluster-use-wsl]] for the wider cluster-ops context.
