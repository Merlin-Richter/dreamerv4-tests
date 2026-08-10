# Memmaze community Dreamer 4 mem2mem — final 48-hour production artifacts

- Ferranti job: `430090` (`COMPLETED`, exit `0:0`)
- Run: `memmaze-community-d4-mem2mem-cached-48h`
- W&B run: `ai5rh826`
- Final optimizer step: `303248`
- Exact cumulative training-loop time: `172800.29900704397s`
- Final checkpoint SHA-256: `aff682549e03616e30d46d83e19bfe79b69aa60fb1bfe714ae225fb98872ab93`
- Final checkpoint size: `514477149` bytes
- Cache-manifest SHA-256: `e7a4e57e63e357d1986154a1b6c3cea9f4220b1a716e4a553df8a345fb2f4fcf`

The checkpoint is retained locally at
`checkpoints/memmaze-community-d4-mem2mem/memory-final.pt` and installed beside the accepted vanilla
checkpoint as `checkpoints/memmaze-community-d4/dynamics-mem2mem-final.pt` in the durable community
Dreamer 4 checkout. Both copies were hash-verified.

Artifact SHA-256 values:

```text
aac725d0364d73c6020c56f147ac2a94a5e2353cefd670dfeb4f300cbbbd3aa5  final-checkpoint-summary.json
f2bb10da7cc6e2e6f737f7f5d2e63f8f991a7040f6c114570a8e9006392cceae  resolved-config.json
e55aecbb29c635609e8040ca8581421eb4d5c3fa79d4ef172af708189637cba1  telemetry-summary.json
e45b43d751d4fa8ffbc3db77a2f9cc8a35614120ad7b03f4c04eee842b6a0925  training-clock.jsonl
```

The 64,300,134-byte clock ledger is retained locally and ignored by Git. The three compact JSON
summaries are versioned here. The 171.515 GiB latent cache and hourly checkpoints were not pulled.
