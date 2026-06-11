# HOWTO: Weights & Biases

- Entity: `models-eberhard-karls-universit-t-t-bingen` (user and team of the same
  name; the team entity is the default).
- Projects:
  - `transformer-C-tokenizer` — tokenizer (C) runs
  - `transformer-D-dynamics` — dynamics (D) runs
  - `my-awesome-project` — scratch/integration-test project; ignore.
- Credentials live in `~\_netrc` (`machine api.wandb.ai`). The MCP server
  (`user-wandb`) may return "relogin required"; re-auth and retry — it has
  succeeded on a delayed retry before.
- Always pass `--wandb-name` with a descriptive name. Auto-generated names
  (glorious-dew-1, …) made the Jun-09 iteration runs hard to tell apart.
- Run metadata worth trusting for provenance: `commit` (resolved SHA), `host`
  (ZaubererPC = local laptop; galvani-*/mlcbm* = cluster), `config` (full
  submitted config), `summaryMetrics`.
- Key metrics conventions:
  - Tokenizer: `train/mse`, `val/mse`, `train/lpips`, `train/lpips_scale`,
    `latent_cos` (collapse detector — pairwise latent cosine similarity, lower is
    better; collapse showed ≈1, healthy runs ≈0.03–0.05), `pred_std`.
  - Dynamics: `train/loss`, `val/loss` (shortcut-forcing loss). **Warning:** EXP-007
    showed healthy val/loss with useless rollouts; never judge dynamics by this
    alone.
  - Perf: `perf/samples_per_s`, `perf/train_time_s`.
- Numbers come from the API (MCP `query_wandb_tool` / `wandb.Api()`), not chart
  screenshots.
