# Playable Memory Maze in dynamics-rollout space (local GPU)

**Asked by Merlin directly (2026-07-04).**

Reproduce `external/memory-maze/gui/run_gui.py` (the real playable Memory Maze: pygame window,
keymap noop/↑/←/→/↑←/↑→ = actions 0..5, 6 fps, space=pause, backspace=reset, tab=speedup,
esc=quit) with ONE difference: the rendering is the trained dynamics model's carrying rollout in
the frozen tokenizer's latent space, running locally on the RTX 4070 — i.e. you play inside the
world model instead of MuJoCo.

Requirements:
- Starts with a FULL dynamics context window of a real episode (default `n_ctx` = the model's
  `max_temporal_length`, encoded one-shot, committed via `rollout_init`), replayed on screen so the
  player sees the maze the model saw; then interactive `rollout_step(commit=True)` per tick,
  action from the keyboard.
- Real episodes from the local held-out split `data/memmaze9x9_val12.npy` (+ `_actions` sidecar).
- Local GPU (repo venv, cuda). pygame installed into the repo venv (2.6.1, py3.13 wheel OK).

Done means: `src/interactive/play_memmaze.py` + spec (campaign spec-delegation), selftest mode
proves reset + N rollout steps + render path run clean on cuda with sane per-step latency, and
Merlin can launch it against `checkpoints/memmaze/dynamics_vanilla.pt` (and the mem2mem arms when
they land).

## Result
DONE 2026-07-04. `src/interactive/play_memmaze.py` + `specs/interactive/play_memmaze.md` (campaign
spec-delegation). Faithful run_gui.py twin: same keymap/pacing/panels; reset = one-shot encode of
n_ctx (default = model window 32, up to 64 via long-context prefill) real val12 frames +
`rollout_init` with the TRUE recorded actions, on-screen CTX REPLAY (green border), then one
committed `rollout_step` per tick from held keys; frame = last of a trailing 16-latent temporal
decode (window-invariance probe justifies). Selftests green on the 4070: dummy + real windowed
driver + `--n-ctx 64` (2 window slides), ~108 ms/step ≈ 9.3 fps > the 6 fps target. pygame 2.6.1
installed into the repo venv. Works with `dynamics_vanilla.pt` now; mem2mem arms drop in when they
land (relay carries the replayed maze).
