# Use the venv Python for ALL training (CUDA)

**The bash default `python` is CPU-only torch; training MUST use the project venv.**

- `which python` in this Git-Bash env → `/c/Python314/python` → **torch 2.11.0+cpu, CUDA False**.
  Running training with it silently falls to CPU (~177 s/iteration) and then **segfaults** (CPU build
  + large mmap). The harness may even report "exit 0" while the underlying command exited 139.
- The real env is **`venv/Scripts/python.exe`** → **torch 2.12.0+cu126, CUDA True** (the RTX 4070).

## Rule
Run every training / GPU job with the venv interpreter explicitly:

```bash
venv/Scripts/python.exe -u src/training/train_tokenizer.py --frames gridworld.npy ...
venv/Scripts/python.exe -u src/training/train_dynamics.py ...
```

Pure-CPU work (env sims, datagen, numpy/cv2 eval, the gate tests `test_gridworld*.py`) runs fine on
the default `python` — only torch-GPU code needs the venv. When a run shows GPU 0% / nvidia-smi idle
or absurd s/it, check `torch.cuda.is_available()` with the interpreter you actually launched.

## Smoke-test scope
Don't launch a multi-epoch run over the full 6.9 GB `gridworld.npy` as a "smoke" — even on GPU, the
memmap + shuffle is heavy. For a true smoke, subset episodes or cap steps; promote to full data once
the pipeline is confirmed clean.
