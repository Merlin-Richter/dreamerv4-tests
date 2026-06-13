# GPU / CUDA: use the project venv python

**The orchestrator's bare `python` (C:\Python314\python) is CPU-only.** `torch.cuda.is_available()`
returns False there. The CUDA torch build lives in the project venv:

    venv/Scripts/python.exe    # <- has CUDA (torch.cuda.is_available() == True)

## Rules
- **Any GPU work (training, long rollouts) MUST use `venv/Scripts/python.exe`**, not `python`.
  Run scripts default to it via `PY="${PY:-venv/Scripts/python.exe}"` (see experiments/EXP-012/run.sh).
- Lightweight inference/analysis (probe dry-runs, the EXP-011 diagnostic, smoke tests) runs fine
  on CPU with bare `python` — results reconcile with GPU runs (deterministic-ish inference;
  numerics negligible). EXP-011 was CPU and matched EXP-010's GPU numbers. But it is SLOW for
  anything heavy — prefer the venv python when a GPU helps.
- A 100-epoch dynamics train is ~2.6h on the 4070 via the venv; on CPU it is infeasible. Always
  confirm `venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"` is True
  before launching a training run, and grep the train.log for the first `Epoch 1` line to confirm
  it is actually progressing (and on GPU, ~90s/epoch — CPU would be minutes/epoch).

## Why
The cluster wrappers (T-003) are not built, so all training currently runs locally on the 4070.
The harness shell's default interpreter is a separate system Python without the CUDA wheel.
