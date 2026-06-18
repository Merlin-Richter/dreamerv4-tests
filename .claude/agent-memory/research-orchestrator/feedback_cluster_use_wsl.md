---
name: cluster-use-wsl
description: Cluster wrapper scripts (scripts/) and the ssh master socket MUST run in WSL, not Git Bash/PowerShell
metadata:
  type: feedback
---

Run ALL `scripts/` cluster wrappers — and the `open_master.sh` master socket — in **WSL**, never
Git Bash or PowerShell.

**Why:** WSL, Git Bash (MSYS2), and PowerShell are separate ssh stacks with separate $HOME and
separate unix-socket namespaces. The ControlMaster socket (the "2FA once, reuse" model) is only
shared if the human opening it and the orchestrator running the verbs use the SAME env. We hit a
spurious `ERROR: AUTH_DEAD` because Merlin opened the master in WSL while the orchestrator's Bash
tool ran the verb in Git Bash (couldn't see the WSL socket). Decided in D-036. WSL is also the most
robust ssh+rsync stack here and Merlin's natural env. PowerShell can't run the bash scripts and
Windows-native ssh has no ControlMaster at all.

**How to apply:** The orchestrator's Bash tool defaults to Git Bash, so invoke every cluster verb
through WSL explicitly:
`wsl.exe -e bash -lc "cd /mnt/c/Users/richt/OneDrive/Desktop/Code/transformer && bash scripts/<verb> ..."`.
Deliberate split (don't "fix" it): local 4070 *training* stays in Windows/Git-Bash (CUDA venv
`venv/Scripts/python.exe`); only *cluster orchestration* is WSL. See HOWTO/cluster.md + scripts/README.md.
