# tasks/ — file-per-task backlog

Each task is one markdown file; **the filename is its short description**
(e.g. `add-bg-colour-to-recall.md`). The file's contents are the details + what "done" means.

State = which folder it's in:
- `backlog/`     — Merlin adds tasks here.
- `in-progress/` — the AI moves a task here when it starts (`git mv`).
- `done/`        — moved here when finished; append a one-line result to the file.
- `archive/`     — dropped / superseded tasks.

Status at a glance: print the tree (`find tasks -type f` or `ls -R tasks`) — no need to open files.
