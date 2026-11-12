# Notes for agents working on this repository

fastersom is a fast batch self-organising map (SOM) in PyTorch for GPU (CUDA, MPS) and CPU.

- Make changes on a new branch in a new worktree, branched from an up-to-date `main`:
  `git worktree add .claude/worktrees/<branch> -b <branch> origin/main` (gitignored). Skip this when already on a
  feature branch, for read-only tasks, and for edits the user asks for on the current checkout. Remove the worktree
  once its branch is merged.
- Run the tests with `hatch test` and build the docs with `hatch run docs:build`.
- The commit hooks run through prek and call hatch: install both with `uv tool install prek` and `uv tool install hatch`.
- CI runs on CPUs only and the CUDA and Metal kernel tests skip without a GPU: run them on a GPU after changing the
  kernels.
- Describe user-facing changes under "Unreleased" in `CHANGELOG.md`.
