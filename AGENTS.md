# Notes for agents working on this repository

README.md introduces the package; `docs/contributing.md` explains the development setup, tests, documentation and
releases.

## Development

- If you are not already on a feature branch or in a worktree, make changes on a new branch in a new worktree
  rather than on `main`: `git worktree add .claude/worktrees/<branch> -b <branch> origin/main` (the folder is
  gitignored). Branch from an up-to-date `main` unless the work builds on another branch.
- Read-only tasks, and edits the user asks for on the current checkout, need no new branch.
- Remove the worktree (`git worktree remove`) once its branch is merged.

## Checks

- Run the tests with `hatch test` (or `uv run pytest`), and build the documentation with `hatch run docs:build`.
- Commits run the hooks in `.pre-commit-config.yaml` through prek; the mypy hook calls `hatch check types`, so `prek`
  and `hatch` must be installed (e.g. `uv tool install prek hatch`).
- The tests of the fused CUDA and Metal kernels skip without a suitable GPU, and CI runs on CPUs only: run them on an
  NVIDIA or Apple GPU after changing the kernels.

## Changelog

- Describe user-facing changes under "Unreleased" in `CHANGELOG.md`.
