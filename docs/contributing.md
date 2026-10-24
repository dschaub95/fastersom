# Contributing guide

Contributions are welcome, as issues and pull requests on [GitHub](https://github.com/dschaub95/fastersom). This page
summarises how to set up a development environment, run the checks and build the documentation. We assume that you
are familiar with git and with making pull requests on GitHub; for more extensive tutorials, see the
[scientific Python tutorials](https://learn.scientific-python.org/development/tutorials/) or the
[scanpy developer guide](https://scanpy.readthedocs.io/en/latest/dev/index.html).

## Installing dev dependencies

fastersom is managed with [uv](https://docs.astral.sh/uv/). In a clone of the repository,

```bash
uv sync                  # the package and the dev group: pytest, ruff, pre-commit, twine
uv sync --extra doc      # also the dependencies to build the documentation
```

With pip instead:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[doc]" pytest ruff pre-commit
```

## Code style

[Ruff](https://docs.astral.sh/ruff/) lints and formats the code, and [pre-commit](https://pre-commit.com/) runs it
together with uv's lock check and file checks on every commit:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

## Running tests

```bash
uv run pytest
```

The tests of the fused CUDA and Metal kernels run where a suitable GPU is available and skip otherwise. Continuous
integration runs the tests on Python 3.11 to 3.14 and checks ruff and the lock file on every pull request.

## Writing documentation

The documentation is built with [Sphinx](https://www.sphinx-doc.org/) and the
[scverse](https://scverse.org/) layout {cite:p}`Virshup_2023`:

- pages are written in Markdown with [MyST](https://myst-parser.readthedocs.io/),
- the API reference is generated from the docstrings, which are in Google style (`Args:`, `Returns:`), with type hints
  rendered by [sphinx-autodoc-typehints](https://github.com/tox-dev/sphinx-autodoc-typehints),
- tutorials are Jupyter notebooks in `docs/tutorials`, rendered by [myst-nb](https://myst-nb.readthedocs.io/) with
  their stored outputs; they are not executed when the documentation is built, so re-run a notebook after changing it,
- citations come from `docs/references.bib` through
  [sphinxcontrib-bibtex](https://sphinxcontrib-bibtex.readthedocs.io/).

To build the documentation locally:

```bash
uv run --extra doc sphinx-build -W -b html docs docs/_build/html
open docs/_build/html/index.html  # or xdg-open on Linux
```

[Read the Docs](https://fastersom.readthedocs.io) builds the documentation as configured in `.readthedocs.yaml`.

## Publishing a release

1. Update the version in `pyproject.toml` (following [semantic versioning](https://semver.org/)), run `uv lock`, and
   move the changes under "Unreleased" in `CHANGELOG.md` to a section for the new version.
2. Check the package with `uv build && uv run twine check --strict dist/*`.
3. Create a GitHub release with the tag `vX.Y.Z`. The release workflow checks that the tag matches the version, builds
   the package and publishes it to PyPI.
