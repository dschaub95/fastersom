# fastersom

A small, fast batch self-organising map (SOM) in PyTorch: a reimplementation of Kohonen's batch map with a
rectangular grid, gaussian neighbourhood and exponentially decaying neighbourhood width. Its defaults reproduce the
maps of [spatiomic](https://github.com/complextissue/spatiomic)'s `so.dimension.som`. Data stays on the device (CUDA,
MPS or CPU) and the per-epoch update uses per-node sums, so an epoch costs one BMU search plus a (nodes x nodes) matrix
product.

## Install

```bash
uv add fastersom    # or: pip install fastersom
```

## Usage

```python
import numpy as np
from fastersom import SOM

X = np.load("pixels.npy")  # (n, d) or (..., d); numpy, array-like or torch
som = SOM(n_rows=30, n_cols=30, n_epochs=10, random_state=0)  # device: cuda > mps > cpu
som.fit(X)
som.weights_  # (900, d) node weights, node = row * n_cols + col
som.labels_  # BMU index per training sample
labels = som.predict(X)  # BMU index per sample
dist = som.transform(X)  # (n, 900) distance to every node
print(som.quantization_error(X), som.topographic_error(X))
```

Options:

- `metric="cosine"` finds BMUs by cosine similarity. Zero-norm samples have no BMU, get label -1 and are left out of
  the update and the error measures.
- `init="sample"` initialises the nodes from random samples instead of seeded random unit vectors.
- `sigma_initial` and `sigma_final` set the neighbourhood width schedule; the defaults reproduce spatiomic.
- `batch_size` bounds memory (about `batch_size x n_nodes` floats) without changing the algorithm.

`BatchSOM` is an alias for `SOM`.

## Benchmark

480,000 x 34 pixels, 30 x 30 nodes, 10 epochs, fit time only (mean of seeds 0-2):

| implementation               | hardware          | fit time | quantization error |
| ---------------------------- | ----------------- | -------- | ------------------ |
| fastersom                    | NVIDIA A100       | 0.085 s  | 0.169              |
| spatiomic (CuPy)             | NVIDIA A100       | 0.46 s   | 0.169              |
| spatiomic (NumPy)            | 256-core CPU      | 40 s     | 0.169              |

## Algorithm

fastersom follows Kohonen's batch map (T. Kohonen, *Self-Organizing Maps*, 3rd ed., Springer, 2001). Each epoch assigns
every sample to its best matching unit and sets every node to the neighbourhood-weighted mean of the samples, with
neighbourhood `h(j, k) = exp(-||r_j - r_k||^2 / (2 sigma(t)^2))` on the grid coordinates `r` and
`sigma(t) = sigma_initial * (sigma_final / sigma_initial) ** (t / n_epochs)`. There is no learning rate, because it
cancels in the batch update. Only the rectangular topology, gaussian neighbourhood and exponential decay are
supported.

## Development

```bash
uv sync                      # installs the dev group: pytest, ruff, pre-commit, twine
uv run pre-commit install    # ruff, uv-lock and file checks on every commit
uv run pytest
uv build && uv run twine check --strict dist/*
```

## License

Apache-2.0
