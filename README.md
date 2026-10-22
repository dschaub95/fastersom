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
labels, dist = som.predict(X, return_distance=True)  # and the distance to it, in the same pass
dist = som.transform(X)  # (n, 900) distance to every node
umatrix = som.distance_map()  # (30, 30) summed distance to the grid neighbours, max 1
print(som.quantization_error(X), som.topographic_error(X))

som = SOM.from_weights(som.weights_, n_rows=30, n_cols=30)  # fitted map from saved weights, no data needed
```

Options:

- `metric` sets the distance used for BMUs, distances and error measures: `"euclidean"` (default), `"manhattan"`
  (L1), `"cosine"` (1 - cosine similarity) or `"correlation"` (1 - Pearson correlation). Nodes are always updated to
  the neighbourhood-weighted mean of the raw samples (not the median under manhattan, nor centred samples under
  correlation). Zero-norm samples under cosine and constant samples under correlation have no BMU: they get label -1
  and distance NaN in `predict`, distance 1 in `transform`, and are left out of the update and the error measures.
- `init="sample"` initialises the nodes from random samples instead of seeded random unit vectors. An array of shape
  `(n_rows * n_cols, d)` or `(n_rows, n_cols, d)` gives the initial weights instead, e.g. to continue training
  (`SOM(init=som.weights_).fit(X)`); with `n_epochs=0`, `fit` keeps them unchanged.
- `sigma_initial` and `sigma_final` set the neighbourhood width schedule; the defaults reproduce spatiomic.
- `batch_size` bounds memory (about `batch_size x n_nodes` floats) without changing the algorithm.

Fitted maps pickle without device tensors and load on the configured device if available, else on the default one,
so a map fitted on a GPU loads on a CPU-only machine. `BatchSOM` is an alias for `SOM`.

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
