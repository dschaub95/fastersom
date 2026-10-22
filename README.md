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
  `(n_rows * n_cols, d)` or `(n_rows, n_cols, d)` gives the initial weights instead, e.g. for a warm start
  (`SOM(init=som.weights_).fit(X)`); with `n_epochs=0`, `fit` keeps them unchanged. The sigma schedule restarts at
  `sigma_initial`, which with the default largely retrains the map; to continue training instead, pass a small
  `sigma_initial`, e.g. the previous map's `sigma_final`.
- `sigma_initial` and `sigma_final` set the neighbourhood width schedule; the defaults reproduce spatiomic.
- `batch_size` bounds memory (about `batch_size x n_nodes` floats) without changing the algorithm. On CUDA and MPS
  it defaults to the largest batch that still helps speed and fits in `memory_fraction` (default 0.9 on CUDA, 0.5
  on MPS) of the free GPU memory, so it rarely needs tuning.

Fitted maps pickle without device tensors and load on the configured device if available, else on the default one,
so a map fitted on a GPU loads on a CPU-only machine. `BatchSOM` is an alias for `SOM`.

## Benchmark

480,000 x 34 pixels (the spatiomic tutorial's subsample), 30 x 30 nodes, 10 epochs, fit time (mean of seeds 0-2) and
peak GPU memory including the data and the allocator cache, each configuration in its own process
([`benchmarks/memory.py`](benchmarks/memory.py)):

| implementation    | hardware     | batch size         | fit time | peak GPU memory | quantization error |
| ----------------- | ------------ | ------------------ | -------- | --------------- | ------------------ |
| fastersom         | NVIDIA A100  | default (298,261)  | 0.12 s   | 1150 MiB        | 0.168              |
| fastersom         | NVIDIA A100  | 32,768             | 0.12 s   | 204 MiB         | 0.168              |
| fastersom         | NVIDIA A100  | 8,096              | 0.20 s   | 118 MiB         | 0.168              |
| spatiomic (CuPy)  | NVIDIA A100  | default (8,096)    | 0.46 s   | 340 MiB         | 0.169              |
| spatiomic (CuPy)  | NVIDIA A100  | 65,536             | 0.41 s   | 2250 MiB        | 0.169              |
| spatiomic (NumPy) | 256-core CPU | default            | 40 s     | -               | 0.169              |

With less memory than spatiomic's default (204 vs 340 MiB), fastersom fits about 4x faster; at the same batch size it
needs about a third of the memory and is about 2x faster. Beyond about 32,768 samples per batch neither library gets
much faster, so on this small dataset fastersom's default batch size uses more memory for no gain; it only takes what
is free and shrinks on a fuller GPU. The CPU row was measured separately.

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
