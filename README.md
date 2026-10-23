# fastersom

A small, fast batch self-organising map (SOM) in PyTorch: a reimplementation of Kohonen's batch map with a
rectangular grid, gaussian neighbourhood and exponentially decaying neighbourhood width. Its defaults reproduce the
batch-trained maps of [XPySom](https://github.com/Manciukic/xpysom) with a gaussian neighbourhood of initial width
half the grid. Data stays on the device (CUDA, MPS or CPU) and the per-epoch update smooths per-node sums with the
separable gaussian, one small matrix product along each grid axis, so an epoch costs little more than one BMU search.

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
- `sigma_initial` and `sigma_final` set the neighbourhood width schedule; the defaults reproduce XPySom's.
- `batch_size` bounds memory (about `batch_size x n_nodes` floats) without changing the algorithm. On CUDA and MPS
  it defaults to the largest batch that still helps speed and fits in `memory_fraction` (default 0.9 on CUDA, 0.5
  on MPS) of the free GPU memory, so it rarely needs tuning.
- `verbose=True` shows a progress bar over the epochs of `fit`, with an estimate of the remaining time.
- `deterministic=True` makes `fit` give the same map in every run on CUDA and MPS, whose atomic additions otherwise
  vary in order (1.2-1.8x slower on MPS). It is also used under `torch.use_deterministic_algorithms(True)`. The CPU is
  always deterministic; maps from different devices still differ by float rounding.

Fitted maps pickle without device tensors and load on the configured device if available, else on the default one,
so a map fitted on a GPU loads on a CPU-only machine. `BatchSOM` is an alias for `SOM`.

## Benchmark

The spatiomic tutorial's 480,000 x 34 pixels, and 5,000,000 x 60 synthetic samples for the 200 x 200 map; 10 epochs,
mean of 3 seeds (one fit on the CPU at 200 x 200); 16 CPU threads; peak GPU memory includes the data.
`uv run benchmarks/memory.py` downloads and preprocesses the tutorial data, installs XPySom and torchsom into temporary
environments and prints one JSON line per row; `--library`, `--device` and `--size` run a part of it
([`benchmarks/memory.py`](https://github.com/dschaub95/fastersom/blob/main/benchmarks/memory.py)).

| implementation                                       | map       | hardware       | batch size        | fit time | peak GPU memory | quantization error |
| ---------------------------------------------------- | --------- | -------------- | ----------------- | -------- | --------------- | ------------------ |
| fastersom                                            | 30 x 30   | NVIDIA A100    | default (298,261) | 0.12 s   | 1150 MiB        | 0.168              |
| fastersom                                            | 30 x 30   | NVIDIA A100    | 32,768            | 0.10 s   | 204 MiB         | 0.168              |
| fastersom                                            | 30 x 30   | NVIDIA A100    | 8,096             | 0.13 s   | 118 MiB         | 0.168              |
| [XPySom](https://github.com/Manciukic/xpysom) (CuPy) | 30 x 30   | NVIDIA A100    | 8,096             | 0.46 s   | 313 MiB         | 0.169              |
| XPySom (CuPy)                                        | 30 x 30   | NVIDIA A100    | 65,536            | 0.41 s   | 2088 MiB        | 0.169              |
| [torchsom](https://github.com/michelin/TorchSOM)     | 30 x 30   | NVIDIA A100    | 1,024             | 22.2 s   | 44 MiB          | 0.175              |
| fastersom                                            | 30 x 30   | 16 CPU threads | default (8,192)   | 1.8 s    | -               | 0.169              |
| XPySom (NumPy)                                       | 30 x 30   | 16 CPU threads | 8,096             | 25.6 s   | -               | 0.168              |
| torchsom                                             | 30 x 30   | 16 CPU threads | 1,024             | 36.2 s   | -               | 0.176              |
| fastersom                                            | 100 x 100 | NVIDIA A100    | default (26,843)  | 0.50 s   | 1114 MiB        | 0.162              |
| fastersom                                            | 100 x 100 | NVIDIA A100    | 8,096             | 0.52 s   | 400 MiB         | 0.162              |
| XPySom (CuPy)                                        | 100 x 100 | NVIDIA A100    | 8,096             | 2.8 s    | 2846 MiB        | 0.162              |
| fastersom                                            | 100 x 100 | 16 CPU threads | default (8,192)   | 62.4 s   | -               | 0.162              |
| fastersom                                            | 200 x 200 | NVIDIA A100    | default (6,710)   | 29.4 s   | 3294 MiB        | 0.380              |
| fastersom                                            | 200 x 200 | NVIDIA A100    | 8,096             | 28.7 s   | 3718 MiB        | 0.380              |
| XPySom (CuPy)                                        | 200 x 200 | NVIDIA A100    | 8,096             | 118.8 s  | 12292 MiB       | 0.380              |
| fastersom                                            | 200 x 200 | 16 CPU threads | default (8,192)   | 43 min   | -               | 0.368              |

At equal or less GPU memory fastersom is about 4x faster than XPySom on a 30 x 30 map, 5x faster on a 100 x 100 map
(with a seventh of the memory) and 4x faster on a 200 x 200 map (with under a third of the memory), and 14x faster on
the CPU. torchsom updates the map per mini-batch, so it needs small batches and is far slower.

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
