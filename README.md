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
- `batch_size` bounds memory (about `batch_size x n_nodes` floats, `batch_size x n_features` with the fused CUDA
  kernels below) without changing the algorithm. On CUDA and MPS it defaults to the largest batch that still helps
  speed and fits in `memory_fraction` (default 0.9 on CUDA, 0.5 on MPS) of the free GPU memory, so it rarely needs
  tuning.
- `verbose=True` shows a progress bar over the epochs of `fit`, with an estimate of the remaining time.
- `deterministic=True` makes `fit` give the same map in every run on CUDA and MPS, whose atomic additions otherwise
  vary in order. On CUDA it sums the samples per node as 64-bit fixed-point integers (on an A100 up to 2x slower on
  small maps, 3-10% from 100 x 100 nodes), on MPS with a matrix product (1.2-1.8x slower). It is also used under
  `torch.use_deterministic_algorithms(True)`. The CPU is always deterministic; maps from different devices still differ
  by float rounding.

On CUDA with Triton >= 3.5 (in torch's Linux wheels from torch 2.9), the BMU search runs in fused kernels that never
store the batch x nodes distances: on tensor cores under euclidean, cosine and correlation on A100, H100 and B200 GPUs
(compute capability 8.0, 9.0 and 10.0), accurate to float32 by 3xTF32, and else in float32 under manhattan and up to
128 features. On an A100 this makes `fit` under euclidean 2.3-3x faster than torch's matrix product from 30 x 30 to
200 x 200 nodes, with a third to a sixth of the memory, and 240x faster under manhattan. The BMUs differ from the
torch path's only on near-ties. Only the A100 has been tested; GPUs that report 9.0 with cut-down tensor cores, such
as the H20, are likely slower on tensor cores. The first use of a feature count compiles its kernel (2-4 s, then
cached on disk by Triton), which needs a C compiler and the Python headers (e.g. `python3-dev`, or a uv-managed
Python), and checks it on a tiny search. If either fails, fastersom warns and falls back from the tensor-core kernel
to the float32 kernel and from that to the torch path, which ROCm, MPS and the CPU always use.

Fitted maps pickle without device tensors and load on the configured device if available, else on the default one,
so a map fitted on a GPU loads on a CPU-only machine. `BatchSOM` is an alias for `SOM`.

## Benchmark

480,000 x 34 pixels, 10 epochs, mean of 3 seeds; peak GPU memory includes the data
([`benchmarks/memory.py`](https://github.com/dschaub95/fastersom/blob/main/benchmarks/memory.py)).

| implementation                                       | map       | hardware     | batch size        | fit time | peak GPU memory | quantization error |
| ---------------------------------------------------- | --------- | ------------ | ----------------- | -------- | --------------- | ------------------ |
| fastersom                                            | 30 x 30   | NVIDIA A100  | default (480,000) | 0.048 s  | 234 MiB         | 0.168              |
| fastersom                                            | 30 x 30   | NVIDIA A100  | 32,768            | 0.054 s  | 90 MiB          | 0.168              |
| fastersom                                            | 30 x 30   | NVIDIA A100  | 8,096             | 0.13 s   | 90 MiB          | 0.168              |
| [XPySom](https://github.com/Manciukic/xpysom) (CuPy) | 30 x 30   | NVIDIA A100  | 8,096             | 0.47 s   | 313 MiB         | 0.169              |
| XPySom (CuPy)                                        | 30 x 30   | NVIDIA A100  | 65,536            | 0.41 s   | 2088 MiB        | 0.169              |
| [torchsom](https://github.com/michelin/TorchSOM)     | 30 x 30   | NVIDIA A100  | 1,024             | 22.3 s   | 44 MiB          | 0.175              |
| fastersom                                            | 30 x 30   | 256-core CPU | default (8,192)   | 1.8 s    | -               | 0.169              |
| XPySom (NumPy)                                       | 30 x 30   | 256-core CPU | 8,096             | 42.3 s   | -               | 0.168              |
| torchsom                                             | 30 x 30   | 256-core CPU | 1,024             | 39.2 s   | -               | 0.176              |
| fastersom                                            | 100 x 100 | NVIDIA A100  | default (480,000) | 0.18 s   | 298 MiB         | 0.162              |
| fastersom                                            | 100 x 100 | NVIDIA A100  | 8,096             | 0.21 s   | 110 MiB         | 0.162              |
| XPySom (CuPy)                                        | 100 x 100 | NVIDIA A100  | 8,096             | 2.8 s    | 2846 MiB        | 0.162              |
| fastersom                                            | 100 x 100 | 256-core CPU | default (8,192)   | 52.7 s   | -               | 0.162              |

With less GPU memory fastersom is about 10x faster than XPySom on a 30 x 30 map and 13x faster on a 100 x 100 map
(at batch size 8,096, with a twenty-fifth of the memory), and over 20x faster on the CPU. torchsom updates the map per
mini-batch, so it needs small batches and is far slower.

[`benchmarks/fused.py`](https://github.com/dschaub95/fastersom/blob/main/benchmarks/fused.py) compares fit time and
peak memory on CUDA over map sizes, feature counts, metrics and `deterministic` between versions.

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
