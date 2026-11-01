# fastersom

[![PyPI](https://img.shields.io/pypi/v/fastersom)](https://pypi.org/project/fastersom/)
[![Downloads](https://static.pepy.tech/badge/fastersom)](https://pepy.tech/projects/fastersom)
[![Python](https://img.shields.io/pypi/pyversions/fastersom)](https://pypi.org/project/fastersom/)
[![Tests](https://github.com/dschaub95/fastersom/actions/workflows/test.yaml/badge.svg)](https://github.com/dschaub95/fastersom/actions/workflows/test.yaml)
[![Documentation](https://img.shields.io/readthedocs/fastersom/latest)](https://fastersom.readthedocs.io)
[![License](https://img.shields.io/pypi/l/fastersom)](https://github.com/dschaub95/fastersom/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A small, fast batch self-organising map (SOM) in PyTorch: a reimplementation of Kohonen's batch map with a
rectangular grid, gaussian neighbourhood and exponentially decaying neighbourhood width. It computes the same batch update as
[XPySom](https://github.com/Manciukic/xpysom), with a gaussian neighbourhood of initial width half the grid that
decays to a tenth of that, at most one grid unit. Data stays on the device (CUDA, MPS or CPU) and the per-epoch update smooths per-node sums with the
separable gaussian, one small matrix product along each grid axis, so an epoch costs little more than one BMU search.

The [documentation](https://fastersom.readthedocs.io) has a user guide, tutorials and the API reference.

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
- `init="sample"` initialises the nodes from random samples instead of seeded random unit vectors, and `init="pca"`
  on a grid in the plane of the samples' first two principal components. An array of shape
  `(n_rows * n_cols, d)` or `(n_rows, n_cols, d)` gives the initial weights instead, e.g. for a warm start
  (`SOM(init=som.weights_).fit(X)`); with `n_epochs=0`, `fit` keeps them unchanged. The sigma schedule restarts at
  `sigma_initial`, which with the default largely retrains the map; to continue training instead, pass a small
  `sigma_initial`, e.g. the previous map's `sigma_final`.
- `sigma_initial` and `sigma_final` set the neighbourhood width schedule. `sigma_final` defaults to
  `min(0.1 * sigma_initial, 1)`, so that large maps are not smoothed beyond the resolution of their units;
  `sigma_final=0.1 * sigma_initial` reproduces XPySom's schedule.
- `batch_size` bounds memory (about `batch_size x n_nodes` floats, `batch_size x n_features` with the fused
  [CUDA](#fused-cuda-kernels) or [Metal](#fused-metal-kernels) kernels) without changing the algorithm. On CUDA and MPS
  it defaults to the largest batch that still helps speed and fits in `memory_fraction` (default 0.9 on CUDA, 0.5 on
  MPS) of the free GPU memory, so it rarely needs tuning.
- `verbose=True` shows a progress bar over the epochs of `fit`, with an estimate of the remaining time.
- `deterministic=True` makes `fit` give the same map in every run on CUDA and MPS, whose atomic additions otherwise
  vary in order. On CUDA it sums the samples per node as 64-bit fixed-point integers (on an A100 up to 2x slower on
  small maps, 3-10% from 100 x 100 nodes), on MPS likewise with the [Metal kernels](#fused-metal-kernels) (on an M3
  Max 1.4x slower at 10 x 10 nodes, 1.14x at 30 x 30, the same from 100 x 100) and else with a matrix product (1.2-1.8x
  slower). It is also used under
  `torch.use_deterministic_algorithms(True)`. The CPU is always deterministic; maps from different devices still differ
  by float rounding.
- `fused=False` turns off the fused [CUDA](#fused-cuda-kernels) and [Metal](#fused-metal-kernels) kernels of the BMU
  search and takes the torch path instead, without compiling anything. It only matters on CUDA and MPS.
- `precision="float16"` lets `fit`'s BMU search round samples and nodes to 11 significant bits, summing the products in
  float32, as half inputs to the [Metal kernels](#fused-metal-kernels) on MPS, scaled by powers of 2 into half's range.
  It applies under euclidean, cosine and correlation from 16 features; elsewhere, and on CUDA, the search stays float32,
  as do `labels_`, `predict`, `transform` and the error measures. The BMUs of near-tied samples flip, more on larger
  maps (of clustered 0-1 pixels about 1% at 10 x 10 nodes and 10-35% from 30 x 30), which gives an equally good map
  rather than the same one: in our tests the same quantization error, but a topographic error up to 1.4x higher on
  synthetic clusters. It is meant for GPUs whose half products are much faster than float32 ones, such as the M5's,
  where it is untested; on an M3 Max `fit` is only 1.00-1.05x faster.

Fitted maps pickle without device tensors and load on the configured device if available, else on the default one,
so a map fitted on a GPU loads on a CPU-only machine. `BatchSOM` is an alias for `SOM`.

### Fused CUDA kernels

On CUDA with Triton >= 3.5 (in torch's Linux wheels from torch 2.9), the BMU search runs in fused kernels that never
store the batch x nodes distances: on tensor cores under euclidean, cosine and correlation on A100, H100 and B200 GPUs
(compute capability 8.0, 9.0 and 10.0) and, from 128 features on, on Ada GPUs (8.9), accurate to float32 by 3xTF32, and
else in float32 under manhattan and up to 128 features. On an A100 this makes `fit` under euclidean 2.3-3x faster than
torch's matrix product from 30 x 30 to 200 x 200 nodes, with a third to a sixth of the memory, and 240x faster under
manhattan. The BMUs differ from the torch path's only on near-ties. On an RTX 6000 Ada, whose TF32 tensor cores do only
2x the multiply-adds of its float32 cores, they were slower than the float32 kernel at 34 features, and from 196
features on a fit took 0.56-0.72x as long as with cuBLAS, hence the threshold; GPUs that report 9.0 with cut-down tensor
cores, such as the H20, are likely slower on them. The first use of a feature count compiles its kernel (2-4 s, then
cached on disk by Triton), which needs a C compiler and the Python headers (e.g. `python3-dev`, or a uv-managed Python),
and checks it on a tiny search. If either fails, fastersom warns and falls back from the tensor-core kernel to the
float32 kernel and from that to the torch path, which ROCm, GPUs below compute capability 7.0 (which Triton does not
support) and the CPU always use.

### Fused Metal kernels

On MPS (Apple GPUs, torch >= 2.6; tested with 2.8 and 2.14), the BMU search runs in fused Metal kernels, in float32
like the torch path: on the simdgroup matrix units under euclidean, cosine and correlation at any feature count, and on
the ALUs under manhattan up to 512 features. Apple GPUs before the M5 have no faster matrix precision than float32 to
trade for it, so there is no counterpart to 3xTF32. On an M3 Max (30-core GPU), with 34 features, this makes `fit` 4-7x
faster than torch's matrix product from 30 x 30 to 300 x 300 nodes; the search alone is 1.2-10x faster from 5 to 300
features and 10 x 10 to 500 x 500 nodes, and the search under manhattan 60-260x. The kernels compile from source at
runtime, in about 0.2 s per feature count with no Xcode or compiler needed, and pass a similar exact self-check, or
warn and leave the torch path. With `precision="float16"` they take half inputs, which on an M3 Max are no faster; the
M5's neural accelerators, which Metal 4's tensor operations reach, would be the next step.

## Benchmark

The spatiomic tutorial's 480,000 x 34 pixels, and 5,000,000 x 60 synthetic samples for the 200 x 200 map; 10 epochs,
mean of 3 seeds (one fit on the CPU at 200 x 200); 16 CPU threads; peak GPU memory includes the data.
`uv run benchmarks/memory.py` downloads and preprocesses the tutorial data, installs XPySom and torchsom into temporary
environments and prints one JSON line per row, on CUDA for fastersom with and without the fused kernels, with
the first fit's time in `first_fit_s`; `--library`, `--device` and `--size` run a part of it
([`benchmarks/memory.py`](https://github.com/dschaub95/fastersom/blob/main/benchmarks/memory.py)).

| implementation                                       | map       | hardware       | batch size          | fit time | peak GPU memory | quantization error |
| ---------------------------------------------------- | --------- | -------------- | ------------------- | -------- | --------------- | ------------------ |
| fastersom                                            | 30 x 30   | NVIDIA A100    | default (480,000)   | 0.046 s  | 234 MiB         | 0.168              |
| fastersom, `fused=False`                             | 30 x 30   | NVIDIA A100    | default (298,261)   | 0.11 s   | 1150 MiB        | 0.168              |
| fastersom                                            | 30 x 30   | NVIDIA A100    | 32,768              | 0.052 s  | 90 MiB          | 0.168              |
| fastersom, `fused=False`                             | 30 x 30   | NVIDIA A100    | 32,768              | 0.10 s   | 204 MiB         | 0.168              |
| fastersom                                            | 30 x 30   | NVIDIA A100    | 8,096               | 0.13 s   | 90 MiB          | 0.168              |
| fastersom, `fused=False`                             | 30 x 30   | NVIDIA A100    | 8,096               | 0.13 s   | 118 MiB         | 0.168              |
| [XPySom](https://github.com/Manciukic/xpysom) (CuPy) | 30 x 30   | NVIDIA A100    | 8,096               | 0.46 s   | 313 MiB         | 0.169              |
| XPySom (CuPy)                                        | 30 x 30   | NVIDIA A100    | 65,536              | 0.41 s   | 2088 MiB        | 0.169              |
| [torchsom](https://github.com/michelin/TorchSOM)     | 30 x 30   | NVIDIA A100    | 1,024               | 22.2 s   | 44 MiB          | 0.175              |
| fastersom                                            | 30 x 30   | 16 CPU threads | default (8,192)     | 1.8 s    | -               | 0.169              |
| XPySom (NumPy)                                       | 30 x 30   | 16 CPU threads | 8,096               | 25.6 s   | -               | 0.168              |
| torchsom                                             | 30 x 30   | 16 CPU threads | 1,024               | 36.2 s   | -               | 0.176              |
| fastersom                                            | 100 x 100 | NVIDIA A100    | default (480,000)   | 0.18 s   | 298 MiB         | 0.162              |
| fastersom, `fused=False`                             | 100 x 100 | NVIDIA A100    | default (26,843)    | 0.51 s   | 2138 MiB        | 0.162              |
| fastersom                                            | 100 x 100 | NVIDIA A100    | 8,096               | 0.21 s   | 110 MiB         | 0.162              |
| fastersom, `fused=False`                             | 100 x 100 | NVIDIA A100    | 8,096               | 0.52 s   | 400 MiB         | 0.162              |
| XPySom (CuPy)                                        | 100 x 100 | NVIDIA A100    | 8,096               | 2.8 s    | 2846 MiB        | 0.162              |
| fastersom                                            | 100 x 100 | 16 CPU threads | default (8,192)     | 62.4 s   | -               | 0.162              |
| fastersom                                            | 200 x 200 | NVIDIA A100    | default (1,118,481) | 11.8 s   | 2244 MiB        | 0.380              |
| fastersom, `fused=False`                             | 200 x 200 | NVIDIA A100    | default (6,710)     | 29.4 s   | 3294 MiB        | 0.380              |
| fastersom                                            | 200 x 200 | NVIDIA A100    | 8,096               | 12.3 s   | 1346 MiB        | 0.380              |
| fastersom, `fused=False`                             | 200 x 200 | NVIDIA A100    | 8,096               | 28.7 s   | 3718 MiB        | 0.380              |
| XPySom (CuPy)                                        | 200 x 200 | NVIDIA A100    | 8,096               | 118.8 s  | 12292 MiB       | 0.380              |
| fastersom                                            | 200 x 200 | 16 CPU threads | default (8,192)     | 43 min   | -               | 0.380              |

With less GPU memory fastersom is about 10x faster than XPySom on a 30 x 30 map, 16x faster on a 100 x 100 map
(with a tenth of the memory) and 10x faster on a 200 x 200 map (with a ninth of the memory), and 14x faster on the CPU.
torchsom updates the map per mini-batch, so it needs small batches and is far slower.

The fused CUDA kernels make fastersom 2-3x faster than its torch path (`fused=False`) with a seventh to two thirds of
the memory, except at small batches on small maps. Fit times exclude the first fit in a process, which compiles and
checks the kernels: with an empty Triton cache it takes 2.4-2.9 s longer than with `fused=False` (30 x 30: 3.0 s
instead of 0.6 s), and with the kernels cached on disk by an earlier run 0.6 s longer.

## Algorithm

fastersom follows Kohonen's batch map (T. Kohonen, *Self-Organizing Maps*, 3rd ed., Springer, 2001). Each epoch assigns
every sample to its best matching unit and sets every node to the neighbourhood-weighted mean of the samples, with
neighbourhood `h(j, k) = exp(-||r_j - r_k||^2 / (2 sigma(t)^2))` on the grid coordinates `r` and
`sigma(t) = sigma_initial * (sigma_final / sigma_initial) ** (t / n_epochs)`. There is no learning rate, because it
cancels in the batch update. Only the rectangular topology, gaussian neighbourhood and exponential decay are
supported.

## Development

The repository follows the [scverse cookiecutter template](https://github.com/scverse/cookiecutter-scverse) and
receives its updates through [cruft](https://cruft.github.io/cruft/) (`.cruft.json`). With [hatch](https://hatch.pypa.io/):

```bash
uvx prek install             # ruff, biome, pyproject-fmt, mypy and file checks on every commit
hatch test                   # the tests (hatch test --all: every Python version of the matrix)
hatch check types            # mypy
hatch run docs:build         # the documentation
```

See the [contributing guide](https://fastersom.readthedocs.io/en/latest/contributing.html) for details.

## License

Apache-2.0
