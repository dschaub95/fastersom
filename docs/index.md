# fastersom

A small, fast batch self-organising map (SOM) in [PyTorch](https://pytorch.org): a reimplementation of Kohonen's
batch map {cite:p}`Kohonen_2001` with a rectangular grid, by default with a gaussian neighbourhood and exponentially
decaying neighbourhood width; other neighbourhoods, grid distances and width schedules, such as FlowSOM's, are
[options](guide/neighbourhoods.md). It computes the same batch update as [XPySom](https://github.com/Manciukic/xpysom)
{cite:p}`Mancini_2020`, with a gaussian neighbourhood of initial width half the grid that decays to a tenth of that, at
most one grid unit. On maps up to 40 x 40 nodes the defaults thus reproduce XPySom's batch-trained maps; on larger maps,
pass `sigma_final=0.1 * sigma_initial` for XPySom's schedule.

The data stays on the device (CUDA, Apple's MPS or the CPU), the best matching unit (BMU) search runs in fused
kernels on NVIDIA and Apple GPUs, and the per-epoch update smooths per-node sums with the separable gaussian, one small
matrix product along each grid axis, so an epoch costs little more than one BMU search. The interface follows
scikit-learn: `fit`, `predict`, `transform` and fitted attributes ending in `_`.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} User guide {octicon}`book;1em;`
:link: guide/index
:link-type: doc

Devices and fused kernels, batch size and memory, reproducibility and precision, distances, map size and epochs.
:::

:::{grid-item-card} Tutorials {octicon}`play;1em;`
:link: tutorials/index
:link-type: doc

Clustering mass cytometry like FlowSOM, and pixel clustering of multiplexed images, with AnnData and scanpy.
:::

:::{grid-item-card} API reference {octicon}`code;1em;`
:link: api
:link-type: doc

The {class}`~fastersom.SOM` class, its parameters, methods and fitted attributes.
:::

:::{grid-item-card} GitHub {octicon}`mark-github;1em;`
:link: https://github.com/dschaub95/fastersom

Source code, issues and the benchmark scripts.
:::
::::

## Installation

fastersom needs Python 3.12 or newer and PyTorch 2.3 or newer.

```bash
pip install fastersom
# or
uv add fastersom
```

Install PyTorch for your accelerator first if pip's default wheel does not match it (see
[pytorch.org](https://pytorch.org/get-started/locally/)). The fused CUDA kernels need Triton 3.5 or newer, which
torch's Linux wheels include from torch 2.9; the fused Metal kernels need torch 2.6 or newer on MPS. See
[](guide/devices.md).

## Quick start

```python
import numpy as np
from fastersom import SOM

X = np.random.default_rng(0).random((100_000, 20), dtype=np.float32)  # (n, d) or (..., d)

som = SOM(n_rows=30, n_cols=30, n_epochs=10, random_state=0)  # device: cuda > mps > cpu
som.fit(X)

som.weights_  # (900, 20) node weights, node = row * n_cols + col
som.labels_  # BMU index per training sample
labels = som.predict(X)  # BMU index per sample
labels, dist = som.predict(X, return_distance=True)  # and the distance to it, in the same pass
dist = som.transform(X[:1000])  # (1000, 900) distance to every node
umatrix = som.distance_map()  # (30, 30) U-matrix, max 1
print(som.quantization_error(X), som.topographic_error(X))

som = SOM.from_weights(som.weights_, n_rows=30, n_cols=30)  # fitted map from saved weights, no data needed
```

`X` can be a NumPy array, any array-like, a torch tensor or a DLPack array such as CuPy; arrays with more than two
dimensions, such as images of shape `(height, width, channels)`, are flattened over all but the last axis. Fitted maps
pickle without device tensors, so a map fitted on a GPU loads on a CPU-only machine.

## Algorithm

Each epoch assigns every sample $x_i$ to its BMU $c(x_i)$ and sets every node to the neighbourhood-weighted mean of
the samples,

$$
w_k = \frac{\sum_i h(c(x_i), k)\, x_i}{\sum_i h(c(x_i), k)}, \qquad
h(j, k) = \exp\left(-\frac{\lVert r_j - r_k \rVert^2}{2 \sigma(t)^2}\right),
$$

with grid coordinates $r$ and $\sigma(t) = \sigma_\text{initial} (\sigma_\text{final} / \sigma_\text{initial})^{t /
n_\text{epochs}}$, where $t = 0, \ldots, n_\text{epochs} - 1$; a node whose denominator is not positive keeps its
weights. There is no learning rate, because it cancels in the batch update {cite:p}`Kohonen_2001,Vesanto_2000`. These
are the defaults; [](guide/neighbourhoods.md) describes the other neighbourhood functions, grid distances and width
schedules. Only the rectangular topology is supported.

```{toctree}
:hidden: true
:maxdepth: 2

guide/index
tutorials/index
api
changelog
contributing
references
```
