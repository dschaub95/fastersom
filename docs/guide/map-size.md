# Map size and epochs

There is no single right map size or number of epochs; they depend on the data and on what the map is for. This page
collects what the parameters do and how they affect cost, and some conventions from the literature.

## Map size

`n_rows` and `n_cols` set the grid, with `n_rows * n_cols` nodes. Node `k` sits at row `k // n_cols` and column
`k % n_cols`, so `weights_.reshape(n_rows, n_cols, -1)` gives the map as an image.

What the nodes are used for decides how many are needed:

- **Nodes as an intermediate clustering.** FlowSOM {cite:p}`Van_Gassen_2015` clusters cells into a 10 x 10 grid by
  default and then groups the 100 nodes into metaclusters, typically more than the number of expected populations.
  Pixel clustering of multiplexed images often uses larger maps, such as 30 x 30 or more, followed by graph clustering
  (for example Leiden {cite:p}`Traag_2019`) of the nodes. Both are shown in the [tutorials](../tutorials/index.md).
- **Nodes as a summary or visualisation of the data.** The SOM Toolbox {cite:p}`Vesanto_2000` chooses about
  $5 n^{0.54}$ nodes for $n$ samples by default, with the ratio of the grid's sides set from the two largest
  principal components of the data.

More nodes lower the quantization error and resolve smaller groups of samples, but leave fewer samples per node, and
more nodes stay empty or represent few samples. The error measures and the U-matrix help compare sizes:

```python
for size in (10, 20, 30):
    som = SOM(size, size, random_state=0).fit(X)
    occupied = len(set(som.labels_[som.labels_ >= 0]))
    print(size, som.quantization_error(X), som.topographic_error(X), occupied)
```

### Cost

Each epoch searches the BMU of every sample, `n_samples x n_nodes x n_features` multiply-adds, which dominates the
time of `fit`; the update costs `n_nodes x (n_rows + n_cols) x n_features`, as the gaussian is applied along the grid
rows and columns separately. Doubling both sides of the grid thus makes `fit` about four times slower. Memory grows
with the number of nodes only on the torch path (see [](batch-size.md)).

## Neighbourhood width

The neighbourhood width $\sigma$ decays exponentially from `sigma_initial` towards `sigma_final`, in grid units. The
defaults are `sigma_initial = ceil(min(n_rows, n_cols) / 2) / 2` (7.5 for a 30 x 30 map, 2.5 for 10 x 10) and
`sigma_final = 0.1 * sigma_initial`, which reproduce XPySom's batch maps {cite:p}`Mancini_2020`. A large initial
width orders the map globally in the first epochs; a small final width lets the nodes fit the data locally at the end.
A larger `sigma_final` gives a smoother map with a higher quantization error.

## Epochs

`n_epochs` is the number of batch updates, each from all samples. The batch map typically converges in far fewer
passes over the data than the online SOM {cite:p}`Kohonen_2001`; the default is 10. Because the width schedule is
spread over `n_epochs`, changing it changes the schedule as well as the training time: the map is not simply trained
for longer. With `n_epochs=0`, `fit` keeps the initial weights.

The error measures help to choose a value:

```python
for n_epochs in (5, 10, 20, 40):
    som = SOM(30, 30, n_epochs=n_epochs, random_state=0).fit(X)
    print(n_epochs, som.quantization_error(X), som.topographic_error(X))
```

### Warm starts

An array as `init` starts from given weights, for example a map fitted on a subsample:

```python
coarse = SOM(30, 30, random_state=0).fit(X[::10])
som = SOM(30, 30, init=coarse.weights_, sigma_initial=1.0).fit(X)
```

The width schedule restarts at `sigma_initial`. With the default this largely retrains the map; to continue training
instead, pass a small `sigma_initial`, such as the previous map's `sigma_final`.
