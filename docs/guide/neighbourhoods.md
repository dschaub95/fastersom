# Neighbourhoods and width schedules

Each epoch sets every node $k$ to the neighbourhood-weighted mean of the samples,

$$
w_k = \frac{\sum_i h\big(d(r_{c(x_i)}, r_k); \sigma(t)\big)\, x_i}{\sum_i h\big(d(r_{c(x_i)}, r_k); \sigma(t)\big)},
$$

where $c(x_i)$ is the BMU of sample $x_i$, $r_j$ the grid position (row, column) of node $j$, $d$ a distance on the
grid, $h$ the neighbourhood function and $\sigma(t)$ the neighbourhood width in epoch $t = 0, \ldots, n_\text{epochs} -
1$. A node whose denominator is not positive keeps its weights. Three parameters choose $d$, $h$ and $\sigma(t)$; the
defaults are the gaussian on the euclidean grid distance with an exponentially decaying width.

## Grid distance

`grid_metric` sets the distance between the grid positions of two nodes that lie $\Delta_r$ rows and $\Delta_c$ columns
apart:

| `grid_metric` | $d$ | nodes within $d \le 1$ |
| --- | --- | --- |
| `"euclidean"` (default) | $\sqrt{\Delta_r^2 + \Delta_c^2}$ | 4 neighbours |
| `"chebyshev"` | $\max(\lvert\Delta_r\rvert, \lvert\Delta_c\rvert)$ | 8 neighbours (a square) |
| `"manhattan"` | $\lvert\Delta_r\rvert + \lvert\Delta_c\rvert$ | 4 neighbours (a diamond) |

`grid_metric` is a distance on the grid, unlike `metric`, which is the distance between samples and nodes. The
topographic error and the U-matrix always count the 8 surrounding nodes as neighbours.

## Neighbourhood functions

`neighbourhood` sets $h(d; \sigma)$. For all of them $\sigma$ is in grid units: the standard deviation of the gaussian
and the mexican hat, the radius of the others, and both for the cut gaussian. A constant factor in $h$ would cancel in
the update, so none is normalised.

| `neighbourhood` | $h(d; \sigma)$ | zero for |
| --- | --- | --- |
| `"gaussian"` (default) | $\exp\left(-\frac{d^2}{2\sigma^2}\right)$ | never |
| `"bubble"` | $1$ if $d \le \sigma$, else $0$ | $d > \sigma$ |
| `"triangle"` | $\max\left(0, 1 - \frac{d}{\sigma}\right)$ | $d \ge \sigma$ |
| `"epanechnikov"` | $\max\left(0, 1 - \frac{d^2}{\sigma^2}\right)$ | $d \ge \sigma$ |
| `"cut_gaussian"` | $\exp\left(-\frac{d^2}{2\sigma^2}\right)$ if $d \le \sigma$, else $0$ | $d > \sigma$ |
| `"mexican_hat"` | $\left(1 - \frac{d^2}{\sigma^2}\right) \exp\left(-\frac{d^2}{2\sigma^2}\right)$ | $d = \sigma$; negative beyond |

At $\sigma = 0$ every function leaves only the winner, $h(d; 0) = 1$ if $d = 0$ and else 0, the limit of each as
$\sigma \to 0$. An epoch with $\sigma = 0$ is then a step of batch k-means (Lloyd's algorithm): every node with samples
moves to their mean. The same holds for the bubble and the cut gaussian with any $\sigma < 1$, and for the triangle and
the Epanechnikov function with $\sigma \le 1$.

The mexican hat weights nodes beyond $\sigma$ negatively, which suits the online SOM better than the batch map: where
the weights of a node's neighbourhood nearly cancel, its denominator is small and the node can move far outside the
data. Nodes whose denominator is 0 or negative keep their weights.

## Width schedules

`sigma_schedule` sets how $\sigma$ moves from `sigma_initial` ($\sigma_0$) to `sigma_final` ($\sigma_1$), with $s$
running from 0 in the first epoch to 1 at the end set by `schedule_end`:

| `sigma_schedule` | $\sigma(t)$ | requires |
| --- | --- | --- |
| `"exponential"` (default) | $\sigma_0 \left(\sigma_1 / \sigma_0\right)^s$ | $\sigma_0, \sigma_1 > 0$ |
| `"linear"` | $\sigma_0 + (\sigma_1 - \sigma_0)\, s$ | $\sigma_0, \sigma_1 \ge 0$ |
| `"inverse"` | $\sigma_0 / \left(1 + (\sigma_0 / \sigma_1 - 1)\, s\right)$ | $\sigma_0, \sigma_1 > 0$ |

An exponential or inverse decay never reaches 0, so a schedule that ends at width 0 is linear (or given as a sequence,
below).

| `schedule_end` | $s$ | last epoch |
| --- | --- | --- |
| `"last_epoch"` (default under linear and inverse) | $t / (n_\text{epochs} - 1)$ | $\sigma_1$ |
| `"after_last_epoch"` (default under exponential) | $t / n_\text{epochs}$ | wider than $\sigma_1$ |

With `"last_epoch"` the last epoch uses `sigma_final`: with `sigma_final=0` it updates every node from its own samples
only, a step of batch k-means. With `"after_last_epoch"`, $\sigma_1$ is the width the schedule would reach one epoch
after the last, as in MiniSom: a linear decay to 0 then ends at $\sigma_0 / n_\text{epochs}$, which on a large map is
still several units (with FlowSOM's initial radius and 10 epochs 2.9 units at 50 x 50 nodes, 4.6 at 80 x 80).
fastersom's exponential default has always ended after the last epoch, and keeps doing so, so that its maps do not
change. A single epoch uses $\sigma_0$ under either end.

Any other schedule can be given as the widths themselves, a sequence of `n_epochs` values $\ge 0$, with
`sigma_initial`, `sigma_final` and `schedule_end` left at `None`:

```python
import numpy as np
from fastersom import SOM

som = SOM(10, 10, n_epochs=10, sigma_schedule=6 * (1 - (np.arange(10) + 0.5) / 10))  # linear, epoch midpoints
```

The defaults are `sigma_initial = ceil(min(n_rows, n_cols) / 2) / 2` and `sigma_final = min(0.1 * sigma_initial, 1)`
for every neighbourhood and schedule (see [](map-size.md)). They were chosen for the gaussian, whose $\sigma$ is a
standard deviation; as a radius they make a narrower neighbourhood, and with a compact function the last epochs, once
$\sigma \le 1$, are k-means steps. `sigma_initial="flowsom"` instead takes FlowSOM's initial radius
{cite:p}`Van_Gassen_2015`: the 0.67 quantile (R's default, type 7) of the grid distances between all pairs of nodes,
each node with itself included. FlowSOM measures them under chebyshev, which gives 6 for its 10 x 10 grid;
fastersom measures them under `grid_metric`.

## Settings of other implementations

The batch map has no learning rate: in Kohonen's batch map it would cancel in the update. Only the neighbourhood and
its width carry over from online SOMs, whose schedules count samples rather than epochs, so the correspondences below
hold per epoch.

FlowSOM (R) {cite:p}`Van_Gassen_2015`
: Bubble on the chebyshev grid distance, inclusive at the radius, which decreases linearly from the 0.67 quantile of
  the grid distances to 0 over `rlen` passes through the data:

  ```python
  SOM(10, 10, n_epochs=10, neighbourhood="bubble", grid_metric="chebyshev", sigma_schedule="linear",
      sigma_initial="flowsom", sigma_final=0)
  ```

  The radius falls from 6 (for 10 x 10 nodes) in the first epoch to 0 in the last, which is a batch k-means step,
  and stays fixed within each epoch. FlowSOM shrinks it after every sample instead, from 6 at the start of the first
  pass to almost 0 at the end of the last; its radius at the start of every pass is that of
  `schedule_end="after_last_epoch"`, and its radius in the middle of every pass is the sequence in the example above
  (for `rlen=10`). Once below 1, FlowSOM's radius is set to 0.5, which like any radius below 1 leaves only the winner.
  FlowSOM numbers its nodes along `xdim` first, so its `xdim` corresponds to `n_cols`.

MiniSom
: The gaussian (on the euclidean grid distance) and the mexican hat match. MiniSom's bubble updates the nodes with
  $\lvert\Delta_r\rvert < \sigma$ and $\lvert\Delta_c\rvert < \sigma$, the chebyshev bubble with a radius just below
  $\sigma$ (for an integer $\sigma$, `bubble` with $\sigma - 1$). MiniSom's triangle is the product of a triangle along
  the rows and one along the columns, which none of the functions here is. Its sigma decays end after the last
  iteration and correspond, with `schedule_end="after_last_epoch"`, to `sigma_schedule="inverse"` with
  `sigma_final=sigma_initial / 3` (`asymptotic_decay`, the default) or `sigma_final=1` (`inverse_decay_to_one`), and
  to `sigma_schedule="linear"` with `sigma_final=1` (`linear_decay_to_one`). MiniSom's batch training mixes the new weights with the old ones by its
  learning rate, so it matches the batch map only with a learning rate of 1 that does not decay.

XPySom {cite:p}`Mancini_2020`
: The gaussian and the mexican hat with `std_coeff * sigma` (by default half its `sigma`) as $\sigma$, and the
  exponential schedule from `std_coeff * sigma` to `std_coeff * sigmaN`, or to a tenth of the initial width with
  `sigmaN=0`, as fastersom's default before version 0.6.0. Its `asymptotic` decay is `sigma_schedule="inverse"` with
  `sigma_final=sigma_initial / 3` and `schedule_end="after_last_epoch"`. Its bubble and triangle are MiniSom's, with
  `sigma` itself as the width. Its linear schedule reaches `sigmaN` in the last epoch, as `sigma_schedule="linear"` does
  by default. XPySom keeps a node only where its denominator is 0, and divides by negative ones of the mexican hat,
  where fastersom keeps the node.

SOM Toolbox {cite:p}`Vesanto_2000`
: The batch map's `gaussian`, `cutgauss`, `ep` and `bubble` are `"gaussian"`, `"cut_gaussian"`, `"epanechnikov"` and
  `"bubble"` on the euclidean grid distance (with `'lattice', 'rect'`; its default lattice is hexagonal), with a
  radius that decreases linearly and reaches `radius_fin` in the last epoch, as `sigma_schedule="linear"` does by
  default.

## Cost

The gaussian on the euclidean grid distance and the bubble on the chebyshev one factor into a part along the rows and
one along the columns, so the update takes one small matrix product along each grid axis, about
`n_nodes x (n_rows + n_cols) x n_features` multiply-adds per epoch. The other combinations take one matrix product per
offset along the shorter grid axis at which the neighbourhood is not zero, up to `n_nodes^2 x n_features`
multiply-adds per epoch, holding one Toeplitz matrix of `max(n_rows, n_cols)^2` floats at a time (on a `1 x n` map
that is the whole `n_nodes x n_nodes` matrix). Where only the winner has a weight, such as for the bubble below a
radius of 1, the update skips the product. This is still small next to the BMU search, `n_samples x n_nodes x
n_features`, whenever there are more samples than nodes: on an M3 Max with 35 features, at most 0.11 s per epoch at
300 x 300 nodes (gaussian on the chebyshev distance with $\sigma = 75$) and 3 ms with a triangle of radius 1.
