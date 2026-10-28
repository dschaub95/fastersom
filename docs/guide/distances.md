# Distances

## Metrics

`metric` sets the distance used for the BMU search, `predict(return_distance=True)`, `transform` and the error
measures:

| `metric` | distance between sample $x$ and node $w$ | range |
| --- | --- | --- |
| `"euclidean"` (default) | $\lVert x - w \rVert_2$ | $[0, \infty)$ |
| `"manhattan"` | $\lVert x - w \rVert_1 = \sum_j \lvert x_j - w_j \rvert$ | $[0, \infty)$ |
| `"cosine"` | $1 - \dfrac{x \cdot w}{\lVert x \rVert \lVert w \rVert}$ | $[0, 2]$ |
| `"correlation"` | $1 - r(x, w)$, the cosine distance of $x$ and $w$ centred over their features (Pearson correlation $r$) | $[0, 2]$ |

Under every metric, `fit` updates each node to the neighbourhood-weighted *mean* of the raw samples: under manhattan
this is not the median that would minimise the L1 distance, and under correlation the samples are averaged as given,
not centred. The metric changes which node a sample is assigned to, not how the nodes are averaged.

### Samples without a BMU

Under cosine, a sample of all zeros has no direction, and under correlation a constant sample has no correlation.
Such samples have no BMU:

- `predict` returns the label -1 for them, and the distance NaN with `return_distance=True`; `labels_` likewise holds
  -1.
- `transform` gives them distance 1 to every node.
- `fit` leaves them out of the update, `init="sample"` does not pick them, `init="pca"` leaves them out of the
  principal components, and the error measures leave them out.

Euclidean and manhattan distances are defined for every sample.

## Choosing a metric

fastersom does not scale the features, and every feature counts in the distance in its own units. Under euclidean and
manhattan, features with larger ranges therefore dominate. Common preprocessing is a variance-stabilising transform
such as $\operatorname{arcsinh}(x / 5)$ for mass cytometry, or clipping to percentiles and scaling each channel to
$[0, 1]$ for imaging data.

- **euclidean** compares absolute values. It is the metric of Kohonen's SOM and of most SOM-based clustering, such as
  FlowSOM {cite:p}`Van_Gassen_2015`.
- **manhattan** sums absolute differences, so a large difference in a single feature counts less than under
  euclidean.
- **cosine** compares directions and ignores each sample's overall scale, for example a pixel's total intensity, so
  samples with the same profile at different intensities share a node.
- **correlation** also ignores each sample's offset across features, comparing only the shape of its profile.

## Distances computed from a fitted map

`predict(X, return_distance=True)`
: Each sample's BMU and its distance to it, from the same pass over the data.

`transform(X)`
: The distance of every sample to every node, a dense `(n_samples, n_nodes)` float32 array (see [](batch-size.md)).

`quantization_error(X)`
: The mean distance of the samples to their BMUs, the mean of `predict(X, return_distance=True)[1]` over the samples
  with a BMU. Lower means the nodes represent the data more closely; it decreases with more nodes.

`topographic_error(X)`
: The fraction of samples whose best and second-best nodes are not adjacent on the grid, counting the 8 surrounding
  nodes as adjacent. Lower means the map preserves the neighbourhoods of the data better.

`distance_map()`
: The U-matrix, shape `(n_rows, n_cols)`: each node's summed euclidean distance to its up to 8 grid neighbours,
  normalised to a maximum of 1, as MiniSom's `distance_map(scaling="sum")`. It uses the euclidean distance whatever
  the metric. High values mark borders between groups of similar nodes.

```python
import matplotlib.pyplot as plt

plt.imshow(som.distance_map(), cmap="viridis")
plt.colorbar(label="normalised distance to neighbours")
```
