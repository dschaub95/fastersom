# k-means

{class}`~fastersom.KMeans` clusters the samples by Lloyd's algorithm {cite:p}`Lloyd_1982`: each iteration assigns every
sample to its nearest centroid and moves every centroid to the mean of its samples. The assignment uses the same
nearest-node search as {class}`~fastersom.SOM`, including its fused kernels on CUDA and MPS, and the same summation,
including `deterministic`. Without a neighbourhood, the centroids represent the samples more closely than a map's
nodes do, but they are not ordered on a grid.

```python
from fastersom import KMeans

km = KMeans(n_clusters=100, random_state=0).fit(X)
km.cluster_centers_  # shape (100, n_features)
km.labels_  # cluster of every sample
km.predict(Y)  # nearest centroid of new samples
```

## Initialisation

`init="k-means++"` (the default) is greedy k-means++ {cite:p}`Arthur_2007` as in scikit-learn: from a random sample,
every further centroid is the best of `n_local_trials` samples (by default $2 + \lfloor \ln k \rfloor$), each drawn with
probability proportional to its squared distance to the nearest centroid so far, best by the total that remains. It
takes $k$ steps, each a pass over the samples it seeds on, so it seeds on a random subset of `init_size` samples (by
default $\max(100{,}000, 10k)$; all samples if there are fewer, which gives the same centroids as seeding on all of
them). Without the subset, $10^4$ centroids of millions of samples would take hours; on $10^6$ samples of 34 features,
1,000 centroids took 4.2 s on the CPU instead of 38.7 s, with the same held-out error after 10 iterations. The random
numbers are drawn up front and the choices kept on the device, so that the seeding never waits for it.
`init="random"` draws distinct samples at random; an array gives the initial centroids.

## Convergence and empty clusters

The fit stops after `max_iter` iterations, or earlier when the centroids move, in total squared distance, by at most
`tol` times the mean variance of the features (scikit-learn's criterion), or not at all. A centroid that gets no samples
moves to the sample farthest from its own centroid, which leaves that centroid (`empty_clusters="farthest"`, as
scikit-learn does), or stays where it is (`"keep"`).

From the same initial centroids, the result matches scikit-learn's `KMeans(algorithm="lloyd")` up to float rounding;
on GPUs, the order of the summation can flip near-tied assignments, as for the map.

## Centroids only

`labels_` and `inertia_` need one more pass over the samples after the last iteration. `compute_labels=False` skips it
when only the centroids are needed; {meth}`~fastersom.KMeans.predict` assigns samples later.

## Metrics

Under `"cosine"` and `"correlation"` it is spherical k-means: the samples are compared and averaged scaled to unit
length (and, for correlation, centred over the features), and the centroids are directions. `"manhattan"` is not
offered, as the mean does not minimise the L1 distance.
