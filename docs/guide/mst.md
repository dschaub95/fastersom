# Minimum spanning trees

{func}`~fastersom.minimum_spanning_tree` computes the minimum spanning tree (MST) of the rows of an array, such as the
nodes of a fitted map or k-means centroids: the tree of `n - 1` edges that connects all `n` rows at the least total
distance. Such trees of SOM nodes serve, for example, to lay out and to cluster the nodes in the FlowSOM workflow
{cite:p}`Van_Gassen_2015`. {meth}`SOM.minimum_spanning_tree <fastersom.SOM.minimum_spanning_tree>` takes the tree of
a fitted map's nodes, in its metric and on its device.

```python
from fastersom import SOM, minimum_spanning_tree

som = SOM(30, 30, random_state=0).fit(X)
tree = som.minimum_spanning_tree()
tree.edges  # node pairs, shape (899, 2), sorted by distance
tree.weights  # their distances in the map's metric

tree = minimum_spanning_tree(centroids, metric="cosine")
```

## Algorithm and memory

The tree is found by Borůvka's algorithm {cite:p}`Boruvka_1926`. Each round finds, for every row, its nearest row in
another component (at first every row is a component of its own); every component keeps the shortest of these edges,
and the components merge along them. Every component merges in every round, so there are at most
$\lceil \log_2 n \rceil$ rounds, and in practice few (5-6 in the table below). A round's search is the BMU search of
a map whose nodes are the rows, with the pairs within a component left out: blocks of rows are compared with all rows
by a matrix product, as on the torch path of the map (see [](devices.md)). Rows whose nearest row is still in another
component after a merge keep it without a new search.

No `n x n` matrix is formed. Beyond the rows themselves, the search holds one block of `batch_size x n` scores: by
default $2^{24}$ of them on the CPU (128 MB in float64), and on GPUs as many as the map's torch path holds for `n`
nodes, fewer if they do not fit in `memory_fraction` of the free memory. A dense float64 distance matrix, by
comparison, takes 0.8 GB for 10,000 rows and 12.8 GB for 40,000.

On an M3 Max (CPU, 8 threads), for clustered rows of 34 features:

| rows | rounds | `exact=True` | `exact=False` | peak memory beyond the rows |
| --- | --- | --- | --- | --- |
| 1,000 | 5 | 0.01 s | 0.01 s | < 20 MB |
| 10,000 | 6 | 0.45 s | 0.23 s | about 0.3 GB |
| 40,000 | 6 | 5.3 s | 3.7 s | about 0.3 GB |

The time grows with the square of the number of rows. On CUDA and MPS the search also takes the torch path; fused
kernels for it, which would hold no scores, are not part of fastersom yet.

## Exactness

The result does not depend on the device, the batch size or the number of threads. Edges are compared by their float64
distance, computed in the same way on every device, with ties broken by the lower and then the higher row index. Under
this strict order the minimum spanning tree is unique, and with `exact=True` (the default) it is the tree returned:
the minimum spanning tree of the rows as float32 (which X is rounded to, as for the map), with float64 distances.

The search itself scores in float32 on GPUs and in float64 on the CPU, and keeps each row's best two rows. Where the
two are closer than a rigorous bound of the scores' rounding error, the row is searched again in float64, and its
nearest row is chosen among all whose scores lie within that bound of the best. In float64 hardly any rows of trained
maps or centroids need this (none in the timings below), in float32 about 1-7% in the first round (in tests on the
CPU); on near-ties, such as an untrained grid or rows on a lattice, most rows do, which costs up to about one more
float64 search per round. On MPS, which has no float64, these searches run on the CPU. Where torch may compute float32
matrix products in TF32 or bfloat16 (see {func}`torch.set_float32_matmul_precision` and the per-backend
`fp32_precision` settings), the bound widens accordingly.

Each row's bound depends on its own length and on a radius fitted to its best score, beyond which no row can be its
nearest, so a few rows far from the others (outliers, or a long tail of intensities) do not loosen the bound of every
other row; only the far rows themselves may need the float64 search. Exact duplicates are joined before the search and
add neither searches nor candidates; a float64 search holds at most $2^{22}$ candidate pairs at once.

`exact=False` takes the search's choices as they are, in float32 on every device. Near-tied edges can then differ from
the exact tree, which gives a spanning tree of nearly the same total distance (within $10^{-6}$ in the tests), on the
CPU in 0.5-0.7 times the time.

### Identical rows and ties

Identical rows are joined by edges of distance 0. `scipy.sparse.csgraph.minimum_spanning_tree` of a dense distance
matrix, by contrast, takes zeros for missing edges: there each copy of a row is joined to another row instead, and the
tree is longer. Rows that tie exactly get the edges of the lowest indices, so that `n` identical rows give a star
around row 0.

## Metrics

`metric` takes the metrics of {class}`~fastersom.SOM`:

| `metric` | edge weight | tree of |
| --- | --- | --- |
| `"euclidean"` (default) | $\lVert a - b \rVert_2$ | the rows |
| `"manhattan"` | $\lVert a - b \rVert_1$ | the rows |
| `"cosine"` | $1 - \cos(a, b)$ | the rows scaled to unit length |
| `"correlation"` | $1 - r(a, b)$ | the rows centred over their features and scaled to unit length |

For rows of unit length, $1 - \cos(a, b) = \lVert a - b \rVert^2 / 2$ grows with their euclidean distance, so the
cosine and correlation trees are the euclidean trees of the scaled (and centred) rows, which fastersom computes in
float64 before the search. A row of zeros has no cosine distance and a constant row no correlation distance; such rows
raise a {class}`ValueError`.

## Using the tree

{class}`~fastersom.SpanningTree` holds `edges`, `weights` and `n_nodes`. Its methods take memory in proportion to the
number of nodes, beyond their output:

`degrees()`, `leaves()`
: The number of edges at every node, and the nodes with a single edge.

`hops(sources, targets=None)`, `path_lengths(sources, targets=None)`
: The number of edges, or their summed distances, on the path from each source to every node (or to the targets).

`labels(n_clusters=None, distance_threshold=None)`
: Single-linkage clusters: cutting the `n_clusters - 1` longest edges, or all edges of at least `distance_threshold`,
  leaves the clusters as components, as in scikit-learn's `AgglomerativeClustering(linkage="single")`. They are numbered
  in the order of their smallest node.

`linkage()`
: The single-linkage hierarchy in the format of `scipy.cluster.hierarchy.linkage(X, "single")`, for
  `scipy.cluster.hierarchy.dendrogram` or `fcluster`.

`to_sparse()`
: The tree as a `scipy.sparse.csr_array` (needs scipy), with edges of distance 0 as explicit zeros.

fastersom does not lay out the tree. Graph libraries take the edges directly, for example igraph, whose tree layouts
take time in proportion to the number of nodes:

```python
import igraph as ig

graph = ig.Graph(n=tree.n_nodes, edges=tree.edges.tolist(), edge_attrs={"weight": tree.weights.tolist()})
layout = graph.layout_reingold_tilford_circular(root=[0])
```
