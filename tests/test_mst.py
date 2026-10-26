import pickle
from functools import cache

import numpy as np
import pytest
import torch

from fastersom import SOM, KMeans, SpanningTree, _mst, minimum_spanning_tree

scipy_distance = pytest.importorskip("scipy.spatial.distance")
csgraph = pytest.importorskip("scipy.sparse.csgraph")

GPUS = [
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")),
    pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")),
]
DEVICES = ["cpu", *GPUS]
# scipy's names of the metrics, as the trees' orders: euclidean by the squared distance, so that no two distinct
# squares round to the same root
ORDERS = {"euclidean": "sqeuclidean", "manhattan": "cityblock", "cosine": "cosine", "correlation": "correlation"}


def reference(X, metric="euclidean"):
    """Kruskal's tree under the strict order (distance, lower index, higher index) of the float64 distances.

    scipy's minimum_spanning_tree of the dense matrix of every pair's rank in that order: the ranks are distinct and
    positive, so its tree is unique, and no edge of distance 0 goes missing (scipy takes zeros for missing edges).
    """
    X = np.asarray(X, np.float64)
    distances = scipy_distance.squareform(scipy_distance.pdist(X, ORDERS[metric]))
    i, j = np.triu_indices(len(X), 1)
    order = np.lexsort((j, i, distances[i, j]))
    ranks = np.zeros_like(distances)
    ranks[i[order], j[order]] = np.arange(1, len(order) + 1)
    tree = csgraph.minimum_spanning_tree(ranks).tocoo()
    edges = np.sort(np.stack([tree.row, tree.col], 1), 1).astype(np.int64)
    edges = edges[np.argsort(ranks[edges[:, 0], edges[:, 1]])]
    weights = distances[edges[:, 0], edges[:, 1]]
    return edges, np.sqrt(weights) if metric == "euclidean" else weights


def assert_tree_equal(tree, edges, weights, rtol=1e-13):
    np.testing.assert_array_equal(tree.edges, edges)
    np.testing.assert_allclose(tree.weights, weights, rtol=rtol, atol=1e-15)
    assert tree.edges.dtype == np.int64
    assert tree.weights.dtype == np.float64


def blobs(n=4000, n_features=6, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(8, n_features)) * 3
    return (centres[rng.integers(0, 8, n)] + rng.normal(scale=0.4, size=(n, n_features))).astype(np.float32)


def points(kind):
    """Inputs of the kinds a tree is taken of: iid gaussian, clusters, a trained map's nodes, k-means centroids."""
    return _points(kind).copy()


@cache
def _points(kind):
    rng = np.random.default_rng(1)
    if kind == "gaussian":
        return rng.normal(size=(500, 10)).astype(np.float32)
    if kind == "mixture":  # 30 clusters of intrinsic dimension 4 in 34 features
        centres, bases = rng.normal(size=(30, 34)) * 4, rng.normal(size=(30, 4, 34)) / 2
        labels = rng.integers(0, 30, 1000)
        X = centres[labels] + np.einsum("kr,krd->kd", rng.normal(size=(1000, 4)), bases[labels])
        return X.astype(np.float32)
    if kind == "som":
        return SOM(20, 20, n_epochs=5, random_state=0, device="cpu").fit(blobs()).weights_
    return KMeans(150, random_state=0, max_iter=20, device="cpu").fit(blobs()).cluster_centers_


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["gaussian", "mixture", "som", "kmeans"])
@pytest.mark.parametrize("batch_size", [None, 37])
def test_matches_scipy(device, kind, batch_size):
    X = points(kind)
    tree = minimum_spanning_tree(X, device=device, batch_size=batch_size)
    assert tree.n_nodes == len(X)
    assert_tree_equal(tree, *reference(X))


@pytest.mark.parametrize("device", DEVICES)
def test_matches_scipy_dense_matrix(device):
    # without ties or zeros, scipy's tree of the dense distance matrix itself
    X = np.random.default_rng(2).normal(size=(3000, 8)).astype(np.float32)
    distances = scipy_distance.squareform(scipy_distance.pdist(X.astype(np.float64)))
    expected = csgraph.minimum_spanning_tree(distances).tocoo()
    del distances
    edges = np.sort(np.stack([expected.row, expected.col], 1), 1)
    order = np.lexsort((edges[:, 1], edges[:, 0], expected.data))
    tree = minimum_spanning_tree(X, device=device)
    assert_tree_equal(tree, edges[order], expected.data[order])


@pytest.mark.parametrize("device", DEVICES)
def test_duplicates_and_ties_follow_the_strict_order(device):
    # an integer lattice, whose distances tie exactly, with 50 of its points twice
    lattice = np.stack(np.meshgrid(*[np.arange(6)] * 3, indexing="ij"), -1).reshape(-1, 3)
    X = np.concatenate([lattice, lattice[::4][:50]]).astype(np.float32)
    X = np.concatenate([X, np.zeros((len(X), 2), np.float32)], 1)
    tree = minimum_spanning_tree(X, device=device, batch_size=64)
    assert_tree_equal(tree, *reference(X), rtol=0)
    assert (tree.weights == 0).sum() == 50
    # scipy's tree of the dense matrix takes distances of 0 for missing edges, and so joins each copy at distance 1
    dense = csgraph.minimum_spanning_tree(scipy_distance.squareform(scipy_distance.pdist(X.astype(np.float64))))
    assert dense.sum() == tree.weights.sum() + 50


@pytest.mark.parametrize("device", DEVICES)
def test_identical_points_give_a_star(device):
    tree = minimum_spanning_tree(np.full((30, 4), 0.1, np.float32), device=device)
    np.testing.assert_array_equal(tree.edges, np.stack([np.zeros(29), np.arange(1, 30)], 1))
    np.testing.assert_array_equal(tree.weights, np.zeros(29))


def test_tiny_inputs():
    tree = minimum_spanning_tree(np.ones((1, 3)), device="cpu")
    assert tree.edges.shape == (0, 2)
    assert tree.n_nodes == 1
    tree = minimum_spanning_tree([[0.0, 0.0], [3.0, 4.0]], device="cpu")
    np.testing.assert_array_equal(tree.edges, [[0, 1]])
    np.testing.assert_array_equal(tree.weights, [5.0])
    X = np.random.default_rng(3).normal(size=(200, 1)).astype(np.float32)  # one feature: the sorted values' chain
    order = np.argsort(X[:, 0])
    expected = {tuple(sorted(pair)) for pair in zip(order[:-1].tolist(), order[1:].tolist(), strict=True)}
    assert {tuple(edge) for edge in minimum_spanning_tree(X, device="cpu").edges.tolist()} == expected


def test_flattens_leading_dimensions():
    weights = SOM(6, 5, n_epochs=2, random_state=0, device="cpu").fit(blobs()).weights_
    flat = minimum_spanning_tree(weights, device="cpu")
    grid = minimum_spanning_tree(torch.as_tensor(weights).reshape(6, 5, -1), device="cpu")
    np.testing.assert_array_equal(grid.edges, flat.edges)


@pytest.mark.parametrize("device", DEVICES)
def test_fast_mode_gives_a_near_minimal_tree(device):
    X = points("mixture")
    tree = minimum_spanning_tree(X, exact=False, device=device)
    exact = reference(X)
    assert tree.edges.shape == (len(X) - 1, 2)
    assert (tree.edges[:, 0] < tree.edges[:, 1]).all()
    assert len(np.unique(tree.labels(n_clusters=1))) == 1  # connected
    assert tree.weights.sum() == pytest.approx(exact[1].sum(), rel=1e-6)


def near_ties():
    """A 40 x 40 lattice moved by about 1e-5: its nearest neighbours tie within float32's rounding of the scores."""
    rng = np.random.default_rng(0)
    lattice = np.stack(np.meshgrid(np.arange(40), np.arange(40), indexing="ij"), -1).reshape(-1, 2)
    return (lattice + rng.normal(scale=1e-5, size=lattice.shape)).astype(np.float32)


def test_certification_rechecks_near_ties_of_float32(monkeypatch):
    X = near_ties()
    expected = reference(X)
    monkeypatch.setattr(_mst, "_search_dtype", lambda device, exact: torch.float32)
    rechecked = []
    recheck = _mst._Recheck.__call__
    monkeypatch.setattr(
        _mst._Recheck, "__call__", lambda self, rows, labels: rechecked.append(len(rows)) or recheck(self, rows, labels)
    )
    assert_tree_equal(minimum_spanning_tree(X, device="cpu"), *expected)
    assert 0 < sum(rechecked) < 3 * len(X)
    # without the certification the float32 choices give another tree, so the test above is not vacuous
    monkeypatch.setattr(
        _mst._TorchSearch, "_separated", lambda self, rows, best, second, first: np.ones(len(rows), bool)
    )
    uncertified = minimum_spanning_tree(X, device="cpu")
    assert not np.array_equal(uncertified.edges, expected[0])
    assert uncertified.weights.sum() == pytest.approx(expected[1].sum(), rel=1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("metric", ["manhattan", "cosine", "correlation"])
def test_metrics_match_scipy(device, metric):
    X = points("gaussian")[:300] + 0.5
    tree = minimum_spanning_tree(X, metric, device=device)
    assert_tree_equal(tree, *reference(X, metric), rtol=1e-12 if metric == "manhattan" else 1e-9)


@pytest.mark.parametrize("metric", ["cosine", "correlation"])
def test_cosine_and_correlation_are_trees_of_unit_rows(metric):
    X = points("gaussian").astype(np.float64)
    rows = X - X.mean(1, keepdims=True) if metric == "correlation" else X
    rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    tree, euclidean = minimum_spanning_tree(X, metric, device="cpu"), minimum_spanning_tree(rows, device="cpu")
    np.testing.assert_array_equal(tree.edges, euclidean.edges)
    np.testing.assert_allclose(tree.weights, euclidean.weights**2 / 2, rtol=1e-6)
    # the scale (and, for correlation, an offset) of a row does not change the tree
    moved = X * np.linspace(0.5, 3, len(X))[:, None] + (7.0 if metric == "correlation" else 0.0)
    np.testing.assert_array_equal(
        minimum_spanning_tree(moved.astype(np.float32), metric, device="cpu").edges, tree.edges
    )


def test_invariances():
    # values on a grid of 2^-8, so that the offset of 1e4 keeps them exact in float32
    rng = np.random.default_rng(4)
    X = (rng.integers(-512, 512, size=(400, 5)) / 256).astype(np.float32)
    tree = minimum_spanning_tree(X, device="cpu")
    assert_tree_equal(tree, *reference(X), rtol=0)
    offset = minimum_spanning_tree(X + np.float32(1e4), device="cpu")
    np.testing.assert_array_equal(offset.edges, tree.edges)
    np.testing.assert_array_equal(offset.weights, tree.weights)
    scaled = minimum_spanning_tree(X * 2, device="cpu")
    np.testing.assert_array_equal(scaled.edges, tree.edges)
    np.testing.assert_array_equal(scaled.weights, tree.weights * 2)
    X = points("gaussian")
    permutation = rng.permutation(len(X))
    permuted = minimum_spanning_tree(X[permutation], device="cpu")
    edges = np.sort(permutation[permuted.edges], 1)
    tree = minimum_spanning_tree(X, device="cpu")
    assert {tuple(edge) for edge in edges.tolist()} == {tuple(edge) for edge in tree.edges.tolist()}


@pytest.mark.parametrize(("metric", "row"), [("cosine", 0.0), ("correlation", 2.5)])
def test_rows_without_a_distance_raise(metric, row):
    X = points("gaussian")
    X[7] = row
    with pytest.raises(ValueError, match=f"no {metric} distance"):
        minimum_spanning_tree(X, metric, device="cpu")


def test_input_validation():
    X = points("gaussian")
    with pytest.raises(ValueError, match="metric must be one of"):
        minimum_spanning_tree(X, "chebyshev", device="cpu")
    X[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        minimum_spanning_tree(X, device="cpu")
    with pytest.raises(ValueError, match="batch_size"):
        minimum_spanning_tree(points("gaussian"), batch_size=0, device="cpu")
    with pytest.raises(ValueError, match="memory_fraction"):
        minimum_spanning_tree(points("gaussian"), memory_fraction=2, device="cpu")
    with pytest.raises(ValueError, match="X must have shape"):
        minimum_spanning_tree(np.ones(5), device="cpu")


def test_blocks_bound_the_memory(monkeypatch):
    # every block of scores holds batch_size rows x n_points, in the search and in the rechecks alike
    shapes = []
    scores = _mst._TorchSearch.scores
    monkeypatch.setattr(
        _mst._TorchSearch, "scores", lambda self, rows, labels: shapes.append(len(rows)) or scores(self, rows, labels)
    )
    X = np.concatenate([points("som"), points("som")[:100]])  # duplicates, which need rechecks
    tree = minimum_spanning_tree(X, device="cpu", batch_size=50)
    assert max(shapes) == 50
    assert_tree_equal(tree, *reference(X))
    # by default 2^24 scores per block on the CPU, whatever the number of points
    engine = SOM(1, 1, device="cpu")
    engine._setup()
    for n in (1000, 40_000, 1_000_000):
        assert _mst._block(engine, torch.empty(n, 34)) * n <= 1 << 24


def test_derived_outputs_match_scipy():
    hierarchy = pytest.importorskip("scipy.cluster.hierarchy")
    sklearn_cluster = pytest.importorskip("sklearn.cluster")
    X = points("mixture")[:600]
    tree = minimum_spanning_tree(X, device="cpu")
    Z = hierarchy.linkage(X.astype(np.float64), "single")
    np.testing.assert_array_equal(tree.linkage()[:, [0, 1, 3]], Z[:, [0, 1, 3]])
    np.testing.assert_allclose(tree.linkage()[:, 2], Z[:, 2], rtol=1e-12)

    def first_seen(labels):  # labels renumbered in the order of their first node, to compare partitions
        return np.unique(labels, return_index=True)[1].argsort().argsort()[np.unique(labels, return_inverse=True)[1]]

    for n_clusters in (1, 5, 30, 599, 600):
        labels = tree.labels(n_clusters=n_clusters)
        assert labels.dtype == np.int64
        assert labels.max() == n_clusters - 1
        np.testing.assert_array_equal(labels, first_seen(labels))  # numbered by smallest member
        np.testing.assert_array_equal(labels, first_seen(hierarchy.fcluster(Z, n_clusters, "maxclust")))
    for n_clusters in (5, 30):
        theirs = sklearn_cluster.AgglomerativeClustering(n_clusters, linkage="single").fit(X)
        np.testing.assert_array_equal(tree.labels(n_clusters=n_clusters), first_seen(theirs.labels_))
    threshold = float(np.sort(tree.weights)[-12])  # an edge's weight: edges of at least the threshold are cut
    theirs = sklearn_cluster.AgglomerativeClustering(None, linkage="single", distance_threshold=threshold).fit(X)
    np.testing.assert_array_equal(tree.labels(distance_threshold=threshold), first_seen(theirs.labels_))
    assert tree.labels(distance_threshold=threshold).max() == 12

    graph = tree.to_sparse()
    sources, targets = np.array([0, 17, 599, 17]), np.array([3, 0, 250])
    hops = csgraph.shortest_path(graph, unweighted=True, indices=sources)
    lengths = csgraph.shortest_path(graph, indices=sources)
    assert tree.hops(sources).dtype == np.int32
    np.testing.assert_array_equal(tree.hops(sources), hops)
    np.testing.assert_array_equal(tree.hops(sources, targets), hops[:, targets])
    np.testing.assert_array_equal(tree.hops(17), hops[1])
    np.testing.assert_allclose(tree.path_lengths(sources), lengths, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tree.path_lengths(599, targets), lengths[2, targets], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(tree.degrees(), np.diff(graph.indptr))
    np.testing.assert_array_equal(tree.leaves(), np.flatnonzero(np.diff(graph.indptr) == 1))


def test_hand_built_trees():
    path = SpanningTree(np.array([[0, 1], [1, 2], [2, 3], [3, 4]]), np.array([1.0, 2.0, 3.0, 4.0]), 5)
    np.testing.assert_array_equal(path.degrees(), [1, 2, 2, 2, 1])
    np.testing.assert_array_equal(path.leaves(), [0, 4])
    np.testing.assert_array_equal(path.hops(np.arange(5)), np.abs(np.arange(5)[:, None] - np.arange(5)))
    np.testing.assert_array_equal(path.path_lengths(4), [10.0, 9.0, 7.0, 4.0, 0.0])
    np.testing.assert_array_equal(path.labels(n_clusters=2), [0, 0, 0, 0, 1])
    np.testing.assert_array_equal(path.labels(distance_threshold=3), [0, 0, 0, 1, 2])
    np.testing.assert_array_equal(path.linkage(), [[0, 1, 1, 2], [2, 5, 2, 3], [3, 6, 3, 4], [4, 7, 4, 5]])
    star = SpanningTree(np.array([[0, 1], [0, 2], [0, 3]]), np.ones(3), 4)
    np.testing.assert_array_equal(star.degrees(), [3, 1, 1, 1])
    np.testing.assert_array_equal(star.leaves(), [1, 2, 3])
    np.testing.assert_array_equal(star.hops(1), [1, 0, 2, 2])
    single = SpanningTree(np.zeros((0, 2), np.int64), np.zeros(0), 1)
    np.testing.assert_array_equal(single.hops(0), [0])
    assert single.linkage().shape == (0, 4)
    np.testing.assert_array_equal(single.labels(n_clusters=1), [0])


def test_to_sparse_keeps_edges_of_weight_zero():
    X = np.concatenate([points("gaussian")[:100]] * 2)
    tree = minimum_spanning_tree(X, device="cpu")
    graph = tree.to_sparse()
    assert graph.nnz == 2 * (len(X) - 1)
    assert (graph.data == 0).sum() == 2 * 100
    assert (graph != graph.T).nnz == 0
    upper = tree.to_sparse(symmetric=False)
    assert upper.nnz == len(X) - 1
    rows, cols = upper.nonzero()
    assert (rows < cols).all()
    np.testing.assert_allclose(upper.sum(), tree.weights.sum())


def test_tree_validation():
    with pytest.raises(ValueError, match="n_nodes - 1 edges"):
        SpanningTree(np.array([[0, 1]]), np.ones(1), 3)
    with pytest.raises(ValueError, match="n_nodes must be"):
        SpanningTree(np.zeros((0, 2)), np.zeros(0), 0)
    with pytest.raises(ValueError, match="cycle"):
        SpanningTree(np.array([[0, 1], [1, 0], [2, 3]]), np.ones(3), 4)
    with pytest.raises(ValueError, match="node indices"):
        SpanningTree(np.array([[0, 1], [1, 4]]), np.ones(2), 3)
    tree = minimum_spanning_tree(points("gaussian"), device="cpu")
    with pytest.raises(ValueError, match="exactly one"):
        tree.labels()
    with pytest.raises(ValueError, match="exactly one"):
        tree.labels(n_clusters=2, distance_threshold=1.0)
    with pytest.raises(ValueError, match="n_clusters must be"):
        tree.labels(n_clusters=501)
    with pytest.raises(ValueError, match="sources must be node indices"):
        tree.hops([0, 500])
    with pytest.raises(ValueError, match="targets must be node indices"):
        tree.path_lengths(0, [0.5])
    with pytest.raises(ValueError, match="nan"):
        tree.labels(distance_threshold=np.nan)
    assert tree.hops([]).shape == (0, 500)


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_methods_of_fitted_models(metric):
    som = SOM(8, 7, n_epochs=3, metric=metric, random_state=0, device="cpu").fit(blobs())
    expected = minimum_spanning_tree(som.weights_, metric, device="cpu")
    for model in (som, pickle.loads(pickle.dumps(som))):
        tree = model.minimum_spanning_tree()
        np.testing.assert_array_equal(tree.edges, expected.edges)
        np.testing.assert_array_equal(tree.weights, expected.weights)
    kmeans = KMeans(40, metric=metric, random_state=0, max_iter=10, device="cpu").fit(blobs())
    expected = minimum_spanning_tree(kmeans.cluster_centers_, metric, device="cpu")
    for model in (kmeans, pickle.loads(pickle.dumps(kmeans))):
        tree = model.minimum_spanning_tree()
        np.testing.assert_array_equal(tree.edges, expected.edges)
        np.testing.assert_array_equal(tree.weights, expected.weights)
    fast = minimum_spanning_tree(som.weights_, metric, exact=False, device="cpu")
    np.testing.assert_array_equal(som.minimum_spanning_tree(exact=False).edges, fast.edges)
    with pytest.raises(RuntimeError, match="not fitted"):
        SOM(3, 3).minimum_spanning_tree()
    with pytest.raises(RuntimeError, match="not fitted"):
        KMeans(3).minimum_spanning_tree()


def float32_search(monkeypatch):
    """The GPUs' float32 search, on the CPU; returns the list of the rows rechecked per round."""
    monkeypatch.setattr(_mst, "_search_dtype", lambda device, exact: torch.float32)
    rechecked = []
    recheck = _mst._Recheck.__call__
    monkeypatch.setattr(
        _mst._Recheck, "__call__", lambda self, rows, labels: rechecked.append(len(rows)) or recheck(self, rows, labels)
    )
    return rechecked


def test_outliers_do_not_loosen_every_rows_bound(monkeypatch):
    X = points("mixture")
    X[7] = 1e6  # one point far from all others
    rechecked = float32_search(monkeypatch)
    assert_tree_equal(minimum_spanning_tree(X, device="cpu"), *reference(X))
    assert sum(rechecked) < 0.2 * len(X)  # with one bound for all rows, every row of every round


def test_long_tailed_rows(monkeypatch):
    X = np.random.default_rng(3).lognormal(0, 2.5, size=(600, 8)).astype(np.float32)
    rechecked = float32_search(monkeypatch)
    assert_tree_equal(minimum_spanning_tree(X, device="cpu"), *reference(X))
    assert sum(rechecked) < len(X)


def test_overflowing_float32_scores_are_rechecked(monkeypatch):
    X = np.random.default_rng(4).normal(size=(200, 5)).astype(np.float32) * np.float32(1e20)
    float32_search(monkeypatch)
    tree = minimum_spanning_tree(X, device="cpu")
    expected = reference(X)
    np.testing.assert_array_equal(tree.edges, expected[0])


def test_duplicates_are_joined_without_a_search(monkeypatch):
    rng = np.random.default_rng(5)
    X = np.repeat(rng.normal(size=(5, 4)).astype(np.float32), 400, axis=0)[rng.permutation(2000)]
    searched = []
    nearest = _mst._TorchSearch.nearest
    monkeypatch.setattr(
        _mst._TorchSearch,
        "nearest",
        lambda self, rows, *args, **kwargs: searched.append(len(self.points)) or nearest(self, rows, *args, **kwargs),
    )
    tree = minimum_spanning_tree(X, device="cpu")
    assert_tree_equal(tree, *reference(X))
    assert max(searched) == 5
    assert (tree.weights == 0).sum() == 1995


def test_unit_roundoff_under_the_per_backend_precision_api(monkeypatch):
    def legacy():
        raise RuntimeError("mix of the legacy and new APIs")

    monkeypatch.setattr(torch, "get_float32_matmul_precision", legacy)
    for device in ("cpu", "cuda", "mps"):
        assert _mst._unit_roundoff(torch.float32, torch.device(device), matmul=True) in (2.0**-24, 2.0**-11, 2.0**-8)
    assert _mst._unit_roundoff(torch.float64, torch.device("cpu"), matmul=True) == 2.0**-53


def test_methods_do_not_take_the_models_batch_size():
    som = SOM(10, 10, n_epochs=2, random_state=0, device="cpu", batch_size=2**16).fit(blobs())
    assert_tree_equal(som.minimum_spanning_tree(), *reference(som.weights_))
