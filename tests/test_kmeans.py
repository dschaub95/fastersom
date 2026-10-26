import pickle

import numpy as np
import pytest
import torch

from fastersom import KMeans

GPUS = [
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")),
    pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")),
]
SPHERICAL = ["cosine", "correlation"]


@pytest.fixture(scope="module")
def data():
    # gaussian blobs with mixed-sign features and distinct directions; no constant or all-zero rows
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(6, 5)) * 3
    X = centers[rng.integers(0, 6, 3000)] + rng.normal(scale=0.4, size=(3000, 5))
    return X.astype(np.float32)


def compared(X, metric):
    """The rows as the metric compares them, in float64: unit length, also centred for correlation."""
    X = X.astype(np.float64)
    if metric == "correlation":
        X = X - X.mean(1, keepdims=True)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def reference_spherical(X, init, metric, iterations):
    """Spherical k-means written out in float64: nearest centroid by cosine similarity, then the mean direction."""
    Y, centroids = compared(X, metric), init.astype(np.float64)
    for _ in range(iterations):
        directions = (
            compared(centroids, metric)
            if metric == "correlation"
            else centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        )
        labels = (Y @ directions.T).argmax(1)
        for k in range(len(centroids)):
            if (labels == k).any():
                centroids[k] = Y[labels == k].mean(0)
    return centroids


def test_matches_scikit_learn_lloyd(data):
    sklearn_cluster = pytest.importorskip("sklearn.cluster")
    init = data[np.random.default_rng(1).choice(len(data), 6, replace=False)]
    ours = KMeans(6, init=init, max_iter=100, tol=0, device="cpu").fit(data)
    theirs = sklearn_cluster.KMeans(6, init=init, n_init=1, max_iter=100, tol=0, algorithm="lloyd").fit(data)
    np.testing.assert_allclose(ours.cluster_centers_, theirs.cluster_centers_, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(ours.labels_, theirs.labels_)
    assert ours.n_iter_ == theirs.n_iter_
    assert ours.inertia_ == pytest.approx(theirs.inertia_, rel=1e-5)


def test_empty_clusters_move_to_the_farthest_samples_as_in_scikit_learn(data):
    sklearn_cluster = pytest.importorskip("sklearn.cluster")
    init = data[:6].copy()
    init[5] = 100.0  # gets no samples in the first iteration
    ours = KMeans(6, init=init, max_iter=30, tol=0, device="cpu").fit(data)
    theirs = sklearn_cluster.KMeans(6, init=init, n_init=1, max_iter=30, tol=0, algorithm="lloyd").fit(data)
    np.testing.assert_allclose(ours.cluster_centers_, theirs.cluster_centers_, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(ours.labels_, theirs.labels_)
    kept = KMeans(6, init=init, max_iter=30, tol=0, empty_clusters="keep", device="cpu").fit(data)
    np.testing.assert_array_equal(kept.cluster_centers_[5], init[5])


def test_fixed_iterations_match_scikit_learn(data):
    sklearn_cluster = pytest.importorskip("sklearn.cluster")
    init = data[np.random.default_rng(6).choice(len(data), 6, replace=False)] * 0.5  # still moving after two
    ours = KMeans(6, init=init, max_iter=2, tol=0, device="cpu").fit(data)
    theirs = sklearn_cluster.KMeans(6, init=init, n_init=1, max_iter=2, tol=0, algorithm="lloyd").fit(data)
    np.testing.assert_allclose(ours.cluster_centers_, theirs.cluster_centers_, rtol=1e-5, atol=1e-5)
    assert ours.n_iter_ == 2


@pytest.mark.parametrize("metric", SPHERICAL)
def test_spherical_matches_reference(data, metric):
    init = data[np.random.default_rng(2).choice(len(data), 6, replace=False)]
    fitted = KMeans(6, init=init, metric=metric, max_iter=5, tol=0, device="cpu").fit(data)
    expected = reference_spherical(data, init, metric, fitted.n_iter_)
    np.testing.assert_allclose(fitted.cluster_centers_, expected, rtol=1e-4, atol=1e-5)


def test_manhattan_is_not_offered(data):
    with pytest.raises(ValueError, match="metric"):
        KMeans(3, metric="manhattan", device="cpu").fit(data)


def test_centroids_only(data):
    with_labels = KMeans(6, random_state=0, device="cpu").fit(data)
    alone = KMeans(6, random_state=0, compute_labels=False, device="cpu").fit(data)
    np.testing.assert_array_equal(alone.cluster_centers_, with_labels.cluster_centers_)
    assert not hasattr(alone, "labels_")
    assert not hasattr(alone, "inertia_")
    np.testing.assert_array_equal(alone.predict(data), with_labels.labels_)
    np.testing.assert_array_equal(alone.fit_predict(data), with_labels.labels_)


def test_kmeans_plus_plus_finds_separated_blobs():
    rng = np.random.default_rng(3)
    centers = np.eye(8, 8) * 20
    X = (centers[rng.integers(0, 8, 4000)] + rng.normal(scale=0.5, size=(4000, 8))).astype(np.float32)
    fitted = KMeans(8, random_state=0, device="cpu").fit(X)
    distance = np.linalg.norm(fitted.cluster_centers_[:, None] - centers[None], axis=-1)
    assert sorted(distance.argmin(1)) == list(range(8))  # one centroid per blob
    assert distance.min(1).max() < 0.5


def test_kmeans_plus_plus_is_seeded_and_picks_samples(data):
    first = KMeans(6, max_iter=1, random_state=4, device="cpu")._initial_centroids(*_engine_and_data(data))
    again = KMeans(6, max_iter=1, random_state=4, device="cpu")._initial_centroids(*_engine_and_data(data))
    torch.testing.assert_close(first, again)
    rows = {tuple(row) for row in data.tolist()}
    assert all(tuple(row) in rows for row in first.tolist())
    assert len({tuple(row) for row in first.tolist()}) == 6


def _engine_and_data(data):
    model = KMeans(6, device="cpu")
    engine = model._engine()
    return engine, engine._check_X(data)


def test_random_init_draws_distinct_samples(data):
    model = KMeans(6, init="random", random_state=0, device="cpu")
    centroids = model._initial_centroids(*_engine_and_data(data))
    assert len(torch.unique(centroids, dim=0)) == 6


def test_transform_and_predict_agree(data):
    fitted = KMeans(6, random_state=0, device="cpu").fit(data)
    distances = fitted.transform(data)
    assert distances.shape == (len(data), 6)
    np.testing.assert_array_equal(distances.argmin(1), fitted.predict(data))
    assert fitted.inertia_ == pytest.approx(float((distances.min(1).astype(np.float64) ** 2).sum()), rel=1e-4)


def test_input_validation(data):
    with pytest.raises(ValueError, match="init must have shape"):
        KMeans(6, init=data[:5], device="cpu").fit(data)
    with pytest.raises(ValueError, match="init must be one of"):
        KMeans(6, init="pca", device="cpu").fit(data)
    with pytest.raises(ValueError, match="needs as many samples"):
        KMeans(10, device="cpu").fit(data[:5])
    with pytest.raises(ValueError, match="max_iter"):
        KMeans(6, max_iter=0, device="cpu").fit(data)
    fitted = KMeans(6, random_state=0, device="cpu").fit(data)
    with pytest.raises(ValueError, match="features"):
        fitted.predict(data[:, :4])
    with pytest.raises(RuntimeError, match="not fitted"):
        KMeans(6).predict(data)


def test_pickle_round_trip(data):
    fitted = KMeans(6, random_state=0, device="cpu").fit(data)
    loaded = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(loaded.predict(data), fitted.labels_)


@pytest.mark.parametrize("device", GPUS)
@pytest.mark.parametrize("metric", ["euclidean", *SPHERICAL])
@pytest.mark.parametrize("deterministic", [False, True])
def test_gpu_matches_cpu(data, device, metric, deterministic):
    init = data[np.random.default_rng(5).choice(len(data), 6, replace=False)]
    cpu = KMeans(6, init=init, metric=metric, max_iter=20, device="cpu").fit(data)
    gpu = KMeans(6, init=init, metric=metric, max_iter=20, device=device, deterministic=deterministic).fit(data)
    np.testing.assert_allclose(gpu.cluster_centers_, cpu.cluster_centers_, rtol=1e-4, atol=1e-4)
    assert (gpu.labels_ == cpu.labels_).mean() > 0.999


def test_empty_clusters_take_only_samples_with_a_centroid():
    X = np.zeros((10, 3), dtype=np.float32)
    X[0] = [1, 0, 0]  # the only sample with a centroid under cosine; two centroids get none
    centroids = KMeans(3, init=np.eye(3, dtype=np.float32), metric="cosine", max_iter=1, device="cpu").fit(X)
    np.testing.assert_array_equal(centroids.cluster_centers_, [[1, 0, 0], [1, 0, 0], [0, 0, 1]])