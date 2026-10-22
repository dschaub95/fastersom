import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fastersom import SOM

CPU = "cpu"
METRICS = ["euclidean", "manhattan", "cosine", "correlation"]


@pytest.fixture(scope="module")
def data():
    # gaussian blobs with mixed-sign features; no all-zero rows (those make cosine BMUs undefined)
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(8, 6)) * 2
    X = centers[rng.integers(0, 8, 5000)] + rng.normal(scale=0.3, size=(5000, 6))
    return X.astype(np.float32)


def brute_force_distances(X, weights, metric):
    """Distance of every sample to every node in float64, by broadcasting."""
    X, weights = X.astype(np.float64)[:, None], weights.astype(np.float64)[None]
    if metric == "euclidean":
        return np.sqrt(((X - weights) ** 2).sum(-1))
    if metric == "manhattan":
        return np.abs(X - weights).sum(-1)
    if metric == "correlation":
        X, weights = X - X.mean(-1, keepdims=True), weights - weights.mean(-1, keepdims=True)
    return 1 - (X * weights).sum(-1) / (np.linalg.norm(X, axis=-1) * np.linalg.norm(weights, axis=-1))


def random_init(n_rows, n_cols, n_features, seed=0):
    """fastersom's seeded init="random" weights, in float64."""
    weights = np.random.RandomState(seed).rand(n_rows, n_cols, n_features) * 2 - 1
    return (weights / np.linalg.norm(weights, axis=-1, keepdims=True)).reshape(n_rows * n_cols, -1)


def reference_som(X, n_rows, n_cols, sigma_initial, sigma_final, epochs, metric="euclidean", seed=0, weights=None):
    """Kohonen's batch map written out per sample in float64, from the given or fastersom's seeded random weights."""
    weights = random_init(n_rows, n_cols, X.shape[1], seed) if weights is None else weights.astype(np.float64)
    coords = np.array([(r, c) for r in range(n_rows) for c in range(n_cols)])
    grid_sq_dist = ((coords[:, None] - coords[None]) ** 2).sum(-1)
    X = X.astype(np.float64)
    for t in range(epochs):
        sigma = sigma_initial * (sigma_final / sigma_initial) ** (t / epochs)
        bmus = brute_force_distances(X, weights, metric).argmin(1)
        h = np.exp(-grid_sq_dist[bmus] / (2 * sigma**2))  # (n_samples, n_nodes)
        denominator = h.sum(0)
        weights = np.where(denominator[:, None] > 0, h.T @ X / denominator[:, None], weights)
    return weights


def reference_topographic_error(X, weights, n_cols):
    best2 = np.argsort(brute_force_distances(X, weights, "euclidean"), axis=1)[:, :2]
    rows, cols = np.divmod(best2, n_cols)
    return ((np.abs(rows[:, 0] - rows[:, 1]) > 1) | (np.abs(cols[:, 0] - cols[:, 1]) > 1)).mean()


def test_attributes_and_shapes(data):
    som = SOM(5, 4, n_epochs=2, random_state=0, device=CPU).fit(data)
    assert som.weights_.shape == (20, 6)
    assert som.weights_.dtype == np.float32
    assert som.n_features_in_ == 6
    assert som.labels_.shape == (5000,)
    assert som.labels_.dtype == np.int64
    dist = som.transform(data)
    assert dist.shape == (5000, 20)
    assert dist.dtype == np.float32


def test_input_types_and_flattening(data):
    weights = SOM(5, 4, n_epochs=2, random_state=0, device=CPU).fit(data).weights_
    som = SOM(5, 4, n_epochs=2, random_state=0, device=CPU).fit(torch.from_numpy(data).reshape(50, 100, 6))
    np.testing.assert_allclose(som.weights_, weights, atol=1e-6)
    assert som.predict(data.reshape(50, 100, 6)).shape == (5000,)
    assert som.transform(data.reshape(50, 100, 6)).shape == (5000, 20)
    som.fit(data.tolist())  # array-like
    np.testing.assert_allclose(som.weights_, weights, atol=1e-6)
    som.fit(torch.from_numpy(data).requires_grad_())
    np.testing.assert_allclose(som.weights_, weights, atol=1e-6)
    som.fit(np.ascontiguousarray(data[::-1])[::-1])  # negative strides
    np.testing.assert_allclose(som.weights_, weights, atol=1e-6)
    for x in (data, data[::-1].copy()[::-1]):  # only through DLPack, like CuPy; negative strides
        som.fit(SimpleNamespace(__dlpack__=x.__dlpack__, __dlpack_device__=x.__dlpack_device__, copy=x.copy))
        np.testing.assert_allclose(som.weights_, weights, atol=1e-6)


def test_float64_default_dtype(data):
    weights = SOM(5, 4, n_epochs=2, random_state=0, device=CPU).fit(data).weights_
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        som = SOM(5, 4, n_epochs=2, random_state=0, device=CPU).fit(data)
    finally:
        torch.set_default_dtype(default)
    np.testing.assert_allclose(som.weights_, weights, atol=1e-6)


def test_euclidean_with_large_offset():
    # ||w||^2 - 2 x.w rounds nearby nodes to the same score in float32 unless samples and nodes are centred first
    X = np.array([[1e4], [1e4 + 0.01], [1e4 + 1]], np.float32)
    som = SOM.from_weights(X, 1, 3, device=CPU)
    np.testing.assert_array_equal(som.predict(X), [0, 1, 2])
    np.testing.assert_allclose(som.transform(X), brute_force_distances(X, som.weights_, "euclidean"), atol=1e-6)


def test_input_validation(data):
    som = SOM(5, 4, n_epochs=1, device=CPU)
    for method in (som.predict, som.transform, som.quantization_error, som.topographic_error):
        with pytest.raises(RuntimeError, match="not fitted"):
            method(data)
    with pytest.raises(ValueError, match="shape"):
        som.fit(data[0])
    with pytest.raises(ValueError, match="shape"):
        som.fit(data[:0])
    bad = data.copy()
    bad[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        som.fit(bad)
    som.fit(data)
    with pytest.raises(ValueError, match="features"):
        som.predict(data[:, :3])
    som.fit(data[:, :3])  # refitting resets n_features_in_
    assert som.weights_.shape == (20, 3)


def test_invalid_parameters(data):
    for kwargs, match in [
        (dict(metric="chebyshev"), "metric"),
        (dict(init="pca"), "init"),
        (dict(sigma_initial=0), "sigma"),
        (dict(sigma_final=-1), "sigma"),
        (dict(sigma_initial=np.nan), "sigma"),
        (dict(n_rows=0), "n_rows"),
        (dict(n_cols=2.5), "n_cols"),
        (dict(n_epochs=-1), "n_epochs"),
        (dict(batch_size=0), "batch_size"),
    ]:
        with pytest.raises(ValueError, match=match):
            SOM(**{"n_rows": 5, "n_cols": 4, **kwargs}, device=CPU).fit(data)


@pytest.mark.parametrize("metric", METRICS)
def test_matches_brute_force(data, metric):
    som = SOM(6, 6, n_epochs=3, init="sample", metric=metric, random_state=0, device=CPU).fit(data)
    dist = brute_force_distances(data, som.weights_, metric)
    # float32 vs float64 may flip near-ties; require agreement on all but a handful and equal distances
    labels, distances = som.predict(data, return_distance=True)
    assert (labels == dist.argmin(1)).mean() > 0.999
    np.testing.assert_allclose(dist[np.arange(len(data)), labels], dist.min(1), atol=1e-4)
    np.testing.assert_allclose(distances, dist.min(1), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(labels, som.predict(data))
    np.testing.assert_allclose(som.transform(data), dist, rtol=1e-5, atol=1e-4)
    assert som.quantization_error(data) == pytest.approx(dist.min(1).mean(), abs=1e-5)
    assert som.quantization_error(data) == pytest.approx(distances.astype(np.float64).mean(), rel=1e-12)


@pytest.mark.parametrize("metric", METRICS)
def test_matches_reference(data, metric):
    som_kwargs = dict(sigma_initial=1.25, sigma_final=0.125, metric=metric, random_state=0, device=CPU)

    # one epoch from the same initial weights
    ref = reference_som(data, 10, 10, 1.25, 0.125, 1, metric)
    som = SOM(10, 10, n_epochs=1, **som_kwargs).fit(data)
    np.testing.assert_allclose(som.weights_, ref, atol=1e-4)

    # full schedule over several epochs; float32 vs float64 may flip near-tied BMUs, so compare map quality
    ref = reference_som(data, 10, 10, 1.25, 0.125, 3, metric)
    som = SOM(10, 10, n_epochs=3, **som_kwargs).fit(data)
    ref_dist = brute_force_distances(data, ref, metric)
    assert (som.labels_ == ref_dist.argmin(1)).mean() > 0.99
    assert som.quantization_error(data) == pytest.approx(ref_dist.min(1).mean(), rel=1e-2)
    if metric == "euclidean":
        assert som.topographic_error(data) == pytest.approx(reference_topographic_error(data, ref, 10), abs=1e-2)

    # default sigmas: sigma_initial = ceil(min(n_rows, n_cols) / 2) / 2 and sigma_final = 0.1 * sigma_initial
    som = SOM(6, 6, n_epochs=2, metric=metric, random_state=0, device=CPU).fit(data)
    ref = reference_som(data, 6, 6, 1.5, 0.15, 2, metric)
    assert som.quantization_error(data) == pytest.approx(
        brute_force_distances(data, ref, metric).min(1).mean(), rel=1e-3
    )


def test_correlation_with_large_offset():
    # node 0 is a perfect match shifted by 1e7, where a float32 row mean rounds by up to 1
    som = SOM.from_weights(
        np.array([[1e7, 1e7, 1e7 + 1], [0.1, 0, 1]], np.float32), 1, 2, metric="correlation", device=CPU
    )
    labels, distances = som.predict(np.array([[0, 0, 1]], np.float32), return_distance=True)
    assert labels[0] == 0
    assert distances[0] == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize(("metric", "constant"), [("cosine", 0.0), ("correlation", 2.7)])
def test_rows_without_bmu(data, metric, constant):
    # zero rows under cosine and constant (zero-variance) rows under correlation have no BMU
    invalid = data.copy()
    invalid[::10] = constant
    valid = data[np.arange(len(data)) % 10 != 0]
    som = SOM(6, 6, n_epochs=3, metric=metric, random_state=0, device=CPU).fit(invalid)
    ref = SOM(6, 6, n_epochs=3, metric=metric, random_state=0, device=CPU).fit(valid)
    # they are left out of the update and the error measures
    np.testing.assert_allclose(som.weights_, ref.weights_, atol=1e-6)
    assert (som.labels_[::10] == -1).all()
    labels, distances = som.predict(invalid, return_distance=True)
    assert (labels[::10] == -1).all()
    assert np.isnan(distances[::10]).all()
    np.testing.assert_array_equal(np.delete(som.labels_, slice(None, None, 10)), ref.labels_)
    np.testing.assert_allclose(
        np.delete(distances, slice(None, None, 10)), ref.predict(valid, return_distance=True)[1], atol=1e-6
    )
    np.testing.assert_array_equal(som.transform(invalid)[::10], 1)
    assert som.quantization_error(invalid) == pytest.approx(ref.quantization_error(valid), rel=1e-6)
    assert som.topographic_error(invalid) == pytest.approx(ref.topographic_error(valid), rel=1e-6)

    # init="sample" never picks such a row and needs at least one other row
    som = SOM(6, 6, init="sample", metric=metric, random_state=0, device=CPU)
    som._setup()
    weights = som._init_weights(torch.from_numpy(invalid))
    assert (som._has_bmu(weights)).all()
    with pytest.raises(ValueError, match="with a BMU"):
        SOM(6, 6, init="sample", metric=metric, device=CPU).fit(np.full((10, 6), constant))


@pytest.mark.parametrize("metric", ["euclidean", "correlation"])  # without and with rows dropped per batch
def test_batch_size_does_not_change_results(data, metric):
    kwargs = dict(n_rows=6, n_cols=6, metric=metric, random_state=0, device=CPU)
    full = SOM(n_epochs=1, **kwargs).fit(data)
    for batch_size in (7, 4999):  # many batches; a last batch of one sample
        som = SOM(n_epochs=1, batch_size=batch_size, **kwargs).fit(data)
        np.testing.assert_allclose(som.weights_, full.weights_, atol=1e-5)
        np.testing.assert_array_equal(som.labels_, full.labels_)
        np.testing.assert_allclose(som.transform(data), full.transform(data), atol=1e-5)
        np.testing.assert_allclose(
            som.predict(data, return_distance=True)[1], full.predict(data, return_distance=True)[1], atol=1e-5
        )
        assert som.quantization_error(data) == pytest.approx(full.quantization_error(data), rel=1e-5)
        assert som.topographic_error(data) == pytest.approx(full.topographic_error(data), abs=1e-3)


def test_single_node(data):
    som = SOM(1, 1, n_epochs=2, device=CPU).fit(data)
    np.testing.assert_allclose(som.weights_[0], data.mean(0), atol=1e-5)
    assert (som.labels_ == 0).all()
    assert som.topographic_error(data) == 0.0
    np.testing.assert_array_equal(som.distance_map(), [[0]])


@pytest.mark.parametrize("shape", ["flat", "grid"])
def test_init_weights(data, shape):
    weights = random_init(5, 4, 6)
    init = weights if shape == "flat" else weights.reshape(5, 4, 6)
    # the seeded random initialisation passed as an array gives the very same map
    som = SOM(5, 4, n_epochs=3, init=init, device=CPU).fit(data)
    ref = SOM(5, 4, n_epochs=3, random_state=0, device=CPU).fit(data)
    np.testing.assert_array_equal(som.weights_, ref.weights_)
    np.testing.assert_array_equal(som.labels_, ref.labels_)

    # n_epochs=0 keeps the given weights exactly, as a copy
    init = ref.weights_.copy() if shape == "flat" else ref.weights_.reshape(5, 4, 6).copy()
    som = SOM(5, 4, n_epochs=0, init=init, device=CPU).fit(data)
    init[:] = 0
    np.testing.assert_array_equal(som.weights_, ref.weights_)
    np.testing.assert_array_equal(som.labels_, ref.labels_)
    np.testing.assert_array_equal(som.transform(data), ref.transform(data))

    # from_weights builds the same fitted map without data; labels_ needs training samples
    som = SOM.from_weights(torch.from_numpy(ref.weights_), 5, 4, device=CPU)
    np.testing.assert_array_equal(som.weights_, ref.weights_)
    np.testing.assert_array_equal(som.predict(data), ref.labels_)
    assert som.quantization_error(data) == ref.quantization_error(data)
    assert not hasattr(som, "labels_")
    np.testing.assert_array_equal(som.fit(data).labels_, ref.labels_)


def test_invalid_init_weights(data):
    for init, match in [
        (np.zeros((19, 6)), "shape"),
        (np.zeros((4, 5, 6)), "shape"),
        (np.zeros(20), "shape"),
        (np.zeros((20, 0)), "shape"),
        (np.full((20, 6), np.nan), "NaN"),
        (np.zeros((20, 3)), "features"),
    ]:
        with pytest.raises(ValueError, match=match):
            SOM(5, 4, init=init, device=CPU).fit(data)
    with pytest.raises(ValueError, match="shape"):
        SOM.from_weights(np.zeros((19, 6)), 5, 4, device=CPU)


def test_distance_map():
    # summed euclidean distances to the 8-neighbours, worked out by hand, on a 2 x 3 grid
    weights = np.array([[0, 0], [1, 0], [3, 0], [0, 0], [0, 0], [3, 4]], np.float32)
    distance_map = SOM.from_weights(weights, 2, 3, device=CPU).distance_map()
    expected = np.array([[1, 1 + 2 + 1 + 1 + np.sqrt(20), 2 + 3 + 4], [1, 1 + 3 + 5, np.sqrt(20) + 4 + 5]])
    expected /= expected.max()
    np.testing.assert_allclose(distance_map, expected, rtol=1e-6)
    assert SOM.from_weights(np.ones((6, 2)), 2, 3, device=CPU).distance_map().max() == 0


def test_pickle_round_trip(data, monkeypatch):
    som = SOM(5, 4, n_epochs=2, random_state=0, device=CPU).fit(data)
    labels, distances = som.predict(data, return_distance=True)
    loaded = pickle.loads(pickle.dumps(som))
    np.testing.assert_array_equal(loaded.weights_, som.weights_)
    np.testing.assert_array_equal(loaded.labels_, som.labels_)
    np.testing.assert_array_equal(loaded.predict(data, return_distance=True)[0], labels)
    np.testing.assert_array_equal(loaded.predict(data, return_distance=True)[1], distances)

    # a map configured for a device the loading machine lacks falls back to the default device
    som.device = "cuda"
    payload = pickle.dumps(som)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    loaded = pickle.loads(payload)
    assert loaded.device == "cuda"
    np.testing.assert_array_equal(loaded.predict(data), labels)
    loaded.device = None
    np.testing.assert_array_equal(loaded.fit(data).weights_, som.weights_)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("metric", METRICS)
def test_cuda_matches_cpu(data, metric):
    kwargs = dict(n_rows=6, n_cols=6, metric=metric, random_state=0)
    cpu = SOM(n_epochs=1, device=CPU, **kwargs).fit(data)
    cuda = SOM(n_epochs=1, device="cuda", **kwargs).fit(data)
    np.testing.assert_allclose(cuda.weights_, cpu.weights_, atol=1e-5)
    np.testing.assert_allclose(cuda.transform(data), cpu.transform(data), atol=1e-4)
    cpu = SOM(n_epochs=3, device=CPU, **kwargs).fit(data)
    cuda = SOM(n_epochs=3, device="cuda", **kwargs).fit(data)
    assert (cuda.labels_ == cpu.labels_).mean() > 0.95
    assert cuda.quantization_error(data) == pytest.approx(cpu.quantization_error(data), rel=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_pickle_loads_without_cuda(data, monkeypatch):
    som = SOM(5, 4, n_epochs=2, random_state=0, device="cuda").fit(data)
    payload = pickle.dumps(som)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    loaded = pickle.loads(payload)
    np.testing.assert_array_equal(loaded.weights_, som.weights_)
    assert (loaded.predict(data) == som.labels_).mean() > 0.999
    np.testing.assert_allclose(loaded.distance_map(), som.distance_map(), rtol=1e-5)
