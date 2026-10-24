import pickle
import subprocess
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
        assert som.transform(data[:10]).dtype == np.float32
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
        (dict(memory_fraction=0), "memory_fraction"),
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

    # non-square, so rows and columns of the separable neighbourhood cannot be swapped
    ref = reference_som(data, 4, 9, 1.25, 0.125, 1, metric)
    som = SOM(4, 9, n_epochs=1, **som_kwargs).fit(data)
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


def test_verbose_progress_bar(data, capsys):
    quiet = SOM(5, 4, n_epochs=3, random_state=0, device=CPU).fit(data)
    assert capsys.readouterr().err == ""
    verbose = SOM(5, 4, n_epochs=3, random_state=0, device=CPU, verbose=True).fit(data)
    assert "3/3" in capsys.readouterr().err
    np.testing.assert_array_equal(verbose.weights_, quiet.weights_)


def test_pickle_round_trip(data, monkeypatch):
    som = SOM(5, 4, n_epochs=2, random_state=0, device=CPU).fit(data)
    labels, distances = som.predict(data, return_distance=True)
    loaded = pickle.loads(pickle.dumps(som))
    np.testing.assert_array_equal(loaded.weights_, som.weights_)
    np.testing.assert_array_equal(loaded.labels_, som.labels_)
    np.testing.assert_array_equal(loaded.predict(data, return_distance=True)[0], labels)
    np.testing.assert_array_equal(loaded.predict(data, return_distance=True)[1], distances)
    state = som.__getstate__()
    del state["memory_fraction"], state["verbose"], state["deterministic"]  # as pickled by 0.2
    loaded = SOM.__new__(SOM)
    loaded.__setstate__(state)
    np.testing.assert_array_equal(loaded.predict(data), labels)

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


GPUS = [
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")),
    pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")),
]


@pytest.mark.parametrize("device", GPUS)
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("deterministic", [False, True])
def test_gpu_matches_cpu(data, metric, device, deterministic):
    kwargs = dict(n_rows=6, n_cols=6, metric=metric, random_state=0)
    cpu = SOM(n_epochs=1, device=CPU, **kwargs).fit(data)
    gpu = SOM(n_epochs=1, device=device, deterministic=deterministic, **kwargs).fit(data)
    np.testing.assert_allclose(gpu.weights_, cpu.weights_, atol=1e-5)
    np.testing.assert_allclose(gpu.transform(data), cpu.transform(data), atol=1e-4)
    cpu = SOM(n_epochs=3, device=CPU, **kwargs).fit(data)
    gpu = SOM(n_epochs=3, device=device, deterministic=deterministic, **kwargs).fit(data)
    assert (gpu.labels_ == cpu.labels_).mean() > 0.95
    assert gpu.quantization_error(data) == pytest.approx(cpu.quantization_error(data), rel=1e-2)


@pytest.mark.parametrize("device", GPUS)
def test_deterministic(device, monkeypatch):
    # many samples per node, so that the order of atomic additions varies between runs without deterministic
    X = np.random.default_rng(0).normal(size=(200_000, 6)).astype(np.float32)
    kwargs = dict(n_rows=4, n_cols=4, n_epochs=3, random_state=0, device=device)
    weights = SOM(deterministic=True, **kwargs).fit(X).weights_
    for _ in range(3):
        np.testing.assert_array_equal(SOM(deterministic=True, **kwargs).fit(X).weights_, weights)

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # else CUDA matmuls raise under deterministic algorithms
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        np.testing.assert_array_equal(SOM(**kwargs).fit(X).weights_, weights)
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_deterministic_keeps_small_values():
    # fixed point scaled for the 1e16 would round every 1 to 0; node 0 holds only the 1s
    X = np.ones((1024, 1), np.float32)
    X[-1] = 1e16
    som = SOM(1, 2, n_epochs=1, sigma_initial=0.1, init=[[1], [1e16]], device="cuda", deterministic=True).fit(X)
    assert som.weights_[0, 0] == pytest.approx(1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_pickle_loads_without_cuda(data, monkeypatch):
    som = SOM(5, 4, n_epochs=2, random_state=0, device="cuda").fit(data)
    payload = pickle.dumps(som)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    loaded = pickle.loads(payload)
    np.testing.assert_array_equal(loaded.weights_, som.weights_)
    assert (loaded.predict(data) == som.labels_).mean() > 0.999
    np.testing.assert_allclose(loaded.distance_map(), som.distance_map(), rtol=1e-5)


@pytest.fixture(scope="module")
def fused():
    """Skips unless Triton compiles the fused CUDA kernels here (CUDA, Triton with Gluon, a C compiler and the Python
    headers); returns fastersom._som."""
    from fastersom import _som

    if not torch.cuda.is_available() or _som._triton(torch.device("cuda")) is None:
        pytest.skip("no CUDA, no Triton with Gluon, or Triton cannot compile")
    return _som


def torch_path(monkeypatch):
    monkeypatch.setattr(SOM, "_kernel", lambda self, n_features, top2=False: None)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(
    ("n_samples", "n_features", "n_rows", "n_cols"),
    [(5000, 6, 7, 9), (3001, 34, 30, 30), (1500, 196, 9, 9), (500, 160, 3, 3)],
)
def test_fused_matches_torch(fused, monkeypatch, metric, n_samples, n_features, n_rows, n_cols):
    # sizes off the tiles. Float32 kernel: 6 features without its loop over 16, 34 and 196 with a remainder, 160 without
    # (manhattan only, as 9 nodes stay off tensor cores); tensor cores (on GPUs with fast TF32): 34 and 196 features
    rng = np.random.default_rng(0)
    X = (rng.normal(size=(n_samples, n_features)) + rng.normal(size=n_features) * 3).astype(np.float32)
    weights = X[rng.choice(n_samples, n_rows * n_cols)] + rng.normal(scale=0.1, size=(n_rows * n_cols, n_features))
    som = SOM.from_weights(weights.astype(np.float32), n_rows, n_cols, metric=metric, device="cuda")
    n_nodes = n_rows * n_cols
    if metric != "manhattan" and n_nodes > 64 and n_nodes * n_features >= 8192:
        fast_tf32 = torch.cuda.get_device_capability() in fused._FAST_TF32
        assert som._kernel(n_features) == ("tensor-core" if fast_tf32 else "float32" if n_features <= 128 else None)
    else:
        assert som._kernel(n_features) == ("float32" if metric == "manhattan" or n_features <= 128 else None)
    labels, distances = som.predict(X, return_distance=True)
    top2 = som._search(som._weights, som._kernel(n_features, top2=True), top2=True)(torch.as_tensor(X, device="cuda"))
    torch_path(monkeypatch)
    ref_labels, ref_distances = som.predict(X, return_distance=True)
    # float32 sums in another order may flip near-ties only
    assert (labels == ref_labels).mean() > 0.999
    np.testing.assert_allclose(distances, ref_distances, rtol=1e-5, atol=1e-5)
    # the two best nodes in order, by the torch path's scores, as their order is arbitrary on exact ties; up to
    # rounding, which on tensor cores is about that of float32 relative to the largest score
    ref_scores = som._scores(torch.as_tensor(X, device="cuda"), som._weights)
    assert (top2[:, 0] != top2[:, 1]).all()
    ref_top2 = ref_scores.topk(2, 1, largest=False).values
    atol = 1e-5 * max(1, ref_scores.abs().max().item() / 10)
    torch.testing.assert_close(ref_scores.gather(1, top2), ref_top2, rtol=1e-5, atol=atol)


def test_fused_ties_and_offsets(fused):
    # duplicate nodes tie exactly: the first wins, as with torch's argmin
    weights = np.array([[3, 1, 2], [1, 2, 3], [3, 1, 2], [1, 2, 3], [2, 3, 1]], np.float32)
    for metric in METRICS:
        som = SOM.from_weights(weights, 1, 5, metric=metric, device="cuda")
        np.testing.assert_array_equal(som.predict(weights[[1, 0, 4, 3, 2]]), [1, 0, 4, 1, 0])
    # also when the copies fall in different parts of the nodes, which few samples split across programs
    weights = np.random.default_rng(0).normal(size=(20, 30, 6)).astype(np.float32)
    weights[19, 29] = weights[12, 5] = weights[0, 10]
    som = SOM.from_weights(weights, 20, 30, device="cuda")
    np.testing.assert_array_equal(som.predict(np.repeat(weights[:1, 10], 3, 0)), [10, 10, 10])
    # as test_euclidean_with_large_offset and test_correlation_with_large_offset (the 10 x 10 map: on tensor cores
    # where they are fast)
    X = (1e4 + np.random.default_rng(0).normal(scale=0.01, size=(100, 100))).astype(np.float32)
    np.testing.assert_array_equal(SOM.from_weights(X, 10, 10, device="cuda").predict(X), np.arange(100))
    som = SOM.from_weights(
        np.array([[1e7, 1e7, 1e7 + 1], [0.1, 0, 1]], np.float32), 1, 2, metric="correlation", device="cuda"
    )
    labels, distances = som.predict(np.array([[0, 0, 1]], np.float32), return_distance=True)
    assert labels[0] == 0
    assert distances[0] == pytest.approx(0, abs=1e-6)
    assert np.isnan(som.topographic_error(np.ones((10, 3), np.float32)))  # no sample has a BMU, so none is searched


@pytest.fixture
def fresh(fused, monkeypatch):
    """fastersom._som with no kernel checked yet, and the checks of other tests restored afterwards."""
    monkeypatch.setattr(fused, "_FAILED", set())
    fused._triton.cache_clear()
    fused._works.cache_clear()
    yield fused
    monkeypatch.undo()
    fused._triton.cache_clear()
    fused._works.cache_clear()


def test_fused_compile_failure_falls_back(data, fresh, monkeypatch):
    som = SOM.from_weights(data[:36], 6, 6, device="cuda")
    expected = som.predict(data)

    def fail(*args):  # its message is the whole compiler command, so the warning says why instead
        raise subprocess.CalledProcessError(1, ["gcc", "driver.c", "-I/include"])

    monkeypatch.setattr(fresh._triton(torch.device("cuda")), "search", fail)
    fresh._triton.cache_clear()
    with pytest.warns(RuntimeWarning, match=r"\((\S+Python\.h not found|.* 1)\)") as record:
        assert som._kernel(6) is None
    assert "-I/include" not in str(record[0].message)
    assert record[0].filename == __file__  # the caller's line, not fastersom's
    # the batch size is chosen for the torch path, before any batch is searched
    assert som._batch_size_for(torch.as_tensor(data), som._kernel(6)) <= 2**28 // 36  # the fused one: 2^26 / 6
    np.testing.assert_array_equal(som.predict(data), expected)  # no second warning


def test_fused_self_check_ignores_torch_defaults(fresh):
    som = SOM.from_weights(np.eye(900, 34, dtype=np.float32), 30, 30, device="cuda")
    expected = "tensor-core" if torch.cuda.get_device_capability() in fresh._FAST_TF32 else "float32"
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float64)
    try:
        assert som._kernel(34) == expected
    finally:
        torch.set_default_device(None)
        torch.set_default_dtype(torch.float32)


def test_fused_kernel_failure_degrades(data, fresh, monkeypatch):
    # wrong BMUs from the tensor-core kernel (made available here on any GPU) fail its self-check, which leaves the
    # float32 kernel; a failure of that one leaves the torch path
    kernels = fresh._triton(torch.device("cuda"))
    monkeypatch.setattr(fresh, "_FAST_TF32", [torch.cuda.get_device_capability()])
    bmus = kernels._bmus
    monkeypatch.setattr(kernels, "_bmus", lambda *args: bmus(*args) // 2 if args[6] else bmus(*args))
    X = np.random.default_rng(0).normal(size=(1000, 20)).astype(np.float32)
    som = SOM.from_weights(X[:900], 30, 30, device="cuda")
    with pytest.warns(RuntimeWarning, match=r"tensor-core BMU kernel failed its self-check \(RuntimeError: wrong"):
        assert som._kernel(20) == "float32"
    np.testing.assert_array_equal(som.predict(X)[:900], np.arange(900))  # no second warning
    monkeypatch.setattr(kernels, "_bmus", lambda *args: bmus(*args) + 1)
    with pytest.warns(RuntimeWarning, match=r"float32 BMU kernel failed its self-check"):
        assert som._kernel(33) is None
