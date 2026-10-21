import numpy as np
import pytest
import torch

from fastersom import SOM

CPU = "cpu"


@pytest.fixture(scope="module")
def data():
    # gaussian blobs with mixed-sign features; no all-zero rows (those make cosine BMUs undefined)
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(8, 6)) * 2
    X = centers[rng.integers(0, 8, 5000)] + rng.normal(scale=0.3, size=(5000, 6))
    return X.astype(np.float32)


def brute_force_distances(X, weights, metric):
    X, weights = X.astype(np.float64), weights.astype(np.float64)
    if metric == "euclidean":
        return np.linalg.norm(X[:, None] - weights[None], axis=-1)
    cos = X @ weights.T / np.linalg.norm(X, axis=1, keepdims=True) / np.linalg.norm(weights, axis=1)
    return 1 - cos


def reference_som(X, n_rows, n_cols, sigma_initial, sigma_final, epochs, metric="euclidean", seed=0):
    """Kohonen's batch map written out per sample in float64, from fastersom's seeded random initialisation."""
    rng = np.random.RandomState(seed)
    weights = rng.rand(n_rows, n_cols, X.shape[1]) * 2 - 1
    weights = (weights / np.linalg.norm(weights, axis=-1, keepdims=True)).reshape(n_rows * n_cols, -1)
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
    np.testing.assert_array_equal(som.predict(data), som.labels_)
    dist = som.transform(data)
    assert dist.shape == (5000, 20)
    assert dist.dtype == np.float32
    assert (dist >= 0).all()
    # the BMU is a nearest node; argmin may pick another one where nodes are near-tied
    np.testing.assert_allclose(dist[np.arange(len(data)), som.labels_], dist.min(1), atol=1e-5)
    assert som.quantization_error(data) == pytest.approx(dist.min(1).mean(), rel=1e-5)
    assert 0 <= som.topographic_error(data) <= 1


def test_torch_input_and_flattening(data):
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


def test_float64_default_dtype(data):
    weights = SOM(5, 4, n_epochs=2, random_state=0, device=CPU).fit(data).weights_
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        som = SOM(5, 4, n_epochs=2, random_state=0, device=CPU).fit(data)
    finally:
        torch.set_default_dtype(default)
    np.testing.assert_allclose(som.weights_, weights, atol=1e-6)


def test_exact_distances_with_large_offset():
    # the dot-product shortcut used for BMU search cancels catastrophically here; transform must not
    X = np.array([[1e5, 1e5], [1e5 + 1, 1e5]], np.float32)
    som = SOM(1, 2, n_epochs=1, init="sample", sigma_initial=0.01, random_state=0, device=CPU).fit(X)
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
        (dict(metric="manhattan"), "metric"),
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


def test_determinism(data):
    a = SOM(6, 6, n_epochs=3, random_state=3, device=CPU).fit(data).weights_
    b = SOM(6, 6, n_epochs=3, random_state=3, device=CPU).fit(data).weights_
    c = SOM(6, 6, n_epochs=3, random_state=4, device=CPU).fit(data).weights_
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_matches_brute_force(data, metric):
    som = SOM(6, 6, n_epochs=3, init="sample", metric=metric, random_state=0, device=CPU).fit(data)
    dist = brute_force_distances(data, som.weights_, metric)
    # float32 vs float64 may flip near-ties; require agreement on all but a handful and equal distances
    assert (som.predict(data) == dist.argmin(1)).mean() > 0.999
    np.testing.assert_allclose(som.transform(data), dist, atol=1e-4)
    assert som.quantization_error(data) == pytest.approx(dist.min(1).mean(), abs=1e-5)


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
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


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_negative_inputs(data, metric):
    negative = data - data.max() - 1  # every feature strictly negative
    assert (negative < 0).all()
    som = SOM(6, 6, n_epochs=3, metric=metric, random_state=0, device=CPU).fit(negative)
    assert np.isfinite(som.weights_).all()
    if metric == "euclidean":  # weights are convex combinations of the samples
        assert (som.weights_ < 0).all()
    dist = brute_force_distances(negative, som.weights_, metric)
    assert (som.labels_ == dist.argmin(1)).mean() > 0.999
    assert som.quantization_error(negative) == pytest.approx(dist.min(1).mean(), abs=1e-4)
    # default sigmas: sigma_initial = ceil(min(n_rows, n_cols) / 2) / 2 and sigma_final = 0.1 * sigma_initial
    ref = reference_som(negative, 6, 6, 1.5, 0.15, 3, metric)
    ref_dist = brute_force_distances(negative, ref, metric)
    assert som.quantization_error(negative) == pytest.approx(ref_dist.min(1).mean(), rel=1e-3)
    som = SOM(6, 6, n_epochs=1, metric=metric, random_state=0, device=CPU).fit(negative)
    np.testing.assert_allclose(som.weights_, reference_som(negative, 6, 6, 1.5, 0.15, 1, metric), atol=1e-4)


def test_cosine_scale_invariance(data):
    som = SOM(6, 6, n_epochs=3, metric="cosine", random_state=0, device=CPU).fit(data)
    scale = np.random.default_rng(1).uniform(0.01, 100, size=(len(data), 1)).astype(np.float32)
    np.testing.assert_array_equal(som.predict(data * scale), som.labels_)
    np.testing.assert_allclose(som.transform(data * scale), som.transform(data), atol=1e-5)
    dist = som.transform(data)
    assert (dist >= 0).all()
    assert (dist <= 2).all()


def test_cosine_zero_rows(data):
    with_zeros = data.copy()
    with_zeros[::10] = 0
    nonzero = data[np.arange(len(data)) % 10 != 0]
    som = SOM(6, 6, n_epochs=3, metric="cosine", random_state=0, device=CPU).fit(with_zeros)
    ref = SOM(6, 6, n_epochs=3, metric="cosine", random_state=0, device=CPU).fit(nonzero)
    # zero rows have no BMU, are left out of the update and the error measures
    np.testing.assert_allclose(som.weights_, ref.weights_, atol=1e-6)
    assert (som.labels_[::10] == -1).all()
    assert (som.predict(with_zeros)[::10] == -1).all()
    np.testing.assert_array_equal(np.delete(som.labels_, slice(None, None, 10)), ref.labels_)
    np.testing.assert_array_equal(som.transform(with_zeros)[::10], 1)
    assert som.quantization_error(with_zeros) == pytest.approx(ref.quantization_error(nonzero), rel=1e-6)
    assert som.topographic_error(with_zeros) == pytest.approx(ref.topographic_error(nonzero), rel=1e-6)

    # init="sample" never picks a zero row and needs at least one non-zero row
    som = SOM(6, 6, init="sample", metric="cosine", random_state=0, device=CPU)
    som._setup()
    assert (som._init_weights(torch.from_numpy(with_zeros)).norm(dim=1) > 0).all()
    with pytest.raises(ValueError, match="non-zero"):
        SOM(6, 6, init="sample", metric="cosine", device=CPU).fit(np.zeros((10, 6)))


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_batch_size_does_not_change_results(data, metric):
    kwargs = dict(n_rows=6, n_cols=6, metric=metric, random_state=0, device=CPU)
    full = SOM(n_epochs=1, **kwargs).fit(data)
    full_3 = SOM(n_epochs=3, **kwargs).fit(data)
    for batch_size in (1, 7, 4999, 10**6):
        som = SOM(n_epochs=1, batch_size=batch_size, **kwargs).fit(data)
        np.testing.assert_allclose(som.weights_, full.weights_, atol=1e-5)
        np.testing.assert_array_equal(som.labels_, full.labels_)
        np.testing.assert_allclose(som.transform(data), full.transform(data), atol=1e-5)
        assert som.quantization_error(data) == pytest.approx(full.quantization_error(data), rel=1e-5)
        assert som.topographic_error(data) == pytest.approx(full.topographic_error(data), abs=1e-3)
        # over several epochs rounding differences can flip near-tied BMUs, so only require the same map quality
        som = SOM(n_epochs=3, batch_size=batch_size, **kwargs).fit(data)
        assert (som.labels_ == full_3.labels_).mean() > 0.95
        assert som.quantization_error(data) == pytest.approx(full_3.quantization_error(data), rel=1e-2)


def test_single_node(data):
    som = SOM(1, 1, n_epochs=2, device=CPU).fit(data)
    np.testing.assert_allclose(som.weights_[0], data.mean(0), atol=1e-5)
    assert (som.labels_ == 0).all()
    assert som.topographic_error(data) == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
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
