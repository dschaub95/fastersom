import pickle
import subprocess
import sys
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


NEIGHBOURHOODS = {  # h(d; sigma) for sigma > 0
    "gaussian": lambda d, s: np.exp(-(d**2) / (2 * s**2)),
    "bubble": lambda d, s: (d <= s).astype(np.float64),
    "triangle": lambda d, s: np.maximum(0, 1 - d / s),
    "epanechnikov": lambda d, s: np.maximum(0, 1 - d**2 / s**2),
    "cut_gaussian": lambda d, s: np.exp(-(d**2) / (2 * s**2)) * (d <= s),
    "mexican_hat": lambda d, s: (1 - d**2 / s**2) * np.exp(-(d**2) / (2 * s**2)),
}
GRID_METRICS = ["euclidean", "chebyshev", "manhattan"]
SCHEDULES = {  # sigma(t) for t = 0, ..., n_epochs - 1
    "exponential": lambda s0, s1, t, n: s0 * (s1 / s0) ** (t / n),
    "linear": lambda s0, s1, t, n: s0 + (s1 - s0) * t / n,
    "inverse": lambda s0, s1, t, n: s0 / (1 + (s0 / s1 - 1) * t / n),
}


def grid_distances(n_rows, n_cols, grid_metric):
    """Distance on the grid between every pair of nodes, (n_nodes, n_nodes)."""
    coords = np.array([(r, c) for r in range(n_rows) for c in range(n_cols)])
    offsets = np.abs(coords[:, None] - coords[None])
    if grid_metric == "chebyshev":
        return offsets.max(-1).astype(np.float64)
    if grid_metric == "manhattan":
        return offsets.sum(-1).astype(np.float64)
    return np.sqrt((offsets**2).sum(-1))


def neighbourhood_matrix(n_rows, n_cols, sigma, neighbourhood="gaussian", grid_metric="euclidean"):
    """H[j, k] = h(d(r_j, r_k); sigma); at sigma 0 the identity."""
    d = grid_distances(n_rows, n_cols, grid_metric)
    return (d == 0).astype(np.float64) if sigma == 0 else NEIGHBOURHOODS[neighbourhood](d, sigma)


def reference_som(
    X,
    n_rows,
    n_cols,
    sigma_initial,
    sigma_final,
    epochs,
    metric="euclidean",
    seed=0,
    weights=None,
    neighbourhood="gaussian",
    grid_metric="euclidean",
    sigma_schedule="exponential",
):
    """Kohonen's batch map written out per sample in float64, from the given or fastersom's seeded random weights."""
    weights = random_init(n_rows, n_cols, X.shape[1], seed) if weights is None else weights.astype(np.float64)
    X = X.astype(np.float64)
    # the default schedule_end: the exponential schedule ends one epoch after the last, the others in it
    span = epochs if sigma_schedule == "exponential" else max(epochs - 1, 1)
    for t in range(epochs):
        sigma = SCHEDULES[sigma_schedule](sigma_initial, sigma_final, t, span)
        bmus = brute_force_distances(X, weights, metric).argmin(1)
        h = neighbourhood_matrix(n_rows, n_cols, sigma, neighbourhood, grid_metric)[bmus]  # (n_samples, n_nodes)
        denominator = h.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):  # nodes with denominator 0 keep their weights
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
        ({"metric": "chebyshev"}, "metric"),
        ({"init": "linear"}, "init"),
        ({"sigma_initial": 0}, "sigma"),
        ({"sigma_final": -1}, "sigma"),
        ({"sigma_initial": np.nan}, "sigma"),
        ({"n_rows": 0}, "n_rows"),
        ({"n_cols": 2.5}, "n_cols"),
        ({"n_epochs": -1}, "n_epochs"),
        ({"batch_size": 0}, "batch_size"),
        ({"memory_fraction": 0}, "memory_fraction"),
        ({"precision": "bfloat16"}, "precision"),
        ({"neighbourhood": "hexagonal"}, "neighbourhood"),
        ({"grid_metric": "hexagonal"}, "grid_metric"),
        ({"sigma_schedule": "cosine"}, "sigma_schedule"),
        ({"sigma_initial": "quantile"}, "sigma_initial"),
        ({"sigma_final": 0}, "sigma_schedule='exponential'"),
        ({"sigma_initial": 0, "sigma_schedule": "inverse"}, "sigma_schedule='inverse'"),
        ({"sigma_initial": -1, "sigma_schedule": "linear"}, "sigma"),
        ({"sigma_final": np.inf, "sigma_schedule": "linear"}, "sigma"),
        ({"sigma_final": True, "sigma_schedule": "linear"}, "sigma"),
        ({"sigma_initial": "flowsom", "n_rows": 1, "n_cols": 1}, "sigma_schedule='exponential'"),
        ({"sigma_schedule": [2.0, 1.0]}, "n_epochs = 10 finite"),
        ({"sigma_schedule": [-1.0], "n_epochs": 1}, "sigma_schedule"),
        ({"sigma_schedule": [np.nan], "n_epochs": 1}, "sigma_schedule"),
        ({"sigma_schedule": [[1.0]]}, "sigma_schedule"),
        ({"sigma_schedule": [1.0], "sigma_final": 0.5}, "must be None"),
        ({"sigma_schedule": [1.0], "schedule_end": "last_epoch"}, "must be None"),
        ({"schedule_end": "end"}, "schedule_end"),
        ({"sigma_initial": [1, 2]}, "sigma_initial"),
        ({"sigma_final": 1j}, "sigma_final"),
        ({"sigma_final": np.complex128(2 + 1j)}, "sigma_final"),
        ({"sigma_initial": 10**400}, "sigma_initial"),
        ({"sigma_initial": b"2.5"}, "sigma_initial"),
        ({"sigma_schedule": ["wide"]}, "sigma_schedule"),
        ({"sigma_schedule": (sigma for sigma in [1.0])}, "sigma_schedule"),
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
    som_kwargs = {"sigma_initial": 1.25, "sigma_final": 0.125, "metric": metric, "random_state": 0, "device": CPU}

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

    # default sigmas: sigma_initial = ceil(min(n_rows, n_cols) / 2) / 2 and sigma_final = min(0.1 * sigma_initial, 1)
    som = SOM(6, 6, n_epochs=2, metric=metric, random_state=0, device=CPU).fit(data)
    ref = reference_som(data, 6, 6, 1.5, 0.15, 2, metric)
    assert som.quantization_error(data) == pytest.approx(
        brute_force_distances(data, ref, metric).min(1).mean(), rel=1e-3
    )
    for size, sigma_initial, sigma_final in ((30, 7.5, 0.75), (42, 10.5, 1.0)):  # below and above the cap
        som = SOM(size, size, metric=metric, device=CPU)
        som._setup()
        assert som._sigmas == pytest.approx(
            [sigma_initial * (sigma_final / sigma_initial) ** (t / 10) for t in range(10)]
        )


def neighbourhood_som(n_rows, n_cols, device=CPU, **kwargs):
    som = SOM(n_rows, n_cols, device=device, **kwargs)
    som._setup()
    return som


def smooth(som, values, sigma):
    """H @ values with fit's neighbourhood matrix of width sigma."""
    return som._smoother(sigma)(torch.as_tensor(values, device=som._device)).cpu().numpy()


@pytest.mark.parametrize("neighbourhood", NEIGHBOURHOODS)
@pytest.mark.parametrize("grid_metric", GRID_METRICS)
def test_neighbourhood_matrix(neighbourhood, grid_metric):
    # the update's H @ values, separable along the grid axes or not, on non-square, flat and square grids; widths at
    # distances between nodes (ties with the cut-offs, which count as inside) and just below them
    rng = np.random.default_rng(0)
    for n_rows, n_cols in ((4, 7), (7, 4), (1, 6), (5, 1), (6, 6)):
        som = neighbourhood_som(n_rows, n_cols, neighbourhood=neighbourhood, grid_metric=grid_metric)
        values = rng.normal(size=(n_rows * n_cols, 3)).astype(np.float32)
        for sigma in (0, 1e-20, 0.05, 0.5, 1, np.sqrt(2), 1.5, np.sqrt(5), 3 - 1e-9, 3, 20):
            expected = neighbourhood_matrix(n_rows, n_cols, sigma, neighbourhood, grid_metric) @ values
            np.testing.assert_allclose(smooth(som, values, sigma), expected, atol=1e-5, err_msg=f"sigma={sigma}")


def test_neighbourhood_values():
    # the centre node's row of H on a 3 x 3 grid, worked out by hand: [[diagonal, edge, diagonal], [edge, 1, edge], ...]
    cases = [
        ("gaussian", "euclidean", 1, np.exp(-0.5), np.exp(-1)),
        ("gaussian", "chebyshev", 1, np.exp(-0.5), np.exp(-0.5)),
        ("gaussian", "manhattan", 1, np.exp(-0.5), np.exp(-2)),
        ("bubble", "chebyshev", 1, 1, 1),
        ("bubble", "euclidean", 1, 1, 0),
        ("bubble", "euclidean", np.sqrt(2), 1, 1),  # inclusive at d == sigma
        ("bubble", "manhattan", 1.9, 1, 0),
        ("bubble", "chebyshev", 0.99, 0, 0),
        ("triangle", "euclidean", 2, 0.5, 1 - np.sqrt(2) / 2),
        ("triangle", "manhattan", 2, 0.5, 0),
        ("epanechnikov", "euclidean", 2, 0.75, 0.5),
        ("epanechnikov", "chebyshev", 1, 0, 0),
        ("cut_gaussian", "euclidean", 1, np.exp(-0.5), 0),
        ("cut_gaussian", "chebyshev", 1, np.exp(-0.5), np.exp(-0.5)),
        ("mexican_hat", "euclidean", 1, 0, -np.exp(-1)),
        ("mexican_hat", "euclidean", 2, 0.75 * np.exp(-1 / 8), 0.5 * np.exp(-1 / 4)),
    ]
    for neighbourhood, grid_metric, sigma, edge, diagonal in cases:
        som = neighbourhood_som(3, 3, neighbourhood=neighbourhood, grid_metric=grid_metric)
        row = smooth(som, np.eye(9, dtype=np.float32), sigma)[4].reshape(3, 3)
        expected = [[diagonal, edge, diagonal], [edge, 1, edge], [diagonal, edge, diagonal]]
        np.testing.assert_allclose(row, expected, atol=1e-7, err_msg=f"{neighbourhood}, {grid_metric}, {sigma}")


def test_sigma_schedules():
    for name, schedule in SCHEDULES.items():
        for sigma_initial, sigma_final in ((5.0, 0.5), (3.0, 1.0), (0.5, 4.0), (7.3, 1.0)):
            for n_epochs in (0, 1, 2, 10):
                kwargs = {"n_epochs": n_epochs, "sigma_initial": sigma_initial, "sigma_final": sigma_final}
                for end, span in (("last_epoch", max(n_epochs - 1, 1)), ("after_last_epoch", n_epochs)):
                    som = neighbourhood_som(10, 10, sigma_schedule=name, schedule_end=end, **kwargs)
                    expected = [schedule(sigma_initial, sigma_final, t, span) for t in range(n_epochs)]
                    np.testing.assert_allclose(som._sigmas, expected, rtol=1e-12)
                    assert som._sigmas[:1] == (sigma_initial,)[:n_epochs]
                    if end == "last_epoch" and n_epochs > 1:
                        assert som._sigmas[-1] == sigma_final  # exactly, not by formula (7.3 to 1 in 2)
                # by default the exponential schedule ends after the last epoch (as until 0.7.0), the others in it
                default = neighbourhood_som(10, 10, sigma_schedule=name, **kwargs)._sigmas
                end = "after_last_epoch" if name == "exponential" else "last_epoch"
                assert default == neighbourhood_som(10, 10, sigma_schedule=name, schedule_end=end, **kwargs)._sigmas

    # MiniSom's sigma decays: asymptotic_decay, inverse_decay_to_one and linear_decay_to_one
    sigma, n_epochs, t = 5.0, 10, np.arange(10)
    for kwargs, minisom in [
        ({"sigma_schedule": "inverse", "sigma_final": sigma / 3}, sigma / (1 + t / (n_epochs / 2))),
        ({"sigma_schedule": "inverse", "sigma_final": 1}, sigma / (1 + t * (sigma - 1) / n_epochs)),
        ({"sigma_schedule": "linear", "sigma_final": 1}, sigma + t * (1 - sigma) / n_epochs),
    ]:
        som = neighbourhood_som(
            10, 10, n_epochs=n_epochs, sigma_initial=sigma, schedule_end="after_last_epoch", **kwargs
        )
        np.testing.assert_allclose(som._sigmas, minisom, rtol=1e-12)
    # FlowSOM's radius from r0 to 0 in the last epoch, or r0 - r0 * k / (rlen * n) at the start of pass t, k = t * n
    flowsom = {"sigma_initial": "flowsom", "sigma_final": 0, "sigma_schedule": "linear", "grid_metric": "chebyshev"}
    som = neighbourhood_som(10, 10, n_epochs=n_epochs, **flowsom)
    np.testing.assert_allclose(som._sigmas, 6 - 6 * t / (n_epochs - 1), rtol=1e-12, atol=1e-15)
    assert som._sigmas[-1] == 0
    som = neighbourhood_som(10, 10, n_epochs=n_epochs, schedule_end="after_last_epoch", **flowsom)
    np.testing.assert_allclose(som._sigmas, 6 - 6 * t / n_epochs, rtol=1e-12)

    # or the widths themselves
    widths = [3, 2.5, 0.0]
    som = neighbourhood_som(10, 10, n_epochs=3, sigma_schedule=np.array(widths))
    assert som._sigmas == (3, 2.5, 0)


def r_quantile(x, p):
    """R's quantile(x, p) of type 7, its default, as written in R's stats::quantile."""
    x = np.sort(np.ravel(x))
    index = (len(x) - 1) * p
    lo, hi = int(np.floor(index)), int(np.ceil(index))
    h = index - lo
    return x[lo] if h == 0 or x[hi] == x[lo] else (1 - h) * x[lo] + h * x[hi]


def test_flowsom_initial_radius():
    # FlowSOM's quantile(nhbrdist, 0.67) of the n_nodes x n_nodes grid distances, which is 6 for its 10 x 10 default
    flowsom = {"sigma_initial": "flowsom", "sigma_schedule": "linear"}
    assert neighbourhood_som(10, 10, grid_metric="chebyshev", **flowsom)._sigmas[0] == 6
    for n_rows, n_cols in ((10, 10), (7, 13), (1, 5), (1, 1), (12, 3)):
        for grid_metric in GRID_METRICS:
            som = neighbourhood_som(n_rows, n_cols, n_epochs=1, grid_metric=grid_metric, **flowsom)
            assert som._sigmas[0] == r_quantile(grid_distances(n_rows, n_cols, grid_metric), 0.67)


def test_default_neighbourhood(data):
    som = SOM(6, 6, n_epochs=3, random_state=0, device=CPU).fit(data)
    explicit = {"neighbourhood": "gaussian", "grid_metric": "euclidean", "sigma_schedule": "exponential"}
    explicit = SOM(6, 6, n_epochs=3, random_state=0, device=CPU, **explicit).fit(data)
    np.testing.assert_array_equal(explicit.weights_, som.weights_)
    # the schedule's widths given as a sequence
    widths = list(neighbourhood_som(6, 6, n_epochs=3)._sigmas)
    np.testing.assert_array_equal(
        SOM(6, 6, n_epochs=3, random_state=0, device=CPU, sigma_schedule=widths).fit(data).weights_, som.weights_
    )


@pytest.mark.parametrize("neighbourhood", NEIGHBOURHOODS)
@pytest.mark.parametrize("grid_metric", GRID_METRICS)
def test_neighbourhoods_match_reference(data, neighbourhood, grid_metric):
    kwargs = {"neighbourhood": neighbourhood, "grid_metric": grid_metric, "sigma_schedule": "linear"}
    for n_rows, n_cols in ((6, 6), (4, 9)):
        ref = reference_som(data, n_rows, n_cols, 2.5, 0, 1, **kwargs)
        som = SOM(n_rows, n_cols, n_epochs=1, sigma_initial=2.5, sigma_final=0, random_state=0, device=CPU, **kwargs)
        # rtol for the mexican hat, whose small denominators extrapolate nodes far beyond the data
        np.testing.assert_allclose(som.fit(data).weights_, ref, rtol=1e-5, atol=1e-4)

    # the whole schedule over several epochs; float32 vs float64 may flip near-tied BMUs, so compare map quality. Only
    # for the gaussian: a compact neighbourhood leaves nodes with a handful of samples, whose flips change the map
    if neighbourhood != "gaussian":
        return
    ref = reference_som(data, 6, 6, 2.5, 0.5, 3, **kwargs)
    som = SOM(6, 6, n_epochs=3, sigma_initial=2.5, sigma_final=0.5, random_state=0, device=CPU, **kwargs).fit(data)
    ref_dist = brute_force_distances(data, ref, "euclidean")
    assert (som.labels_ == ref_dist.argmin(1)).mean() > 0.99
    assert som.quantization_error(data) == pytest.approx(ref_dist.min(1).mean(), rel=1e-3)


def lloyd(X, weights, n_iterations):
    """Batch k-means (Lloyd's algorithm) in float64; centroids without samples stay."""
    X, weights = X.astype(np.float64), weights.astype(np.float64)
    for _ in range(n_iterations):
        labels = brute_force_distances(X, weights, "euclidean").argmin(1)
        counts = np.bincount(labels, minlength=len(weights))[:, None]
        sums = np.zeros_like(weights)
        np.add.at(sums, labels, X)
        weights = np.where(counts > 0, sums / np.maximum(counts, 1), weights)
    return weights


def test_sigma_zero_is_batch_kmeans(data):
    init = data[np.random.default_rng(1).choice(len(data), 12, replace=False)]
    kwargs = {"init": init, "sigma_initial": 0, "sigma_final": 0, "sigma_schedule": "linear", "device": CPU}
    for neighbourhood in NEIGHBOURHOODS:
        som = SOM(3, 4, n_epochs=5, neighbourhood=neighbourhood, **kwargs).fit(data)
        np.testing.assert_allclose(som.weights_, lloyd(data, init, 5), atol=1e-4)

    # FlowSOM's settings: the bubble's radius reaches 0 in the last epoch, which updates every node from its own
    # samples only, a batch k-means step, also on a large map whose previous epoch is still wide
    flowsom = {"neighbourhood": "bubble", "grid_metric": "chebyshev", "random_state": 0, "device": CPU}
    som = SOM(50, 50, n_epochs=10, sigma_initial="flowsom", sigma_final=0, sigma_schedule="linear", **flowsom)
    widths = som.fit(data)._sigmas
    assert widths[-2] > 1
    assert widths[-1] == 0
    before = SOM(50, 50, n_epochs=9, sigma_schedule=widths[:-1], **flowsom).fit(data)
    last = SOM(50, 50, n_epochs=1, sigma_schedule=widths[-1:], init=before.weights_, **flowsom).fit(data)
    np.testing.assert_array_equal(last.weights_, som.weights_)
    labels = before.predict(data)  # the BMUs of the last epoch
    occupied = np.unique(labels)
    centroids = np.stack([data[labels == k].mean(0) for k in occupied])
    np.testing.assert_allclose(som.weights_[occupied], centroids, atol=1e-5)
    empty = np.setdiff1d(np.arange(2500), occupied)
    np.testing.assert_array_equal(som.weights_[empty], before.weights_[empty])


def test_inverse_schedule_from_a_tiny_width():
    # 1 + (s0 / s1 - 1) rounds to 0 for s0 = 1e-20 and s1 = 1, which the inverse schedule must not divide by
    for end, last in (("last_epoch", 1.0), ("after_last_epoch", 2e-20)):
        kwargs = {"sigma_initial": 1e-20, "sigma_final": 1, "sigma_schedule": "inverse", "schedule_end": end}
        sigmas = neighbourhood_som(4, 4, n_epochs=2, **kwargs)._sigmas
        assert sigmas[0] == 1e-20
        assert sigmas[1] == pytest.approx(last, rel=1e-12)


def test_widths_as_scalars():
    # any real scalar, also a 0-d array or tensor, is a width
    tensors = [torch.tensor(2.5, dtype=torch.bfloat16)]
    if torch.cuda.is_available():
        tensors.append(torch.tensor(2.5, device="cuda"))
    for sigma in (2, np.float32(2.5), np.array(2.5), torch.tensor(2.5), *tensors):
        assert neighbourhood_som(4, 4, n_epochs=1, sigma_initial=sigma, sigma_final=sigma)._sigmas == (float(sigma),)
    # but not a complex one
    for sigma in (torch.tensor(2.5 + 0j), torch.tensor(2.5, dtype=torch.complex64)):
        with pytest.raises(ValueError, match="sigma_initial"):
            neighbourhood_som(4, 4, n_epochs=1, sigma_initial=sigma)


def test_neighbourhood_options_pickle(data):
    for kwargs in (
        {"sigma_initial": "flowsom", "sigma_final": 0, "sigma_schedule": "linear"},
        {"sigma_schedule": np.array([2.0, 0.5])},
    ):
        som = SOM(
            5, 4, n_epochs=2, neighbourhood="bubble", grid_metric="chebyshev", random_state=0, device=CPU, **kwargs
        )
        som.fit(data)
        loaded = pickle.loads(pickle.dumps(som))
        assert loaded._sigmas == som._sigmas
        np.testing.assert_array_equal(loaded.fit(data).weights_, som.weights_)


def test_mexican_hat_keeps_nodes_without_positive_weight():
    # with a wide mexican hat, nodes far from the only occupied one get a negative denominator and keep their weights
    X = np.zeros((10, 2), np.float32)
    init = np.arange(24, dtype=np.float32).reshape(12, 2)
    som = SOM(1, 12, n_epochs=1, init=init, sigma_initial=2, neighbourhood="mexican_hat", device=CPU).fit(X)
    far = np.arange(12) >= 2  # d >= sigma from node 0, the BMU of every sample: denominators <= 0
    np.testing.assert_array_equal(som.weights_[far], init[far])
    np.testing.assert_array_equal(som.weights_[~far], 0)


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
    kwargs = {"n_rows": 6, "n_cols": 6, "metric": metric, "random_state": 0, "device": CPU}
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
    for key in ("memory_fraction", "verbose", "deterministic", "fused", "precision"):  # pickled by 0.2
        del state[key]
    for key in ("neighbourhood", "grid_metric", "sigma_schedule", "schedule_end", "stream"):  # pickled by 0.7.0
        del state[key]
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
    kwargs = {"n_rows": 6, "n_cols": 6, "metric": metric, "random_state": 0}
    cpu = SOM(n_epochs=1, device=CPU, **kwargs).fit(data)
    gpu = SOM(n_epochs=1, device=device, deterministic=deterministic, **kwargs).fit(data)
    np.testing.assert_allclose(gpu.weights_, cpu.weights_, atol=1e-5)
    np.testing.assert_allclose(gpu.transform(data), cpu.transform(data), atol=1e-4)
    cpu = SOM(n_epochs=3, device=CPU, **kwargs).fit(data)
    gpu = SOM(n_epochs=3, device=device, deterministic=deterministic, **kwargs).fit(data)
    assert (gpu.labels_ == cpu.labels_).mean() > 0.95
    assert gpu.quantization_error(data) == pytest.approx(cpu.quantization_error(data), rel=1e-2)


@pytest.mark.parametrize("device", GPUS)
def test_gpu_neighbourhoods_match_cpu(data, device):
    for neighbourhood, grid_metric, n_rows, n_cols in (
        ("bubble", "chebyshev", 5, 7),
        ("triangle", "euclidean", 5, 7),
        ("mexican_hat", "manhattan", 7, 5),
        ("gaussian", "chebyshev", 1, 12),
    ):
        kwargs = {"n_rows": n_rows, "n_cols": n_cols, "n_epochs": 1, "random_state": 0, "sigma_schedule": "linear"}
        kwargs |= {"neighbourhood": neighbourhood, "grid_metric": grid_metric, "sigma_initial": "flowsom"}
        cpu = SOM(device=CPU, **kwargs).fit(data)
        gpu = SOM(device=device, deterministic=True, **kwargs).fit(data)
        np.testing.assert_allclose(gpu.weights_, cpu.weights_, atol=1e-5)
        np.testing.assert_array_equal(SOM(device=device, deterministic=True, **kwargs).fit(data).weights_, gpu.weights_)


@pytest.mark.parametrize("device", GPUS)
def test_deterministic(device, monkeypatch):
    # many samples per node, so that the order of atomic additions varies between runs without deterministic
    X = np.random.default_rng(0).normal(size=(200_000, 6)).astype(np.float32)
    kwargs = {"n_rows": 4, "n_cols": 4, "n_epochs": 3, "random_state": 0, "device": device}
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


@pytest.mark.parametrize("device", GPUS)
def test_deterministic_keeps_small_values(device):
    # fixed point scaled for the 1e16 would round every 1 to 0; node 0 holds only the 1s
    X = np.ones((1024, 1), np.float32)
    X[-1] = 1e16
    som = SOM(1, 2, n_epochs=1, sigma_initial=0.1, init=[[1], [1e16]], device=device, deterministic=True).fit(X)
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


@pytest.mark.parametrize("p", [1, 2])
def test_cdist_matches_torch(p):
    # on MPS one feature at a time, as torch's cdist there holds samples x nodes x features floats
    from fastersom._som import _cdist

    rng = np.random.default_rng(0)
    x, y = (torch.as_tensor(rng.normal(size=(n, 7)).astype(np.float32)) for n in (50, 30))
    ref = torch.cdist(x.double(), y.double(), p=p).float()
    for device in ["cpu", *(["mps"] if torch.backends.mps.is_available() else [])]:
        torch.testing.assert_close(_cdist(x.to(device), y.to(device), p).cpu(), ref)


@pytest.fixture(scope="module")
def metal_som():
    """Skips unless torch compiles Metal kernels here (MPS, torch >= 2.6); returns fastersom._som."""
    from fastersom import _som

    if not torch.backends.mps.is_available() or _som._metal(torch.device("mps")) is None:
        pytest.skip("no MPS or no torch.mps.compile_shader")
    return _som


SEARCH_SIZES = [(5000, 6, 7, 9), (3001, 34, 30, 30), (1500, 196, 9, 9), (500, 160, 3, 3)]


def assert_fused_matches_torch(som, X, device, agreement=0.999):
    """The fused kernel's BMUs, distances and best two nodes against som's torch path."""
    n_features = X.shape[1]
    labels, distances = som.predict(X, return_distance=True)
    top2 = som._search(som._weights, som._kernel(n_features, top2=True), top2=True)(torch.as_tensor(X, device=device))
    som.fused = False
    ref_labels, ref_distances = som.predict(X, return_distance=True)
    # float32 sums in another order may flip near-ties only. So the torch path's scores of the BMUs found are its
    # best, up to rounding, which on tensor cores is about that of float32 relative to the largest score
    ref_scores = som._scores(torch.as_tensor(X, device=device), som._weights)
    atol = 1e-5 * max(1, ref_scores.abs().max().item() / 10)
    assert (labels == ref_labels).mean() > agreement
    found = ref_scores.gather(1, torch.as_tensor(labels, device=device)[:, None])[:, 0]
    torch.testing.assert_close(found, ref_scores.amin(1), rtol=1e-5, atol=atol)
    np.testing.assert_allclose(distances, ref_distances, rtol=1e-5, atol=1e-5)
    # the two best nodes in order, by the torch path's scores, as their order is arbitrary on exact ties
    assert (top2[:, 0] != top2[:, 1]).all()
    ref_top2 = ref_scores.topk(2, 1, largest=False).values
    torch.testing.assert_close(ref_scores.gather(1, top2), ref_top2, rtol=1e-5, atol=atol)


def search_case(metric, n_samples, n_features, n_rows, n_cols, device):
    """Samples with an offset and a map of samples plus noise, for assert_fused_matches_torch."""
    rng = np.random.default_rng(0)
    X = (rng.normal(size=(n_samples, n_features)) + rng.normal(size=n_features) * 3).astype(np.float32)
    weights = X[rng.choice(n_samples, n_rows * n_cols)] + rng.normal(scale=0.1, size=(n_rows * n_cols, n_features))
    return X, SOM.from_weights(weights.astype(np.float32), n_rows, n_cols, metric=metric, device=device)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(("n_samples", "n_features", "n_rows", "n_cols"), SEARCH_SIZES)
def test_fused_matches_torch(fused, metric, n_samples, n_features, n_rows, n_cols):
    # sizes off the tiles. Float32 kernel: 6 features without its loop over 16, 34 and 196 with a remainder, 160 without
    # (manhattan only, as 9 nodes stay off tensor cores); tensor cores (on GPUs with fast TF32): 34 and 196 features
    X, som = search_case(metric, n_samples, n_features, n_rows, n_cols, "cuda")
    n_nodes = n_rows * n_cols
    if metric != "manhattan" and n_nodes > 64 and n_nodes * n_features >= 8192:
        fast_tf32 = torch.cuda.get_device_capability() in fused._FAST_TF32
        assert som._kernel(n_features) == ("tensor-core" if fast_tf32 else "float32" if n_features <= 128 else None)
    else:
        assert som._kernel(n_features) == ("float32" if metric == "manhattan" or n_features <= 128 else None)
    assert_fused_matches_torch(som, X, "cuda")


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(("n_samples", "n_features", "n_rows", "n_cols"), [*SEARCH_SIZES, (700, 64, 5, 5)])
def test_metal_matches_torch(metal_som, metric, n_samples, n_features, n_rows, n_cols):
    # bmu_dot at 6, 34 and 64 features (64 adds the bias after the products, as it has no padding to ride in),
    # bmu_staged at 160 (a multiple of its chunks of 16, so that the bias takes a chunk of its own) and 196; bmu_l1 with
    # 8, 3 and 2 samples per thread. Agreement 0.99, as at 160 features one of the 500 samples is a near-tie
    X, som = search_case(metric, n_samples, n_features, n_rows, n_cols, "mps")
    assert som._kernel(n_features) == "metal"
    assert metal_som._metal(torch.device("mps"))._use_staged(n_rows * n_cols, n_features) == (n_features > 64)
    assert_fused_matches_torch(som, X, "mps", agreement=0.99)


def assert_ties_and_offsets(device):
    # duplicate nodes tie exactly: the first wins, as with torch's argmin
    weights = np.array([[3, 1, 2], [1, 2, 3], [3, 1, 2], [1, 2, 3], [2, 3, 1]], np.float32)
    for metric in METRICS:
        som = SOM.from_weights(weights, 1, 5, metric=metric, device=device)
        np.testing.assert_array_equal(som.predict(weights[[1, 0, 4, 3, 2]]), [1, 0, 4, 1, 0])
    # also when the copies fall in different parts of the nodes, which few samples split across programs
    weights = np.random.default_rng(0).normal(size=(20, 30, 6)).astype(np.float32)
    weights[19, 29] = weights[12, 5] = weights[0, 10]
    for metric in ("euclidean", "manhattan"):
        som = SOM.from_weights(weights, 20, 30, metric=metric, device=device)
        np.testing.assert_array_equal(som.predict(np.repeat(weights[:1, 10], 3, 0)), [10, 10, 10])
    # as test_euclidean_with_large_offset and test_correlation_with_large_offset (the 10 x 10 map: on tensor cores
    # where they are fast)
    X = (1e4 + np.random.default_rng(0).normal(scale=0.01, size=(100, 100))).astype(np.float32)
    np.testing.assert_array_equal(SOM.from_weights(X, 10, 10, device=device).predict(X), np.arange(100))
    som = SOM.from_weights(
        np.array([[1e7, 1e7, 1e7 + 1], [0.1, 0, 1]], np.float32), 1, 2, metric="correlation", device=device
    )
    labels, distances = som.predict(np.array([[0, 0, 1]], np.float32), return_distance=True)
    assert labels[0] == 0
    assert distances[0] == pytest.approx(0, abs=1e-6)
    assert np.isnan(som.topographic_error(np.ones((10, 3), np.float32)))  # no sample has a BMU, so none is searched


def test_fused_ties_and_offsets(fused):
    assert_ties_and_offsets("cuda")


def float16_case(device, monkeypatch, module):
    """Pixel-like data of 20 features, 0-1 scaled with clusters, a float32 and a float16 fit of a 24 x 24 map, and
    which of the float16 fit's searches (by module.search) were in half."""
    rng = np.random.default_rng(0)
    centres = rng.random((12, 20))
    X = np.clip(centres[rng.integers(12, size=20_000)] + rng.normal(scale=0.05, size=(20_000, 20)), 0, 1)
    X = X.astype(np.float32)
    kwargs = {"n_rows": 24, "n_cols": 24, "random_state": 0, "device": device}
    ref = SOM(**kwargs).fit(X)
    search, halves = module.search, []

    def spy(nodes, *args, **kw):
        if len(nodes) == 24 * 24:  # not the self-checks' searches
            halves.append(kw.get("bound") is not None)
        return search(nodes, *args, **kw)

    monkeypatch.setattr(module, "search", spy)
    som = SOM(precision="float16", **kwargs).fit(X)
    monkeypatch.undo()
    return X, ref, som, halves


def assert_float16_fit(X, ref, som, halves):
    # every epoch's search in half, then labels_ in float32; an equally good map, not the same one
    assert halves == [True] * som.n_epochs + [False]
    assert not np.array_equal(som.weights_, ref.weights_)
    assert som.quantization_error(X) == pytest.approx(ref.quantization_error(X), rel=0.02)
    assert abs(som.topographic_error(X) - ref.topographic_error(X)) < 0.02


def test_metal_float16(metal_som, monkeypatch):
    assert_float16_fit(*float16_case("mps", monkeypatch, metal_som._metal(torch.device("mps"))))
    som = SOM(24, 24, metric="manhattan", device="mps")
    som._setup()
    assert som._kernel(20, half=True) == "metal"  # as with fewer than 16 features
    som = SOM(24, 24, device="mps")
    som._setup()
    assert som._kernel(15, half=True) == "metal"


def test_metal_float16_falls_back_beyond_half_range(data, metal_som, monkeypatch):
    # the nodes' scales would leave float32's exponents, so fit searches in float32
    module, halves = metal_som._metal(torch.device("mps")), []
    search = module.search
    monkeypatch.setattr(module, "search", lambda *args, **kw: halves.append(kw.get("bound")) or search(*args, **kw))
    SOM(6, 6, n_epochs=1, precision="float16", device="mps").fit(np.tile(data, 4) * 2.0**31)
    assert halves
    assert all(bound is None for bound in halves)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_deterministic_without_fused_kernels():
    # on MPS without the Metal kernels, the deterministic sums take a one-hot matrix product
    X = np.random.default_rng(0).normal(size=(20_000, 6)).astype(np.float32)
    kwargs = {
        "n_rows": 4,
        "n_cols": 4,
        "n_epochs": 2,
        "random_state": 0,
        "device": "mps",
        "fused": False,
        "deterministic": True,
    }
    np.testing.assert_array_equal(SOM(**kwargs).fit(X).weights_, SOM(**kwargs).fit(X).weights_)


def test_metal_ties_and_offsets(metal_som):
    assert_ties_and_offsets("mps")


def test_metal_manhattan_up_to_512_features(metal_som):
    # beyond 512 features a threadgroup's samples no longer fit in its threadgroup memory
    for n_features, expected in ((512, "metal"), (513, None)):
        som = SOM.from_weights(np.eye(4, n_features, dtype=np.float32), 2, 2, metric="manhattan", device="mps")
        assert som._kernel(n_features) == expected


@pytest.fixture
def fresh_metal(metal_som, monkeypatch):
    """fastersom._som with no Metal kernel checked yet, and the checks of other tests restored afterwards."""
    monkeypatch.setattr(metal_som, "_FAILED", set())
    metal_som._works.cache_clear()
    yield metal_som
    monkeypatch.undo()
    metal_som._works.cache_clear()


def test_metal_failure_falls_back(data, fresh_metal, monkeypatch):
    # a kernel that fails its self-check (or does not compile) warns once, at the caller, and leaves the torch path
    som = SOM.from_weights(data[:36], 6, 6, device="mps", fused=False)
    expected = som.predict(data)
    som.fused = True

    def fail(*args):
        raise RuntimeError("wrong BMUs in a self-check")

    monkeypatch.setattr(fresh_metal._metal(torch.device("mps")), "check", fail)
    with pytest.warns(
        RuntimeWarning, match=r"Metal BMU kernel failed to compile or its self-check \(RuntimeError"
    ) as w:
        assert som._kernel(6) is None
    assert w[0].filename == __file__
    np.testing.assert_array_equal(som.predict(data), expected)  # no second warning


def test_metal_out_of_memory_is_no_failure(fresh_metal, monkeypatch):
    # MPS raises out of memory as a plain RuntimeError, which must not turn the kernels off for the session
    def fail(*args):
        raise RuntimeError("MPS backend out of memory (MPS allocated: 1.04 GiB)")

    monkeypatch.setattr(fresh_metal._metal(torch.device("mps")), "check", fail)
    with pytest.raises(RuntimeError, match="out of memory"):
        fresh_metal._works(torch.device("mps"), "metal", 6, "euclidean", False)
    assert not fresh_metal._FAILED


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


@pytest.mark.parametrize(
    ("device", "compiler"),
    [
        pytest.param("cuda", "_triton", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")),
        pytest.param("mps", "_metal", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")),
    ],
)
def test_fused_off_never_compiles(data, monkeypatch, device, compiler):
    from fastersom import _som

    def fail(device):
        raise AssertionError("compiled")

    monkeypatch.setattr(_som, compiler, fail)
    som = SOM(6, 6, n_epochs=1, metric="manhattan", device=device, fused=False).fit(data)
    som.predict(data)
    som.topographic_error(data)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fused_skips_old_gpus(data, monkeypatch):
    # below compute capability 7.0, which Triton does not support, the torch path runs without compiling or warning
    from fastersom import _som

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (6, 0))
    _som._triton.cache_clear()
    try:
        assert _som._triton(torch.device("cuda")) is None
        SOM(6, 6, n_epochs=1, device="cuda").fit(data)
    finally:
        _som._triton.cache_clear()


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
    assert som._batch_size_for(data.shape[1], som._kernel(6)) <= 2**28 // 36  # the fused one: 2^26 / 6
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


def reference_pca_grid(X, n_rows, n_cols, metric="euclidean"):
    """The SOM Toolbox's linear initialisation in float64, of the rows as the metric compares them."""
    X = X.astype(np.float64)
    if metric in ("cosine", "correlation"):
        X = X[np.linalg.norm(X, axis=1) > 0] if metric == "cosine" else X[X.max(1) > X.min(1)]
        if metric == "correlation":
            X = X - X.mean(1, keepdims=True)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    mean = X.mean(0)
    values, vectors = np.linalg.eigh(np.cov(X, rowvar=False).reshape(X.shape[1], X.shape[1]))
    values, vectors = values[::-1].clip(0), vectors[:, ::-1]
    vectors = vectors * np.sign(vectors[np.abs(vectors).argmax(0), np.arange(vectors.shape[1])])
    axes = [vectors[:, k] * np.sqrt(values[k]) if k < X.shape[1] else np.zeros(X.shape[1]) for k in range(2)]
    row_axis, col_axis = (axes[0], axes[1]) if n_rows >= n_cols else (axes[1], axes[0])

    def positions(n):
        return np.linspace(-1, 1, n) if n > 1 else np.zeros(1)

    grid = mean + positions(n_rows)[:, None, None] * row_axis + positions(n_cols)[None, :, None] * col_axis
    return grid.reshape(n_rows * n_cols, -1)


@pytest.mark.parametrize(("n_rows", "n_cols"), [(3, 5), (5, 3), (4, 4), (1, 6), (6, 1)])
def test_pca_init_matches_reference(data, n_rows, n_cols):
    weights = SOM(n_rows, n_cols, n_epochs=0, init="pca", device=CPU).fit(data).weights_
    np.testing.assert_allclose(weights, reference_pca_grid(data, n_rows, n_cols), rtol=1e-5, atol=1e-5)


def test_pca_init_spans_one_standard_deviation():
    rng = np.random.default_rng(0)
    rotation, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    X = ((rng.normal(size=(20000, 5)) * [3.0, 2.0, 1.0, 0.5, 0.2]) @ rotation.T + 4).astype(np.float32)
    grid = SOM(7, 5, n_epochs=0, init="pca", device=CPU).fit(X).weights_.reshape(7, 5, 5)
    np.testing.assert_allclose(grid.reshape(-1, 5).mean(0), X.mean(0), atol=0.05)
    long_axis, short_axis = (grid[-1, 2] - grid[0, 2]) / 2, (grid[3, -1] - grid[3, 0]) / 2
    assert abs(long_axis @ rotation[:, 0]) / np.linalg.norm(long_axis) > 0.999  # along the first component
    assert abs(short_axis @ rotation[:, 1]) / np.linalg.norm(short_axis) > 0.999
    assert np.linalg.norm(long_axis) == pytest.approx(3.0, rel=0.02)
    assert np.linalg.norm(short_axis) == pytest.approx(2.0, rel=0.02)


@pytest.mark.parametrize("metric", ["cosine", "correlation"])
def test_pca_init_of_compared_rows(data, metric):
    X = data.copy()
    X[:10] = 0.0 if metric == "cosine" else 1.5  # rows without a BMU take no part
    weights = SOM(4, 6, n_epochs=0, init="pca", metric=metric, device=CPU).fit(X).weights_
    np.testing.assert_allclose(weights, reference_pca_grid(X, 4, 6, metric), rtol=1e-4, atol=1e-5)


def test_pca_init_with_one_feature_and_training(data):
    one = SOM(3, 4, n_epochs=0, init="pca", device=CPU).fit(data[:, :1]).weights_
    np.testing.assert_allclose(one, reference_pca_grid(data[:, :1], 3, 4), rtol=1e-5, atol=1e-5)
    trained = SOM(6, 6, n_epochs=5, init="pca", device=CPU).fit(data)
    assert trained.quantization_error(data) < SOM(6, 6, n_epochs=0, init="pca", device=CPU).fit(
        data
    ).quantization_error(data)


def test_pca_init_of_a_large_offset():
    weights = SOM(3, 1, n_epochs=0, init="pca", device=CPU).fit(np.full((100000, 2), 1e7, np.float32)).weights_
    np.testing.assert_array_equal(weights, np.full((3, 2), 1e7, np.float32))


@pytest.mark.parametrize(
    ("platform", "expected"), [("linux", (41943, 2097, 512, 8192)), ("darwin", (65536, 8388, 8192, 8192))]
)
def test_cpu_batch_size_by_nodes_and_features(monkeypatch, platform, expected):
    monkeypatch.setattr(sys, "platform", platform)
    sizes = []
    for (n_rows, n_cols), n_features in (((10, 10), 34), ((10, 10), 2000), ((100, 100), 34), ((200, 200), 34)):
        som = SOM(n_rows, n_cols, device=CPU)
        som._setup()
        sizes.append(som._batch_size_for(n_features, None))
    assert tuple(sizes) == expected


STREAM_KWARGS = {"n_rows": 6, "n_cols": 5, "n_epochs": 3, "random_state": 0, "batch_size": 512}


class Rows:
    """Rows of an array behind slicing and a shape only, as an h5py or zarr dataset."""

    def __init__(self, X):
        self.shape, self._X = X.shape, X

    def __getitem__(self, rows):
        assert isinstance(rows, slice)
        return self._X[rows]


@pytest.mark.parametrize("device", [CPU, *GPUS])
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("init", ["random", "sample", "pca"])
def test_stream_matches_in_memory(data, device, metric, init):
    # the same batches in the same order give the same map bit for bit, on GPUs with fixed-point sums
    X = data.copy()
    X[::10] = 0  # without a BMU under cosine and correlation, which init="sample" and "pca" leave out
    kwargs = {**STREAM_KWARGS, "metric": metric, "init": init, "device": device, "deterministic": True}
    expected = SOM(**kwargs).fit(X).weights_
    np.testing.assert_array_equal(SOM(stream=True, **kwargs).fit(X).weights_, expected)


def test_stream_inputs(data, tmp_path, monkeypatch):
    expected = SOM(**STREAM_KWARGS, device=CPU).fit(data).weights_
    path = tmp_path / "X.npy"
    np.save(path, data.astype(np.float64).reshape(50, 100, 6))
    memmap = np.load(path, mmap_mode="r")  # read-only, float64, reshaped as a view
    for X in (memmap, np.load(path, mmap_mode="r+"), torch.from_numpy(data), data.tolist(), Rows(data)):
        som = SOM(**STREAM_KWARGS, device=CPU, stream=True).fit(X)
        np.testing.assert_array_equal(som.weights_, expected)
        assert not hasattr(som, "labels_")
    # float64 rounds to float32 as in memory
    X = data + np.random.default_rng(0).normal(scale=1e-6, size=data.shape)
    np.testing.assert_array_equal(
        SOM(**STREAM_KWARGS, device=CPU, stream=True).fit(X).weights_, SOM(**STREAM_KWARGS, device=CPU).fit(X).weights_
    )
    # the CPU's sums in parts of several batches, the last one shorter
    from fastersom import _som

    monkeypatch.setattr(_som, "_CPU_SUM_ROWS", 1300)
    np.testing.assert_array_equal(
        SOM(**STREAM_KWARGS, device=CPU, stream=True).fit(data).weights_,
        SOM(**STREAM_KWARGS, device=CPU).fit(data).weights_,
    )


def test_stream_input_validation(data, tmp_path):
    som = SOM(**STREAM_KWARGS, device=CPU, stream=True)
    for X in (data[0], data[:0]):
        with pytest.raises(ValueError, match="shape"):
            som.fit(X)
    path = tmp_path / "X.npy"
    np.save(path, data.reshape(50, 100, 6))
    for X in (np.load(path, mmap_mode="r").transpose(1, 0, 2), Rows(data.reshape(50, 100, 6))):
        with pytest.raises(ValueError, match="C-contiguous"):  # reshaping would read all of X
            som.fit(X)
    for value in (np.nan, np.inf):
        bad = data.copy()
        bad[4321, 2] = value  # in the last of several batches
        with pytest.raises(ValueError, match="NaN"):
            som.fit(bad)
    with pytest.raises(ValueError, match="with a BMU"):
        SOM(init="sample", metric="cosine", device=CPU, stream=True).fit(np.zeros((10, 6)))


def test_stream_pickle(data):
    som = SOM(**STREAM_KWARGS, device=CPU, stream=True).fit(data)
    loaded = pickle.loads(pickle.dumps(som))
    assert loaded.stream
    np.testing.assert_array_equal(loaded.fit(data).weights_, som.weights_)
    np.testing.assert_array_equal(som.fit_predict(data), SOM(**STREAM_KWARGS, device=CPU).fit(data).labels_)


@pytest.mark.parametrize("device", [CPU, *GPUS])
@pytest.mark.parametrize("metric", METRICS)
def test_stream_predict_transform_and_errors(data, device, metric):
    X = data.copy()
    X[::10] = 0  # without a BMU under cosine and correlation
    kwargs = {**STREAM_KWARGS, "metric": metric, "device": device, "deterministic": True}
    som = SOM(**kwargs).fit(X)
    streamed = SOM(**kwargs, stream=True).fit(X)
    labels, distances = streamed.predict(X, return_distance=True)
    np.testing.assert_array_equal(labels, som.labels_)
    np.testing.assert_array_equal(streamed.predict(X), som.labels_)
    np.testing.assert_array_equal(distances, som.predict(X, return_distance=True)[1])
    np.testing.assert_array_equal(streamed.fit_predict(X), som.labels_)
    np.testing.assert_array_equal(streamed.transform(X), som.transform(X))
    assert streamed.quantization_error(X) == som.quantization_error(X)
    # streamed, the samples with a BMU are regrouped into the batches of the search in memory
    assert streamed.topographic_error(X) == som.topographic_error(X)


@pytest.mark.parametrize("device", [CPU, *GPUS])
@pytest.mark.parametrize("metric", METRICS)
def test_stream_kept_batches(data, device, metric):
    # the batches of data[has_bmu].split(batch_size), also across batches without any sample with a BMU
    X = data.copy()
    X[::7] = 0
    X[600:1700] = 0
    som = SOM(**STREAM_KWARGS, metric=metric, device=device).fit(data)
    streamed = SOM.from_weights(som.weights_, 6, 5, metric=metric, device=device, stream=True)
    has_bmu = som._has_bmu(torch.as_tensor(X))
    kept = torch.as_tensor(X) if has_bmu is None else torch.as_tensor(X)[has_bmu]
    batches = [batch.cpu() for batch in streamed._kept_batches(streamed._check_X(X), 512)]
    assert [len(batch) for batch in batches] == [len(batch) for batch in kept.split(512)]
    np.testing.assert_array_equal(torch.cat(batches).numpy(), kept.numpy())
    assert streamed.topographic_error(X) == som.topographic_error(X)


def test_stream_predict_input_validation(data, tmp_path):
    som = SOM(**STREAM_KWARGS, device=CPU, stream=True).fit(data)
    path = tmp_path / "X.npy"
    np.save(path, data.reshape(50, 100, 6))
    np.testing.assert_array_equal(som.predict(np.load(path, mmap_mode="r")), som.predict(data))
    bad = data.copy()
    bad[4321, 2] = np.nan
    for method in (som.predict, som.transform, som.quantization_error, som.topographic_error):
        with pytest.raises(ValueError, match="NaN"):  # at the end of the pass
            method(bad)
        with pytest.raises(ValueError, match="features"):
            method(data[:, :3])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_stream_cuda_copies(data, tmp_path, monkeypatch):
    # through pinned buffers smaller than a batch, so that batches arrive in several copies, the last one shorter
    from fastersom import _som

    monkeypatch.setattr(_som, "_PINNED_VALUES", 1000)  # 166 rows of 6 features
    kwargs = {**STREAM_KWARGS, "device": "cuda", "deterministic": True}
    som = SOM(**kwargs).fit(data)
    path = tmp_path / "X.npy"
    np.save(path, data.astype(np.float64))
    for X in (np.load(path, mmap_mode="r"), torch.from_numpy(data), Rows(data)):
        streamed = SOM(**kwargs, stream=True).fit(X)
        np.testing.assert_array_equal(streamed.weights_, som.weights_)
        np.testing.assert_array_equal(streamed.predict(X), som.labels_)
    bad = data.copy()
    bad[4321, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        SOM(**kwargs, stream=True).fit(bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_stream_results_off_the_device(data, metric):
    # streamed, the per-sample results go to the host batch by batch: the peak device memory does not grow with X
    som = SOM(**STREAM_KWARGS, metric=metric, device="cuda", stream=True).fit(data)
    for method in (lambda X: som.predict(X, return_distance=True), som.quantization_error, som.topographic_error):
        peaks = []
        for X in (data, np.tile(data, (80, 1))):  # 400000 samples, 4.8 MB of BMUs and distances
            method(X)  # once before, for the memory kept between calls, such as cuBLAS's workspace
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            method(X)
            peaks.append(torch.cuda.max_memory_allocated() - before)
        assert peaks[1] <= 2 * peaks[0] < 1 << 20


def test_nth_true():
    from fastersom._som import _nth_true

    mask = np.random.default_rng(0).random(3_000_000) < 0.3
    first = int(mask[: 1 << 20].sum())  # the first block's True values
    ranks = np.r_[np.random.default_rng(1).integers(0, mask.sum(), 1000), 0, first - 1, first, mask.sum() - 1]
    np.testing.assert_array_equal(_nth_true(mask, ranks), np.flatnonzero(mask)[ranks])
