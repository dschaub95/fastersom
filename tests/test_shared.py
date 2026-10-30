"""Tests of the module _shared, kept byte-identical in fastersom and its sister k-means package (like _shared itself)."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
import torch

# the package's _shared, found under src, so that this file is the same in both packages
(PACKAGE,) = {path.parent.name for path in (Path(__file__).parents[1] / "src").glob("*/_shared.py")}
shared = import_module(f"{PACKAGE}._shared")
CPU = torch.device("cpu")
DEVICES = (
    ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)


def exact_grid(n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Samples and 32 codebook vectors (four of them equal) on a grid that float32 keeps exactly, a mean in 1/32.

    Returns them and every sample's nearest codebook vector by brute force, the lowest index on ties.
    """
    rng = np.random.default_rng(11)
    X = rng.integers(-4, 5, size=(n_samples, 3)).astype(np.float32)
    codebook = rng.integers(-4, 5, size=(32, 3)).astype(np.float32)
    codebook[[9, 20, 31]] = codebook[4]
    distances = ((X[:, None].astype(np.float64) - codebook) ** 2).sum(2)
    assert ((distances == distances.min(1, keepdims=True)).sum(1) > 1).mean() > 0.1  # many ties
    return X, codebook, distances.argmin(1)  # numpy's argmin: the first index on ties


def sums_of(X: np.ndarray, labels: np.ndarray, n_vectors: int) -> np.ndarray:
    sums = np.zeros((n_vectors, X.shape[1]), dtype=np.float64)
    np.add.at(sums, labels, X)
    return sums


@pytest.mark.parametrize("centred", [False, True])
def test_cpu_passes_find_the_nearest_codebook_vector(centred):
    # over several parts, the last one partial
    X, codebook, expected = exact_grid(3 * shared._CPU_MIN_PART_ROWS + 5)
    data = torch.as_tensor(X)
    search = shared._nearest(shared._euclidean_scorer(torch.as_tensor(codebook), centred), CPU)
    assigned, stepped = (torch.empty(len(X), dtype=torch.long) for _ in range(2))
    with shared._cpu_pool(CPU) as pool:
        shared._assign(data, search, assigned, 1000, pool=pool)
        sums, counts = shared._cpu_step(data, search, stepped, 32, 1000, pool)
    np.testing.assert_array_equal(assigned, expected)
    np.testing.assert_array_equal(stepped, expected)
    np.testing.assert_array_equal(counts, np.bincount(expected, minlength=32))
    np.testing.assert_array_equal(sums, sums_of(X, expected, 32))


@pytest.mark.parametrize("device", DEVICES)
def test_torch_path_finds_the_nearest_codebook_vector(device):
    X, codebook, expected = exact_grid(5000)
    data = torch.as_tensor(X, device=device)
    search = shared._nearest(shared._euclidean_scorer(torch.as_tensor(codebook, device=device)), torch.device(device))
    labels = torch.empty(len(X), dtype=torch.long, device=device)
    shared._assign(data, search, labels, 1000)
    np.testing.assert_array_equal(labels.cpu(), expected)


def test_cpu_step_leaves_out_samples_without_a_nearest_codebook_vector():
    X, codebook, expected = exact_grid(2 * shared._CPU_MIN_PART_ROWS + 5)
    keep = X[:, 0] > 0
    data = torch.as_tensor(X)
    search = shared._nearest(shared._euclidean_scorer(torch.as_tensor(codebook)), CPU)
    labels = torch.empty(len(X), dtype=torch.long)
    with shared._cpu_pool(CPU) as pool:
        sums, counts = shared._cpu_step(data, search, labels, 32, 1000, pool, lambda batch: batch[:, 0] > 0)
    np.testing.assert_array_equal(labels, np.where(keep, expected, 32))
    np.testing.assert_array_equal(counts, np.bincount(expected[keep], minlength=32))
    np.testing.assert_array_equal(sums, sums_of(X[keep], expected[keep], 32))


def test_cpu_threads_do_not_change_results():
    # over several parts, the last one partial
    X = np.random.default_rng(6).normal(size=(4 * shared._CPU_MIN_PART_ROWS + 123, 5)).astype(np.float32)
    data = torch.as_tensor(X)
    search = shared._nearest(shared._euclidean_scorer(data[:30].clone()), CPU)
    results = []
    threads = torch.get_num_threads()
    try:
        for n_threads in (1, 4):
            torch.set_num_threads(n_threads)
            labels, assigned = (torch.empty(len(X), dtype=torch.long) for _ in range(2))
            with shared._cpu_pool(CPU) as pool:
                sums, counts = shared._cpu_step(data, search, labels, 30, 1000, pool)
                shared._assign(data, search, assigned, 1000, pool=pool)
            assert torch.get_num_threads() == n_threads  # restored after the passes
            results.append((sums, counts, labels, assigned))
    finally:
        torch.set_num_threads(threads)
    for one, four in zip(*results, strict=True):
        assert torch.equal(one, four)
    assert torch.equal(results[0][2], results[0][3])


def test_cpu_map_restores_threads_on_an_error():
    def fail(part, out, chunk):
        raise ValueError("a part failed")

    data = torch.zeros(3 * shared._CPU_MIN_PART_ROWS, 1)
    threads = torch.get_num_threads()
    with shared._cpu_pool(CPU) as pool:
        results = shared._cpu_map(pool, fail, data, torch.empty(len(data)), 1000)
        with pytest.raises(ValueError, match="a part failed"):
            list(results)
    assert torch.get_num_threads() == threads


def test_cpu_map_keeps_at_most_two_parts_per_thread_ahead():
    started = []
    data = torch.zeros(40 * shared._CPU_MIN_PART_ROWS, 1)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with shared._cpu_pool(CPU) as pool:
            ahead = [
                len(started) - consumed
                for consumed, _ in enumerate(
                    shared._cpu_map(pool, lambda *_: started.append(1), data, torch.empty(len(data)), 1000)
                )
            ]
    finally:
        torch.set_num_threads(threads)
    assert len(ahead) == 40
    assert max(ahead) <= 2 * 2


def test_cpu_batch_size_by_codebook_vectors_and_features():
    # 2^20 values per chunk, from 256 rows (fewer within 2^22 values beyond 16,384 codebook vectors or features) to 8192
    shapes = ((100, 34), (100, 2000), (10_000, 34), (40_000, 34))
    assert tuple(shared._cpu_batch_size(k, d) for k, d in shapes) == (8192, 524, 256, 104)
