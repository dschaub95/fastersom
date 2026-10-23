"""Fit time and peak GPU memory of fastersom, XPySom and torchsom at several map and batch sizes.

Needs XPySom, CuPy and torchsom besides fastersom, and the spatiomic tutorial's preprocessed subsample
(480,000 x 34 pixels) as a .npy file. Every configuration runs in its own process, so the memory peaks do not mix:

    python benchmarks/memory.py subsample_processed.npy

Peak memory includes the data on the GPU and the allocator cache (torch's reserved memory, CuPy's memory pool).
XPySom runs with the parameters spatiomic's ``so.dimension.som`` passes it, and torchsom with spatiomic's neighbourhood
schedule.
"""

import json
import subprocess
import sys
import time
from math import ceil

import numpy as np

EPOCHS = 10
SEEDS = [0, 1, 2]
# (library, device, batch size, map size); None is fastersom's default batch size, chosen from the free memory on GPUs
CONFIGS = [
    ("fastersom", "cuda", None, 30),
    ("fastersom", "cuda", 32768, 30),
    ("fastersom", "cuda", 8096, 30),
    ("xpysom", "cuda", 8096, 30),  # spatiomic's default
    ("xpysom", "cuda", 65536, 30),
    ("torchsom", "cuda", 1024, 30),  # its fastest; mini-batch updates, so the batch size also changes the map
    ("fastersom", "cpu", None, 30),
    ("xpysom", "cpu", 8096, 30),
    ("torchsom", "cpu", 1024, 30),
    ("fastersom", "cuda", None, 100),
    ("fastersom", "cuda", 8096, 100),
    ("xpysom", "cuda", 8096, 100),
    ("fastersom", "cpu", None, 100),
]


def fit(library: str, device: str, batch_size: int | None, size: int, X: np.ndarray, seed: int):
    """Fits one size x size map and returns its weights, shape (n_nodes, n_features)."""
    if library == "fastersom":
        from fastersom import SOM

        som = SOM(size, size, n_epochs=EPOCHS, random_state=seed, device=device, batch_size=batch_size)
        return som.fit(X).weights_
    if library == "xpysom":
        import cupy
        from xpysom import XPySom

        som = XPySom(
            size,
            size,
            X.shape[1],
            sigma=ceil(size / 2),
            sigmaN=0,
            learning_rate=0.2,
            learning_rateN=3e-3,
            neighborhood_function="gaussian",
            xp=cupy if device == "cuda" else np,
            n_parallel=batch_size,
            random_seed=seed,
        )
        som.train(X, EPOCHS)
        return som.get_weights().reshape(-1, X.shape[1])
    import torch
    from torchsom import SOM

    som = SOM(
        size, size, num_features=X.shape[1], epochs=EPOCHS, batch_size=batch_size, device=device, random_seed=seed
    )
    # XPySom's gaussian has std sigma / 2 and decays as 15 * 0.1 ** (t / epochs) here; torchsom's std is sigma
    som.sigma_schedule = torch.tensor([0.5 * ceil(size / 2) * 0.1 ** (t / EPOCHS) for t in range(EPOCHS)])
    som.fit(torch.from_numpy(X), verbose=False)
    return som.weights.detach().reshape(-1, X.shape[1])


def run(library: str, device: str, batch_size: int | None, size: int, path: str) -> dict:
    import cupy
    import torch

    from fastersom import SOM

    X = np.load(path).astype(np.float32)
    pool = cupy.get_default_memory_pool()
    times = []
    for seed in [SEEDS[0], *SEEDS]:  # the first run warms up
        pool.free_all_blocks()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        weights = fit(library, device, batch_size, size, X, seed)
        cupy.cuda.Device().synchronize()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    peak = pool.total_bytes() + torch.cuda.max_memory_reserved()  # CuPy's pool keeps every block it allocated
    error = SOM.from_weights(weights, size, size, device="cpu").quantization_error(X)
    return {
        "library": library,
        "device": device,
        "nodes": f"{size}x{size}",
        "batch_size": batch_size or "default",
        "fit_s": round(float(np.mean(times[1:])), 3),
        "peak_gpu_MiB": round(peak / 2**20) if device == "cuda" else None,
        "quantization_error": round(error, 3),
    }


if __name__ == "__main__":
    if len(sys.argv) == 6:  # one configuration, in a child process
        library, device, batch_size, size, path = sys.argv[1:]
        print(json.dumps(run(library, device, int(batch_size) or None, int(size), path)))
    else:
        for library, device, batch_size, size in CONFIGS:
            command = [sys.executable, __file__, library, device, str(batch_size or 0), str(size), sys.argv[1]]
            print(subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip().splitlines()[-1])
