"""Fit time and peak GPU memory of fastersom and spatiomic (XPySom on CuPy) at several batch sizes.

Needs spatiomic and CuPy besides fastersom, and the spatiomic tutorial's preprocessed subsample (480,000 x 34 pixels)
as a .npy file. Every configuration runs in its own process, so the memory peaks do not mix:

    python benchmarks/memory.py subsample_processed.npy

Peak memory includes the data on the GPU and the allocator cache (torch's reserved memory, CuPy's memory pool).
"""

import json
import subprocess
import sys
import time

import numpy as np

# (library, batch size); None is the library's default (spatiomic: 8096, fastersom: chosen from the free memory)
CONFIGS = [
    ("spatiomic", None),
    ("spatiomic", 65536),
    ("fastersom", None),
    ("fastersom", 32768),
    ("fastersom", 8096),
]
SEEDS = [0, 1, 2]


def run(library: str, batch_size: int | None, path: str) -> dict:
    X = np.load(path).astype(np.float32)
    times = []
    for seed in [SEEDS[0], *SEEDS]:  # the first run warms up
        if library == "spatiomic":
            import cupy
            import spatiomic as so

            cupy.get_default_memory_pool().free_all_blocks()
            kwargs = {} if batch_size is None else {"parallel_count": batch_size}
            som = so.dimension.som(node_count=(30, 30), dimension_count=X.shape[1], use_gpu=True, seed=seed, **kwargs)
            start = time.perf_counter()
            som.fit(X, iteration_count=10)
            cupy.cuda.Device().synchronize()
            times.append(time.perf_counter() - start)
            peak = cupy.get_default_memory_pool().total_bytes()  # the pool keeps every block it allocated
            nodes = np.asarray(som.get_nodes())
        else:
            import torch

            from fastersom import SOM

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            som = SOM(30, 30, n_epochs=10, random_state=seed, device="cuda", batch_size=batch_size)
            start = time.perf_counter()
            som.fit(X)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            peak = torch.cuda.max_memory_reserved()
            nodes = som.weights_
    from fastersom import SOM

    error = SOM.from_weights(nodes.reshape(-1, X.shape[1]), 30, 30, device="cpu").quantization_error(X)
    return {
        "library": library,
        "batch_size": batch_size or "default",
        "fit_s": round(float(np.mean(times[1:])), 3),
        "peak_gpu_MiB": round(peak / 2**20),
        "quantization_error": round(error, 3),
    }


if __name__ == "__main__":
    if len(sys.argv) == 4:  # one configuration, in a child process
        library, batch_size, path = sys.argv[1:]
        print(json.dumps(run(library, int(batch_size) or None, path)))
    else:
        for library, batch_size in CONFIGS:
            command = [sys.executable, __file__, library, str(batch_size or 0), sys.argv[1]]
            print(subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip().splitlines()[-1])
