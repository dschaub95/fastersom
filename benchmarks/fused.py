"""Fit time and peak GPU memory of fastersom on CUDA over map sizes, feature counts, metrics and deterministic.

Measures the fastersom that is importable, so running it once on an earlier version (e.g. with that version's source
on PYTHONPATH) and once on this one compares the two on the same settings. Every configuration runs in its own
process, so the memory peaks do not mix:

    python benchmarks/fused.py subsample_processed.npy [batch_size] > after.jsonl

34 features are the spatiomic tutorial's preprocessed subsample (480,000 x 34 pixels, as in memory.py); 256 and 1024
features are gaussian blobs, 200,000 and 100,000 samples. batch_size is "default" (fastersom's choice), "torch" (the
torch path's default on an empty A100, min(n_samples, 2^28 / n_nodes), to compare versions at the same batch size) or
a number. The fit time is the median of 3 fits after a warm-up fit (which compiles the kernels); peak memory is
torch's max_memory_reserved, which includes the data and the allocator cache.
"""

import json
import subprocess
import sys
import time

import numpy as np

EPOCHS = 10
# (map size, n_features, metric, deterministic)
CONFIGS = [
    *(
        (size, 34, metric, deterministic)
        for size in (10, 30, 100, 200)
        for metric in ("euclidean", "manhattan", "cosine", "correlation")
        for deterministic in (False, True)
    ),
    *(
        (size, d, "euclidean", deterministic)
        for d in (256, 1024)
        for size in (30, 100)
        for deterministic in (False, True)
    ),
    *((size, d, "manhattan", False) for d in (256, 1024) for size in (30, 100)),
]
SLOW_S = 60  # a fit slower than this is measured once, without warm-up


def load(path: str, n_features: int):
    import torch

    if n_features == 34:
        return torch.as_tensor(np.load(path), dtype=torch.float32, device="cuda")
    n_samples = {256: 200_000, 1024: 100_000}[n_features]
    generator = torch.Generator(device="cuda").manual_seed(0)
    centres = torch.randn(50, n_features, device="cuda", generator=generator) * 2
    picks = torch.randint(0, 50, (n_samples,), device="cuda", generator=generator)
    return centres[picks] + 0.5 * torch.randn(n_samples, n_features, device="cuda", generator=generator)


def run(size: int, n_features: int, metric: str, deterministic: bool, batch_size: str, path: str) -> dict:
    import torch

    import fastersom
    from fastersom import SOM

    X = load(path, n_features)
    if batch_size == "torch":
        batch = min(len(X), (1 << 28) // size**2)
    else:
        batch = None if batch_size == "default" else int(batch_size)
    kwargs = {"n_epochs": EPOCHS, "metric": metric, "random_state": 0, "device": "cuda", "batch_size": batch}
    if deterministic:
        kwargs["deterministic"] = True
    times = []
    while len(times) < 4 and not (times and times[0] > SLOW_S):  # the first fit warms up
        # the previous map's tensors can keep a cached block of the allocator in use, so the next fit would need a new
        # one: on the torch path at 100 x 100 nodes, 2138 instead of 1114 MiB
        som = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        som = SOM(size, size, **kwargs).fit(X)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    if batch is None:  # the default batch size fit chose (its signature differs between versions)
        kernel = (som._kernel(n_features),) if hasattr(som, "_kernel") else ()
        batch = som._batch_size_for(X, *kernel)
    return {
        "version": fastersom.__file__,
        "nodes": f"{size}x{size}",
        "n_features": n_features,
        "metric": metric,
        "deterministic": deterministic,
        "batch_size": batch,
        "fit_s": round(float(np.median(times[1:] or times)), 3),
        "peak_MiB": round(torch.cuda.max_memory_reserved() / 2**20),
        "quantization_error": round(som.quantization_error(X), 4),
    }


if __name__ == "__main__":
    if len(sys.argv) == 7:  # one configuration, in a child process
        size, n_features, metric, deterministic, batch_size, path = sys.argv[1:]
        print(json.dumps(run(int(size), int(n_features), metric, deterministic == "1", batch_size, path)))
    else:
        path, batch_size = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "default"
        for size, n_features, metric, deterministic in CONFIGS:
            args = [str(size), str(n_features), metric, str(int(deterministic)), batch_size, path]
            result = subprocess.run([sys.executable, __file__, *args], capture_output=True, text=True)
            print(
                result.stdout.strip().splitlines()[-1]
                if result.returncode == 0
                else json.dumps({"config": args[:5], "error": result.stderr.strip().splitlines()[-1]}),
                flush=True,
            )
