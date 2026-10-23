"""Fit time and peak GPU memory of fastersom, XPySom and torchsom at several map and batch sizes.

Runs on the spatiomic tutorial's preprocessed subsample (480,000 x 34 pixels), which the first run downloads and
prepares (see ``data.py``), and on 5,000,000 x 60 synthetic samples for a large 200 x 200 map. Every configuration runs
in its own process limited to ``--cpus`` cores (default 16), so the memory peaks do not mix, with XPySom, CuPy and
torchsom installed into temporary environments on top of this project's:

    uv run benchmarks/memory.py                                   # every configuration
    uv run benchmarks/memory.py --library fastersom --device cpu  # a part of them

Set ``CUPY`` to the CuPy wheel matching your CUDA version (default cupy-cuda12x==14.2.0). Peak memory includes the
data on the GPU and the allocator cache (torch's reserved memory, CuPy's memory pool). XPySom runs with the parameters
spatiomic's ``so.dimension.som`` passes it, and torchsom with spatiomic's neighbourhood schedule.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from math import ceil
from pathlib import Path

import numpy as np
from data import SPATIOMIC  # benchmarks/data.py

HERE = Path(__file__).parent
DATA = HERE / "data" / "subsample_processed.npy"
EPOCHS = 10
SEEDS = [0, 1, 2]
# (library, device, batch size, map size); None is fastersom's default batch size, chosen from the free memory on GPUs.
# 200 x 200 maps run on the synthetic data, the others on the tutorial's.
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
    ("fastersom", "cuda", None, 200),
    ("fastersom", "cuda", 8096, 200),
    ("xpysom", "cuda", 8096, 200),
    ("fastersom", "cpu", None, 200),  # one fit without a warm-up, as it takes about 45 minutes
]
# packages each library needs besides this project's environment; XPySom on the GPU also needs CuPy
PACKAGES = {"fastersom": [], "xpysom": ["xpysom==1.0.7"], "torchsom": ["torchsom==1.3.1"]}
# uv.lock does not cover these packages, so their dependencies resolve as of this date
EXCLUDE_NEWER = "2026-10-23T00:00:00Z"


def fit(library: str, device: str, batch_size: int | None, size: int, X: np.ndarray, seed: int):
    """Fits one size x size map and returns its weights, shape (n_nodes, n_features)."""
    if library == "fastersom":
        from fastersom import SOM

        som = SOM(size, size, n_epochs=EPOCHS, random_state=seed, device=device, batch_size=batch_size)
        return som.fit(X).weights_
    if library == "xpysom":
        from xpysom import XPySom

        if device == "cuda":
            import cupy as xp
        else:
            xp = np
        som = XPySom(
            size,
            size,
            X.shape[1],
            sigma=ceil(size / 2),
            sigmaN=0,
            learning_rate=0.2,
            learning_rateN=3e-3,
            neighborhood_function="gaussian",
            xp=xp,
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


def synthetic(n_samples: int = 5_000_000, n_features: int = 60, n_clusters: int = 50) -> np.ndarray:
    """Gaussian clusters in [0, 1]^n_features, like normalised pixel intensities."""
    rng = np.random.default_rng(0)
    centres = rng.random((n_clusters, n_features), dtype=np.float32)
    X = rng.standard_normal((n_samples, n_features), dtype=np.float32)
    X *= 0.05
    X += centres[rng.integers(n_clusters, size=n_samples)]
    return np.clip(X, 0, 1, out=X)


def run(library: str, device: str, batch_size: int | None, size: int, path: str) -> dict:
    import torch

    from fastersom import SOM

    X = synthetic() if size == 200 else np.load(path).astype(np.float32)
    cuda, pool = device == "cuda", None
    if cuda and library == "xpysom":
        import cupy

        pool = cupy.get_default_memory_pool()
    times = []
    # the first of several runs warms up; a single fit uses the last seed, whose map the other rows report
    seeds = SEEDS[-1:] if size == 200 and not cuda else [SEEDS[0], *SEEDS]
    for seed in seeds:
        if pool:
            pool.free_all_blocks()
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        weights = fit(library, device, batch_size, size, X, seed)
        if pool:
            cupy.cuda.Device().synchronize()
        if cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    peak = None
    if cuda:  # CuPy's pool keeps every block it allocated
        peak = round(((pool.total_bytes() if pool else 0) + torch.cuda.max_memory_reserved()) / 2**20)
    error = SOM.from_weights(weights, size, size, device=device).quantization_error(X)
    return {
        "library": library,
        "device": device,
        "hardware": torch.cuda.get_device_name() if cuda else f"{torch.get_num_threads()} CPU threads",
        "nodes": f"{size}x{size}",
        "batch_size": batch_size or "default",
        "fit_s": round(float(np.mean(times[1:] or times)), 3),
        "fits": len(times[1:] or times),
        "peak_gpu_MiB": peak,
        "quantization_error": round(error, 3),
    }


def uv_run(packages: list[str], script: Path, *args: str) -> subprocess.CompletedProcess:
    """Runs the script in a temporary environment of this project's plus the given packages."""
    command = [os.environ.get("UV", "uv"), "run", "--frozen", f"--exclude-newer={EXCLUDE_NEWER}"]
    command += ["--project", str(HERE.parent), *(f"--with={p}" for p in packages)]
    return subprocess.run([*command, "python", str(script), *args], capture_output=True, text=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=DATA, help="preprocessed subsample, prepared if missing")
    parser.add_argument("--library", nargs="+", choices=list(PACKAGES), default=list(PACKAGES))
    parser.add_argument("--device", nargs="+", choices=["cuda", "cpu"], default=["cuda", "cpu"])
    parser.add_argument("--size", nargs="+", type=int, choices=[30, 100, 200], default=[30, 100, 200])
    parser.add_argument("--cpus", type=int, default=16, help="CPU cores and threads per configuration")
    args = parser.parse_args()
    configs = [c for c in CONFIGS if c[0] in args.library and c[1] in args.device and c[3] in args.size]

    # the child processes inherit the CPU affinity (Linux only) and the thread counts
    cpus = min(args.cpus, os.cpu_count())
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[: args.cpus])
        cpus = len(os.sched_getaffinity(0))
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ[name] = str(cpus)

    if not args.data.exists() and any(size != 200 for *_, size in configs):
        print(f"preparing {args.data}", file=sys.stderr)
        args.data.parent.mkdir(parents=True, exist_ok=True)
        result = uv_run([SPATIOMIC], HERE / "data.py", str(args.data))
        if result.returncode:
            sys.exit(result.stderr)

    cupy = os.environ.get("CUPY", "cupy-cuda12x==14.2.0")
    failed = 0
    for library, device, batch_size, size in configs:
        packages = PACKAGES[library] + ([cupy] if library == "xpysom" and device == "cuda" else [])
        config = [library, device, str(batch_size or 0), str(size)]
        result = uv_run(packages, Path(__file__), "--run", *config, str(args.data))
        if result.returncode:
            failed += 1
            print(f"{' '.join(config)} failed:\n{result.stderr}", file=sys.stderr)
        else:
            print(result.stdout.strip().splitlines()[-1], flush=True)
    sys.exit(f"{failed} configurations failed" if failed else 0)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--run"]:  # one configuration, in a child process
        library, device, batch_size, size, path = sys.argv[2:]
        print(json.dumps(run(library, device, int(batch_size) or None, int(size), path)))
    else:
        main()
