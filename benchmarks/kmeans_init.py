"""Times KMeans' k-means++ initialisation alone, against scikit-learn's kmeans_plusplus on the CPU.

Run: .venv/bin/python benchmarks/kmeans_init.py [--device cuda|cpu] [--out results/kmeans_init.jsonl]

The data are Gaussian blobs as in kmeans_cuml.py/kmeans_cpu.py. Each case gets one warmup, then up to three timed
repeats, stopping early once 30 s have been spent on it. Times cover _initial_centroids only (data already on the
device), with the device synchronised before and after.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from fastersom import KMeans

SHAPES = {
    "cuda": [(n, d, k) for n in (100_000, 1_000_000) for d in (16, 64, 256) for k in (10, 100, 1000)]
    + [(10_000_000, 64, 1000), (10_000_000, 256, 1000)],
    "cpu": [(100_000, d, k) for d in (16, 64, 256) for k in (10, 100, 1000)],
}


def blobs(n: int, d: int, k: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    centres = rng.standard_normal((k, d), dtype=np.float32) * (4 / np.sqrt(2 * d))
    data = rng.standard_normal((n, d), dtype=np.float32)
    data += centres[rng.integers(k, size=n)]
    return data


def timed(fn, device: str, budget: float = 30.0) -> list[float]:
    def once() -> float:
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - start

    spent = once()  # warmup
    times = []
    while len(times) < 3 and spent < budget:
        times.append(once())
        spent += times[-1]
    return times or [spent]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "kmeans_init.jsonl")
    args = parser.parse_args()
    torch.set_num_threads(8)
    threadpool_limits(8)  # scikit-learn's OpenMP and BLAS
    torch.backends.cuda.matmul.allow_tf32 = False
    args.out.parent.mkdir(exist_ok=True)
    for n, d, k in SHAPES[args.device]:
        X = blobs(n, d, k)
        km = KMeans(k, random_state=0, device=args.device)
        engine = km._engine()
        data = engine._check_X(X)
        record = {"device": args.device, "n": n, "d": d, "k": k}
        record["fastersom_s"] = timed(
            lambda km=km, engine=engine, data=data: km._initial_centroids(engine, data), args.device
        )
        if args.device == "cpu":
            from sklearn.cluster import kmeans_plusplus

            record["sklearn_s"] = timed(lambda X=X, k=k: kmeans_plusplus(X, k, random_state=0), "cpu")
        del data, engine
        if args.device == "cuda":
            torch.cuda.empty_cache()
        line = json.dumps(record)
        print(line, flush=True)
        with args.out.open("a") as file:
            file.write(line + "\n")


if __name__ == "__main__":
    main()
