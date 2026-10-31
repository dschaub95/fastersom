# Devices and fused kernels

## Choosing the device

`device` takes anything {class}`torch.device` accepts, such as `"cuda"`, `"cuda:1"`, `"mps"` or `"cpu"`. By default
(`device=None`) fastersom uses CUDA if available, else MPS (Apple GPUs), else the CPU.

```python
from fastersom import SOM

som = SOM(30, 30, device="cuda:1")
```

`fit` converts `X` to float32 and copies it to the device once; all epochs then run there, so the data,
`n_samples x n_features x 4` bytes, has to fit in the device's memory next to the working memory of the BMU search
(see [](batch-size.md)), unless it is streamed (see below). `predict`, `transform` and the error measures copy their
input in the same way. A fitted map pickles without device tensors: unpickled, it moves to the configured device if
that is available, else to the default one, so a map fitted on a GPU loads on a CPU-only machine.

## Data larger than the device's memory

With `stream=True`, `X` stays on the host and every epoch reads it batch by batch, so that only a few batches are on
the device at a time. The batch update needs no more than that: each epoch adds the batches into per-node sums and
counts, whose size does not depend on the number of samples.

```python
import numpy as np
from fastersom import SOM

X = np.load("X.npy", mmap_mode="r")  # on disk, read as needed
som = SOM(30, 30, stream=True).fit(X)
labels = som.predict(X)
```

- `X` can be anything of shape `(n_samples, n_features)` whose row slices `X[i:j]` are NumPy arrays or torch tensors:
  a NumPy array or memmap, an h5py or zarr dataset, or a CPU tensor. fastersom reads nothing but such slices, and needs
  neither h5py nor zarr itself. Inputs of shape `(..., n_features)` are flattened only as C-contiguous NumPy arrays or
  tensors, whose reshape is a view.
- Each batch is converted to float32 as it is read. On CUDA a thread reads it into one of two pinned 64 MB buffers
  (a larger batch in several parts) and a second CUDA stream copies it to the GPU, both while the previous batch is
  searched. On MPS each batch is copied as it is, and on the CPU it is used where it was read.
- `fit` reads `X` once before the epochs, to check that it is finite and to find its range per feature, which
  `deterministic=True` needs. `init="pca"` reads it twice more, for the mean and the covariance; `init="sample"` reads
  only the rows it picks, and under cosine and correlation notes in the first pass which samples have a BMU (a byte
  per sample on the host).
- With the same batches, a streamed fit gives the same map as one in memory, bit for bit wherever a fit is
  deterministic (on the CPU, and with `deterministic=True`). Pass an explicit `batch_size` to compare the two, as the
  default depends on the free memory.
- `fit` does not set `labels_`, which would hold an int64 per sample; `predict(X)` returns them, and `fit_predict`
  calls it. `predict`, `transform` and the error measures read `X` once, and raise for values that are not finite only
  at the end of that pass. `predict` copies each batch's BMUs and distances (12 bytes per sample) to the host as they
  are found, on CUDA through pinned memory without waiting for each copy, and the error measures sum over each batch,
  so that the device holds nothing per sample.

The streamed fit is bound by reading the data rather than by the search. On an A100, a 32 x 32 map on 20 million
samples x 50 features (4 GB) in float32, from a memmap in the page cache, took 0.33 s per epoch, about 12 GB/s, against
0.11-0.15 s with the data in GPU memory: the reading thread copies from the page cache at 9-15 GB/s, the copies to the
GPU run at 25 GB/s. Data that is not in the page cache is read at the speed of the disk.

## What runs where

Most of the time of `fit` goes into the best matching unit (BMU) search: the distance of every sample to every node,
and the nearest node per sample. The straightforward way, which fastersom calls the *torch path*, computes a
`batch x nodes` matrix of scores with a matrix product and then takes its minimum per row. On NVIDIA and Apple GPUs
fastersom instead uses *fused kernels*, which compute the scores in tiles and keep only each sample's best node (or
best two, for the topographic error), so that the scores never reach GPU memory. They are used by `fit`, `predict`,
`quantization_error` and `topographic_error`; `transform` returns every distance and so always takes the torch path.

| device | BMU search | requirements |
| --- | --- | --- |
| CUDA | fused Triton (Gluon) kernels | Triton >= 3.5, a C compiler and the Python headers; compute capability >= 7.0, not ROCm |
| MPS (Apple GPUs) | fused Metal kernels | torch >= 2.6 |
| CPU | torch path | - |

When a fused kernel is not available, or fails, the search falls back to the torch path, which gives the same map up to
float rounding (BMUs can differ on near-ties).

### CUDA

On CUDA with Triton >= 3.5, which torch's Linux wheels include from torch 2.9, the search runs in one of two fused
kernels:

- On GPUs of compute capability 8.0, 9.0 and 10.0 (A100, H100, B200), under the euclidean, cosine and correlation
  metrics, for maps of over 64 nodes and `n_nodes x n_features >= 8192`, a kernel on the tensor cores at any feature
  count. It is accurate to float32 by computing every product from three TF32 products (3xTF32).
- Otherwise a float32 kernel: under manhattan at any feature count, and under the other metrics up to 128 features,
  beyond which cuBLAS's matrix product is as fast and the torch path is used.

On an A100, this makes `fit` under euclidean 2.3-3x faster than the torch path from 30 x 30 to 200 x 200 nodes, with
a third to a sixth of the memory, and 240x faster under manhattan, whose torch path takes `torch.cdist`. Only the A100
has been tested; GPUs that report compute capability 9.0 with cut-down tensor cores, such as the H20, also get the
tensor-core kernel, which is likely slower there.

The first use of a feature count compiles its kernel, in 2-4 s; Triton caches it on disk for later processes.
Compiling needs a C compiler and the Python headers (for example the `python3-dev` package, or a Python installed by
uv). Each kernel is then checked on a tiny search with exactly known results. If compiling or the check fails,
fastersom warns with a {class}`RuntimeWarning` and falls back one step: from the tensor-core kernel to the float32
kernel (up to 128 features, else directly to the torch path), and from that to the torch path. ROCm, whose 64-thread
warps the kernels' tiling does not cover, and GPUs below compute capability 7.0, which Triton does not support, always
use the torch path.

### Apple GPUs (MPS)

On MPS with torch >= 2.6, the search runs in fused Metal kernels, in float32 like the torch path: on the simdgroup
matrix units under euclidean, cosine and correlation at any feature count, and on the ALUs under manhattan up to 512
features. They compile from source at runtime with `torch.mps.compile_shader`, in about 0.2 s per feature count and
without Xcode, and pass a similar exact self-check, or warn and leave the torch path.

On an M3 Max (30-core GPU) with 34 features, this makes `fit` 4-7x faster than the torch path from 30 x 30 to
300 x 300 nodes; the search alone is 1.2-10x faster from 5 to 300 features and 10 x 10 to 500 x 500 nodes, and 60-260x
under manhattan.

MPS shares its memory with the system and other applications. fastersom's default batch size uses at most half of the
memory torch recommends for MPS (see [](batch-size.md)); PyTorch's own limit can be lowered with the environment
variable `PYTORCH_MPS_HIGH_WATERMARK_RATIO`.

### CPU

The CPU always takes the torch path. It uses as many threads as torch does, which {func}`torch.set_num_threads`
changes. The CPU is deterministic (see [](reproducibility.md)).

## Checking and disabling the fused kernels

fastersom has no public function that reports which kernel ran; a kernel that cannot be used is either not attempted
(missing requirements) or reported by a {class}`RuntimeWarning` when it fails. To check the requirements:

```python
import torch

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(), "compute capability", torch.cuda.get_device_capability())
    print("ROCm" if torch.version.hip else "CUDA", torch.version.hip or torch.version.cuda)
    try:
        import triton

        print("Triton", triton.__version__)  # the fused kernels need >= 3.5
    except ImportError:
        print("no Triton: the torch path is used")
elif torch.backends.mps.is_available():
    print("Metal kernels available:", hasattr(torch.mps, "compile_shader"))  # torch >= 2.6
else:
    print("CPU: the torch path is used")
```

To make a failing kernel raise instead of falling back silently after its warning, turn the warning into an error:

```python
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("error", RuntimeWarning)
    som.fit(X)
```

The kernels are checked once per process, so this only catches a failure on their first use.

`fused=False` turns the fused kernels off: the search then takes the torch path on every device and nothing is
compiled. This is useful to compare speed or results, or to rule out the kernels when debugging:

```python
import time

for fused in (True, False):
    som = SOM(100, 100, fused=fused, random_state=0)
    som.fit(X[:10_000])  # the first fit compiles and checks the kernels
    start = time.perf_counter()
    som.fit(X)
    print(f"fused={fused}: {time.perf_counter() - start:.2f} s, QE {som.quantization_error(X):.4f}")
```

On GPUs, `fit` returns when its work is done, because it copies `labels_` to the host at the end, so wall-clock times
of `fit` are meaningful without an explicit synchronisation.
