# Batch size and memory

## What `batch_size` changes

`batch_size` is the number of samples processed at once in the BMU search, the per-node sums of `fit`, `predict`,
`transform` and the error measures. It bounds memory and affects speed, but not the algorithm: every epoch still makes
one update from all samples, the same up to the order of float summation. Over several epochs such rounding
differences can flip near-tied BMUs, so maps fitted with different batch sizes (like maps fitted on different devices)
agree in quality rather than bit for bit.

The working memory of the BMU search is about

- `batch_size x n_nodes` floats on the torch path (the score matrix, up to about two copies; on the CPU per thread,
  as each of torch's threads searches chunks of its own), and
- `batch_size x n_features` floats with the fused CUDA or Metal kernels (a few copies of the batch, no scores).

On top of it come the data itself, `n_samples x n_features` float32 values on the device, and the nodes, which are small.

## The default

With `batch_size=None` (the default):

- On the CPU, it is `2**20 / max(n_nodes, n_features)` samples, from 256 to 8192: the chunk that one thread of the
  pool searches at a time (see [](devices.md)), whose score matrix and copies then stay within about 4 MB, which a
  thread's caches serve. Beyond 16,384 nodes or features it is `2**22 / max(n_nodes, n_features)` samples (104 at
  200 x 200 nodes), within 16 MB: glibc gives blocks beyond 32 MB fresh pages on every allocation, and an epoch then
  takes up to 2-4x as long on an x86 CPU (an AMD EPYC 7742).
- On CUDA and MPS, it is the largest batch that still helps speed: `2**28 / n_nodes` samples on the torch path and
  `2**26 / n_features` with the fused kernels. It is reduced if that would not fit in `memory_fraction` of the free GPU
  memory. `memory_fraction` defaults to 0.9 on CUDA and 0.5 on MPS, where the free memory is the memory torch
  recommends for MPS ({func}`torch.mps.recommended_max_memory`) minus what torch has allocated, as the system and
  other applications share it. Before torch 2.5 the MPS default is 65536.

So the default rarely needs tuning. Two examples for a torch-path BMU search (for instance with `fused=False`) on a
GPU with enough free memory:

| map | nodes | default batch | score matrix (float32) |
| --- | --- | --- | --- |
| 30 x 30 | 900 | 298,261 | about 1.0 GiB |
| 100 x 100 | 10,000 | 26,843 | about 1.0 GiB |

With the fused kernels and 34 features, the default batch is 1,973,790 samples, or the whole data if it is smaller,
which takes about 0.25 GiB per copy.

## When to set it

- **Shared GPUs or other work on the GPU:** lower `memory_fraction`, for example `SOM(memory_fraction=0.3)`, or set an
  explicit `batch_size`.
- **Apple silicon:** MPS uses the same memory as the rest of the system. The default of `memory_fraction=0.5` leaves
  room for other applications; PyTorch's overall limit for MPS can be lowered further with the environment variable
  `PYTORCH_MPS_HIGH_WATERMARK_RATIO`, set before torch is imported.
- **Very large maps on the torch path:** the score matrix grows with the number of nodes, so a 200 x 200 map holds
  40,000 floats per sample; the default then picks small batches (6,710 samples at 200 x 200 nodes).
- **The CPU:** every thread holds the scores of its own chunk, so an explicit `batch_size` takes about
  `threads x batch_size x n_nodes` floats (a chunk is at most one part, `n_samples / 96` samples and at least 8192).
  The default keeps them within 4 MB per thread up to 16,384 nodes or features and within 16 MB beyond.

## Other memory to keep in mind

- `transform` returns a dense `n_samples x n_nodes` float32 array on the CPU: 1,000,000 samples on a 30 x 30 map take
  3.6 GB. Call it on slices of the data, or use `predict(X, return_distance=True)`, which returns only the BMU and the
  distance to it.
- `fit` with `deterministic=True` on CUDA converts the batch to 64-bit fixed point in parts of `2**21` values, which
  takes a few times the memory of that part, not of the batch.
- The data is copied to the device as float32. Passing a float32 torch tensor that is already on the device avoids the
  copy.
