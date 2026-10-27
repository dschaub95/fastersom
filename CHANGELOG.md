# Changelog

## Unreleased

- `KMeans`: k-means clustering (Lloyd's algorithm) with the nearest-node search and summation of `SOM`, including the
  fused kernels on CUDA and MPS and `deterministic`. Greedy k-means++ initialisation as in scikit-learn (or random
  samples, or given centroids), scikit-learn's convergence criterion, empty clusters moved to the farthest samples as in
  scikit-learn (or kept, `empty_clusters="keep"`), and spherical k-means under cosine and correlation. From the same
  initial centroids, centroids, labels and the number of iterations match scikit-learn's `KMeans(algorithm="lloyd")` on
  the CPU. `compute_labels=False` skips the pass for `labels_` and `inertia_` when only the centroids are needed.
- `SOM(init="pca")`: linear initialisation as in the SOM Toolbox, a grid in the plane of the first two principal
  components, centred at the mean, with the longer side of the map along the first component and each side spanning one
  standard deviation of its component; under cosine and correlation of the samples as these metrics compare them.
- The summation of a fit is a method of its own (`SOM._sums`), which `KMeans` shares. On the CPU it adds the samples
  of up to 2^20 samples at once rather than batch by batch, as index_add_ spreads over the threads only for many rows,
  and the default batch size is 2^24 / n_nodes, from 8192 to 65536 (8192 before). On 200,000 samples x 34 features on
  an M3 Max, an epoch takes half as long at 10 x 10 nodes (25.5 to 12.9 ms), a quarter less at 22 x 22 and about as
  long at 71 x 71, where the matrix product dominates; a k-means iteration with 500 centroids a third less. Sums now add
  in another order, which changes results only by float rounding.

## 0.6.0 (2026-10-26)

- Python 3.12 or newer is required: support for Python 3.11 is dropped, as in the scverse template, whose
  documentation tools need 3.12.
- `sigma_final` defaults to `min(0.1 * sigma_initial, 1)` instead of `0.1 * sigma_initial`: the final neighbourhood is
  at most one grid unit wide. With the old default, which XPySom uses, a large map ended training with a neighbourhood
  of several units (2.5 at 100 x 100 nodes, 5 at 200 x 200) and was smoothed beyond the resolution of its nodes. On
  mass cytometry (Levine_32dim, 265,627 cells x 32 markers) and multiplexed imaging (2,000,000 pixels x 34 channels),
  the quantization error is 9-21% lower at 100 x 100 and 200 x 200 nodes, the topographic error stays at or below
  0.072 (narrower widths such as 0.5 lower the quantization error further but raise it to about 0.1, 0.25 to 0.24-0.35),
  and on 100 x 100 nodes cells of the gated populations below 1% of the cells land more often on nodes of their own
  (0.82 instead of 0.79). Maps up to 40 x 40 nodes (sigma_initial <= 10, e.g. FlowSOM's 10 x 10) are unchanged. Pass
  `sigma_final=0.1 * sigma_initial` for the old schedule.
- Documentation on Read the Docs (https://fastersom.readthedocs.io): a user guide, two tutorials (clustering mass
  cytometry like FlowSOM, and pixel clustering of multiplexed images) and the API reference, built with Sphinx in the
  layout of the scverse cookiecutter template.
- The repository now follows the [scverse cookiecutter template](https://github.com/scverse/cookiecutter-scverse) as a
  whole, linked with cruft (`.cruft.json`) so that template updates can be applied: hatchling builds the package, hatch
  runs the tests, type checks and documentation build, prek runs the checks (ruff with the template's rules, biome,
  pyproject-fmt, mypy, zizmor), and GitHub Actions test, check the build, autofix pull requests and release. The
  docstrings are in NumPy style. The package is unchanged apart from type annotations that mypy needed, and it now
  depends on session-info2, which the issue template uses for version reports.

## 0.5.0 (2026-10-24)

- On MPS (Apple GPUs, torch >= 2.6; tested with 2.8 and 2.14), the BMU search runs in fused Metal kernels, which keep
  only each sample's best node (or best two) in float32: on the simdgroup matrix units under euclidean, cosine and
  correlation at any feature count (the samples in registers up to 64 features while the nodes fit in 32 MB, else both
  operands streamed in chunks), and on the ALUs under manhattan up to 512 features. Few samples split the nodes over a
  second grid axis. The search alone is 1.2-10x faster than the torch path from 5 to 300 features and 10 x 10 to
  500 x 500 nodes, and 60-260x under manhattan. On an M3 Max (30-core GPU), with synthetic samples of the benchmark
  data's shape (480,000 x 34; the tutorial data was not used), `fit` takes 0.059 s instead of 0.32 s at 30 x 30 nodes
  and 0.52 s instead of 2.15 s at 100 x 100 under euclidean (cosine 3.9x at both), 0.10 s instead of about 34 s under
  manhattan at 30 x 30, and 0.066 s instead of 0.47 s with `deterministic=True`; at 300 x 300 nodes with 300,000
  samples 3.1 s instead of 23 s. The kernels compile from source at runtime with `torch.mps.compile_shader` (about
  0.2 s per feature count) and pass a similar exact self-check, or warn and leave the torch path; `fused=False` keeps
  the torch path.
- On MPS, manhattan distances on the torch path (`fused=False`) and the distances of `transform` add up one feature
  at a time instead of with `cdist`, which there holds batch x nodes x features floats several times over: the default
  batch could ask for over 36 GB (26,843 samples at 100 x 100 nodes and 34 features) and freeze the machine. They now
  take two batch x nodes matrices, at 1.5x the time of `cdist`.
- The new parameter `precision="float16"` lets `fit`'s BMU search round samples and nodes to 11 significant bits
  (float16's), summing the products in float32, under euclidean, cosine and correlation from 16 features with the Metal
  kernels on MPS: as half inputs, scaled by powers of 2 per feature into half's range on the GPU (samples and initial
  weights within 2^-30 to 2^30, else float32), with the bias added in float32; cosine's samples are centred on the mean
  node for it, which halves its flips. The kernels pass an exact self-check (of integers, scaled by 2^-3 to 2^3 per
  feature) or fall back to float32. On CUDA the search stays float32, as do `labels_`, `predict`, `transform` and the
  error measures. Near-tied samples flip BMUs, more on larger maps and depending on the data: on clustered 0-1 pixels of
  34 features about 1% at 10 x 10 nodes and 10-35% from 30 x 30 (60% under cosine without its centring), each within the
  rounding error, with the squared distance to the node found 0.6% above the best on average. In a CPU emulation of such
  pixels quantization and topographic error and Leiden clusters of the nodes stayed within the spread of refitting
  without 1% of the samples; on the synthetic 480,000 x 34 data at 100 x 100 nodes the quantization error is the same
  but the topographic error 0.044 instead of 0.032 (mean of 4 seeds), which a final epoch in float32 does not reduce. On
  an M3 Max, whose half products are no faster than float32 ones, `fit` is 1.00-1.05x faster; it is meant for M5 GPUs,
  where it is untested. BF16 is not offered: in the emulation it reorganised maps from 200 x 200 nodes.
- On MPS with the fused kernels, `deterministic=True` adds the samples per node as 64-bit fixed-point integers, like
  CUDA, by pairs of 32-bit atomic additions with the carry (Apple GPUs have no 64-bit atomic addition), instead of with
  a one-hot matrix product: on an M3 Max 1.4x slower than without at 10 x 10 nodes, 1.14x at 30 x 30 and no slower from
  100 x 100 (300 x 300: 3.0 s), instead of 3.4-4.3x. Zeros, such as clipped pixels, add nothing.
- On MPS, `fit` counts the samples per node in int32, as torch 2.8's `index_add_` there takes no int64 source and
  raised.

## 0.4.0 (2026-10-24)

- On CUDA with Triton >= 3.5 (in torch's Linux wheels from torch 2.9), the BMU search of `fit`, `predict`,
  `quantization_error` and `topographic_error` runs in fused kernels that keep only each sample's best node (or best
  two) instead of writing the batch x nodes score matrix to GPU memory: a float32 Gluon kernel under manhattan, and
  under the other metrics up to 128 features; and under euclidean, cosine and correlation on GPUs of compute capability
  8.0, 9.0 and 10.0 (A100, H100, B200), for maps of over 64 nodes and nodes x features >= 8192, a tensor-core kernel at
  any feature count, accurate to float32 by three TF32 products per product (3xTF32), summed in float32 per chunk of
  up to 32 features. Other cases keep the torch path. On an A100 with the 480,000 x 34 benchmark data, `fit` under
  euclidean is 2.3-3x faster than in 0.3.2 from 30 x 30 to 200 x 200 nodes (100 x 100: 0.17 s instead of 0.47 s;
  cosine and correlation 1.4-3.2x), 5.3x with `deterministic=True` and 240x under manhattan, which used `cdist` (0.43 s
  instead of 103 s); with 256 and 1024 features 1.3-1.7x. Peak GPU memory drops from 1114-2160 MiB to 234-556 MiB
  (34 features, 30 x 30 to 200 x 200 nodes), as the default batch on CUDA becomes 2^26 / n_features samples. The
  BMUs differ from the torch path's only on near-ties; in our tests the tensor-core kernel's were wrong against float64
  for fewer samples than cuBLAS's (1,764 instead of 5,105 of 200,000 at 128 features and 200 x 200 nodes). Only the A100
  has been tested; GPUs that report 9.0 with cut-down tensor cores, such as the H20, get the tensor-core kernel too,
  which is likely slower there.
- The first use of a feature count compiles its kernel (2-4 s, then cached on disk by Triton), which needs a C
  compiler and the Python headers, and checks it on a tiny search with exactly known scores. If compiling or the check
  fails, fastersom warns and falls back one step: from the tensor-core kernel to the float32 kernel, and from that to
  the torch path. ROCm, GPUs below compute capability 7.0 (which Triton does not support), MPS and the CPU use the
  torch path, and with the new parameter `fused=False` also CUDA, which then compiles nothing.
- `predict(return_distance=True)` and `quantization_error` compute each distance from the BMU's weights for every
  metric, as they did for euclidean, rather than from the search's scores, so that the distances do not depend on which
  kernel found the BMU (same values up to float32 rounding).
- On CUDA, `deterministic=True` adds the samples per node as 64-bit fixed-point integers, whose sum does not depend
  on the order, instead of with a one-hot matrix product, in two words per value, which keep values down to about
  n_samples^2 * 2^-124 of the feature's largest magnitude (smaller ones count as 0): on an A100 up to 2x slower than
  without on 10 x 10 nodes and 3-10% from 100 x 100, instead of 2x. MPS keeps the matrix product.
- Under cosine and correlation, `fit` adds the samples without a BMU as zeros instead of dropping them from each batch,
  which waited for the GPU in every batch: 1.4-1.6x faster with small batches on an A100 (30 x 30 and 100 x 100 nodes,
  `batch_size=8096`) and 1.15-1.25x on the CPU, with the same result.

## 0.3.2 (2026-10-23)

- `deterministic=True` makes `fit` give the same map in every run on CUDA and MPS, by summing the samples per node
  with a matrix product instead of atomic additions, whose order varies (1.2-1.8x slower on MPS). It is also used
  under `torch.use_deterministic_algorithms(True)`, which previously raised an error on MPS.
- `verbose=True` shows a tqdm progress bar over the epochs of `fit`, with an estimate of the remaining time. tqdm is a
  new dependency.
- On MPS, the euclidean BMU search adds the node norms within the matrix product, as a column of ones in the samples
  and a column of node norms in the weights, instead of in place after it. This extra pass over the scores made 0.3.1
  slower than `addmm` from 100 x 100 maps on, whose default batches are small (`fit` 7.4 s instead of 2.2 s for
  100 x 100, 16.3 s instead of 5.4 s for 150 x 150, 480,000 x 34 samples). `fit` and `predict` are now 1.6-3.3x faster
  than in 0.3.1 on every map size and as fast as `addmm` on large maps.

## 0.3.1 (2026-10-23)

- `fit` applies the gaussian neighbourhood along the grid rows and columns separately instead of building the
  nodes x nodes neighbourhood matrix, which took 400 MB for a 100 x 100 map and 6.4 GB for a 200 x 200 map. The update
  is the same up to float32 rounding and costs O(nodes * (n_rows + n_cols)) instead of O(nodes^2).
- `fit` counts the samples per node without `torch.bincount`, which waited for the GPU once per batch on CUDA.
- On MPS, the euclidean BMU search uses a matrix product plus an in-place add instead of `torch.addmm`, which on MPS
  expands the node norms to a second batch x nodes matrix: `fit` and `predict` are about 4x faster on small and
  medium maps (1.7x on 60 x 60) and the BMU search needs half the memory. CUDA and CPU keep `addmm`.

## 0.3.0 (2026-10-22)

- On CUDA and MPS (torch >= 2.5), `batch_size` defaults to 2^28 / n_nodes (where speed levels off), reduced to fit
  `memory_fraction` (new, default 0.9 on CUDA and 0.5 on MPS) of the free GPU memory, instead of 65536: small maps are
  faster, large maps need less memory, and a nearly full GPU no longer fails with an out-of-memory error.
- The NaN/inf check of `X` no longer allocates about 1.7x the size of `X` on the device.
- `fit` holds one nodes x nodes neighbourhood matrix at a time instead of up to four.
- `transform` fills its result on the CPU batch by batch instead of holding it twice on the device.
- Euclidean BMU search centres samples and nodes on the mean node, so data with a large offset relative to the node
  spacing (e.g. raw 16-bit intensities) no longer gets BMUs that are not the nearest node; this affects `fit`,
  `predict`, `quantization_error` and `topographic_error`. Small maps on the CPU are up to about a third slower; large
  maps and GPUs are unaffected.
- `X` and `init` also accept DLPack arrays such as CuPy, converted without leaving the GPU.

## 0.2.0 (2026-10-22)

- `init` also accepts an array of initial weights, shape (n_rows * n_cols, n_features) or (n_rows, n_cols,
  n_features), and `n_epochs=0` is allowed, so fitting keeps the given weights; this also enables warm starts
  (the sigma schedule restarts, so pass a small `sigma_initial` to continue training rather than retrain).
- `SOM.from_weights(weights, n_rows, n_cols, **kwargs)` returns a fitted map without training data (`labels_` is not
  set).
- `predict(X, return_distance=True)` also returns each sample's distance to its BMU, computed in the same pass; its
  mean over samples with a BMU is `quantization_error(X)`.
- `distance_map()` returns the U-matrix: each node's summed euclidean distance to its grid neighbours, normalised to a
  maximum of 1.
- `metric="manhattan"` (L1) and `metric="correlation"` (1 - Pearson correlation); constant samples have no BMU under
  correlation, like zero-norm samples under cosine.
- Pickled maps hold no device tensors and load on the default device if the configured one is unavailable.
- Type hints for `X` and `y`; `predict` is overloaded on `return_distance`.
- Cosine BMU search is about 5x faster on CPU.
- Euclidean and cosine maps are unchanged.

## 0.1.1 (2026-10-21)

- First release: `SOM` (alias `BatchSOM`), a batch self-organising map in PyTorch for CUDA, MPS and CPU.
- scikit-learn-like interface: `fit`, `predict`, `fit_predict`, `transform`, `fit_transform`, `quantization_error`,
  `topographic_error`; fitted attributes `weights_`, `labels_`, `n_features_in_`.
- Euclidean and cosine metrics, `init="random"` or `"sample"`, device-dependent `batch_size`.
- Defaults reproduce spatiomic's `so.dimension.som` maps.
- Tested on Python 3.11-3.14; development tooling: ruff, pre-commit, uv-lock, `twine check --strict`.
