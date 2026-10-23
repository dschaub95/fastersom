# Changelog

## Unreleased

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
