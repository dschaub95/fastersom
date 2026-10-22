# Changelog

## Unreleased

- On CUDA and MPS (torch >= 2.5), `batch_size` defaults to 2^28 / n_nodes (where speed levels off), reduced to fit
  `memory_fraction` (new, default 0.9 on CUDA and 0.5 on MPS) of the free GPU memory, instead of 65536: small maps are
  faster, large maps need less memory, and a nearly full GPU no longer fails with an out-of-memory error.
- The NaN/inf check of `X` no longer allocates about 1.7x the size of `X` on the device.
- `fit` holds one nodes x nodes neighbourhood matrix at a time instead of up to four.
- `transform` fills its result on the CPU batch by batch instead of holding it twice on the device.

## 0.2.1 (2026-10-22)

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
