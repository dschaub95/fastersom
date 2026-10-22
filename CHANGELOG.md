# Changelog

## Unreleased

- `init` also accepts an array of initial weights, shape (n_rows * n_cols, n_features) or (n_rows, n_cols,
  n_features), and `n_epochs=0` is allowed, so fitting keeps the given weights; this also enables warm starts.
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
- Euclidean and cosine maps are unchanged.

## 0.1.1 (2026-10-21)

- First release: `SOM` (alias `BatchSOM`), a batch self-organising map in PyTorch for CUDA, MPS and CPU.
- scikit-learn-like interface: `fit`, `predict`, `fit_predict`, `transform`, `fit_transform`, `quantization_error`,
  `topographic_error`; fitted attributes `weights_`, `labels_`, `n_features_in_`.
- Euclidean and cosine metrics, `init="random"` or `"sample"`, device-dependent `batch_size`.
- Defaults reproduce spatiomic's `so.dimension.som` maps.
- Tested on Python 3.11-3.14; development tooling: ruff, pre-commit, uv-lock, `twine check --strict`.
