# Changelog

## 0.1.1 (2026-10-21)

- First release: `SOM` (alias `BatchSOM`), a batch self-organising map in PyTorch for CUDA, MPS and CPU.
- scikit-learn-like interface: `fit`, `predict`, `fit_predict`, `transform`, `fit_transform`, `quantization_error`,
  `topographic_error`; fitted attributes `weights_`, `labels_`, `n_features_in_`.
- Euclidean and cosine metrics, `init="random"` or `"sample"`, device-dependent `batch_size`.
- Defaults reproduce spatiomic's `so.dimension.som` maps.
- Tested on Python 3.11-3.14; development tooling: ruff, pre-commit, uv-lock, `twine check --strict`.
