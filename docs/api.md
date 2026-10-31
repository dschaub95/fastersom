# API

fastersom has one public class, {class}`~fastersom.SOM`, with a scikit-learn-like interface.
`fastersom.BatchSOM` is an alias of it.

```{eval-rst}
.. module:: fastersom
.. currentmodule:: fastersom

.. autosummary::
    :toctree: generated

    SOM
```

## Overview

Parameters
: `n_rows`, `n_cols` (grid size), `n_epochs`, `sigma_initial`, `sigma_final` (neighbourhood schedule), `init`,
  `metric`, `random_state`, `device`, `batch_size`, `memory_fraction`, `verbose`, `deterministic`, `fused`,
  `precision`, and keyword only `neighbourhood`, `grid_metric`, `sigma_schedule` and `schedule_end` (see
  [](guide/neighbourhoods.md)). See {class}`~fastersom.SOM` and the [user guide](guide/index.md).

Fitting
: {meth}`~fastersom.SOM.fit`, {meth}`~fastersom.SOM.fit_predict`, {meth}`~fastersom.SOM.fit_transform`, and
  {meth}`~fastersom.SOM.from_weights` for a fitted map from saved weights.

Using a fitted map
: {meth}`~fastersom.SOM.predict` (best matching unit, BMU, per sample, optionally with the distance to it),
  {meth}`~fastersom.SOM.transform` (distance of every sample to every node), {meth}`~fastersom.SOM.distance_map`
  (U-matrix).

Quality measures
: {meth}`~fastersom.SOM.quantization_error` and {meth}`~fastersom.SOM.topographic_error`.

Fitted attributes
: `weights_` (node weights, shape `(n_rows * n_cols, n_features)`, node = `row * n_cols + col`), `labels_` (BMU of
  every training sample) and `n_features_in_`.
