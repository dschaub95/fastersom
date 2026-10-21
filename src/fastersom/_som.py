"""Batch self-organising map (Kohonen's batch map) in PyTorch.

Each epoch assigns every sample x_i to its best matching unit c(x_i) and then sets every node to the
neighbourhood-weighted mean of the samples,

    w_k = sum_i h(c(x_i), k) x_i / sum_i h(c(x_i), k),    h(j, k) = exp(-||r_j - r_k||^2 / (2 sigma(t)^2)),

keeping w_k if the denominator is 0. r_j are the node coordinates on the rectangular grid and sigma decays
exponentially, sigma(t) = sigma_initial * (sigma_final / sigma_initial) ** (t / n_epochs).

Grouping samples by BMU gives the same update from per-node sums: with S_j the sum and n_j the count of samples whose
BMU is j, numerator = H @ S and denominator = H @ n, H the (nodes x nodes) neighbourhood matrix. This costs
O(nodes^2) per epoch instead of O(n_samples * nodes).

References:
    T. Kohonen, Self-Organizing Maps, 3rd ed., Springer, 2001.
    J. Vesanto et al., SOM Toolbox for Matlab 5, Helsinki University of Technology, 2000.
"""

from math import ceil, isfinite

import numpy as np
import torch

_METRICS = ("euclidean", "cosine")
_INITS = ("random", "sample")


def _is_int(value, minimum: int) -> bool:
    return isinstance(value, int | np.integer) and not isinstance(value, bool) and value >= minimum


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SOM:
    """Batch self-organising map on a rectangular grid with a gaussian neighbourhood.

    The defaults reproduce the maps of spatiomic's ``so.dimension.som``.

    Args:
        n_rows: Number of grid rows.
        n_cols: Number of grid columns.
        n_epochs: Number of epochs; each makes one weight update from all samples.
        sigma_initial: Initial gaussian neighbourhood width in grid units. Defaults to
            ceil(min(n_rows, n_cols) / 2) / 2.
        sigma_final: Width the exponential schedule decays towards (reached at t = n_epochs), > 0. Defaults to
            0.1 * sigma_initial.
        init: ``"random"`` draws weights uniformly from [-1, 1]^d with numpy's ``RandomState(random_state)`` and
            normalises them to unit length; ``"sample"`` uses randomly chosen samples (non-zero ones for cosine).
        metric: ``"euclidean"`` or ``"cosine"``. Under cosine, zero-norm samples have no BMU: they get label -1, are
            excluded from the update and the error measures, and have cosine distance 1 to every node.
        random_state: Seed for the initialisation.
        device: Torch device. Defaults to cuda, then mps, then cpu.
        batch_size: Samples processed at once in BMU search, aggregation, predict and transform. Affects memory
            (about batch_size x n_nodes floats) and speed only. Each epoch's update is the same up to float
            summation order, but over several epochs such rounding differences (as between devices) can flip
            near-tied BMUs, so maps agree in quality rather than bit for bit. Defaults to 65536 on GPU and 8192
            on CPU.

    Attributes:
        weights_: Node weights, shape (n_rows * n_cols, n_features), row-major (node = row * n_cols + col).
        labels_: BMU of every training sample under the final weights.
        n_features_in_: Number of features seen in fit.
    """

    def __init__(
        self,
        n_rows: int = 30,
        n_cols: int = 30,
        n_epochs: int = 10,
        sigma_initial: float | None = None,
        sigma_final: float | None = None,
        init: str = "random",
        metric: str = "euclidean",
        random_state: int | None = None,
        device: str | torch.device | None = None,
        batch_size: int | None = None,
    ):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_epochs = n_epochs
        self.sigma_initial = sigma_initial
        self.sigma_final = sigma_final
        self.init = init
        self.metric = metric
        self.random_state = random_state
        self.device = device
        self.batch_size = batch_size

    def _setup(self) -> None:
        if self.metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}, got {self.metric!r}")
        if self.init not in _INITS:
            raise ValueError(f"init must be one of {_INITS}, got {self.init!r}")
        for name in ("n_rows", "n_cols", "n_epochs"):
            if not _is_int(getattr(self, name), 1):
                raise ValueError(f"{name} must be an integer >= 1, got {getattr(self, name)!r}")
        if self.batch_size is not None and not _is_int(self.batch_size, 1):
            raise ValueError(f"batch_size must be None or an integer >= 1, got {self.batch_size!r}")
        self._device = torch.device(self.device) if self.device is not None else _default_device()
        self._batch_size = self.batch_size or (8192 if self._device.type == "cpu" else 1 << 16)
        self._sigma0 = ceil(min(self.n_rows, self.n_cols) / 2) / 2 if self.sigma_initial is None else self.sigma_initial
        self._sigma1 = 0.1 * self._sigma0 if self.sigma_final is None else self.sigma_final
        if not all(isfinite(sigma) and sigma > 0 for sigma in (self._sigma0, self._sigma1)):
            raise ValueError("sigma_initial and sigma_final must be finite and > 0")
        self._coords = torch.cartesian_prod(torch.arange(self.n_rows), torch.arange(self.n_cols))
        self._coords = self._coords.to(self._device, torch.float32)
        self._grid_sq_dist = torch.cdist(self._coords, self._coords) ** 2

    def _check_X(self, X) -> torch.Tensor:
        data = torch.as_tensor(
            X.detach() if isinstance(X, torch.Tensor) else np.ascontiguousarray(X),
            dtype=torch.float32,
            device=self._device,
        )
        if data.ndim < 2 or data.numel() == 0:
            raise ValueError(f"X must have shape (n_samples, n_features) or (..., n_features), got {tuple(data.shape)}")
        data = data.reshape(-1, data.shape[-1])
        if hasattr(self, "n_features_in_") and data.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {data.shape[1]} features, but SOM was fitted with {self.n_features_in_}")
        if not torch.isfinite(data).all():
            raise ValueError("X contains NaN or infinite values")
        return data

    def _check_fitted(self) -> None:
        if not hasattr(self, "weights_"):
            raise RuntimeError("This SOM instance is not fitted yet, call fit() first")

    def _init_weights(self, data: torch.Tensor) -> torch.Tensor:
        rng = np.random.RandomState(self.random_state)
        n_nodes = self.n_rows * self.n_cols
        if self.init == "random":
            weights = rng.rand(self.n_rows, self.n_cols, data.shape[1]) * 2 - 1
            weights /= np.linalg.norm(weights, axis=-1, keepdims=True)
            return torch.as_tensor(weights.reshape(n_nodes, -1), dtype=torch.float32, device=self._device)
        candidates = torch.arange(len(data), device=self._device)
        if self.metric == "cosine":
            candidates = candidates[data.norm(dim=1) > 0]
        if len(candidates) == 0:
            raise ValueError("init='sample' with metric='cosine' needs at least one non-zero sample")
        picks = rng.choice(len(candidates), n_nodes, replace=len(candidates) < n_nodes)
        return data[candidates[torch.as_tensor(picks, device=self._device)]].clone()

    def _scores(self, batch: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Per-node score to minimise: ||w||^2 - 2 x.w (euclidean, ||x||^2 dropped) or -x.w / ||w|| (cosine)."""
        if self.metric == "euclidean":
            return torch.addmm((weights**2).sum(1), batch, weights.T, alpha=-2)
        return -(batch @ (weights / weights.norm(dim=1, keepdim=True).clamp_min(1e-12)).T)

    def _bmus(self, data: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """BMU per sample; -1 for zero-norm samples under the cosine metric."""
        bmus = torch.cat([self._scores(batch, weights).argmin(1) for batch in data.split(self._batch_size)])
        if self.metric == "cosine":
            bmus[data.norm(dim=1) == 0] = -1
        return bmus

    def fit(self, X, y=None) -> "SOM":
        """Fits the map from a fresh initialisation.

        Args:
            X: Samples, shape (n_samples, n_features) or (..., n_features); numpy array, array-like or torch tensor.
                Any real values are fine; data is copied to the device once and stays there.
            y: Ignored.

        Returns:
            self
        """
        self._setup()
        for attr in ("weights_", "labels_", "n_features_in_"):
            self.__dict__.pop(attr, None)
        data = self._check_X(X)
        weights = self._init_weights(data)
        n_nodes = weights.shape[0]
        for t in range(self.n_epochs):
            sigma = self._sigma0 * (self._sigma1 / self._sigma0) ** (t / self.n_epochs)
            sums, counts = torch.zeros_like(weights), weights.new_zeros(n_nodes)
            for batch in data.split(self._batch_size):
                bmus = self._scores(batch, weights).argmin(1)
                if self.metric == "cosine":  # zero-norm samples have no BMU and do not contribute
                    keep = batch.norm(dim=1) > 0
                    batch, bmus = batch[keep], bmus[keep]
                sums.index_add_(0, bmus, batch)
                counts += torch.bincount(bmus, minlength=n_nodes)
            h = torch.exp(-self._grid_sq_dist / (2 * sigma**2))
            numerator, denominator = h @ sums, h @ counts
            weights = torch.where(denominator[:, None] > 0, numerator / denominator[:, None], weights)
        self._weights = weights
        self.weights_ = weights.cpu().numpy()
        self.n_features_in_ = data.shape[1]
        self.labels_ = self._bmus(data, weights).cpu().numpy()
        return self

    def predict(self, X) -> np.ndarray:
        """Returns the BMU index (int64) per sample; -1 for zero-norm samples under the cosine metric."""
        self._check_fitted()
        return self._bmus(self._check_X(X), self._weights).cpu().numpy()

    def fit_predict(self, X, y=None) -> np.ndarray:
        """Fits the map and returns labels_."""
        return self.fit(X).labels_

    def _distances(self, batch: torch.Tensor) -> torch.Tensor:
        if self.metric == "euclidean":  # exact, unlike the BMU search, so large offsets do not cancel
            return torch.cdist(batch, self._weights, compute_mode="donot_use_mm_for_euclid_dist")
        cos = -self._scores(batch, self._weights) / batch.norm(dim=1, keepdim=True)
        return torch.where(torch.isfinite(cos), (1 - cos).clamp(0, 2), 1.0)

    def transform(self, X) -> np.ndarray:
        """Returns the distance of every sample to every node, shape (n_samples, n_nodes).

        Euclidean or cosine distances (1 - cosine similarity) depending on the metric. The output is a dense
        n_samples x n_nodes float32 array; for large inputs call it on slices or use predict.
        """
        self._check_fitted()
        data = self._check_X(X)
        return torch.cat([self._distances(batch) for batch in data.split(self._batch_size)]).cpu().numpy()

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fits the map and returns transform(X)."""
        return self.fit(X).transform(X)

    def _top2(self, X) -> tuple[torch.Tensor, torch.Tensor]:
        """Distance to the BMU and indices of the two best nodes, for every sample that has a BMU."""
        self._check_fitted()
        data = self._check_X(X)
        if self.metric == "cosine":
            data = data[data.norm(dim=1) > 0]
        dists, nodes = [], []
        for batch in data.split(self._batch_size):
            best = self._scores(batch, self._weights).topk(min(2, len(self._weights)), dim=1, largest=False)
            if self.metric == "euclidean":
                dists.append((batch - self._weights[best.indices[:, 0]]).norm(dim=1))
            else:
                dists.append((1 + best.values[:, 0] / batch.norm(dim=1)).clamp(0, 2))
            nodes.append(best.indices)
        return torch.cat(dists), torch.cat(nodes)

    def quantization_error(self, X) -> float:
        """Mean distance (in the chosen metric) between each sample and its BMU."""
        dist, _ = self._top2(X)
        return float(dist.cpu().double().mean())  # on the CPU because MPS has no float64

    def topographic_error(self, X) -> float:
        """Fraction of samples whose first and second BMU are not adjacent on the grid (8-neighbourhood)."""
        _, nodes = self._top2(X)
        if nodes.shape[1] < 2:
            return 0.0
        offset = (self._coords[nodes[:, 0]] - self._coords[nodes[:, 1]]).abs().amax(1)
        return float((offset > 1).cpu().double().mean())


BatchSOM = SOM
