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
from typing import Literal, overload

import numpy as np
import torch
from numpy.typing import ArrayLike

_METRICS = ("euclidean", "manhattan", "cosine", "correlation")
_INITS = ("random", "sample")


def _is_int(value, minimum: int) -> bool:
    return isinstance(value, int | np.integer) and not isinstance(value, bool) and value >= minimum


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _is_available(device: torch.device) -> bool:
    if device.type == "cuda":
        return torch.cuda.is_available() and (device.index or 0) < torch.cuda.device_count()
    if device.type == "mps":
        return torch.backends.mps.is_available()
    return device.type == "cpu"


class SOM:
    """Batch self-organising map on a rectangular grid with a gaussian neighbourhood.

    The defaults reproduce the maps of spatiomic's ``so.dimension.som``.

    Args:
        n_rows: Number of grid rows.
        n_cols: Number of grid columns.
        n_epochs: Number of epochs; each makes one weight update from all samples. With 0, fit keeps the initial
            weights.
        sigma_initial: Initial gaussian neighbourhood width in grid units. Defaults to
            ceil(min(n_rows, n_cols) / 2) / 2.
        sigma_final: Width the exponential schedule decays towards (reached at t = n_epochs), > 0. Defaults to
            0.1 * sigma_initial.
        init: ``"random"`` draws weights uniformly from [-1, 1]^d with numpy's ``RandomState(random_state)`` and
            normalises them to unit length; ``"sample"`` uses randomly chosen samples (ones with a BMU); an array of
            shape (n_rows * n_cols, n_features) or (n_rows, n_cols, n_features) gives the initial weights, e.g. another
            map's ``weights_`` for a warm start. The sigma schedule restarts at sigma_initial, which with the default
            largely retrains the map; to continue training, pass a small sigma_initial such as the previous
            sigma_final.
        metric: ``"euclidean"``, ``"manhattan"`` (L1), ``"cosine"`` (1 - cosine similarity) or ``"correlation"``
            (1 - Pearson correlation, the cosine distance of vectors centred over the features). Every metric updates
            the nodes to the neighbourhood-weighted mean of the raw samples (not the median for manhattan, nor the
            centred samples for correlation). Zero-norm samples under cosine and constant samples under correlation
            have no BMU: they get label -1, are excluded from the update and the error measures, and have distance 1
            to every node in transform.
        random_state: Seed for the initialisation.
        device: Torch device. Defaults to cuda, then mps, then cpu.
        batch_size: Samples processed at once in BMU search, aggregation, predict and transform. Affects memory
            (about batch_size x n_nodes floats) and speed only. Each epoch's update is the same up to float
            summation order, but over several epochs such rounding differences (as between devices) can flip
            near-tied BMUs, so maps agree in quality rather than bit for bit. Defaults to 65536 on GPU and 8192
            on CPU.

    Attributes:
        weights_: Node weights, shape (n_rows * n_cols, n_features), row-major (node = row * n_cols + col).
        labels_: BMU of every training sample under the final weights. Not set by from_weights.
        n_features_in_: Number of features seen in fit.
    """

    def __init__(
        self,
        n_rows: int = 30,
        n_cols: int = 30,
        n_epochs: int = 10,
        sigma_initial: float | None = None,
        sigma_final: float | None = None,
        init: str | ArrayLike | torch.Tensor = "random",
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

    @classmethod
    def from_weights(cls, weights: ArrayLike | torch.Tensor, n_rows: int, n_cols: int, **kwargs) -> "SOM":
        """Returns a fitted map with the given weights (shapes as for init), like ``SOM(n_rows, n_cols, n_epochs=0,
        init=weights, **kwargs).fit(X)`` but without data, so ``labels_`` is not set."""
        som = cls(n_rows, n_cols, init=weights, **{"n_epochs": 0, **kwargs})
        som._setup()
        som._set_weights(som._given_weights())
        return som

    def __getstate__(self) -> dict:
        """Keeps the parameters and numpy fitted attributes only, so the pickle does not depend on the device."""
        state = {key: value for key, value in self.__dict__.items() if not key.startswith("_")}
        if isinstance(self.init, torch.Tensor):
            state["init"] = self.init.cpu()
        return state

    def __setstate__(self, state: dict) -> None:
        """Restores a pickled map on the configured device if available (cpu, cuda, mps), else the default one."""
        self.__dict__.update(state)
        if hasattr(self, "weights_"):
            self._setup(fall_back=True)
            self._set_weights(torch.as_tensor(self.weights_, device=self._device))

    def _setup(self, fall_back: bool = False) -> None:
        if self.metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}, got {self.metric!r}")
        if isinstance(self.init, str) and self.init not in _INITS:
            raise ValueError(f"init must be one of {_INITS} or an array of weights, got {self.init!r}")
        for name, minimum in (("n_rows", 1), ("n_cols", 1), ("n_epochs", 0)):
            if not _is_int(getattr(self, name), minimum):
                raise ValueError(f"{name} must be an integer >= {minimum}, got {getattr(self, name)!r}")
        if self.batch_size is not None and not _is_int(self.batch_size, 1):
            raise ValueError(f"batch_size must be None or an integer >= 1, got {self.batch_size!r}")
        self._device = torch.device(self.device) if self.device is not None else _default_device()
        if fall_back and not _is_available(self._device):
            self._device = _default_device()
        self._batch_size = self.batch_size or (8192 if self._device.type == "cpu" else 1 << 16)
        self._sigma0 = ceil(min(self.n_rows, self.n_cols) / 2) / 2 if self.sigma_initial is None else self.sigma_initial
        self._sigma1 = 0.1 * self._sigma0 if self.sigma_final is None else self.sigma_final
        if not all(isfinite(sigma) and sigma > 0 for sigma in (self._sigma0, self._sigma1)):
            raise ValueError("sigma_initial and sigma_final must be finite and > 0")
        self._coords = torch.cartesian_prod(torch.arange(self.n_rows), torch.arange(self.n_cols))
        self._coords = self._coords.to(self._device, torch.float32)

    def _as_tensor(self, X: ArrayLike | torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            X = X.detach()
        elif hasattr(X, "__dlpack__") and not isinstance(X, np.ndarray):  # e.g. CuPy, without leaving the GPU
            try:
                X = torch.from_dlpack(X)
            except RuntimeError:  # torch cannot import negative strides
                X = torch.from_dlpack(X.copy())
        else:  # numpy refuses to export read-only arrays through DLPack
            X = np.ascontiguousarray(X)
        return torch.as_tensor(X, dtype=torch.float32, device=self._device)

    def _check_X(self, X: ArrayLike | torch.Tensor) -> torch.Tensor:
        data = self._as_tensor(X)
        if data.ndim < 2 or data.numel() == 0:
            raise ValueError(f"X must have shape (n_samples, n_features) or (..., n_features), got {tuple(data.shape)}")
        data = data.reshape(-1, data.shape[-1])
        if hasattr(self, "n_features_in_") and data.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {data.shape[1]} features, but SOM was fitted with {self.n_features_in_}")
        if not torch.isfinite(torch.stack(torch.aminmax(data))).all():  # unlike isfinite(data), without a copy of X
            raise ValueError("X contains NaN or infinite values")
        return data

    def _check_fitted(self) -> None:
        if not hasattr(self, "weights_"):
            raise RuntimeError("This SOM instance is not fitted yet, call fit() first")

    def _set_weights(self, weights: torch.Tensor) -> None:
        self._weights = weights
        self.weights_ = weights.cpu().numpy()
        self.n_features_in_ = weights.shape[1]

    def _given_weights(self) -> torch.Tensor:
        """The weights given as init, as a (n_nodes, n_features) copy on the device."""
        weights = self._as_tensor(self.init)
        n_nodes = self.n_rows * self.n_cols
        if weights.shape[:-1] not in ((n_nodes,), (self.n_rows, self.n_cols)) or weights.shape[-1] == 0:
            raise ValueError(
                f"init must have shape (n_nodes, n_features) or (n_rows, n_cols, ...), got {weights.shape}"
            )
        if not torch.isfinite(weights).all():
            raise ValueError("init contains NaN or infinite values")
        return weights.reshape(n_nodes, -1).clone()

    def _init_weights(self, data: torch.Tensor) -> torch.Tensor:
        if not isinstance(self.init, str):
            weights = self._given_weights()
            if weights.shape[1] != data.shape[1]:
                raise ValueError(f"X has {data.shape[1]} features, but init has {weights.shape[1]}")
            return weights
        rng = np.random.RandomState(self.random_state)
        n_nodes = self.n_rows * self.n_cols
        if self.init == "random":
            weights = rng.rand(self.n_rows, self.n_cols, data.shape[1]) * 2 - 1
            weights /= np.linalg.norm(weights, axis=-1, keepdims=True)
            return torch.as_tensor(weights.reshape(n_nodes, -1), dtype=torch.float32, device=self._device)
        has_bmu = self._has_bmu(data)
        candidates = torch.arange(len(data), device=self._device) if has_bmu is None else has_bmu.nonzero()[:, 0]
        if len(candidates) == 0:
            raise ValueError(f"init='sample' with metric={self.metric!r} needs at least one sample with a BMU")
        picks = rng.choice(len(candidates), n_nodes, replace=len(candidates) < n_nodes)
        return data[candidates[torch.as_tensor(picks, device=self._device)]].clone()

    def _has_bmu(self, data: torch.Tensor) -> torch.Tensor | None:
        """Mask of samples with a BMU: non-zero ones (cosine), non-constant ones (correlation); None if all."""
        if self.metric == "cosine":
            return data.norm(dim=1) > 0
        if self.metric == "correlation":
            return data.amax(1) > data.amin(1)
        return None

    def _centre(self, x: torch.Tensor) -> torch.Tensor:
        """Centres the rows over the features under correlation, the cosine metric of centred vectors."""
        if self.metric != "correlation":
            return x
        x = x - x[:, :1]  # exact for nearby values, so a large offset does not spoil the float32 mean
        return x - x.mean(1, keepdim=True)

    def _scores(self, batch: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Per-node score to minimise.

        ||w - c||^2 - 2 (x - c).(w - c) with c the mean node (euclidean, ||x - c||^2 dropped; centring keeps a large
        offset from cancelling), the L1 distance (manhattan) or -x.w / ||w|| of the (for correlation centred) vectors
        (cosine, correlation).
        """
        if self.metric == "euclidean":
            centre = weights.mean(0)
            weights = weights - centre
            return torch.addmm((weights**2).sum(1), batch - centre, weights.T, alpha=-2)
        if self.metric == "manhattan":
            return torch.cdist(batch, weights, p=1)
        weights = self._centre(weights)
        # negate the small weight matrix, not the batch x nodes scores
        return self._centre(batch) @ (weights / -weights.norm(dim=1, keepdim=True).clamp_min(1e-12)).T

    def _score_distances(self, batch: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Distances from the scores of any number of nodes per sample, for every metric but euclidean."""
        if self.metric == "manhattan":
            return scores
        return (1 + scores / self._centre(batch).norm(dim=1, keepdim=True)).clamp(0, 2)

    def _bmus(
        self, data: torch.Tensor, weights: torch.Tensor, return_distance: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """BMU and, if return_distance, distance to it per sample; -1 and NaN for samples without a BMU."""
        bmus, distances = [], []
        for batch in data.split(self._batch_size):
            score, bmu = self._scores(batch, weights).min(1)
            bmus.append(bmu)
            if not return_distance:
                continue
            if self.metric == "euclidean":  # exact, unlike the scores, so large offsets do not cancel
                distances.append((batch - weights[bmu]).norm(dim=1))
            else:
                distances.append(self._score_distances(batch, score[:, None])[:, 0])
        bmu = torch.cat(bmus)
        distance = torch.cat(distances) if return_distance else None
        has_bmu = self._has_bmu(data)
        if has_bmu is not None:
            bmu[~has_bmu] = -1
            if distance is not None:
                distance[~has_bmu] = torch.nan
        return bmu, distance

    def fit(self, X: ArrayLike | torch.Tensor, y: object = None) -> "SOM":
        """Fits the map, starting from the initialisation given by init.

        Args:
            X: Samples, shape (n_samples, n_features) or (..., n_features); numpy array, array-like, torch tensor
                or DLPack array such as CuPy. Any real values are fine; data is copied to the device once and stays
                there.
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
                keep = self._has_bmu(batch)  # samples without a BMU do not contribute
                if keep is not None:
                    batch, bmus = batch[keep], bmus[keep]
                sums.index_add_(0, bmus, batch)
                counts += torch.bincount(bmus, minlength=n_nodes)
            # in place, so large maps hold one nodes x nodes matrix at a time
            h = torch.cdist(self._coords, self._coords).square_().mul_(-0.5 / sigma**2).exp_()
            numerator, denominator = h @ sums, h @ counts
            weights = torch.where(denominator[:, None] > 0, numerator / denominator[:, None], weights)
        self._set_weights(weights)
        self.labels_ = self._bmus(data, weights, return_distance=False)[0].cpu().numpy()
        return self

    @overload
    def predict(self, X: ArrayLike | torch.Tensor, return_distance: Literal[False] = False) -> np.ndarray: ...

    @overload
    def predict(self, X: ArrayLike | torch.Tensor, return_distance: Literal[True]) -> tuple[np.ndarray, np.ndarray]: ...

    def predict(
        self, X: ArrayLike | torch.Tensor, return_distance: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Returns the BMU index (int64) per sample; -1 for samples without a BMU (see metric).

        Args:
            X: Samples, shape (n_samples, n_features) or (..., n_features).
            return_distance: Whether to also return each sample's distance (float32) to its BMU in the metric, NaN
                for samples without a BMU. It comes from the same pass as the BMU search, and its mean over the
                samples with a BMU is quantization_error(X).

        Returns:
            The BMU indices, or the tuple (BMU indices, distances) if return_distance.
        """
        self._check_fitted()
        bmus, distances = self._bmus(self._check_X(X), self._weights, return_distance)
        return (bmus.cpu().numpy(), distances.cpu().numpy()) if return_distance else bmus.cpu().numpy()

    def fit_predict(self, X: ArrayLike | torch.Tensor, y: object = None) -> np.ndarray:
        """Fits the map and returns labels_."""
        return self.fit(X).labels_

    def _distances(self, batch: torch.Tensor) -> torch.Tensor:
        if self.metric == "euclidean":  # exact, unlike the BMU search, so large offsets do not cancel
            return torch.cdist(batch, self._weights, compute_mode="donot_use_mm_for_euclid_dist")
        distances = self._score_distances(batch, self._scores(batch, self._weights))
        has_bmu = self._has_bmu(batch)
        if has_bmu is not None:
            distances[~has_bmu] = 1
        return distances

    def transform(self, X: ArrayLike | torch.Tensor) -> np.ndarray:
        """Returns the distance of every sample to every node, shape (n_samples, n_nodes).

        Distances in the metric; 1 for samples without a BMU. The output is a dense n_samples x n_nodes float32
        array; for large inputs call it on slices or use predict.
        """
        self._check_fitted()
        data = self._check_X(X)
        distances = torch.empty(len(data), len(self._weights))  # on the CPU, as it can exceed device memory
        for batch, rows in zip(data.split(self._batch_size), distances.split(self._batch_size), strict=True):
            rows.copy_(self._distances(batch))
        return distances.numpy()

    def fit_transform(self, X: ArrayLike | torch.Tensor, y: object = None) -> np.ndarray:
        """Fits the map and returns transform(X)."""
        return self.fit(X).transform(X)

    def quantization_error(self, X: ArrayLike | torch.Tensor) -> float:
        """Mean distance (in the metric) between each sample with a BMU and its BMU."""
        self._check_fitted()
        bmus, distances = self._bmus(self._check_X(X), self._weights)
        return float(distances[bmus >= 0].cpu().double().mean())  # on the CPU because MPS has no float64

    def topographic_error(self, X: ArrayLike | torch.Tensor) -> float:
        """Fraction of samples whose first and second BMU are not adjacent on the grid (8-neighbourhood)."""
        self._check_fitted()
        data = self._check_X(X)
        if len(self._weights) < 2:
            return 0.0
        has_bmu = self._has_bmu(data)
        if has_bmu is not None:
            data = data[has_bmu]
        nodes = torch.cat(
            [
                self._scores(batch, self._weights).topk(2, dim=1, largest=False).indices
                for batch in data.split(self._batch_size)
            ]
        )
        offset = (self._coords[nodes[:, 0]] - self._coords[nodes[:, 1]]).abs().amax(1)
        return float((offset > 1).cpu().double().mean())

    def distance_map(self) -> np.ndarray:
        """Returns the U-matrix, shape (n_rows, n_cols).

        Each node's summed euclidean distance to its (up to 8) neighbours on the grid, normalised to a maximum of 1
        (all zeros if all neighbouring nodes coincide), as MiniSom's ``distance_map(scaling="sum")``.
        """
        self._check_fitted()
        grid = self._weights.reshape(self.n_rows, self.n_cols, -1)
        padded = torch.nn.functional.pad(grid, (0, 0, 1, 1, 1, 1), value=torch.nan)  # NaN neighbours beyond the edge
        total = sum(
            (grid - padded[row : row + self.n_rows, col : col + self.n_cols]).norm(dim=-1).nan_to_num()
            for row in range(3)
            for col in range(3)
        )
        total = total / total.max() if total.max() > 0 else total
        return total.cpu().numpy()


BatchSOM = SOM
