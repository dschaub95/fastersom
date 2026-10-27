"""K-means clustering with the nearest-node search and summation of the SOM."""

from collections.abc import Callable
from math import log

import numpy as np
import torch
from numpy.typing import ArrayLike
from tqdm import tqdm

from ._som import SOM, _is_int, _synchronize

_INITS = ("k-means++", "random")
_METRICS = ("euclidean", "cosine", "correlation")
_EMPTY = ("farthest", "keep")


class KMeans:
    """K-means clustering (Lloyd's algorithm) on CPUs and on CUDA and Metal GPUs.

    Each iteration assigns every sample to its nearest centroid with the BMU search of :class:`SOM`, including its
    fused kernels, and moves every centroid to the mean of its samples (see empty_clusters for centroids without).
    Under cosine and correlation it is spherical k-means: the samples are compared, and averaged, scaled to unit length
    (and, for correlation, centred over the features), and the centroids are directions.

    Parameters
    ----------
    n_clusters
        Number of clusters.
    init
        ``"k-means++"`` (greedy k-means++ as in scikit-learn: from a random sample, every further centroid is the best
        of ``n_local_trials`` samples drawn with probability proportional to their squared distance to the nearest
        centroid so far, by the reduction of the total), ``"random"`` (distinct samples drawn at random) or an array of
        shape (n_clusters, n_features) with the initial centroids.
    max_iter
        Largest number of iterations.
    tol
        The fit stops when the centroids move, in total squared distance, by at most tol times the mean variance of the
        features (scikit-learn's criterion), or not at all.
    metric
        ``"euclidean"``, ``"cosine"`` or ``"correlation"`` (see :class:`SOM`). Manhattan is not offered, as the mean
        does not minimise the L1 distance.
    empty_clusters
        What happens to a centroid that gets no samples: ``"farthest"`` moves it to the sample farthest from its own
        centroid, which leaves that centroid's mean (as scikit-learn does; several empty centroids take the farthest
        samples in decreasing order of distance), at the cost of one more pass over the samples in that iteration;
        ``"keep"`` leaves it where it is (as cuML does).
    n_local_trials
        Candidates per centroid of k-means++. Defaults to 2 + int(log(n_clusters)), as in scikit-learn.
    init_size
        Samples that k-means++ seeds on: a random subset of this many, all samples if there are fewer. Seeding reads
        them once per centroid, so its cost grows with init_size x n_clusters x n_local_trials; on all samples it would
        take hours for 10^4 centroids of millions of samples, while the iterations that follow work on all of them.
        Defaults to max(100_000, 10 x n_clusters).
    compute_labels
        Whether fit also assigns the samples to the final centroids, for ``labels_`` and ``inertia_``: one more pass
        over the samples, which a fit for the centroids alone does not need.
    random_state
        Seed for the initialisation.
    device, batch_size, memory_fraction, deterministic, fused
        As for :class:`SOM`.
    verbose
        Whether fit shows a progress bar over the iterations.

    Attributes
    ----------
    cluster_centers_
        Centroids, shape (n_clusters, n_features).
    n_iter_
        Number of iterations run.
    labels_
        Cluster of every training sample under the final centroids (-1 without a nearest centroid, see metric); with
        compute_labels only.
    inertia_
        Sum over the training samples of the squared euclidean distance to their centroid, or of the cosine (or
        correlation) distance; with compute_labels only.
    n_features_in_
        Number of features seen in fit.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: str | ArrayLike | torch.Tensor = "k-means++",
        max_iter: int = 300,
        tol: float = 1e-4,
        metric: str = "euclidean",
        empty_clusters: str = "farthest",
        n_local_trials: int | None = None,
        init_size: int | None = None,
        compute_labels: bool = True,
        random_state: int | None = None,
        device: str | torch.device | None = None,
        batch_size: int | None = None,
        memory_fraction: float | None = None,
        deterministic: bool = False,
        fused: bool = True,
        verbose: bool = False,
    ) -> None:
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.metric = metric
        self.empty_clusters = empty_clusters
        self.n_local_trials = n_local_trials
        self.init_size = init_size
        self.compute_labels = compute_labels
        self.random_state = random_state
        self.device = device
        self.batch_size = batch_size
        self.memory_fraction = memory_fraction
        self.deterministic = deterministic
        self.fused = fused
        self.verbose = verbose

    def _engine(self, fall_back: bool = False) -> SOM:
        """A map of n_clusters x 1 nodes whose search and summation the fit uses (its grid and schedule unused)."""
        for name in ("n_clusters", "max_iter"):
            if not _is_int(getattr(self, name), 1):
                raise ValueError(f"{name} must be an integer >= 1, got {getattr(self, name)!r}")
        if self.metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}, got {self.metric!r}")
        if isinstance(self.init, str) and self.init not in _INITS:
            raise ValueError(f"init must be one of {_INITS} or an array of centroids, got {self.init!r}")
        if self.empty_clusters not in _EMPTY:
            raise ValueError(f"empty_clusters must be one of {_EMPTY}, got {self.empty_clusters!r}")
        if self.n_local_trials is not None and not _is_int(self.n_local_trials, 1):
            raise ValueError(f"n_local_trials must be None or an integer >= 1, got {self.n_local_trials!r}")
        if not self.tol >= 0:
            raise ValueError(f"tol must be >= 0, got {self.tol!r}")
        engine = SOM(
            self.n_clusters,
            1,
            n_epochs=0,
            metric=self.metric,
            random_state=self.random_state,
            device=self.device,
            batch_size=self.batch_size,
            memory_fraction=self.memory_fraction,
            deterministic=self.deterministic,
            fused=self.fused,
        )
        engine._setup(fall_back)
        return engine

    def _initial_centroids(self, engine: SOM, data: torch.Tensor) -> torch.Tensor:
        if not isinstance(self.init, str):
            centroids = engine._as_tensor(self.init)
            if centroids.shape != (self.n_clusters, data.shape[1]):
                raise ValueError(
                    f"init must have shape (n_clusters, n_features) = {(self.n_clusters, data.shape[1])}, "
                    f"got {tuple(centroids.shape)}"
                )
            if not torch.isfinite(centroids).all():
                raise ValueError("init contains NaN or infinite values")
            return centroids.clone()
        # the indices of the samples with a centroid (None for all), of which only those drawn are copied
        has_bmu = engine._has_bmu(data)
        candidates = None if has_bmu is None else has_bmu.nonzero()[:, 0]
        n_candidates = len(data) if candidates is None else len(candidates)
        if n_candidates < self.n_clusters:
            raise ValueError(f"n_clusters={self.n_clusters} needs as many samples with a nearest centroid")

        def samples(rows: np.ndarray) -> torch.Tensor:
            index = torch.as_tensor(rows, device=data.device)
            return data[index if candidates is None else candidates[index]]

        rng = np.random.default_rng(self.random_state)
        if self.init == "random":
            return samples(rng.choice(n_candidates, self.n_clusters, replace=False))
        size = self.init_size if self.init_size is not None else max(100_000, 10 * self.n_clusters)
        if not _is_int(size, self.n_clusters):
            raise ValueError(f"init_size must be an integer >= n_clusters, got {size!r}")
        if n_candidates > size:
            subset = samples(np.sort(rng.choice(n_candidates, size, replace=False)))
        else:
            subset = data if candidates is None else data[candidates]
        return self._kmeans_plus_plus(engine, subset, rng)

    def _kmeans_plus_plus(self, engine: SOM, data: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Greedy k-means++ on the samples as the metric compares them (squared euclidean distances).

        The distances are ||x - m||^2 + ||c - m||^2 - 2 (x - m).(c - m) by a matmul of the samples centred once on
        their mean m (a copy of at most init_size rows, the compared copy itself under cosine and correlation), so
        that a large offset does not cancel.
        """
        trials = self.n_local_trials if self.n_local_trials is not None else 2 + int(log(self.n_clusters))
        batch_size = max(1, (1 << 24) // data.shape[1])
        if self.metric == "euclidean":
            centred = data - data.mean(0)
        else:  # unit rows, compared once rather than at every centroid, and centred in place in that copy
            centred = torch.cat([engine._compared(batch) for batch in data.split(batch_size)])
            centred -= centred.mean(0)
        norms = torch.cat([batch.square().sum(1) for batch in centred.split(batch_size)])

        def squared(centres: torch.Tensor) -> torch.Tensor:
            """Squared distance of every sample to each of the (centred) centres, shape (n_samples, len(centres))."""
            distances = torch.addmm(centres.square().sum(1), centred, centres.T, alpha=-2)
            return distances.add_(norms[:, None]).clamp_min_(0)

        # every draw up front and the choices kept on the device, so that the loop never waits for the device
        first = int(rng.integers(len(data)))
        uniform = rng.uniform(size=(self.n_clusters - 1, trials))
        closest = squared(centred[first : first + 1])[:, 0]
        exact = closest.device.type != "mps"  # float64 sums where the device has them
        draws = torch.as_tensor(uniform, dtype=torch.float64 if exact else closest.dtype, device=closest.device)
        chosen = [torch.full((1,), first, dtype=torch.long, device=data.device)]
        for k in range(1, self.n_clusters):
            cumulative = closest.double().cumsum(0) if exact else closest.cumsum(0)
            rows = torch.searchsorted(cumulative, draws[k - 1] * cumulative[-1]).clamp_(max=len(data) - 1)
            potentials = torch.minimum(closest[:, None], squared(centred.index_select(0, rows)))
            best = potentials.sum(0).argmin(0, keepdim=True)  # indexing with a 0-d tensor would wait for its value
            closest = potentials.index_select(1, best)[:, 0].contiguous()
            chosen.append(rows.index_select(0, best))
        # the samples themselves (euclidean), else their compared form, the centroids' directions
        samples = data.index_select(0, torch.cat(chosen))
        return samples if self.metric == "euclidean" else engine._compared(samples)

    def fit(self, X: ArrayLike | torch.Tensor, y: object = None) -> "KMeans":
        """Fit the centroids, starting from the initialisation given by init.

        Parameters
        ----------
        X
            Samples, shape (n_samples, n_features) or (..., n_features), as for :meth:`SOM.fit`.
        y
            Ignored.

        Returns
        -------
        KMeans
            The fitted model.
        """
        for attr in ("cluster_centers_", "labels_", "inertia_", "n_iter_", "n_features_in_"):
            self.__dict__.pop(attr, None)
        engine = self._engine()
        data = engine._check_X(X)
        centroids = self._initial_centroids(engine, data)
        deterministic = engine._device.type != "cpu" and (
            self.deterministic or torch.are_deterministic_algorithms_enabled()
        )
        kernel = engine._kernel(data.shape[1])
        batch_size = engine._batch_size_for(data, kernel)
        euclidean = self.metric == "euclidean"
        # the mean of the values summed (of the samples with a centroid), for tol and as the origin of the sums
        mean, n_samples = engine._mean(data, batch_size) if self.tol or euclidean else (None, 0)
        threshold = self.tol * self._feature_variance(engine, data, batch_size, mean, n_samples) if self.tol else 0.0
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None if euclidean else engine._compared
        origin = None
        if euclidean:
            assert mean is not None
            low, high = torch.aminmax(data, dim=0)
            centre = mean.float().to(engine._device)
            # deviations from the mean are summed if it lies beyond the range of a feature, so that a large offset does
            # not round the sums; within it the samples' sums round at most about twice as much, and spare a pass over
            # the samples per iteration (up to 5% of an iteration on an A100 at 10^6 x 32, 17% on the CPU at 200,000 x
            # 32 and 100 centroids, none at 1000)
            if bool((centre.abs() > high - low).any()):
                origin = centre
                low, high = low - origin, high - origin  # as transform rounds them
                # into one buffer, which each caller reads before the next call: on the CPU a fresh copy of a part of
                # 2^20 samples costs more in page faults than its sums (36 vs 22 ms at 2^20 x 32)
                buffer = data.new_empty(0)

                def transform(batch: torch.Tensor) -> torch.Tensor:
                    nonlocal buffer
                    if len(buffer) < len(batch):
                        buffer = torch.empty_like(batch)
                    return torch.sub(batch, centre, out=buffer[: len(batch)])

        magnitude = None
        if deterministic:  # the largest |x| per feature of the values summed: unit rows under cosine and correlation
            magnitude = torch.maximum(high, -low) if euclidean else data.new_ones(data.shape[1])
        summing = engine._summing(len(data), kernel, deterministic, magnitude)
        self.n_iter_ = 0
        for _ in tqdm(range(self.max_iter), desc="Fitting k-means", unit="iteration", disable=not self.verbose):
            search = engine._search(centroids, kernel)
            sums, counts = engine._sums(data, search, self.n_clusters, batch_size, summing, transform)
            empty = (counts == 0).nonzero()[:, 0]
            if self.empty_clusters == "farthest" and len(empty):
                sums, counts = self._relocate(engine, data, centroids, sums, counts, empty, transform)
            means = sums / counts[:, None].clamp_min(1)
            moved = torch.where(counts[:, None] > 0, means if origin is None else means + origin, centroids)
            shift = float((moved - centroids).square().sum())
            centroids = moved
            self.n_iter_ += 1
            if self.verbose:
                _synchronize(engine._device)
            if shift <= threshold:
                break
        engine._set_weights(centroids)  # the engine's nodes are the centroids, for predict and transform
        self.cluster_centers_, self.n_features_in_, self._engine_ = engine.weights_, data.shape[1], engine
        if self.compute_labels:
            labels, distances = engine._bmus(data, centroids)
            assert distances is not None
            self.labels_ = labels.cpu().numpy()
            values = distances[labels >= 0].double() if distances.device.type != "mps" else distances[labels >= 0]
            values = values.square() if self.metric == "euclidean" else values
            self.inertia_ = float(values.sum().cpu())
        return self

    def _relocate(
        self,
        engine: SOM,
        data: torch.Tensor,
        centroids: torch.Tensor,
        sums: torch.Tensor,
        counts: torch.Tensor,
        empty: torch.Tensor,
        transform: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Give each empty centroid one of the samples farthest from their centroid, taken from that centroid's sums."""
        labels, distances = engine._bmus(data, centroids)
        assert distances is not None
        # only samples with a centroid, which may be fewer than the empty centroids; those left over stay empty
        candidates = (labels >= 0).nonzero()[:, 0]
        far = candidates[distances[candidates].topk(min(len(empty), len(candidates))).indices]
        values = data[far] if transform is None else transform(data[far])
        sums, counts = sums.clone(), counts.clone()
        for cluster, sample, value in zip(empty.tolist(), far.tolist(), values, strict=False):
            old = int(labels[sample])
            sums[old] -= value
            counts[old] -= 1
            sums[cluster] = value
            counts[cluster] = 1
        return sums, counts

    @staticmethod
    def _feature_variance(
        engine: SOM, data: torch.Tensor, batch_size: int, mean: torch.Tensor | None, n_samples: int
    ) -> float:
        """The mean over the features of the variance of the values summed (of the samples with a centroid).

        About their mean and over their number from :meth:`SOM._mean`, by :meth:`SOM._scatter` as for the covariance
        of :meth:`SOM._pca_weights`.
        """
        if mean is None or n_samples == 0:
            return 0.0
        return float((engine._scatter(data, batch_size, mean, n_samples, full=False) / n_samples).mean())

    def _check_fitted(self) -> SOM:
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("This KMeans instance is not fitted yet, call fit() first")
        return self._engine_

    def predict(self, X: ArrayLike | torch.Tensor) -> np.ndarray:
        """Return the nearest centroid (int64) of every sample; -1 for samples without one (see metric)."""
        return self._check_fitted().predict(X)

    def fit_predict(self, X: ArrayLike | torch.Tensor, y: object = None) -> np.ndarray:
        """Fit the centroids and return the cluster of every sample.

        That is ``labels_``, computed even without compute_labels.
        """
        self.fit(X)
        return self.labels_ if self.compute_labels else self.predict(X)

    def transform(self, X: ArrayLike | torch.Tensor) -> np.ndarray:
        """Return the distance of every sample to every centroid in the metric, shape (n_samples, n_clusters)."""
        return self._check_fitted().transform(X)

    def __getstate__(self) -> dict:
        """Keeps the parameters and numpy fitted attributes only, so the pickle does not depend on the device."""
        state = {key: value for key, value in self.__dict__.items() if not key.startswith("_")}
        if isinstance(self.init, torch.Tensor):
            state["init"] = self.init.cpu()
        return state

    def __setstate__(self, state: dict) -> None:
        """Restores a pickled model on the configured device if available (cpu, cuda, mps), else the default one."""
        self.__dict__.update(state)
        if hasattr(self, "cluster_centers_"):
            self._engine_ = self._engine(fall_back=True)
            self._engine_._set_weights(torch.as_tensor(self.cluster_centers_, device=self._engine_._device))
