"""Batch self-organising map (Kohonen's batch map) in PyTorch.

Each epoch assigns every sample x_i to its best matching unit c(x_i) and then sets every node to the
neighbourhood-weighted mean of the samples,

    w_k = sum_i h(c(x_i), k) x_i / sum_i h(c(x_i), k),    h(j, k) = h(d(r_j, r_k); sigma(t)),

keeping w_k if the denominator is not positive. r_j are the node coordinates on the rectangular grid, d a grid
distance (grid_metric), h a neighbourhood function (neighbourhood) and sigma(t) the width of epoch t (sigma_schedule).
By default h is the gaussian exp(-d^2 / (2 sigma^2)) on the euclidean grid distance and sigma decays exponentially,
sigma(t) = sigma_initial * (sigma_final / sigma_initial) ** (t / n_epochs).

Grouping samples by BMU gives the same update from per-node sums: with S_j the sum and n_j the count of samples whose
BMU is j, numerator = H @ S and denominator = H @ n, H the (nodes x nodes) neighbourhood matrix. The gaussian on the
euclidean grid distance is separable, H = H_rows (x) H_cols (as is the bubble on the chebyshev one), so both products
take one matrix product along the rows and one along the columns, O(nodes * (n_rows + n_cols)) per epoch instead of
O(n_samples * nodes), without ever building H. The other neighbourhoods depend only on the offsets between nodes and
take one Toeplitz matrix product per offset along a grid axis, at most O(nodes^2) per epoch.

References
----------
    T. Kohonen, Self-Organizing Maps, 3rd ed., Springer, 2001.
    J. Vesanto et al., SOM Toolbox for Matlab 5, Helsinki University of Technology, 2000.
"""

import subprocess
import sys
import sysconfig
import warnings
from collections.abc import Callable
from functools import cache
from math import ceil, isfinite
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, cast, overload

import numpy as np
import torch
from numpy.typing import ArrayLike
from tqdm import tqdm

_METRICS = ("euclidean", "manhattan", "cosine", "correlation")
_INITS = ("random", "sample")
_NEIGHBOURHOODS = ("gaussian", "bubble", "triangle", "epanechnikov", "cut_gaussian", "mexican_hat")
_GRID_METRICS = ("euclidean", "chebyshev", "manhattan")
_SCHEDULES = ("exponential", "linear", "inverse")
# Below this neighbourhood width every neighbourhood is the identity in float32: the weight of the nearest other nodes,
# such as the gaussian's exp(-0.5 / sigma^2), is below float32's smallest number (exp(-200)); the gaussian's
# exp(-0.5 / sigma^2 * d^2) would otherwise give 0 * inf = NaN at d = 0 once 0.5 / sigma^2 overflows
_IDENTITY_SIGMA = 0.05
# Up to this many features the float32 kernel beats cuBLAS's matrix product plus argmin for euclidean, cosine and
# correlation (fit on an A100 from 100 to 10,000 nodes: 1.2-2.3x at 34 and 64 features, 1.1-1.2x at 128); beyond it
# the matrix product is compute-bound and cuBLAS catches up (as fast at 192, 0.9x at 256). Manhattan's fused kernel is
# always faster than cdist
_FUSED_MAX_FEATURES = 128
# Compute capabilities of data-centre GPUs (A100, H100, B200), whose TF32 tensor cores do about 8x the multiply-adds of
# their float32 cores, so that three TF32 products per float32 product still win; on GeForce, RTX and L40 GPUs they do
# 1-2x. There the tensor-core kernel searches dot-product scores at any feature count (on an A100, fit 1.5x faster than
# with the float32 kernel at 34 features from 100 x 100 nodes, 1.4-1.8x at 128, 1.3-1.8x faster than cuBLAS beyond).
# The capability does not tell variants with cut-down tensor cores such as the H20 (9.0), where it is likely slower;
# timing both kernels on the device instead could pick differently from run to run, and so change the map
_FAST_TF32 = ((8, 0), (9, 0), (10, 0))
_FAILED: set[tuple[torch.device, str]] = set()  # kernels that failed their self-check on a device
_PRECISIONS = ("float32", "float16")
# precision="float16" searches with fewer features in float32: the scores' rounding error, about 2^-11 ||x - c|| ||v||
# with the ||x - c||^2 dropped from them, then becomes comparable to the distances themselves (in a CPU emulation of
# 0-1 pixels with 5 features, quantization error 2.7% above float32's), and the kernels gain little there anyway
_HALF_MIN_FEATURES = 16
_METAL = ("metal", "metal-float16")
# how a failure's warning names the kernel, and what the kernel leaves
_WARNINGS = {
    "tensor-core": ("tensor-core", "the float32 kernel or the torch path"),
    "metal": ("Metal", "the torch path"),
    "metal-float16": ("Metal float16", "the float32 Metal kernel or the torch path"),
}


def _warn(error: Exception, message: str) -> None:
    """Warns of a fused kernel's failure, at the user's call into fastersom."""
    headers = Path(sysconfig.get_path("include"), "Python.h")
    if not isinstance(error, subprocess.CalledProcessError):
        reason = f"{type(error).__name__}: " + str(error).partition("\n")[0]
    elif headers.exists():  # its message is the whole compiler command; the compiler printed why to stderr
        reason = f"the C compiler exited with status {error.returncode}"
    else:
        reason = f"{headers} not found"
    level = 1
    while sys._getframe(level).f_globals.get("__name__", "").startswith("fastersom."):
        level += 1
    warnings.warn(message.format(reason), RuntimeWarning, stacklevel=level + 1)


@cache
def _triton(device: torch.device) -> ModuleType | None:
    """Load the Triton kernels if they compile for the device.

    fastersom._kernels if Triton compiles kernels for the device, else None: not on ROCm (whose 64-thread warps the
    tiling lacks), not below compute capability 7.0 (which Triton does not support, like torch.compile; on a Pascal
    GP100 its PTX codegen fails), without Triton with Gluon (Triton >= 3.5), and after a warning without a C compiler or
    the Python headers, which Triton needs for its launchers.
    """
    if device.type != "cuda" or torch.version.hip is not None or torch.cuda.get_device_capability(device) < (7, 0):
        return None
    try:
        from fastersom import _kernels
    except Exception:  # noqa: BLE001  # not only ImportError: an older Gluon can fail on its decorators
        return None
    try:
        _kernels.search(torch.ones(1, 1, device=device), None, None, True, False)  # compiles the nodes' copy only
    except torch.OutOfMemoryError:
        raise
    except Exception as error:  # noqa: BLE001  # any compile or launch failure falls back to the next kernel
        _warn(
            error,
            "fastersom's fused CUDA kernels failed ({}), using the slower torch path. Triton needs a C compiler and "
            "the Python headers (e.g. python3-dev, or a uv-managed Python)",
        )
        return None
    return _kernels


@cache
def _metal(device: torch.device) -> ModuleType | None:
    """Load the Metal kernels if torch compiles them for the device.

    fastersom._metal if torch compiles Metal kernels for the device (MPS, torch >= 2.6), else None. The kernels
    compile per feature count, so that one that does not compile fails in _works.
    """
    if device.type != "mps" or not hasattr(torch.mps, "compile_shader"):
        return None
    from fastersom import _metal as kernels

    return kernels


def _out_of_memory(error: Exception) -> bool:
    """Whether the error is torch running out of GPU memory, which on MPS raises a plain RuntimeError."""
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error)


@cache
def _works(device: torch.device, kernel: str, n_features: int, metric: str, top2: bool) -> bool:
    """Whether a fused kernel variant passes its self-check.

    Whether a fused kernel's variant for these arguments passes its self-check (fastersom._kernels.check or
    fastersom._metal.check), before any batch size is chosen for it. A failure warns and turns off the kernel's
    variants not yet checked on the device; those that passed stay on, as they passed an exact check of the very code
    that runs, and switching kernels within a session would change results between fits. Running out of memory is no
    failure of the kernel and raises.
    """
    if (device, kernel) in _FAILED:
        return False
    try:
        manhattan, bias = metric == "manhattan", metric == "euclidean"
        if kernel in _METAL:
            _module(device, kernel).check(device, n_features, manhattan, bias, top2, kernel == "metal-float16")
        else:
            _module(device, kernel).check(device, n_features, kernel == "tensor-core", manhattan, bias, top2)
    except Exception as error:
        if _out_of_memory(error):
            raise
        _FAILED.add((device, kernel))
        name, fallback = _WARNINGS.get(kernel, (kernel, "the torch path"))
        failed = f"{name} BMU kernel failed " + ("to compile or its" if kernel in _METAL else "its")
        _warn(error, f"fastersom's {failed} self-check ({{}}), using {fallback}")
        return False
    return True


def _module(device: torch.device, kernel: str) -> ModuleType:
    """The module of a fused kernel, fastersom._metal or fastersom._kernels, which must be available on the device."""
    module = _metal(device) if kernel in _METAL else _triton(device)
    if module is None:
        raise RuntimeError(f"the {kernel} kernel is not available on {device}")
    return module


def _is_int(value, minimum: int) -> bool:
    return bool(isinstance(value, int | np.integer) and not isinstance(value, bool) and value >= minimum)


def _is_real(value) -> bool:
    return isinstance(value, int | float | np.integer | np.floating) and not isinstance(value, bool)


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


def _cdist(x: torch.Tensor, y: torch.Tensor, p: Literal[1, 2]) -> torch.Tensor:
    """L1 or exact euclidean distance of every row of x to every row of y.

    The L1 (p=1) or exact euclidean (p=2) distance of every row of x to every row of y. On MPS, torch's cdist under
    p=1 and its exact euclidean mode hold the differences of every pair of rows at once, len(x) x len(y) x n_features
    floats (36 GB for 26,843 samples, 10,000 nodes and 34 features), several times over, so there the distances add
    up one feature at a time instead, in two len(x) x len(y) matrices (1.5x the time of cdist at 20,000 samples, 900
    nodes and 34 features).
    """
    if x.device.type != "mps":
        return torch.cdist(x, y, p=p, compute_mode="donot_use_mm_for_euclid_dist")
    out = torch.zeros(len(x), len(y), dtype=x.dtype, device=x.device)
    part = torch.empty_like(out)
    for k in range(x.shape[1]):
        term = torch.sub(x[:, k, None], y[:, k], out=part)
        out += term.abs_() if p == 1 else term.square_()
    return out if p == 1 else out.sqrt_()


def _grid_distance(dr: np.ndarray, dc: np.ndarray, grid_metric: str) -> np.ndarray:
    """Distance in float64 between grid positions dr rows and dc columns apart (integer arrays)."""
    dr, dc = np.abs(dr), np.abs(dc)
    if grid_metric == "chebyshev":
        return np.maximum(dr, dc).astype(np.float64)
    if grid_metric == "manhattan":
        return (dr + dc).astype(np.float64)
    return np.sqrt((dr * dr + dc * dc).astype(np.float64))  # correctly rounded, so d == sigma compares exactly


def _neighbourhood(d: np.ndarray, sigma: float, neighbourhood: str) -> np.ndarray:
    """The neighbourhood function h(d; sigma) in float64, [d == 0] at sigma 0 (the limit of every function)."""
    if sigma == 0:
        return (d == 0).astype(np.float64)
    q = np.square(d / sigma)
    if neighbourhood == "gaussian":
        return np.exp(-0.5 * q)
    if neighbourhood == "bubble":
        return (d <= sigma).astype(np.float64)
    if neighbourhood == "triangle":
        return np.maximum(1 - d / sigma, 0)
    if neighbourhood == "epanechnikov":
        return np.maximum(1 - q, 0)
    if neighbourhood == "cut_gaussian":
        return np.exp(-0.5 * q) * (d <= sigma)
    return (1 - q) * np.exp(-0.5 * q)  # mexican_hat


def _grid_quantile(n_rows: int, n_cols: int, grid_metric: str, p: float) -> float:
    """R's type-7 quantile p of the grid distances of all n_nodes^2 ordered pairs of nodes, as FlowSOM's radius.

    The pairs (j, k) include j == k, as in the distance matrix FlowSOM takes the quantile of. Computed from the
    (2 n_rows - 1) x (2 n_cols - 1) grid offsets, weighted by how many pairs of nodes have each, without the matrix.
    """
    dr, dc = np.arange(1 - n_rows, n_rows), np.arange(1 - n_cols, n_cols)
    distances = _grid_distance(dr[:, None], dc[None], grid_metric).ravel()
    counts = ((n_rows - np.abs(dr))[:, None] * (n_cols - np.abs(dc))[None]).ravel()
    order = np.argsort(distances, kind="stable")
    distances, cumulative = distances[order], np.cumsum(counts[order])
    index = (int(cumulative[-1]) - 1) * p  # 0-based position in the sorted distances
    lo = int(np.floor(index))
    x_lo = distances[np.searchsorted(cumulative, lo, side="right")]
    x_hi = distances[np.searchsorted(cumulative, min(lo + 1, int(cumulative[-1]) - 1), side="right")]
    h = index - lo
    return float(x_lo) if h == 0 or x_hi == x_lo else float((1 - h) * x_lo + h * x_hi)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class SOM:
    """Batch self-organising map on a rectangular grid, by default with a gaussian neighbourhood.

    On CUDA with Triton's Gluon (Triton >= 3.5, part of torch's Linux wheels from torch 2.9), the BMU search of fit,
    predict, quantization_error and topographic_error runs in a fused kernel that never stores the batch x nodes
    scores: under manhattan always, under the other metrics up to 128 features, beyond which cuBLAS's matrix product is
    as fast. It computes in float32 like the torch path, so the BMUs differ only on near-ties. On A100, H100 and B200
    GPUs, the other metrics' search instead runs on tensor cores at any feature count (for maps of over 64 nodes and
    nodes x features >= 8192), with float32 accuracy from three TF32 products per product (3xTF32); its BMUs too
    differ only on near-ties, and match float64's more often than float32's. The first use of a feature count compiles
    the kernel (2-4 s the first time, cached on disk by Triton), which needs a C compiler and the Python headers, and
    checks it on a tiny search; a kernel that fails warns and leaves the next one (tensor cores, float32 kernel, torch
    path). On MPS (torch >= 2.6) the search runs in fused Metal kernels, in float32 on the simdgroup matrix units (on
    the ALUs under manhattan, up to 512 features), compiled at runtime in about 0.2 s per feature count and checked in
    the same way.

    Parameters
    ----------
    n_rows
        Number of grid rows.
    n_cols
        Number of grid columns.
    n_epochs
        Number of epochs; each makes one weight update from all samples. With 0, fit keeps the initial
        weights.
    sigma_initial
        Initial neighbourhood width in grid units: the standard deviation of the gaussian and the mexican hat, the
        radius of the other neighbourhoods. Defaults to ceil(min(n_rows, n_cols) / 2) / 2. ``"flowsom"`` takes
        FlowSOM's initial radius, the 0.67 quantile (R's type 7) of the grid distances between all pairs of nodes,
        each with itself included, under grid_metric (FlowSOM's chebyshev: 6 for 10 x 10 nodes).
    sigma_final
        Width the schedule decays towards (reached at t = n_epochs, one epoch after the last), >= 0 under the linear
        schedule and > 0 under the others. Defaults to min(0.1 * sigma_initial, 1): a tenth of the initial width, at
        most one grid unit, so that a large map is not smoothed beyond the resolution of its units (on 100 x 100
        nodes the quantization error is 9-13% lower than with 0.1 * sigma_initial = 2.5); maps up to 40 x 40 nodes
        are unaffected. Smaller widths lower the quantization error further but raise the topographic error.
        0.1 * sigma_initial reproduces XPySom's schedule.
    init
        ``"random"`` draws weights uniformly from [-1, 1]^d with numpy's ``RandomState(random_state)`` and
        normalises them to unit length; ``"sample"`` uses randomly chosen samples (ones with a BMU); an array of
        shape (n_rows * n_cols, n_features) or (n_rows, n_cols, n_features) gives the initial weights, e.g. another
        map's ``weights_`` for a warm start. The sigma schedule restarts at sigma_initial, which with the default
        largely retrains the map; to continue training, pass a small sigma_initial such as the previous
        sigma_final.
    metric
        ``"euclidean"``, ``"manhattan"`` (L1), ``"cosine"`` (1 - cosine similarity) or ``"correlation"``
        (1 - Pearson correlation, the cosine distance of vectors centred over the features). Every metric updates
        the nodes to the neighbourhood-weighted mean of the raw samples (not the median for manhattan, nor the
        centred samples for correlation). Zero-norm samples under cosine and constant samples under correlation
        have no BMU: they get label -1, are excluded from the update and the error measures, and have distance 1
        to every node in transform.
    random_state
        Seed for the initialisation.
    device
        Torch device. Defaults to cuda, then mps, then cpu.
    batch_size
        Samples processed at once in BMU search, aggregation, predict and transform. Affects memory
        (about batch_size x n_nodes floats, batch_size x n_features with the fused kernel) and speed only. Each
        epoch's update is the same up to float summation order, but over several epochs such rounding differences
        (as between devices) can flip near-tied BMUs, so maps agree in quality rather than bit for bit. Defaults
        to 8192 on CPU and on GPUs to 2^28 / n_nodes (where speed levels off; 2^26 / n_features with the fused
        kernel), fewer if that does not fit in memory_fraction of the free GPU memory (on MPS, of the memory torch
        recommends; 65536 before torch 2.5).
    memory_fraction
        Fraction of the free GPU memory the default batch_size may use, in (0, 1]. Defaults to 0.9
        on CUDA and 0.5 on MPS, whose memory the system and other apps share.
    verbose
        Whether fit shows a progress bar over the epochs, with an estimate of the remaining time. On GPUs, it
        then waits for each epoch to finish before starting the next.
    deterministic
        Whether fit gives the same map in every run on CUDA and MPS, whose atomic additions of the
        per-node sums otherwise vary in order, so that near-tied BMUs can flip between runs. On CUDA, and on MPS
        with the fused Metal kernels, the sums then add the samples as 64-bit fixed-point integers, whose sum does
        not depend on the order, at a small cost (on an M3 Max 1.4x at 10 x 10 nodes, none from 100 x 100); on MPS
        without the kernels they take a matrix product with a one-hot nodes x batch matrix (1.2-1.8x slower).
        Also used when ``torch.use_deterministic_algorithms(True)`` is set. The CPU is always deterministic. Maps
        from different devices still differ by float rounding.
    fused
        Whether the BMU search on CUDA and MPS may use the fused kernels. With False, it never compiles or
        launches them and takes the torch path, as on the CPU.
    precision
        ``"float32"`` or ``"float16"``, with which fit's BMU search may round the samples and nodes to 11
        significant bits for speed, summing the products in float32, as half inputs to the Metal kernels on MPS,
        scaled by powers of 2 per feature into half's range (samples and initial weights within 2^-30 to 2^30).
        Only under euclidean, cosine and correlation from 16 features with the Metal kernels, else float32 (as on
        CUDA); ``labels_``, predict, transform and the error measures stay float32. The BMUs of near-tied samples
        flip, more of them on larger maps and depending on the data (on clustered 0-1 pixels of 34 features about
        1% at 10 x 10 nodes and 10-35% from 30 x 30), each within the rounding error (the squared distance to the
        node found 0.6% above the best on average), which gives an equally good map rather than the same one: the
        same quantization error, and a topographic error up to 1.4x higher on synthetic clusters. On an M3 Max fit
        is only 1.00-1.05x faster, as its half products are no faster than float32 ones; M5 GPUs may gain more,
        which is untested.

    neighbourhood
        Neighbourhood function h(d; sigma) of the grid distance d: ``"gaussian"`` exp(-d^2 / (2 sigma^2)),
        ``"bubble"`` 1 for d <= sigma (inclusive, as FlowSOM and the SOM Toolbox) else 0, ``"triangle"``
        max(0, 1 - d / sigma), ``"epanechnikov"`` max(0, 1 - d^2 / sigma^2), ``"cut_gaussian"`` the gaussian for
        d <= sigma else 0, or ``"mexican_hat"`` (1 - d^2 / sigma^2) exp(-d^2 / (2 sigma^2)), which weights nodes
        beyond sigma negatively (nodes whose denominator is not positive keep their weights; small denominators can
        move nodes far outside the data). At sigma 0 every function leaves only the winner, which makes the epoch a
        batch k-means step. Keyword only.
    grid_metric
        Distance between grid positions dr rows and dc columns apart: ``"euclidean"`` sqrt(dr^2 + dc^2),
        ``"chebyshev"`` max(abs(dr), abs(dc)) (FlowSOM's) or ``"manhattan"`` abs(dr) + abs(dc). The gaussian under
        euclidean and the bubble under chebyshev factor into the grid axes and cost the least; the other combinations
        take up to n_nodes^2 x n_features multiply-adds per epoch, small next to the BMU search when there are more
        samples than nodes. The topographic error and distance_map always use the 8 surrounding nodes. Keyword only.
    sigma_schedule
        How the width moves from sigma_initial (s0) towards sigma_final (s1) over the epochs t = 0, ...,
        n_epochs - 1, with s = t / n_epochs: ``"exponential"`` s0 (s1 / s0)^s, ``"linear"`` s0 + (s1 - s0) s (with
        s1 = 0 FlowSOM's radius), or ``"inverse"`` s0 / (1 + (s0 / s1 - 1) s) (with s1 = s0 / 3 MiniSom's default
        asymptotic decay). A sequence of n_epochs widths >= 0 gives the width of every epoch instead, with
        sigma_initial and sigma_final None. Keyword only.

    Attributes
    ----------
    weights_
        Node weights, shape (n_rows * n_cols, n_features), row-major (node = row * n_cols + col).
    labels_
        BMU of every training sample under the final weights. Not set by from_weights.
    n_features_in_
        Number of features seen in fit.
    """

    def __init__(
        self,
        n_rows: int = 30,
        n_cols: int = 30,
        n_epochs: int = 10,
        sigma_initial: float | str | None = None,
        sigma_final: float | None = None,
        init: str | ArrayLike | torch.Tensor = "random",
        metric: str = "euclidean",
        random_state: int | None = None,
        device: str | torch.device | None = None,
        batch_size: int | None = None,
        memory_fraction: float | None = None,
        verbose: bool = False,
        deterministic: bool = False,
        fused: bool = True,
        precision: str = "float32",
        *,
        neighbourhood: str = "gaussian",
        grid_metric: str = "euclidean",
        sigma_schedule: str | ArrayLike = "exponential",
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
        self.memory_fraction = memory_fraction
        self.verbose = verbose
        self.deterministic = deterministic
        self.fused = fused
        self.precision = precision
        self.neighbourhood = neighbourhood
        self.grid_metric = grid_metric
        self.sigma_schedule = sigma_schedule

    @classmethod
    def from_weights(cls, weights: ArrayLike | torch.Tensor, n_rows: int, n_cols: int, **kwargs) -> "SOM":
        """Create a fitted map from given weights.

        Returns a fitted map with the given weights (shapes as for init), like ``SOM(n_rows, n_cols, n_epochs=0,
        init=weights, **kwargs).fit(X)`` but without data, so ``labels_`` is not set.
        """
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
        state.setdefault("memory_fraction", None)  # pickled before 0.3
        state.setdefault("verbose", False)  # pickled before 0.3.2
        state.setdefault("deterministic", False)  # pickled before 0.3.2
        state.setdefault("fused", True)  # pickled by 0.3.2 and before
        state.setdefault("precision", "float32")  # pickled by 0.4.0 and before
        state.setdefault("neighbourhood", "gaussian")  # pickled by 0.6.0 and before
        state.setdefault("grid_metric", "euclidean")
        state.setdefault("sigma_schedule", "exponential")
        self.__dict__.update(state)
        if hasattr(self, "weights_"):
            self._setup(fall_back=True)
            self._set_weights(torch.as_tensor(self.weights_, device=self._device))

    def _setup(self, fall_back: bool = False) -> None:
        if self.metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}, got {self.metric!r}")
        if self.precision not in _PRECISIONS:
            raise ValueError(f"precision must be one of {_PRECISIONS}, got {self.precision!r}")
        if isinstance(self.init, str) and self.init not in _INITS:
            raise ValueError(f"init must be one of {_INITS} or an array of weights, got {self.init!r}")
        for name, minimum in (("n_rows", 1), ("n_cols", 1), ("n_epochs", 0)):
            if not _is_int(getattr(self, name), minimum):
                raise ValueError(f"{name} must be an integer >= {minimum}, got {getattr(self, name)!r}")
        if self.batch_size is not None and not _is_int(self.batch_size, 1):
            raise ValueError(f"batch_size must be None or an integer >= 1, got {self.batch_size!r}")
        if self.memory_fraction is not None and not 0 < self.memory_fraction <= 1:
            raise ValueError(f"memory_fraction must be None or in (0, 1], got {self.memory_fraction!r}")
        self._device = torch.device(self.device) if self.device is not None else _default_device()
        if fall_back and not _is_available(self._device):
            self._device = _default_device()
        for name, options in (("neighbourhood", _NEIGHBOURHOODS), ("grid_metric", _GRID_METRICS)):
            if getattr(self, name) not in options:
                raise ValueError(f"{name} must be one of {options}, got {getattr(self, name)!r}")
        self._sigmas = self._widths()
        self._coords = torch.cartesian_prod(torch.arange(self.n_rows), torch.arange(self.n_cols))
        self._coords = self._coords.to(self._device, torch.float32)

    def _as_tensor(self, X: ArrayLike | torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            X = X.detach()
        elif hasattr(X, "__dlpack__") and not isinstance(X, np.ndarray):  # e.g. CuPy, without leaving the GPU
            try:
                X = torch.from_dlpack(X)
            except RuntimeError:  # torch cannot import negative strides
                X = torch.from_dlpack(cast(Any, X).copy())
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

    def _kernel(self, n_features: int, top2: bool = False, half: bool = False) -> str | None:
        """Choose the fused kernel of the BMU search.

        The fused kernel of the BMU search, on CUDA "tensor-core" or "float32", on MPS "metal", or None for the
        torch path; with half (fit under precision="float16") first "metal-float16" for dot-product scores from
        _HALF_MIN_FEATURES features. The tensor-core kernel searches dot-product scores on the GPUs of _FAST_TF32,
        for over 64 nodes (a tile of 128 is half empty otherwise) and n_nodes x n_features >= 8192 (below, the float32
        kernel is as fast or faster on an A100); a kernel that fails its self-check leaves the next one.
        """
        if not self.fused:
            return None
        n_nodes, manhattan = self.n_rows * self.n_cols, self.metric == "manhattan"
        half = half and not top2 and not manhattan and n_features >= _HALF_MIN_FEATURES
        kernels = []
        metal = _metal(self._device)
        if metal is not None:
            kernels += ["metal-float16"] if half else []
            kernels += ["metal"] if metal.supported(n_features, manhattan) else []
        elif _triton(self._device) is not None:
            if not manhattan and n_nodes > 64 and n_nodes * n_features >= 8192:
                fast_tf32 = torch.cuda.get_device_capability(self._device) in _FAST_TF32
                kernels += ["tensor-core"] if fast_tf32 else []
            kernels += ["float32"] if manhattan or n_features <= _FUSED_MAX_FEATURES else []
        return next((k for k in kernels if _works(self._device, k, n_features, self.metric, top2)), None)

    def _batch_size_for(self, data: torch.Tensor, kernel: str | None) -> int:
        """kernel: the fused kernel of the BMU search, if any, which holds no scores."""
        if self.batch_size is not None:
            return self.batch_size
        if self._device.type == "cuda":
            cached = torch.cuda.memory_reserved(self._device) - torch.cuda.memory_allocated(self._device)
            free, fraction = torch.cuda.mem_get_info(self._device)[0] + cached, 0.9
        elif self._device.type == "mps" and hasattr(torch.mps, "recommended_max_memory"):  # torch >= 2.5
            free, fraction = torch.mps.recommended_max_memory() - torch.mps.current_allocated_memory(), 0.5
        else:
            return 8192 if self._device.type == "cpu" else 1 << 16
        fraction = fraction if self.memory_fraction is None else self.memory_fraction
        n_features = data.shape[1]
        if kernel is not None:
            # no scores in memory, only up to about 4 copies of the batch (transposed or as two TF32 parts, centred,
            # with the samples without a BMU zeroed, their BMUs' weights), the TF32 parts' features padded to a
            # multiple of 8 or 16, which for few features takes several times their memory; on MPS 4 copies and the
            # one that Metal's bmu_staged pads with the bias column to a multiple of 16 features. Speed levels off long
            # before 2^26 values
            if kernel == "tensor-core":
                per_sample = 16 * (-(-n_features // 16) * 16) + 32
            elif kernel in _METAL:
                per_sample = 4 * (4 * n_features + _module(self._device, kernel).sample_width(n_features)) + 32
            else:
                per_sample = 16 * n_features + 32
            return max(1, min((1 << 26) // n_features, int(fraction * free) // per_sample))
        n_nodes = self.n_rows * self.n_cols
        # BMU search holds at most about 2 floats per sample and node, plus copies of the batch
        return max(1, min((1 << 28) // n_nodes, int(fraction * free) // (8 * (n_nodes + n_features))))

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
            values = rng.rand(self.n_rows, self.n_cols, data.shape[1]) * 2 - 1
            values /= np.linalg.norm(values, axis=-1, keepdims=True)
            return torch.as_tensor(values.reshape(n_nodes, -1), dtype=torch.float32, device=self._device)
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

    def _operands(self, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Nodes v, bias b and centre c with score (x - c).v + b for all metrics but manhattan (None for no b or c).

        The score is ||w - c||^2 - 2 (x - c).(w - c) with c the mean node (euclidean, ||x - c||^2 dropped; centring
        keeps a large offset from cancelling) or -x.w / ||w|| of the (for correlation centred) vectors (cosine,
        correlation). The small weight matrix is scaled and negated, not the batch x nodes scores.
        """
        if self.metric == "euclidean":
            centre = weights.mean(0)
            weights = weights - centre
            return -2 * weights, (weights**2).sum(1), centre
        weights = self._centre(weights)
        return weights / -weights.norm(dim=1, keepdim=True).clamp_min(1e-12), None, None

    def _scores(self, batch: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Per-node score to minimise: the L1 distance (manhattan), else as in _operands."""
        if self.metric == "manhattan":
            return _cdist(batch, weights, 1)
        nodes, bias, centre = self._operands(weights)
        if bias is None:
            return self._centre(batch) @ nodes.T
        assert centre is not None  # euclidean has both
        if batch.device.type == "mps":
            # MPS's addmm expands the bias to a second batch x nodes matrix, and adding it after the matmul is a
            # further pass over the scores, which is slower at the small batches of large maps; a column of ones in
            # the batch and the node norms in the weights make the bias part of a single matmul. On CUDA addmm fuses
            # the bias into the matmul (cuBLASLt epilogue), so it stays there
            ones = batch.new_ones(len(batch), 1)
            return torch.cat([batch - centre, ones], 1) @ torch.cat([nodes, bias[:, None]], 1).T
        return torch.addmm(bias, batch - centre, nodes.T)

    def _search(
        self, weights: torch.Tensor, kernel: str | None, top2: bool = False, magnitude: torch.Tensor | None = None
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Build the BMU search of a batch.

        Returns a function of a batch giving each sample's BMU, or with top2 its best two nodes, shape (batch, 2), by
        the given fused kernel (from _kernel), without the batch x nodes scores, or the torch path. magnitude: for
        "metal-float16", the largest |x| per feature of the samples to come, which bounds what the kernel scales.
        """
        if kernel is None:
            if top2:
                return lambda batch: self._scores(batch, weights).topk(2, 1, largest=False).indices
            return lambda batch: self._scores(batch, weights).argmin(1)
        manhattan = self.metric == "manhattan"
        nodes, bias, centre = (weights, None, None) if manhattan else self._operands(weights)
        if kernel == "metal-float16":
            assert magnitude is not None  # fit passes it under precision="float16"
            if self.metric == "cosine":
                # centred on the mean node, with c . v as the bias, so that half rounds the smaller x - c (on clustered
                # 0-1 pixels half as many near-ties flip as of the uncentred scores)
                centre = weights.mean(0)
                bias = nodes @ centre
            if self.metric == "correlation":  # rows centred twice (see _centre), each within twice the largest |x|
                bound = (4 * magnitude.amax()).expand_as(magnitude)
            else:
                assert centre is not None  # set for euclidean and, above, for cosine
                bound = magnitude + centre.abs()
            search = _module(self._device, kernel).search(nodes, bias, centre, manhattan, top2, bound=bound)
        elif kernel == "metal":
            search = _module(self._device, kernel).search(nodes, bias, centre, manhattan, top2)
        else:
            search = _module(self._device, kernel).search(nodes, bias, centre, manhattan, kernel == "tensor-core", top2)
        return lambda batch: search(self._centre(batch))

    def _score_distances(self, batch: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Distances from the scores of any number of nodes per sample, for every metric but euclidean."""
        if self.metric == "manhattan":
            return scores
        # in place, as transform's scores hold a batch x nodes matrix; callers pass scores they no longer need
        return scores.div_(self._centre(batch).norm(dim=1, keepdim=True)).add_(1).clamp_(0, 2)

    def _bmus(
        self, data: torch.Tensor, weights: torch.Tensor, return_distance: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """BMU and, if return_distance, distance to it per sample; -1 and NaN for samples without a BMU."""
        bmus, distances = [], []
        kernel = self._kernel(data.shape[1])
        search = self._search(weights, kernel)
        nodes = self._operands(weights)[0] if return_distance and self.metric in ("cosine", "correlation") else None
        for batch in data.split(self._batch_size_for(data, kernel)):
            bmu = search(batch)
            bmus.append(bmu)
            if not return_distance:
                continue
            # from the BMUs' weights, not the search's scores: for euclidean exact, so large offsets do not cancel, and
            # in float32 whichever kernel found the BMU
            if self.metric == "euclidean":
                distances.append((batch - weights[bmu]).norm(dim=1))
            elif self.metric == "manhattan":
                distances.append((batch - weights[bmu]).abs().sum(1))
            else:
                assert nodes is not None
                score = (self._centre(batch) * nodes[bmu]).sum(1, keepdim=True)
                distances.append(self._score_distances(batch, score)[:, 0])
        bmu = torch.cat(bmus)
        distance = torch.cat(distances) if return_distance else None
        has_bmu = self._has_bmu(data)
        if has_bmu is not None:
            bmu[~has_bmu] = -1
            if distance is not None:
                distance[~has_bmu] = torch.nan
        return bmu, distance

    def _widths(self) -> tuple[float, ...]:
        """Validates the width parameters and returns the neighbourhood width of every epoch."""
        if not isinstance(self.sigma_schedule, str):  # the widths themselves
            if self.sigma_initial is not None or self.sigma_final is not None:
                raise ValueError("sigma_initial and sigma_final must be None when sigma_schedule gives the widths")
            sigmas = np.asarray(self.sigma_schedule, dtype=np.float64)
            if sigmas.shape != (self.n_epochs,) or not (np.isfinite(sigmas) & (sigmas >= 0)).all():
                raise ValueError(
                    f"sigma_schedule must be one of {_SCHEDULES} or n_epochs = {self.n_epochs} finite widths >= 0, "
                    f"got {self.sigma_schedule!r}"
                )
            return tuple(sigmas.tolist())
        if self.sigma_schedule not in _SCHEDULES:
            raise ValueError(
                f"sigma_schedule must be one of {_SCHEDULES} or a sequence of widths, got {self.sigma_schedule!r}"
            )
        if isinstance(self.sigma_initial, str):
            if self.sigma_initial != "flowsom":
                raise ValueError(f"sigma_initial must be a number, None or 'flowsom', got {self.sigma_initial!r}")
            sigma0 = _grid_quantile(self.n_rows, self.n_cols, self.grid_metric, 0.67)
        elif self.sigma_initial is None:
            sigma0 = ceil(min(self.n_rows, self.n_cols) / 2) / 2
        else:
            sigma0 = self.sigma_initial
        sigma1 = min(0.1 * sigma0, 1.0) if self.sigma_final is None else self.sigma_final
        if not all(_is_real(sigma) and isfinite(sigma) and sigma >= 0 for sigma in (sigma0, sigma1)):
            raise ValueError(f"sigma_initial and sigma_final must be finite and >= 0, got {sigma0!r} and {sigma1!r}")
        if self.sigma_schedule != "linear" and not (sigma0 > 0 and sigma1 > 0):
            raise ValueError(
                f"sigma_schedule={self.sigma_schedule!r} needs sigma_initial and sigma_final > 0, got {sigma0!r} and "
                f"{sigma1!r}; sigma_schedule='linear' allows 0"
            )
        n_epochs = self.n_epochs
        if self.sigma_schedule == "linear":
            return tuple(sigma0 + (sigma1 - sigma0) * t / n_epochs for t in range(n_epochs))
        if self.sigma_schedule == "inverse":
            return tuple(sigma0 / (1 + (sigma0 / sigma1 - 1) * t / n_epochs) for t in range(n_epochs))
        return tuple(sigma0 * (sigma1 / sigma0) ** (t / n_epochs) for t in range(n_epochs))

    def _smoother(self, sigma: float) -> Callable[[torch.Tensor], torch.Tensor]:
        """Build the product H @ values with the neighbourhood matrix of width sigma, H[j, k] = h(d(r_j, r_k); sigma).

        Below a width of 0.05 grid units, H is the identity in float32 for every neighbourhood (the gaussian's
        exp(-0.5 / sigma^2) underflows), and values are returned as they are. The gaussian under the euclidean grid
        metric and the bubble under chebyshev factor into the grid axes, H = H_rows (x) H_cols, and take one small
        matrix product along each. Every other neighbourhood takes _convolve. Whatever goes to the device is made
        here, before fit queues the epoch's BMU search, so that copying it does not wait for the search.
        """
        if sigma < _IDENTITY_SIGMA:
            return lambda values: values
        if (self.neighbourhood, self.grid_metric) == ("gaussian", "euclidean"):

            def factor(n: int) -> torch.Tensor:
                r = torch.arange(n, dtype=torch.float32, device=self._device)
                return (r[:, None] - r).square_().mul_(-0.5 / sigma**2).exp_()

        elif (self.neighbourhood, self.grid_metric) == ("bubble", "chebyshev"):

            def factor(n: int) -> torch.Tensor:  # compared in float64, as _neighbourhood does
                r = np.arange(n)
                return torch.as_tensor(np.abs(r[:, None] - r) <= sigma, dtype=torch.float32, device=self._device)

        else:
            return self._convolve(sigma)
        rows, cols = factor(self.n_rows), factor(self.n_cols)

        def smooth(values: torch.Tensor) -> torch.Tensor:
            grid = (rows @ values.view(self.n_rows, -1)).view(self.n_rows, self.n_cols, -1)
            return (cols @ grid).view(len(values), -1)

        return smooth

    def _convolve(self, sigma: float) -> Callable[[torch.Tensor], torch.Tensor]:
        """Build H @ values for any neighbourhood, as a sum of Toeplitz matrix products along one grid axis.

        H depends only on the offsets of the nodes on the grid, H[(i, c), (i', c')] = K[i - i', c - c'], so with
        the grid as a (rows, cols, n) array V, (H V)[:, c] = sum over offsets e of T_e V[:, c - e], T_e the
        n_rows x n_rows Toeplitz matrix of K[:, e]: one matrix product per column offset (per row offset if there
        are fewer rows, on the transposed grid), skipping the offsets whose weights are all 0, such as those beyond a
        bubble's radius. H itself is never held. K is evaluated in float64 on the CPU, so that the cut-offs compare
        exactly and every device gets the same weights. This takes up to n_nodes^2 x n_features multiply-adds per
        epoch, fewer with a compact neighbourhood, compared to the BMU search's n_samples x n_nodes x n_features.
        """
        n_rows, n_cols = self.n_rows, self.n_cols
        dr, dc = np.arange(1 - n_rows, n_rows), np.arange(1 - n_cols, n_cols)
        table = _neighbourhood(_grid_distance(dr[:, None], dc[None], self.grid_metric), sigma, self.neighbourhood)
        transpose = n_cols > n_rows  # loop over the offsets of the shorter axis, with fewer, larger products
        kernel = torch.as_tensor(table.T if transpose else table, dtype=torch.float32)
        n_long, n_short = kernel.shape[0] // 2 + 1, kernel.shape[1] // 2 + 1
        offsets = [e for e in range(1 - n_short, n_short) if kernel[:, e + n_short - 1].any()]
        kernel = kernel.to(self._device)
        r = torch.arange(n_long, device=self._device)
        toeplitz = r[:, None] - r + n_long - 1  # T_e = kernel[toeplitz, e + n_short - 1]

        def smooth(values: torch.Tensor) -> torch.Tensor:
            width = values.shape[1]
            if transpose:
                grid = values.view(n_rows, n_cols, width).transpose(0, 1).reshape(n_cols, n_rows * width)
            else:
                grid = values.view(n_rows, n_cols * width)
            out = torch.zeros_like(grid)
            for e in offsets:
                lo, hi = max(0, e), n_short + min(0, e)  # output positions c along the shorter axis, reading c - e
                source = grid[:, (lo - e) * width : (hi - e) * width]
                out[:, lo * width : hi * width].addmm_(kernel[toeplitz, e + n_short - 1], source)
            if transpose:
                return out.view(n_cols, n_rows, width).transpose(0, 1).reshape(len(values), width)
            return out.view(len(values), width)

        return smooth

    def fit(self, X: ArrayLike | torch.Tensor, y: object = None) -> "SOM":
        """Fits the map, starting from the initialisation given by init.

        Parameters
        ----------
        X
            Samples, shape (n_samples, n_features) or (..., n_features); numpy array, array-like, torch tensor
            or DLPack array such as CuPy. Any real values are fine; data is copied to the device once and stays
            there.
        y
            Ignored.

        Returns
        -------
        SOM
            The fitted map.
        """
        self._setup()
        for attr in ("weights_", "labels_", "n_features_in_"):
            self.__dict__.pop(attr, None)
        data = self._check_X(X)
        weights = self._init_weights(data)
        deterministic = self._device.type != "cpu" and (
            self.deterministic or torch.are_deterministic_algorithms_enabled()
        )
        magnitude, half = None, self.precision == "float16"
        metal = _metal(self._device) if half else None  # only then, as fused=False must not compile the kernels
        if deterministic or (half and metal is not None):  # the Metal half kernels' scales need it
            magnitude = torch.maximum(data.amax(0), -data.amin(0))  # without a copy of data
        if half and metal is not None:
            assert magnitude is not None
            # the nodes stay among the samples and the initial weights, so these bound all that the half kernels scale
            half = metal.in_half_range(torch.maximum(magnitude, weights.abs().amax(0)))
        kernel = self._kernel(data.shape[1], half=half)
        mps = self._device.type == "mps"
        # deterministic sums add int64 fixed point, which unlike index_add_'s atomic float additions gives the same
        # result in any order: on CUDA with index_add_, on MPS with the Metal kernels' pairs of 32-bit atomic additions
        # (MPS without them, e.g. with fused=False, sums with a one-hot matrix product instead, in a fixed order). Each
        # feature is scaled by a power of 2 that leaves room for n_samples values, which alone would round every value
        # to about n_samples * 2^-62 of the feature's largest magnitude (a node of only small values could sum to 0); a
        # second word adds the rounding remainders, exact in float64, scaled by 2^(62 - log2(n_samples)), so that values
        # are lost only below about n_samples^2 * 2^-124 of the largest
        fixed_point = deterministic and (not mps or kernel in _METAL)
        n_nodes, batch_size = weights.shape[0], self._batch_size_for(data, kernel)
        if fixed_point:
            assert magnitude is not None
            low_shift = 62 - len(data).bit_length()  # remainders within +-0.5, so their sums below 2^61
            # frexp is not implemented on MPS, and n_features values are quick to do on the CPU
            _, exponent = torch.frexp(magnitude.cpu() if mps else magnitude)
            if mps:  # the shifts, and 2^exponent as two float32 factors, which MPS's lack of float64 calls for
                shift = (low_shift - exponent).int().to(self._device)
                split = exponent // 2
                factors = [torch.ldexp(torch.ones(len(split)), e).to(self._device) for e in (split, exponent - split)]
            else:
                scale = data.new_ones(data.shape[1], dtype=torch.float64).ldexp_(low_shift - exponent)
                fixed_rows = max(1, (1 << 21) // data.shape[1])
        for t in tqdm(range(self.n_epochs), desc="Fitting SOM", unit="epoch", disable=not self.verbose):
            smooth = self._smoother(self._sigmas[t])
            sums = torch.zeros_like(weights, dtype=torch.long if fixed_point else None)
            # (empty without fixed point)
            lows = torch.zeros_like(sums) if fixed_point else sums.new_empty(0)
            # int32 on MPS, whose index_add_ in torch 2.8 takes no int64 source
            counts = torch.zeros(n_nodes, dtype=torch.int32 if mps else torch.long, device=weights.device)
            search = self._search(weights, kernel, magnitude=magnitude)
            for batch in data.split(batch_size):
                bmus = search(batch)
                keep = self._has_bmu(batch)
                if keep is not None:  # samples without a BMU add zeros (dropping them waits for the GPU to count them)
                    batch = batch * keep[:, None]
                if fixed_point and mps:
                    _module(self._device, "metal").fixed_sums(batch, bmus, shift, low_shift, sums, lows)
                elif fixed_point:  # in parts of 2^21 values, as the float64 and int64 copies take 6x the memory
                    for rows, part in zip(bmus.split(fixed_rows), batch.split(fixed_rows), strict=True):
                        value = part.double().mul_(scale)
                        high = value.round()
                        sums.index_add_(0, rows, high.long())
                        lows.index_add_(0, rows, value.sub_(high).mul_(2.0**low_shift).round_().long())
                        del value, high  # before the next part's copies
                elif deterministic:  # on MPS without the Metal kernels a matrix product, which sums in a fixed order
                    sums += batch.new_zeros(n_nodes, len(bmus)).scatter_(0, bmus[None], 1.0) @ batch
                else:
                    sums.index_add_(0, bmus, batch)
                counts.index_add_(
                    0, bmus, torch.ones_like(bmus, dtype=counts.dtype) if keep is None else keep.to(counts.dtype)
                )
            if fixed_point and mps:
                sums = (sums.float() + lows.float() * 2.0**-low_shift) * 2.0**-low_shift * factors[0] * factors[1]
            elif fixed_point:
                sums = (sums / scale + lows / (scale * 2.0**low_shift)).float()
            smoothed = smooth(torch.cat([sums, counts[:, None].float()], 1))
            numerator, denominator = smoothed[:, :-1], smoothed[:, -1:]
            weights = torch.where(denominator > 0, numerator / denominator, weights)
            if self.verbose:  # the epoch's work is only queued on GPUs, so the bar would run ahead without waiting
                _synchronize(self._device)
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

        Parameters
        ----------
        X
            Samples, shape (n_samples, n_features) or (..., n_features).
        return_distance
            Whether to also return each sample's distance (float32) to its BMU in the metric, NaN
            for samples without a BMU. It comes from the same pass as the BMU search, and its mean over the
            samples with a BMU is quantization_error(X).

        Returns
        -------
        numpy.ndarray or tuple of numpy.ndarray
            The BMU indices, or the tuple (BMU indices, distances) if return_distance.
        """
        self._check_fitted()
        bmus, distances = self._bmus(self._check_X(X), self._weights, return_distance)
        if distances is None:
            return bmus.cpu().numpy()
        return bmus.cpu().numpy(), distances.cpu().numpy()

    def fit_predict(self, X: ArrayLike | torch.Tensor, y: object = None) -> np.ndarray:
        """Fits the map and returns ``labels_``."""
        return self.fit(X).labels_

    def _distances(self, batch: torch.Tensor) -> torch.Tensor:
        if self.metric == "euclidean":  # exact, unlike the BMU search, so large offsets do not cancel
            return _cdist(batch, self._weights, 2)
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
        batch_size = self._batch_size_for(data, None)
        # on the CPU, as it can exceed device memory
        distances = torch.empty(len(data), len(self._weights), dtype=torch.float32, device="cpu")
        for batch, rows in zip(data.split(batch_size), distances.split(batch_size), strict=True):
            rows.copy_(self._distances(batch))
        return distances.numpy()

    def fit_transform(self, X: ArrayLike | torch.Tensor, y: object = None) -> np.ndarray:
        """Fits the map and returns transform(X)."""
        return self.fit(X).transform(X)

    def quantization_error(self, X: ArrayLike | torch.Tensor) -> float:
        """Mean distance (in the metric) between each sample with a BMU and its BMU."""
        self._check_fitted()
        bmus, distances = self._bmus(self._check_X(X), self._weights)
        assert distances is not None
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
        kernel = self._kernel(data.shape[1], top2=True)
        search = self._search(self._weights, kernel, top2=True)
        nodes = torch.cat([search(batch) for batch in data.split(self._batch_size_for(data, kernel))])
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
