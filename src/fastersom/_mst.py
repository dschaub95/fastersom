"""Exact minimum spanning trees of points (such as a map's nodes) by Borůvka's algorithm.

Each round finds, for every point, its nearest point in another component: a search of blocks of rows against all
points, with the pairs within a component masked, as the BMU search of a map whose nodes are the points. Every
component keeps the lightest edge of its rows under the strict order (weight, lower index, higher index), and the
components merge along the kept edges. Every component merges in every round, so there are at most ceil(log2 n)
rounds, and a row whose nearest point is still in another component keeps it without a new search. The memory is that
of the points and of one block of rows x points scores; no n x n matrix is formed.

Exactness. Edges are weighed by one function W of the points in float64, the squared euclidean (or L1) distance summed
feature by feature in order, which is symmetric and the same on every device. The search scores in float32 on GPUs and
in float64 on the CPU and keeps each row's best two points. A row whose two scores are not separated by more than a
rigorous bound of the scores' rounding error (and of W's) is searched again in float64: among the points whose float64
scores lie within that bound of the best, the nearest by (W, index). Every row's choice is then its nearest point
under (W, index), and the tree is the unique minimum spanning tree of the points under the strict order of W, whatever
the device, batch size or number of threads. Cosine and correlation distances are monotone in the euclidean distance
of the rows scaled to unit length (and centred over the features), whose tree they share.

References
----------
    O. Borůvka, O jistém problému minimálním, Práce moravské přírodovědecké společnosti 3, 1926.
    N. J. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM, 2002 (the error bounds gamma_n).
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import ArrayLike

from ._som import SOM, _cdist, _is_int

_METRICS = ("euclidean", "manhattan", "cosine", "correlation")
_F64 = 2.0**-53  # unit roundoff of float64
# rows x points scores per block on the CPU (128 MB in float64), beyond which the blocks gain no speed
_CPU_SCORES = 1 << 24
# pairs whose W adds up at once, and rows x points per chunk of the tree's distances (each 8 bytes and a few copies)
_PAIRS = 1 << 22


def _gamma(n: int, u: float) -> float:
    """Higham's gamma_n = n u / (1 - n u), the relative error bound of n roundings of unit roundoff u.

    inf where n u >= 1 (from 254 features at bfloat16's u, 2046 at TF32's), where no bound holds: nothing is certified.
    """
    return n * u / (1 - n * u) if n * u < 1 else np.inf


def _unit_roundoff(dtype: torch.dtype, device: torch.device, matmul: bool) -> float:
    """Unit roundoff of a search's scores: float32's, or TF32's or bfloat16's where torch may use them for matmul."""
    if dtype == torch.float64:
        return _F64
    if not matmul:
        return 2.0**-24
    try:
        precision = torch.get_float32_matmul_precision()
    except RuntimeError:  # the per-backend fp32_precision settings are in use: the device's, else the global one
        backend = torch.backends.cuda.matmul if device.type == "cuda" else None
        setting = getattr(backend, "fp32_precision", "none")
        if setting == "none":
            setting = getattr(torch.backends, "fp32_precision", "none")
        return {"ieee": 2.0**-24, "tf32": 2.0**-11}.get(setting, 2.0**-8)  # unknown: as coarse as bfloat16
    if device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32 and precision == "highest":
        precision = "high"
    return {"highest": 2.0**-24, "high": 2.0**-11}.get(precision, 2.0**-8)


def _weights(QT: np.ndarray, a: np.ndarray, b: np.ndarray, manhattan: bool) -> np.ndarray:
    """W of the pairs (a, b): the squared euclidean or the L1 distance in float64, summed over the features in order.

    QT holds the points' features as rows (n_features, n_points). Every operation rounds once, as IEEE prescribes, so
    W is symmetric and depends on nothing but the points.
    """
    out = np.empty(len(a))
    for start in range(0, len(a), _PAIRS):
        rows, cols = a[start : start + _PAIRS], b[start : start + _PAIRS]
        total = np.zeros(len(rows))
        for feature in QT:
            difference = feature[rows] - feature[cols]
            total += np.abs(difference) if manhattan else difference * difference
        out[start : start + _PAIRS] = total
    return out


class _TorchSearch:
    """Masked nearest-point search of the torch path, by blocks of rows x all points.

    nearest() gives each row's nearest point outside its component and whether that choice is certified: whether no
    other point can be as near by W. A fused kernel can take its place through the same interface, given its own
    bound. Euclidean scores are ||y_j||^2 - 2 y_i . y_j of the points y centred on their coordinate-wise median; with
    u the unit roundoff of the scores, r_i = ||y_i||, R = max_j r_j, M_i = r_i + R and d features, they lie within

        e_i = gamma_(d+2) M_i^2 + delta_i (2 M_i + delta_i) + gamma_(d+2)(float64) (M_i + delta_i)^2

    of W_ij - ||y_i||^2: the matrix product's rounding and the bias ||y_j||^2's (bounded by gamma_(d+2) M_i^2, as
    M_i^2 >= 2 r_i R + R^2), the points' rounding to the search's dtype and their centring (delta_i = 2 u (M_i +
    ||p_i|| + max_j ||p_j||) bounds how much that moves a distance), and W's own rounding, plus (d + 2) times the
    smallest normal number for underflow; a further gamma_(d+2) R^2 covers the addition of the bias to the product.
    L1 scores lie within a factor 1 +- g of W_ij, g = gamma_(d+1) + gamma_(d+1)(float64), as differences and sums of
    non-negative values do not underflow. Both bounds are doubled for the rounding of the bounds and of the test itself.

    A few points far from the others (outliers, or a long tail of intensities) would set R and loosen every row's bound,
    so each row's bound uses instead of R a radius R_i >= 2 r_i fitted to its best score b: no point with r_k > R_i
    can be nearer to row i than its best point j, because ||y_i - y_k|| >= r_k - r_i (less the centring's rounding)
    exceeds the largest W_ij that b admits; e_i then holds for every point that can be the nearest. Where r_j > R_i or
    the radius does not exclude the far points, the row takes the bound with R. Non-finite best scores (float32
    overflow) are never certified, nor an infinite second score unless a single point lies outside the component.
    """

    def __init__(self, points: torch.Tensor, manhattan: bool, block: int):
        self.points, self.manhattan, self.block, self.device = points, manhattan, block, points.device
        u = _unit_roundoff(points.dtype, points.device, matmul=not manhattan)
        n_features = points.shape[1]
        if manhattan:
            self.relative = 2 * (_gamma(n_features + 1, u) + _gamma(n_features + 1, _F64))
            return
        # centred on the coordinate-wise median, which far points do not move (any centre gives the same W)
        centre = points.cpu().median(0).values.to(points.device)
        centred = points - centre
        self.nodes, self.bias, self.centred = -2 * centred, (centred * centred).sum(1), centred
        self.norms = centred.cpu().double().norm(dim=1).numpy()  # on the CPU, as MPS has no float64
        self.lengths = points.cpu().double().norm(dim=1).numpy()
        self.u, self.n_features, self.tiny = u, n_features, torch.finfo(points.dtype).tiny
        self.centre_norm = float(centre.cpu().double().norm())
        rows = np.arange(len(points))
        self.error = self._bound(rows, np.full(len(points), self.norms.max()), np.full(len(points), self.lengths.max()))
        if self.device.type == "mps":  # the bias as a column of the nodes, for one matrix product as in SOM._scorer
            self.operand = torch.cat([self.nodes, self.bias[:, None]], 1).T

    def _bound(self, rows: np.ndarray, radius: np.ndarray, length: np.ndarray) -> np.ndarray:
        """2 e_i of the rows for points within radius of the centre (r_j <= radius) and of length at most length."""
        u, d = self.u, self.n_features
        reach = self.norms[rows] + radius
        delta = 2 * u * (reach + self.lengths[rows] + length)
        error = _gamma(d + 2, u) * (reach**2 + radius**2) + delta * (2 * reach + delta)
        error += _gamma(d + 2, _F64) * (reach + delta) ** 2 + (d + 2) * self.tiny
        return 2 * error

    def errors(self, rows: np.ndarray, best: np.ndarray, index: np.ndarray) -> np.ndarray:
        """Each row's bound 2 e_i given its best score and point (see the class): with its own radius where valid."""
        u, r = self.u, self.norms[rows]
        with np.errstate(invalid="ignore", over="ignore"):
            radius = np.minimum(self.norms.max(), np.maximum(2 * r, r + 2 * np.sqrt(np.maximum(best + r * r, 0))))
            # the points' lengths ||p|| <= r + ||mean|| up to the centring's rounding
            length = np.minimum(self.lengths.max(), radius + self.centre_norm + 4 * u * (self.lengths.max() + 1))
            error = self._bound(rows, radius, length)
            # the smallest distance of a point beyond the radius, less the centring's rounding of both points
            gap = radius * (1 - 2 * u) - r - 4 * u * (r + 3 * self.centre_norm + self.lengths.max() * u)
            valid = (self.norms[index] <= radius) & (gap > 0)
            valid &= (best + r * r + error) * (1 + _gamma(self.n_features + 2, _F64)) < gap * gap
        return np.where(valid & np.isfinite(error), error, self.error[rows])

    def scores(self, rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Scores of the rows against all points, inf for the points of the row's own component."""
        if self.manhattan:
            scores = _cdist(self.points[rows], self.points, 1)
        elif self.device.type == "mps":
            batch = self.centred[rows]
            scores = torch.cat([batch, batch.new_ones(len(batch), 1)], 1) @ self.operand
        else:
            scores = torch.addmm(self.bias, self.centred[rows], self.nodes.T)
        return scores.masked_fill_(labels[rows, None] == labels, torch.inf)

    def nearest(self, rows: np.ndarray, labels: torch.Tensor, certify: bool) -> tuple[np.ndarray, np.ndarray | None]:
        """Each row's nearest point outside its component, and with certify whether it is certainly the nearest by W."""
        nearest, certified = np.empty(len(rows), np.int64), np.empty(len(rows), bool) if certify else None
        host_labels = labels.cpu().numpy() if certify else None
        for start in range(0, len(rows), self.block):
            part = rows[start : start + self.block]
            scores = self.scores(torch.as_tensor(part, device=self.device), labels)
            if certify:
                values, indices = scores.topk(2, 1, largest=False)
                best, second = values.cpu().double().numpy().T
                first = indices[:, 0].cpu().numpy()
                nearest[start : start + len(part)] = first
                assert certified is not None
                assert host_labels is not None
                # a second score of inf is certain only where a single point lies outside the row's component (not
                # after an overflow), and a non-finite best never
                outside = len(self.points) - np.bincount(host_labels, minlength=len(self.points))[host_labels[part]]
                finite = np.isfinite(best) & (np.isfinite(second) | ((second == np.inf) & (outside == 1)))
                certified[start : start + len(part)] = finite & self._separated(part, best, second, first)
            else:
                nearest[start : start + len(part)] = scores.argmin(1).cpu().numpy()
            del scores  # before the next block's
        return nearest, certified

    def _separated(self, rows: np.ndarray, best: np.ndarray, second: np.ndarray, first: np.ndarray) -> np.ndarray:
        """Whether the second best score exceeds the best by more than the scores' error."""
        with np.errstate(invalid="ignore"):
            if self.manhattan:
                return second * (1 - self.relative) > best * (1 + self.relative)
            return second - best > 2 * self.errors(rows, best, first)

    def candidates(self, rows: np.ndarray, labels: torch.Tensor) -> torch.Tensor:
        """Mask (rows x points) of the points that can be nearest to each row by W, for a float64 search."""
        scores = self.scores(torch.as_tensor(rows, device=self.device), labels)
        best, first = scores.min(1, keepdim=True)
        if self.manhattan:
            return scores * (1 - self.relative) <= best * (1 + self.relative)
        error = self.errors(rows, best[:, 0].cpu().double().numpy(), first[:, 0].cpu().numpy())
        return scores <= best + 2 * torch.as_tensor(error, dtype=scores.dtype, device=self.device)[:, None]


class _Search(Protocol):
    """A masked nearest-point search as _TorchSearch.nearest, which a fused kernel's can replace."""

    device: torch.device

    def nearest(
        self, rows: np.ndarray, labels: torch.Tensor, certify: bool
    ) -> tuple[np.ndarray, np.ndarray | None]: ...


def _search_dtype(device: torch.device, exact: bool) -> torch.dtype:
    """float64 on the CPU when exact, whose BLAS is half as fast in it but needs almost no rechecks; else float32."""
    return torch.float64 if exact and device.type == "cpu" else torch.float32


def _block(engine: SOM, points: torch.Tensor) -> int:
    """Rows per block of the search.

    batch_size if given, else as many as keep the scores at about 2^24 on the CPU, and on GPUs as many as the torch
    path's BMU search takes (2^28 / n_points, or what fits in memory_fraction of the free memory).
    """
    if engine.batch_size is not None or engine._device.type != "cpu":
        return engine._batch_size_for(points, None)
    return max(1, _CPU_SCORES // len(points))


class _Recheck:
    """The nearest point by (W, index) of rows whose search was not certified, from float64 scores and W."""

    def __init__(self, engine: SOM, Q: torch.Tensor, QT: np.ndarray, block: int):
        # on the device in float64 (half as many rows as the float32 search's), but on the CPU instead of MPS, which
        # has no float64
        device = torch.device("cpu") if engine._device.type == "mps" else engine._device
        if device.type != "cpu":
            block = max(1, block // 2)
        elif engine._device.type != "cpu" and engine.batch_size is None:
            block = max(1, _CPU_SCORES // len(Q))
        manhattan = engine.metric == "manhattan"
        self.search = _TorchSearch(Q.to(device), manhattan, block)
        self.QT, self.manhattan = QT, manhattan

    def __call__(self, rows: np.ndarray, labels: np.ndarray) -> np.ndarray:
        nearest = np.empty(len(rows), np.int64)
        device_labels = torch.as_tensor(labels, device=self.search.device)
        for start in range(0, len(rows), self.search.block):
            part = rows[start : start + self.search.block]
            within = self.search.candidates(part, device_labels)
            counts = within.sum(1).cpu().numpy()
            # at most _PAIRS candidate pairs at once (many near-ties, as of a lattice, would make the list large)
            ends = np.r_[0, np.cumsum(counts)]
            first_row = 0
            while first_row < len(part):
                last = max(first_row + 1, int(np.searchsorted(ends, ends[first_row] + _PAIRS, side="right")) - 1)
                hit_rows, hit_columns = within[first_row:last].nonzero(as_tuple=True)
                position, point = hit_rows.cpu().numpy() + first_row, hit_columns.cpu().numpy()
                weights = _weights(self.QT, part[position], point, self.manhattan)
                order = np.lexsort((point, weights, position))  # per row, the least W, then the lowest index
                first = order[np.r_[True, position[order[1:]] != position[order[:-1]]]]
                nearest[start + position[first]] = point[first]
                first_row = last
            del within
        return nearest


def _find(parent: np.ndarray, node: int) -> int:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def _boruvka(
    n: int, search: _Search, recheck: _Recheck | None, QT: np.ndarray, manhattan: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Edges (n - 1, 2) and their W, sorted by (W, lower index, higher index).

    Components are labelled by their smallest point. Each component's edge is the least of its rows' by the strict
    order; with certified rows these edges form a forest, and without certification (recheck None) the merge in the
    strict order skips those that would close a cycle.
    """
    labels, parent, points = np.arange(n), np.arange(n), np.arange(n)
    nearest, weight = np.zeros(n, np.int64), np.zeros(n)
    rows = points
    found: list[tuple[int, int, float]] = []
    while len(found) < n - 1:
        device_labels = torch.as_tensor(labels, device=search.device)
        choice, certified = search.nearest(rows, device_labels, certify=recheck is not None)
        if recheck is not None and certified is not None and not certified.all():
            choice[~certified] = recheck(rows[~certified], labels)
        nearest[rows] = choice
        weight[rows] = _weights(QT, rows, choice, manhattan)
        low, high = np.minimum(points, nearest), np.maximum(points, nearest)
        order = np.lexsort((high, low, weight, labels))
        kept = order[np.r_[True, labels[order[1:]] != labels[order[:-1]]]]  # each component's least edge
        for edge in kept[np.lexsort((high[kept], low[kept], weight[kept]))]:
            first, second = _find(parent, int(labels[edge])), _find(parent, int(labels[nearest[edge]]))
            if first != second:
                parent[max(first, second)] = min(first, second)
                found.append((low[edge], high[edge], weight[edge]))
        while not np.array_equal(parent[parent], parent):
            parent = parent[parent]
        labels = parent[labels]
        rows = points[labels[nearest] == labels]  # rows whose nearest point joined their component
    low, high = (np.array([edge[k] for edge in found], np.int64) for k in (0, 1))
    weight = np.array([edge[2] for edge in found])
    order = np.lexsort((high, low, weight))
    return np.stack([low[order], high[order]], 1), weight[order]


def minimum_spanning_tree(
    X: ArrayLike | torch.Tensor,
    metric: str = "euclidean",
    *,
    exact: bool = True,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    memory_fraction: float | None = None,
) -> "SpanningTree":
    """Minimum spanning tree of the rows of X, such as a map's nodes or k-means centroids.

    Borůvka's algorithm with a masked nearest-point search per round, in O(n_points) memory beyond one block of
    batch_size x n_points scores (no n_points x n_points matrix) and at most ceil(log2 n_points) rounds of
    O(n_points^2 n_features) each, in practice few (5-6 in the user guide's timings). Exact duplicates are
    joined first, without a search. With exact, the tree is the unique minimum spanning tree under the
    strict order (distance, lower index, higher index) of the float64 distances of the rows as float32, the same on
    every device (see the user guide). Identical rows are joined by edges of weight 0.

    Parameters
    ----------
    X
        Points, shape (n_points, n_features) or (..., n_features), as for :meth:`SOM.fit`; rounded to float32.
    metric
        ``"euclidean"``, ``"manhattan"``, ``"cosine"`` or ``"correlation"`` (see :class:`SOM`). Under cosine and
        correlation, the tree is that of the euclidean distances of the rows scaled to unit length in float64 (and
        centred over the features, for correlation), with the weights ||a - b||^2 / 2 = 1 - cos(a, b); rows of zero
        norm (cosine) or constant rows (correlation) have no such distance and raise (whereas :meth:`SOM.transform`
        gives such samples the distance 1 to every node).
    exact
        Whether every row's nearest point in another component is certified, by searching again in float64 the rows
        whose float32 (on the CPU float64) scores do not separate the best two points by more than their rounding
        error. Without, the search's choices are taken as they are (float32 on every device), which gives a spanning
        tree whose weight is within about the scores' rounding error of the minimum.
    device
        Torch device of the search. Defaults to cuda, then mps, then cpu. The float64 rechecks run on the CPU
        instead of MPS.
    batch_size
        Rows searched at once. Defaults to 2^24 / n_points on the CPU (128 MB of float64 scores), on GPUs as for the
        torch path of :class:`SOM` with n_points nodes.
    memory_fraction
        As for :class:`SOM`.

    Returns
    -------
    SpanningTree
        The n_points - 1 edges and their weights, the distances in the metric.
    """
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}, got {metric!r}")
    manhattan = metric == "manhattan"
    engine = SOM(
        1,
        1,
        n_epochs=0,
        metric="manhattan" if manhattan else "euclidean",
        device=device,
        batch_size=batch_size,
        memory_fraction=memory_fraction,
    )
    engine._setup()
    data = engine._check_X(X)
    n = engine.n_rows = len(data)  # the search's nodes, for its batch size
    Q = data.cpu().double()
    if metric in ("cosine", "correlation"):
        undefined = Q.amax(1) == Q.amin(1) if metric == "correlation" else (Q == 0).all(1)
        if undefined.any():
            kind = "constant" if metric == "correlation" else "all-zero"
            raise ValueError(f"X has {int(undefined.sum())} {kind} rows, which have no {metric} distance")
        Q = Q - Q.mean(1, keepdim=True) if metric == "correlation" else Q
        Q = Q / Q.norm(dim=1, keepdim=True)
    if n == 1:
        return SpanningTree(np.zeros((0, 2), np.int64), np.zeros(0), 1)
    # identical rows: the tree joins each to the first of its kind by an edge of weight 0 (as the strict order does),
    # and the first rows, whose order is that of the rows, by the tree of the distinct rows
    _, first, inverse = np.unique(Q.numpy(), axis=0, return_index=True, return_inverse=True)
    distinct = np.sort(first)
    rank = np.empty(len(first), np.int64)
    rank[np.argsort(first)] = np.arange(len(first))
    representative = distinct[rank[inverse.ravel()]]
    edges, squared = np.zeros((0, 2), np.int64), np.zeros(0)
    if len(distinct) > 1:
        engine.n_rows = len(distinct)
        Q = Q[torch.as_tensor(distinct)]
        QT = np.ascontiguousarray(Q.numpy().T)
        block = _block(engine, Q)
        search = _TorchSearch(Q.to(engine._device, _search_dtype(engine._device, exact)), manhattan, block)
        recheck = _Recheck(engine, Q, QT, block) if exact else None
        edges, squared = _boruvka(len(distinct), search, recheck, QT, manhattan)
        edges = distinct[edges]
    copies = np.flatnonzero(representative != np.arange(n))
    edges = np.concatenate([edges, np.stack([representative[copies], copies], 1)])
    squared = np.concatenate([squared, np.zeros(len(copies))])
    order = np.lexsort((edges[:, 1], edges[:, 0], squared))
    edges, squared = edges[order], squared[order]
    weights = np.sqrt(squared) if metric == "euclidean" else squared if manhattan else squared / 2
    return SpanningTree(edges, weights, n)


@dataclass(frozen=True, eq=False)
class SpanningTree:
    """A spanning tree of n_nodes nodes, as returned by :func:`minimum_spanning_tree`.

    Its methods take O(n_nodes) memory beyond their output. Hop counts and path lengths come from the tree rooted at
    node 0: d(s, t) = depth(s) + depth(t) - 2 depth(lca(s, t)), with each lowest common ancestor found by pointer
    jumping in O(n_nodes log n_nodes) per source. The parameters are kept as attributes of the same names.

    Parameters
    ----------
    edges
        Node pairs, shape (n_nodes - 1, 2), int64, the lower index first, the rows sorted by weight, then by the lower
        and the higher index.
    weights
        Edge weights, shape (n_nodes - 1,), float64: the distances in the metric.
    n_nodes
        Number of nodes.
    """

    edges: np.ndarray
    weights: np.ndarray
    n_nodes: int

    def __post_init__(self) -> None:
        if not _is_int(self.n_nodes, 1):
            raise ValueError(f"n_nodes must be an integer >= 1, got {self.n_nodes!r}")
        if np.shape(self.edges) != (self.n_nodes - 1, 2) or np.shape(self.weights) != (self.n_nodes - 1,):
            raise ValueError("a spanning tree has n_nodes - 1 edges, of shape (n_nodes - 1, 2), and as many weights")
        edges = np.asarray(self.edges)
        if edges.size and (edges.dtype.kind not in "iu" or edges.min() < 0 or edges.max() >= self.n_nodes):
            raise ValueError(f"edges must be node indices in [0, {self.n_nodes})")
        parent = np.arange(self.n_nodes)
        for a, b in edges.tolist():  # n_nodes - 1 edges without a cycle connect all nodes
            first, second = _find(parent, a), _find(parent, b)
            if first == second:
                raise ValueError("the edges contain a cycle, so they are no spanning tree")
            parent[max(first, second)] = min(first, second)

    def degrees(self) -> np.ndarray:
        """Number of edges at every node, shape (n_nodes,), int64."""
        return np.bincount(np.asarray(self.edges).ravel(), minlength=self.n_nodes).astype(np.int64)

    def leaves(self) -> np.ndarray:
        """Indices of the nodes with a single edge, in increasing order."""
        return np.flatnonzero(self.degrees() == 1)

    def hops(self, sources: ArrayLike, targets: ArrayLike | None = None) -> np.ndarray:
        """Number of edges on the path between nodes, int32.

        Shape (len(sources), n_nodes), or (len(sources), len(targets)) for the given targets; one row for a single
        source.
        """
        return self._between(sources, targets, self._rooted[3], np.int32)

    def path_lengths(self, sources: ArrayLike, targets: ArrayLike | None = None) -> np.ndarray:
        """Summed weights of the edges on the path between nodes, float64; shapes as for hops.

        Computed from the distances to node 0, so the absolute error grows with the depth of the tree (about the
        float64 rounding of the summed weights from the root), not with the path's own length.
        """
        return np.maximum(self._between(sources, targets, self._rooted[4], np.float64), 0)

    def labels(self, n_clusters: int | None = None, distance_threshold: float | None = None) -> np.ndarray:
        """Single-linkage clusters: the components left after cutting the heaviest edges.

        Cuts the n_clusters - 1 last edges (the heaviest, ties broken as in edges), or every edge of weight at least
        distance_threshold, as scikit-learn's AgglomerativeClustering(linkage="single"); give one of the two. Returns
        int64 labels, the clusters numbered in the order of their smallest node.
        """
        if (n_clusters is None) == (distance_threshold is None):
            raise ValueError("give exactly one of n_clusters and distance_threshold")
        if n_clusters is not None:
            if not _is_int(n_clusters, 1) or n_clusters > self.n_nodes:
                raise ValueError(f"n_clusters must be an integer in [1, {self.n_nodes}], got {n_clusters!r}")
            kept = np.asarray(self.edges)[: self.n_nodes - n_clusters]
        else:
            assert distance_threshold is not None  # one of the two is given
            if np.isnan(distance_threshold):
                raise ValueError("distance_threshold must not be nan")
            kept = np.asarray(self.edges)[np.asarray(self.weights) < distance_threshold]
        parent = np.arange(self.n_nodes)
        for a, b in kept.tolist():
            first, second = _find(parent, a), _find(parent, b)
            parent[max(first, second)] = min(first, second)
        while not np.array_equal(parent[parent], parent):
            parent = parent[parent]
        return np.unique(parent, return_inverse=True)[1].astype(np.int64)

    def linkage(self) -> np.ndarray:
        """Single-linkage hierarchy in scipy's format, shape (n_nodes - 1, 4), float64.

        Row k merges the clusters in its first two columns (nodes below n_nodes, else the cluster of row
        id - n_nodes; the lower id first) at the height in the third, into a cluster of the size in the fourth, as
        ``scipy.cluster.hierarchy.linkage(X, "single")``.
        """
        n = self.n_nodes
        Z = np.empty((n - 1, 4))
        parent, cluster, size = np.arange(n), np.arange(n), np.ones(n, np.int64)
        for k, (a, b) in enumerate(np.asarray(self.edges).tolist()):
            first, second = _find(parent, a), _find(parent, b)
            ids = sorted((cluster[first], cluster[second]))
            root, other = min(first, second), max(first, second)
            parent[other], cluster[root], size[root] = root, n + k, size[root] + size[other]
            Z[k] = ids[0], ids[1], self.weights[k], size[root]
        return Z

    def to_sparse(self, symmetric: bool = True) -> Any:
        """The tree as a scipy.sparse.csr_array of shape (n_nodes, n_nodes), with the weights as values.

        Both (i, j) and (j, i) with symmetric, else only i < j. Edges of weight 0 are stored as explicit zeros, which
        tools that treat zeros as missing edges drop. Needs scipy.
        """
        from scipy import sparse

        a, b = np.asarray(self.edges).T
        weights = np.asarray(self.weights, np.float64)
        if symmetric:
            a, b, weights = np.r_[a, b], np.r_[b, a], np.r_[weights, weights]
        return sparse.csr_array((weights, (a, b)), shape=(self.n_nodes, self.n_nodes))

    def _nodes(self, nodes: ArrayLike, name: str) -> np.ndarray:
        array = np.asarray(nodes)
        if array.size == 0 and array.ndim == 1:
            return np.zeros(0, np.int64)
        if array.ndim > 1 or array.dtype.kind not in "iu" or ((array < 0) | (array >= self.n_nodes)).any():
            raise ValueError(f"{name} must be node indices in [0, {self.n_nodes}), got {nodes!r}")
        return array.astype(np.int64)

    def _between(self, sources: ArrayLike, targets: ArrayLike | None, lengths: np.ndarray, dtype: type) -> np.ndarray:
        """lengths[s] + lengths[t] - 2 lengths[lca(s, t)] for the sources s and targets t."""
        single = np.ndim(sources) == 0
        starts = np.atleast_1d(self._nodes(sources, "sources"))
        ends = np.arange(self.n_nodes) if targets is None else np.atleast_1d(self._nodes(targets, "targets"))
        position, end, parent = self._rooted[:3]
        nodes = np.arange(self.n_nodes)
        out: np.ndarray = np.empty((len(starts), len(ends)), dtype)
        chunk = max(1, _PAIRS // self.n_nodes)
        for first in range(0, len(starts), chunk):
            part = starts[first : first + chunk, None]
            # the ancestors of each source (and itself) point to themselves, every other node to its parent; jumping
            # then leads every node to its lowest ancestor among them, its lowest common ancestor with the source
            ancestor = np.where((position <= position[part]) & (position[part] < end), nodes, parent)
            while not np.array_equal(jumped := np.take_along_axis(ancestor, ancestor, 1), ancestor):
                ancestor = jumped
            out[first : first + len(part)] = lengths[part] + lengths[ends] - 2 * lengths[ancestor[:, ends]]
        return out[0] if single else out

    @cached_property
    def _rooted(self) -> tuple[np.ndarray, ...]:
        """The tree rooted at node 0.

        Per node its position in a depth-first preorder, the end of its subtree's positions, its parent (the root its
        own), and its depth in edges and in weight.
        """
        n = self.n_nodes
        edges, weights = np.asarray(self.edges), np.asarray(self.weights, np.float64)
        ends, others = np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]
        order = np.argsort(ends, kind="stable")
        starts = np.searchsorted(ends[order], np.arange(n + 1)).tolist()
        neighbours, lengths = others[order].tolist(), np.r_[weights, weights][order].tolist()
        parent, depth, distance = [0] * n, [0] * n, [0.0] * n
        seen, preorder, stack = [False] * n, [], [0]
        seen[0] = True
        while stack:  # depth first, so that every subtree takes consecutive positions
            node = stack.pop()
            preorder.append(node)
            for k in range(starts[node], starts[node + 1]):
                child = neighbours[k]
                if not seen[child]:
                    seen[child] = True
                    parent[child], depth[child], distance[child] = node, depth[node] + 1, distance[node] + lengths[k]
                    stack.append(child)
        if len(preorder) < n:
            raise ValueError("the edges do not connect all nodes, so they are no spanning tree")
        size = [1] * n
        for node in reversed(preorder[1:]):
            size[parent[node]] += size[node]
        position = np.empty(n, np.int64)
        position[preorder] = np.arange(n)
        return position, position + size, np.array(parent), np.array(depth), np.array(distance)
