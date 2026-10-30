"""The torch path of the nearest-codebook-vector search, and the CPU's passes over the samples on a pool of threads.

Kept byte-identical in fastersom and its sister k-means package: change both together and compare the two checkouts
with scripts/check_shared.py. The codebook vectors are what the samples are searched against; a pass is one search
(and, in a fit, one sum per codebook vector) over all samples.

The torch path minimises the score ||w - c||^2 - 2 (x - c).(w - c) over the codebook vectors w, with c their mean: the
squared distance less ||x - c||^2, which is the same for every codebook vector. Centring on c keeps a large offset common
to the samples and the codebook vectors from cancelling the products. Ties go to the lowest index.

On the CPU, a pass runs on one pool of threads (_cpu_pool) in fixed parts of the samples (_CPU_PARTS), each searched in
chunks by one thread with torch on a single thread (_cpu_map).
"""

from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager

import torch

# the CPU's fixed parts: n_samples / _CPU_PARTS samples each, at least _CPU_MIN_PART_ROWS, each searched (and in a fit
# summed) by one thread into sums of its own, which are then added in the parts' order, so that the sums do not depend
# on the number of threads. From 96 x 2^13 samples on, 96 parts are 4 per thread at 24 threads, 8 at 12 and 12 at 8, so
# that the threads finish together. Each part and each chunk costs a few torch calls, each of which releases and retakes
# the GIL, about 10-20 us of serial time under contention; small parts add parts and, at few codebook vectors, shrink the
# chunk below the default of 8192 rows. Hence at least 2^13 rows: on an M3 Max with 10 threads, a pass over 3 x 10^5
# samples x 40 features at 100 codebook vectors took 1.5-1.7x as long with parts of 2^12 rows. On a 24-core Threadripper
# PRO 5965WX at 24 threads, these parts, with min for argmin, one pool per fit and (k-means) the centre in the bias, made
# a pass 1.07-1.97x as fast as parts of 2^14 rows with argmin, the centre subtracted and a pool per pass (10^5 to 2 x
# 10^6 samples x 40 features, 100 to 6400 codebook vectors)
_CPU_PARTS = 96
_CPU_MIN_PART_ROWS = 1 << 13
SearchFunction = Callable[[torch.Tensor], torch.Tensor]


def _cpu_batch_size(n_vectors: int, n_features: int) -> int:
    """Default rows of a chunk on the CPU, which one thread of the pool searches at a time (see _cpu_map).

    A chunk's scores, and its copies, within 2^20 values (4 MB), which a thread's caches serve; at least 256 rows, as
    fewer only add calls, but within 2^22 values (16 MB) beyond 16,384 codebook vectors or features, as glibc maps blocks
    above 32 MB afresh on every allocation, whose page faults made a pass take up to 2-4x as long (AMD EPYC 7742).
    """
    largest = max(n_vectors, n_features)
    return max(1, min(8192, max(min(256, (1 << 22) // largest), (1 << 20) // largest)))


def _euclidean_operands(codebook: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectors v = -2 (w - c), bias b = ||w - c||^2 and centre c of the score (x - c).v + b (see the module)."""
    centre = codebook.mean(0)
    shifted = codebook - centre
    return -2 * shifted, shifted.square().sum(1), centre


def _euclidean_scorer(codebook: torch.Tensor, centred: bool = False) -> SearchFunction:
    """The scores (x - c).v + b of a batch against every codebook vector (see _euclidean_operands), to minimise.

    The operands are computed once for all batches, not per batch. centred: whether the batches come from samples
    centred by their mean, among which the codebook vectors lie. With no large offset left to cancel, the CPU then takes
    the centre into the bias once, (x - c).v + b = x.v + (b - c.v), instead of subtracting it from every batch; on
    uncentred samples that would lose nearby codebook vectors to rounding at large offsets.
    """
    vectors, bias, centre = _euclidean_operands(codebook)
    if codebook.device.type == "mps":
        # MPS's addmm expands the bias to a second batch x codebook matrix, and adding it after the matrix product is a
        # further pass over the scores; a column of ones in the batch and the bias in the vectors make it part of a
        # single matrix product. On CUDA addmm fuses the bias into the matrix product (cuBLASLt epilogue)
        extended = torch.cat([vectors, bias[:, None]], 1).T
        return lambda batch: torch.cat([batch - centre, batch.new_ones(len(batch), 1)], 1) @ extended
    transposed = vectors.T
    if centred and codebook.device.type == "cpu":
        folded = bias - centre @ transposed
        return lambda batch: torch.addmm(folded, batch, transposed)
    return lambda batch: torch.addmm(bias, batch - centre, transposed)


def _nearest(scores: SearchFunction, device: torch.device) -> SearchFunction:
    """The search of each sample's nearest codebook vector in a batch on the device, the lowest index on ties.

    scores: the scores of a batch against every codebook vector, to minimise. On the CPU, min(1)'s indices: on one
    thread, argmin(1) took 55-80% of a chunk's search (Threadripper PRO, M3 Max; 100 to 40,000 codebook vectors x 40
    features), min(1) 27-47% less (M3 Max), and a k-means iteration at 24 threads was 1.08-1.20x as fast (Threadripper
    PRO). Both return the first minimal index on ties, as torch documents. GPUs keep argmin(1): on MPS min(1) took
    1.5-2x as long (M3 Max, 100 to 40,000 codebook vectors); on CUDA it was not measured.
    """
    if device.type != "cpu":
        return lambda batch: scores(batch).argmin(1)
    return lambda batch: scores(batch).min(1).indices


def _assign(
    data: torch.Tensor,
    search: SearchFunction,
    labels: torch.Tensor,
    batch_size: int,
    operands: list | None = None,
    pool: ThreadPoolExecutor | None = None,
) -> None:
    """Writes the search's result for every sample to labels, such as its nearest codebook vector.

    With a pool (on the CPU, see _cpu_pool), part by part on its threads (see _cpu_map); else batch by batch, the search
    taking each batch's operand (operands: e.g. the batches as a fused kernel loads them; by default the batches).
    """
    if pool is not None:
        for _ in _cpu_map(pool, lambda part, out, chunk: _assign(part, search, out, chunk), data, labels, batch_size):
            pass
        return
    for operand, out in zip(operands or data.split(batch_size), labels.split(batch_size), strict=True):
        out.copy_(search(operand))


@contextmanager
def _cpu_pool(device: torch.device) -> Iterator[ThreadPoolExecutor | None]:
    """A pool of torch.get_num_threads() threads for _cpu_map while the block runs, None on other devices.

    One pool serves a whole call such as a fit or a predict, not a new one per pass; it is shut down on leaving, also on
    an error.
    """
    if device.type != "cpu":
        yield None
        return
    with ThreadPoolExecutor(torch.get_num_threads()) as pool:
        yield pool


def _cpu_map[T](
    pool: ThreadPoolExecutor,
    work: Callable[[torch.Tensor, torch.Tensor, int], T],
    data: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> Iterator[T]:
    """work(samples, labels, rows per chunk) of every fixed part (see _CPU_PARTS) on the pool, in the parts' order.

    Each of the pool's threads runs torch on a single thread, as the small chunks of a part gain nothing from torch's
    threads: torch is set to one thread while the parts run and to its number of threads again afterwards (also on an
    error), so that the work between passes, such as the codebook update, keeps all of torch's threads. At most two
    parts per thread are submitted ahead of the one consumed, so that per-part sums (codebook vectors x features each)
    stay within that many rather than _CPU_PARTS of them. Consume the results fully, which restores torch's threads.
    """
    rows = max(_CPU_MIN_PART_ROWS, -(-len(data) // _CPU_PARTS))
    chunk = min(batch_size, rows)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        pending: deque[Future[T]] = deque()
        for part, out in zip(data.split(rows), labels.split(rows), strict=True):
            if len(pending) == 2 * threads:
                yield pending.popleft().result()
            pending.append(pool.submit(work, part, out, chunk))
        while pending:
            yield pending.popleft().result()
    finally:
        torch.set_num_threads(threads)


def _cpu_step(
    data: torch.Tensor,
    search: SearchFunction,
    labels: torch.Tensor,
    n_vectors: int,
    batch_size: int,
    pool: ThreadPoolExecutor,
    has_nearest: Callable[[torch.Tensor], torch.Tensor | None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assigns the samples and sums them per codebook vector on the CPU, part by part on the pool (see _cpu_map).

    Returns the sums and the counts of the samples of each of the n_vectors codebook vectors. Each fixed part is searched
    and summed in chunks of batch_size by one thread into sums of its own, which are added in the parts' order: the same
    sums at any number of threads. has_nearest: a function of a batch giving the mask of its samples that have a
    nearest codebook vector, or None if all do (without it, all do). The others get label n_vectors and go to an extra
    row of the sums, which is dropped, rather than being copied out of their chunk.
    """
    size = n_vectors + (has_nearest is not None)

    def part_sums(part: torch.Tensor, out: torch.Tensor, chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
        sums = part.new_zeros(size, part.shape[1])
        for batch, rows in zip(part.split(chunk), out.split(chunk), strict=True):
            found = search(batch)
            keep = None if has_nearest is None else has_nearest(batch)
            if keep is not None:
                found.masked_fill_(~keep, n_vectors)
            rows.copy_(found)
            sums.index_add_(0, found, batch)
        return sums, torch.bincount(out, minlength=size)

    sums = data.new_zeros(size, data.shape[1])
    counts = torch.zeros(size, dtype=torch.long)
    for part_sum, part_count in _cpu_map(pool, part_sums, data, labels, batch_size):
        sums += part_sum
        counts += part_count
    return sums[:n_vectors], counts[:n_vectors]
