"""Fused BMU search on CUDA: scores and argmin in one Gluon (Triton) kernel.

The torch path writes the batch x nodes score matrix to GPU memory and reads it back for the argmin, which with few
features costs more than the matrix product itself, and computes manhattan distances with cdist, which is 100-200x
slower than a matrix product. Here each program keeps a tile of scores for 128 samples x 64 nodes in registers, built
as a sum of outer products over the features in float32, and only the best node (or two) per sample leaves the chip.
Gluon sets the tiling explicitly: 8 x 8 scores per thread take 16 loads per 64 multiply-adds, whereas plain Triton
gives each thread a column of samples for one node, a load per multiply-add, at under half the speed.

On GPUs whose TF32 tensor cores far outpace their float32 cores, dot-product scores instead come from _tc_kernel, a
plain Triton kernel of the same structure on tensor cores, as 3xTF32 (three TF32 products per float32 product). Its
loop over the features takes chunks of 32 where their pipeline buffers fit in a block's shared memory, else of 16.
"""

from collections.abc import Callable
from functools import cache

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.runtime.errors import OutOfResources

if not hasattr(gl, "static_range"):  # Gluon of Triton 3.4 (torch 2.8) lacks it
    raise ImportError("fastersom's fused kernels need Triton >= 3.5")

# (size_per_thread, threads_per_warp, warps_per_cta) of the 128 x 64 score tile, with 4 warps; the fastest or within
# 3% of it of the tilings tried on an A100, for both kernels, 34-1024 features and 100-10,000 nodes
_TILING = ((8, 8), (8, 4), (2, 2))
BLOCK_M, BLOCK_N = (s * t * w for s, t, w in zip(*_TILING, strict=True))
_LAYOUT = gl.BlockedLayout(*map(list, _TILING), [1, 0])  # built once, as building it per launch takes about 20 us
_TC_BLOCK = 128  # samples and nodes per tile of the tensor-core kernel, with 8 warps
_PAD = 128  # samples and nodes are padded to a multiple of every tile
# features per chunk of the tensor-core kernel at least: Triton before 3.8 takes dots of float32 over 16 or more only
_TC_MIN_K = 8 if tuple(map(int, triton.__version__.split(".")[:2])) >= (3, 8) else 16
# features per chunk of the tensor-core loop at most, per device: 32, or 16 where the pipeline buffers of chunks of 32
# exceed a block's shared memory, which Triton tells when it launches the kernel. Compiled by Triton 3.8, chunks of 32
# take 128-144 KB for compute capability 8.x, within the A100's 163 KB but not the 99 KB of Ada (8.9) and consumer
# Ampere (8.6), and 240 KB for 10.0, beyond the B200's 227 KB; chunks of 16 take at most 120 KB
_TC_LOOP_K: dict[torch.device, int] = {}
# counts that only bound masks and loops; Triton would otherwise compile a variant each for a count of 1, a multiple of
# 16 and others, e.g. again for a fit's last batch or a single sample
_COUNTS = ["n_samples", "n_nodes"]


@gluon.jit(do_not_specialize=_COUNTS)
def _bmu_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    bmu_ptr,
    score_ptr,
    n_samples,
    n_nodes,
    n_features,
    x_stride,
    w_stride,
    split_nodes,
    MANHATTAN: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    TOP2: gl.constexpr,
    REMAINDER: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LAYOUT: gl.constexpr,
):
    rows_layout: gl.constexpr = gl.SliceLayout(1, LAYOUT)
    cols_layout: gl.constexpr = gl.SliceLayout(0, LAYOUT)
    rows = gl.program_id(0) * BLOCK_M + gl.arange(0, BLOCK_M, layout=rows_layout)
    best = gl.full([BLOCK_M], float("inf"), gl.float32, rows_layout)
    best_node = gl.full([BLOCK_M], 0, gl.int32, rows_layout)
    second, second_node = best, best_node  # with TOP2 only
    # the second grid axis splits the nodes, so that few samples on a large map still fill the GPU
    first = gl.program_id(1) * split_nodes
    for start in range(first, gl.minimum(first + split_nodes, n_nodes), BLOCK_N):
        cols = start + gl.arange(0, BLOCK_N, layout=cols_layout)
        scores = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, LAYOUT)
        # x and w are transposed (features x samples, features x nodes), so a feature of a tile is contiguous; moving
        # the pointers instead of computing feature * stride cannot overflow int32
        x_k = x_ptr + rows
        w_k = w_ptr + cols
        # 16 features per step: as fast as unrolling up to 64 features in full on an A100, or faster (1.1-1.4x at 48
        # to 64), and a compiled variant per remainder rather than per feature count
        for _ in range(0, n_features - REMAINDER, 16):
            for _k in gl.static_range(16):
                scores = _accumulate(scores, x_k, w_k, MANHATTAN)
                x_k += x_stride
                w_k += w_stride
        for _k in gl.static_range(REMAINDER):
            scores = _accumulate(scores, x_k, w_k, MANHATTAN)
            x_k += x_stride
            w_k += w_stride
        if HAS_BIAS:  # after the products, like cuBLAS's addmm, which for few features then gives the same scores
            scores += gl.expand_dims(gl.load(bias_ptr + cols), 0)
        nodes = gl.expand_dims(cols, 0)
        best, best_node, second, second_node = _merge(
            scores, nodes, n_nodes, best, best_node, second, second_node, TOP2
        )
    _store(bmu_ptr, score_ptr, rows, n_samples, best, best_node, second, second_node, TOP2)


@triton.jit
def _argmin(scores, nodes):
    """Minimum of each row and the first node with it."""
    best = tl.min(scores, 1)
    return best, tl.min(tl.where(scores == best[:, None], nodes, 2147483647), 1)


@triton.jit
def _merge(scores, nodes, n_nodes, best, best_node, second, second_node, TOP2: tl.constexpr):
    """Merge a tile of scores into the running best nodes per sample.

    Merges a tile of scores (samples x nodes) into the running best node (and with TOP2 second best) per sample;
    shared by both kernels, as plain Triton also runs on Gluon's tensors.
    """
    scores = tl.where(nodes < n_nodes, scores, float("inf"))
    tile_best, tile_node = _argmin(scores, nodes)
    better = tile_best < best  # the running best are earlier nodes, so they win ties, as in torch's argmin
    if TOP2:  # the tile's second best is its best without its first; of two pairs, the earlier wins ties
        tile_second, tile_second_node = _argmin(tl.where(nodes == tile_node[:, None], float("inf"), scores), nodes)
        # if better, whether the old best stays second, else whether the old second does
        stays = tl.where(better, best <= tile_second, second <= tile_best)
        second_node = tl.where(
            stays, tl.where(better, best_node, second_node), tl.where(better, tile_second_node, tile_node)
        )
        second = tl.where(stays, tl.where(better, best, second), tl.where(better, tile_second, tile_best))
    return tl.where(better, tile_best, best), tl.where(better, tile_node, best_node), second, second_node


@triton.jit
def _store(bmu_ptr, score_ptr, rows, n_samples, best, best_node, second, second_node, TOP2: tl.constexpr):
    mask = rows < n_samples
    if TOP2:  # per part of the nodes, reduced by the caller
        out = tl.program_id(1) * 2 * n_samples + rows
        tl.store(bmu_ptr + out, best_node.to(tl.int64), mask=mask)
        tl.store(score_ptr + out, best, mask=mask)
        tl.store(bmu_ptr + out + n_samples, second_node.to(tl.int64), mask=mask)
        tl.store(score_ptr + out + n_samples, second, mask=mask)
    else:
        # the programs of the parts of the nodes take an atomic minimum of one int64 per sample, the score's bits
        # flipped to order as integers (negatives below positives) above the node, so ties go to the first node in any
        # order, with no reduction over the parts afterwards (which costs more than the search on small batches)
        bits = tl.where(best == 0, 0.0, best).to(tl.int32, bitcast=True)  # -0.0 ties with 0.0
        key = ((bits ^ ((bits >> 31) & 0x7FFFFFFF)).to(tl.int64) << 32) | best_node.to(tl.int64)
        tl.atomic_min(bmu_ptr + rows, key, mask=mask, sem="relaxed")


@gluon.jit
def _accumulate(scores, x_k, w_k, MANHATTAN: gl.constexpr):
    x = gl.expand_dims(gl.load(x_k), 1)
    w = gl.expand_dims(gl.load(w_k), 0)
    if MANHATTAN:
        return scores + gl.abs(x - w)
    return scores + x * w


@triton.jit
def _tf32_split(a):
    """a = big + small, big rounded to TF32 (10 of float32's 23 mantissa bits; to nearest, ties away from zero)."""
    big = ((a.to(tl.int32, bitcast=True) + 0x1000) & -0x2000).to(tl.float32, bitcast=True)
    return big, a - big


@triton.jit
def _tc_chunk(scores, x_rows, w_cols, k, w_stride, x_half, w_half):
    """Add the tensor-core products over a chunk of features to the scores.

    Adds the products over a chunk of features: x . w = big . big + big . small + small . big up to the tiny
    small . small, each product of TF32 parts exact. Their sum over the chunk is added to the scores in float32, as the
    tensor cores truncate their sums to the precision of the running sum: summed over all features, the scores' errors
    were 15-40x those of float32 sums at 200-1024 features; per chunk of at most 32, 1-3x, and the BMUs then match
    float64's more often than the float32 kernel's.
    """
    x = x_rows + k[None, :]
    w = w_cols + k[:, None] * w_stride
    x_big, x_small, w_big, w_small = tl.load(x), tl.load(x + x_half), tl.load(w), tl.load(w + w_half)
    chunk = tl.dot(x_small, w_big, input_precision="tf32")
    chunk = tl.dot(x_big, w_small, chunk, input_precision="tf32")
    return scores + tl.dot(x_big, w_big, chunk, input_precision="tf32")


@triton.jit(do_not_specialize=_COUNTS)
def _tc_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    bmu_ptr,
    score_ptr,
    n_samples,
    n_nodes,
    width,
    w_stride,
    x_half,
    w_half,
    split_nodes,
    n_loop,
    LOOP: tl.constexpr,
    LOOP_K: tl.constexpr,
    N16: tl.constexpr,
    N8: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    TOP2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """_bmu_kernel's search of x . w + bias on tensor cores, as 3xTF32 (see _tc_chunk), in tiles of 128 x 128."""
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x_rows = x_ptr + rows[:, None].to(tl.int64) * width
    best = tl.full([BLOCK], float("inf"), tl.float32)
    best_node = tl.zeros([BLOCK], tl.int32)
    second, second_node = best, best_node  # with TOP2 only
    first = tl.program_id(1) * split_nodes
    for start in range(first, tl.minimum(first + split_nodes, n_nodes), BLOCK):
        cols = start + tl.arange(0, BLOCK)
        w_cols = w_ptr + cols[None, :]
        scores = tl.zeros([BLOCK, BLOCK], tl.float32)
        # a pipelined loop over chunks of LOOP_K features (only if any: an empty one costs up to 1.5x), then the rest
        # in chunks of 16 and 8
        if LOOP:
            for i in tl.range(0, n_loop, num_stages=3):
                scores = _tc_chunk(scores, x_rows, w_cols, i * LOOP_K + tl.arange(0, LOOP_K), w_stride, x_half, w_half)
        for i in tl.static_range(N16):
            k = n_loop * LOOP_K + 16 * i + tl.arange(0, 16)
            scores = _tc_chunk(scores, x_rows, w_cols, k, w_stride, x_half, w_half)
        if N8:
            k = n_loop * LOOP_K + 16 * N16 + tl.arange(0, 8)
            scores = _tc_chunk(scores, x_rows, w_cols, k, w_stride, x_half, w_half)
        if HAS_BIAS:
            scores += tl.load(bias_ptr + cols)[None, :]
        best, best_node, second, second_node = _merge(
            scores, cols[None, :], n_nodes, best, best_node, second, second_node, TOP2
        )
    _store(bmu_ptr, score_ptr, rows, n_samples, best, best_node, second, second_node, TOP2)


def _tc_chunks(width: int, largest: int = 32) -> tuple[int, int, int, int]:
    """Split a multiple of 8 features into the chunks of the tensor-core loop.

    (chunks in the pipelined loop, their size, chunks of 16, of 8) of a multiple of 8 features: a loop over chunks
    of 32, the fastest on an A100 from 34 to 256 features (16 below 32 features, or if largest is 16).
    """
    size = min(largest, 32 if width >= 32 else 16)
    n_loop, rest = divmod(width, size)
    return n_loop, size, rest // 16, rest % 16 // 8


@triton.jit(do_not_specialize=["n"])
def _copy_kernel(
    x_ptr,
    centre_ptr,
    out_ptr,
    n,
    n_features,
    width,
    x_row,
    x_col,
    out_row,
    out_col,
    part,
    HAS_CENTRE: tl.constexpr,
    SPLIT: tl.constexpr,
):
    # a tile of 64 rows x 32 features of x - centre to out_row * row + out_col * feature, zeros beyond n rows and
    # n_features features; a stride of 1 tells Triton which way out is contiguous, so that for a transposed copy it
    # converts the tile through shared memory. With SPLIT, the big TF32 part and, part elements on, the small one,
    # rounded too, as TF32 products ignore its lower 13 bits
    rows = tl.program_id(0) * 64 + tl.arange(0, 64)
    cols = tl.program_id(1) * 32 + tl.arange(0, 32)
    mask = (rows[:, None] < n) & (cols[None, :] < n_features)
    tile = tl.load(x_ptr + rows[:, None].to(tl.int64) * x_row + cols[None, :] * x_col, mask=mask, other=0.0)
    if HAS_CENTRE:
        tile = tl.where(mask, tile - tl.load(centre_ptr + cols, mask=cols < n_features)[None, :], 0.0)
    out = out_ptr + rows[:, None].to(tl.int64) * out_row + cols[None, :].to(tl.int64) * out_col
    if SPLIT:
        tile, small = _tf32_split(tile)
        tl.store(out + part, _tf32_split(small)[0], mask=cols[None, :] < width)
    tl.store(out, tile, mask=cols[None, :] < width)


def _padded(x: torch.Tensor, centre: torch.Tensor | None, tensor_cores: bool, nodes: bool = False) -> torch.Tensor:
    """Lay out samples or nodes as the kernels load them.

    x - centre (x of shape (n, n_features), any strides) as the kernels load it: for the float32 kernel transposed,
    (1, n_features, n'); for the tensor-core kernel as its big and small TF32 parts, (2, n', width), or for nodes
    (2, width, n'), the order in which it loads them the fastest. n' >= n is a multiple of the tiles and width >=
    n_features one of 8 (16 before Triton 3.8), with zeros beyond, so that loads need no masks. About 8x faster than
    torch's copy of x.T, which reads or writes with a stride of n_features.
    """
    n, n_features = x.shape
    n_out = -(-n // _PAD) * _PAD
    width = -(-n_features // _TC_MIN_K) * _TC_MIN_K if tensor_cores else n_features
    if nodes or not tensor_cores:
        out, strides = x.new_empty(2 if tensor_cores else 1, width, n_out), (1, n_out)
    else:
        out, strides = x.new_empty(2, n_out, width), (width, 1)
    with torch.cuda.device(x.device):  # Triton launches on the current device
        _copy_kernel[(n_out // 64, -(-width // 32))](
            x,
            x if centre is None else centre,
            out,
            n,
            n_features,
            width,
            *x.stride(),
            *strides,
            out.stride(0),
            centre is not None,
            tensor_cores,
        )
    return out


def _tc_launch(grid: tuple[int, int], args: tuple, split_nodes: int, largest: int, bias: bool, top2: bool) -> None:
    """Launch _tc_kernel on args (x, w, ...) of _bmus, in chunks of at most largest features."""
    x, w = args[0], args[1]
    n_loop, loop_k, n16, n8 = _tc_chunks(x.shape[2], largest)
    _tc_kernel[grid](
        *args,
        x.shape[2],
        w.stride(1),
        x.stride(0),
        w.stride(0),
        split_nodes,
        n_loop,
        n_loop > 0,
        loop_k,
        n16,
        n8,
        bias,
        top2,
        _TC_BLOCK,
        num_warps=8,
    )


@cache
def _n_programs(device: torch.device) -> int:
    return 8 * torch.cuda.get_device_properties(device).multi_processor_count


def _bmus(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    n_samples: int,
    n_nodes: int,
    manhattan: bool,
    tensor_cores: bool,
    top2: bool,
) -> torch.Tensor:
    if n_samples == 0:  # e.g. none of the samples has a BMU, which leaves no programs to split the nodes over
        return torch.empty((0, 2) if top2 else (0,), dtype=torch.long, device=x.device)
    block_m, block_n = (_TC_BLOCK, _TC_BLOCK) if tensor_cores else (BLOCK_M, BLOCK_N)
    # split the nodes into as few parts as give each streaming multiprocessor about 8 programs
    n_blocks, n_tiles = -(-n_samples // block_m), -(-n_nodes // block_n)
    tiles_per_split = -(-n_tiles // min(n_tiles, -(-_n_programs(x.device) // n_blocks)))
    n_splits = -(-n_tiles // tiles_per_split)
    if top2:
        bmu = torch.empty(n_splits * 2, n_samples, dtype=torch.long, device=x.device)
        score = torch.empty(n_splits * 2, n_samples, dtype=torch.float32, device=x.device)
    else:  # the keys of the atomic minimum, in place of both
        bmu = score = torch.full((n_samples,), torch.iinfo(torch.long).max, device=x.device)
    args = (x, w, w if bias is None else bias, bmu, score, n_samples, n_nodes)  # the bias is not read without one
    with torch.cuda.device(x.device):  # Triton launches on the current device
        if tensor_cores:
            grid, split_nodes = (n_blocks, n_splits), tiles_per_split * block_n
            largest = _TC_LOOP_K.get(x.device, 32)
            try:
                _tc_launch(grid, args, split_nodes, largest, bias is not None, top2)
            except OutOfResources:  # raised before the kernel runs, which leaves bmu and score as they were
                if largest == 16 or x.shape[2] < 32:  # already in chunks of 16
                    raise
                _TC_LOOP_K[x.device] = 16  # for the device's self-checks and searches from now on
                _tc_launch(grid, args, split_nodes, 16, bias is not None, top2)
        else:
            _bmu_kernel[(n_blocks, n_splits)](
                *args,
                x.shape[1],
                x.stride(1),
                w.stride(1),
                tiles_per_split * block_n,
                manhattan,
                bias is not None,
                top2,
                x.shape[1] % 16,
                BLOCK_M,
                BLOCK_N,
                _LAYOUT,
                num_warps=4,
            )
    if not top2:
        return bmu.bitwise_and_(0xFFFFFFFF)
    if n_splits == 1:
        return bmu.T
    return bmu.gather(0, score.topk(2, 0, largest=False).indices).T


def search(
    nodes: torch.Tensor,
    bias: torch.Tensor | None,
    centre: torch.Tensor | None,
    manhattan: bool,
    tensor_cores: bool,
    top2: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the BMU search of a CUDA kernel variant.

    Returns a function of samples x, shape (n, n_features), giving each one's BMU (int64): the node j with the
    least (x - centre) . nodes_j + bias_j, or ||x - nodes_j||_1 if manhattan, the first on ties; with top2 the best
    two, shape (n, 2), in either order on ties, as torch's topk. The tensor-core kernel does not take manhattan.
    """
    w = _padded(nodes, None, tensor_cores, nodes=True)
    if bias is not None:
        bias = torch.nn.functional.pad(bias, (0, w.shape[-1] - len(nodes)))
    return lambda x: _bmus(_padded(x, centre, tensor_cores), w, bias, len(x), len(nodes), manhattan, tensor_cores, top2)


def check(device: torch.device, n_features: int, tensor_cores: bool, manhattan: bool, bias: bool, top2: bool) -> None:
    """Check a CUDA kernel variant on a small search.

    Raises unless the variant of a kernel for these arguments finds the exact BMUs of a small search (with bias, a
    bias and a centre, as for euclidean). Its values are -1, 0 or 1, for the tensor-core kernel up to 2000 features
    plus a small TF32 part (-2^-13, 0 or 2^-13) on alternate features of the samples and of the nodes, never both, as
    3xTF32 drops small x small; every sum is then a multiple of 2^-13 below 2^11, exact in float32 in any order. Each
    node is there twice, in different parts of the nodes, so that ties must go to the first. As the kernels do not
    specialise on the counts, this compiles the very variant that the search then runs. The reference is on the CPU,
    as cuBLAS raises under torch.use_deterministic_algorithms without CUBLAS_WORKSPACE_CONFIG.
    """
    generator = torch.Generator().manual_seed(0)
    # an explicit device and dtype, which torch.set_default_device or set_default_dtype would change otherwise
    x, nodes, offsets = torch.randint(-1, 2, (3, 256, n_features), generator=generator, device="cpu").double()
    if tensor_cores and n_features <= 2000:
        small = torch.randint(-1, 2, (2, 256, n_features), generator=generator, device="cpu") * 2.0**-13
        x[:, ::2] += small[0, :, ::2]
        nodes[:, 1::2] += small[1, :, 1::2]
    nodes[128:] = nodes[:128]
    scores = torch.cdist(x, nodes, p=1) if manhattan else x @ nodes.T + (offsets[:, 0] if bias else 0)
    b = offsets[:, 0].float().to(device) if bias else None
    centre = None if b is None else b.new_zeros(n_features)
    found = search(nodes.float().to(device), b, centre, manhattan, tensor_cores, top2)(x.float().to(device)).cpu()
    if top2:
        best2 = scores.topk(2, 1, largest=False).values
        exact = torch.equal(scores.gather(1, found).sort(1).values, best2) and bool((found[:, 0] != found[:, 1]).all())
    else:
        exact = torch.equal(found, scores.argmin(1))
    if not exact:
        raise RuntimeError("wrong BMUs in a self-check")
