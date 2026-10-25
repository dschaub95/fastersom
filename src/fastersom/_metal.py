"""Fused BMU search on Apple GPUs (MPS): scores and argmin in one Metal kernel.

As on CUDA (see fastersom._kernels), the torch path writes the batch x nodes score matrix to memory and reads it back
for the argmin, and with few features its matrix product reaches 1.5-3 TFLOP/s of the 10 of an M3 Max; manhattan
distances take cdist, which is 60-260x slower still. Here the dot products run on the simdgroup matrix units
(simdgroup_multiply_accumulate on 8 x 8 fragments, in float32 as the torch path) and only each sample's best node (or
two) leaves the chip: bmu_dot keeps each simdgroup's samples in registers and reads the nodes from the caches; with
many features or nodes bmu_staged streams chunks of the nodes through threadgroup memory, shared by a threadgroup's
simdgroups, and reads the samples from device memory. Manhattan distances take the ALUs instead (bmu_l1), in tiles of
up to 8 samples x 4 nodes per thread.

Before the M5, Apple GPUs have no faster matrix precision to trade for float32 (on an M3 Max half and bfloat matrices
are as fast as float, about 10 TFLOP/s), so unlike the CUDA tensor-core kernel the search computes in float32; only
fit under precision="float16" takes half inputs, summed in float32 (the _half variants). fixed_sums adds fit's
deterministic per-node sums as 64-bit fixed point. The kernels compile from source at runtime
(torch.mps.compile_shader, no Xcode needed), once per feature count, in about 0.2 s; the system caches the compiled
pipelines on disk.
"""

from collections.abc import Callable
from functools import cache
from itertools import product

import torch

# up to this many fragments of 8 features (with the bias column) the samples stay in registers (bmu_dot); beyond, the
# register file no longer holds them with the scores, and bmu_staged streams both operands in chunks instead (from 64
# to 300 features on an M3 Max 6.6-7 TFLOP/s, where bmu_dot's samples in threadgroup memory reach 1.4-6). bmu_dot also
# reads all nodes once per 8 samples, from the caches while they fit: beyond _REGISTER_NODE_BYTES of nodes it falls to
# half the speed of bmu_staged, which reads them once per 64 (500 x 500 nodes of 60 features: 3.2 instead of 7.0
# TFLOP/s), while below it is 1.05-1.4x faster (6.6 instead of 5.3 at 300 x 300 nodes of 34 features)
_REGISTER_FRAGMENTS = 8
_REGISTER_NODE_BYTES = 32 << 20
# bmu_dot: nodes per block in fragments of 8, simdgroups per threadgroup (of 8 samples each); the fastest of 1-3 x 2-8
# fragments per simdgroup and 2-8 simdgroups tried on an M3 Max at 34 and 60 features and 900 and 10,000 nodes (6.9-8
# TFLOP/s): the smallest tile leaves the most registers for simdgroups in flight, while the 4 x 4 fragments of a GEMM
# tile spill (0.5 TFLOP/s)
_DOT_TN, _DOT_SG = 2, 4
# bmu_staged: samples and nodes per simdgroup in fragments of 8, simdgroups per threadgroup, features per chunk; the
# fastest of 1-4 x 2-8 fragments, 2-8 simdgroups and chunks of 8-32 features tried at 100 and 300 features on 10,000
# nodes, or within 3% of it
_STAGED_TM, _STAGED_TN, _STAGED_SG, _STAGED_BK = 2, 8, 4, 16
# bmu_l1: threads per threadgroup, threads across a block's nodes, nodes per thread; with 8 samples per thread the
# fastest of 2-8 samples x 2-8 nodes and 4-16 threads across nodes tried at 34 features on 900 nodes, 1.8 billion
# differences per second, near the ALUs' rate at 2 instructions each
_L1_THREADS, _L1_TX, _L1_RN = 128, 8, 4
_THREADGROUP_MEMORY = 32768  # bytes, the most Metal allows a threadgroup on Apple GPUs
# threads per launch: torch.mps.compile_shader takes a grid's width modulo 2^32 without an error, so that larger
# batches (about 2^29-2^31 samples) run in parts
_MAX_THREADS = 1 << 31
# threadgroups below which the nodes are split over a second grid axis, so that few samples on a large map still use
# the whole GPU: about 34 per core of an M3 Max's 30; smaller Apple GPUs then split more than they need, which costs
# only the reduction of the parts
_MIN_THREADGROUPS = 1024

_SOURCE = r"""
// D, the tiling constants (DOT_*, ST_*, L1_*) and whether this library has bmu_dot and bmu_l1 (DOT, L1) come in a
// header from _library

// bmu_dot: fragments of 8 features, with the bias as a column of ones in the samples if the padding has room
constant constexpr bool DOT_BIAS_COLUMN = D % 8 != 0;
constant constexpr int DOT_KF = (D + 7) / 8, DOT_KP = DOT_KF * 8, DOT_BM = DOT_SG * 8, DOT_BN = DOT_TN * 8;
// bmu_staged: the features, a column of ones for the bias and zeros to a multiple of ST_BK
constant constexpr int ST_KP = (D + ST_BK) / ST_BK * ST_BK, ST_BM = ST_SG * ST_TM * 8, ST_BN = ST_TN * 8;
// bmu_l1: L1_RM samples x L1_RN nodes per thread, L1_TX threads across a block's nodes
constant constexpr int L1_TY = L1_THREADS / L1_TX, L1_BM = L1_TY * L1_RM, L1_BN = L1_TX * L1_RN;

// whether (s, j) comes before (t, k): the lower score, on ties the earlier node
inline bool before(float s, int j, float t, int k) { return s < t || (s == t && j < k); }

// adds node j with score s to a thread's best (and with TOP2 second best); nodes come in increasing order, so the
// earlier node stays first on ties
template <bool TOP2>
inline void update(float s, int j, thread float& b1, thread int& j1, thread float& b2, thread int& j2) {
    bool first = s < b1;
    if (TOP2) {
        bool second = s < b2;
        b2 = first ? b1 : (second ? s : b2);
        j2 = first ? j1 : (second ? j : j2);
    }
    b1 = first ? s : b1;
    j1 = first ? j : j1;
}

// merges the best (two) of the lanes whose index differs by mask; both lanes end with the same result
template <bool TOP2>
inline void merge(ushort mask, thread float& b1, thread int& j1, thread float& b2, thread int& j2) {
    float c1 = simd_shuffle_xor(b1, mask);
    int k1 = simd_shuffle_xor(j1, mask);
    if (TOP2) {
        float c2 = simd_shuffle_xor(b2, mask);
        int k2 = simd_shuffle_xor(j2, mask);
        if (before(c1, k1, b1, j1)) {  // the other lane's best is first, then the better of ours and its second
            bool ours = before(b1, j1, c2, k2);
            b2 = ours ? b1 : c2;
            j2 = ours ? j1 : k2;
            b1 = c1;
            j1 = k1;
        } else if (before(c1, k1, b2, j2)) {
            b2 = c1;
            j2 = k1;
        }
    } else if (before(c1, k1, b1, j1)) {
        b1 = c1;
        j1 = k1;
    }
}

// a sample's best node (and second) of the nodes of this program's part, to out[part][row] (with TOP2 to
// out[2 * part][row] and out[2 * part + 1][row])
template <bool TOP2>
inline void store(device long* node, device float* score, long n, long row, uint part, float b1, int j1, float b2,
                  int j2) {
    long out = (TOP2 ? 2 * long(part) : long(part)) * n + row;
    node[out] = j1;
    score[out] = b1;
    if (TOP2) {
        node[out + n] = j2;
        score[out + n] = b2;
    }
}

// the elements of an 8 x 8 simdgroup matrix that a lane holds (thread_elements): row frag_row, columns frag_col and
// frag_col + 1, so that the 4 lanes of a row differ in bits 0 and 3 of the lane. Apple does not document this layout
// (MLX's steel kernels assume the same); check() would catch another
inline int frag_row(ushort lane) { return (lane / 4 & 4) + (lane / 2 % 4); }
inline int frag_col(ushort lane) { return (lane / 4 & 2) * 2 + lane % 2 * 2; }

// merges a fragment row's best (two) over its 4 lanes and stores them from the first
template <bool TOP2>
inline void store_row(ushort lane, device long* node, device float* score, long n, long row, uint part, float b1,
                      int j1, float b2, int j2) {
    merge<TOP2>(1, b1, j1, b2, j2);
    merge<TOP2>(8, b1, j1, b2, j2);
    if ((lane & 9) == 0 && row < n) store<TOP2>(node, score, n, row, part, b1, j1, b2, j2);
}

#if DOT
// the node j of least (x - centre) . w_j + bias_j per sample x. w holds the nodes in blocks of DOT_BN, as fragments
// (block, DOT_KF, DOT_TN, 8 features, 8 nodes), with the bias in feature D if DOT_BIAS_COLUMN, else in bias; padded
// nodes have an infinite bias. In half (precision float16) the samples are multiplied by scale (a power of 2 per
// feature) and the nodes by another, so that every feature's products come out times 1 / scale[D], and the bias always
// comes after the products, in float32. p: samples, x's row stride, blocks of nodes, blocks per part of the nodes (the
// second grid axis)
template <bool CENTRE, bool TOP2, typename T>
kernel void bmu_dot(
    device const float* x [[buffer(0)]],
    device const float* centre [[buffer(1)]],
    device const T* w [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    device long* out_node [[buffer(4)]],
    device float* out_score [[buffer(5)]],
    constant long* p [[buffer(6)]],
    device const float* scale [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort tid [[thread_index_in_threadgroup]])
{
    constexpr bool HALF = is_same<T, half>::value, COLUMN = DOT_BIAS_COLUMN && !HALF;
    const long n = p[0], ld = p[1], n_blocks = p[2], per_part = p[3];
    const float unscale = HALF ? scale[D] : 1.0f;
    // the threadgroup's samples minus the centre, with the column of ones and zeros beyond, then each simdgroup's 8
    // into registers
    threadgroup T xs[DOT_BM * DOT_KP];
    const long row0 = long(tg.x) * DOT_BM;
    for (int i = tid; i < DOT_BM * DOT_KP; i += DOT_SG * 32) {
        int r = i / DOT_KP, c = i % DOT_KP;
        long s = row0 + r;
        float v = 0.0f;
        if (s < n) {
            if (c < D) v = (x[s * ld + c] - (CENTRE ? centre[c] : 0.0f)) * (HALF ? scale[c] : 1.0f);
            else if (COLUMN && c == D) v = 1.0f;
        }
        xs[i] = T(v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<T, 8, 8> a[DOT_KF];
    for (int kf = 0; kf < DOT_KF; kf++) simdgroup_load(a[kf], xs + sg * 8 * DOT_KP + kf * 8, DOT_KP);

    const int col = frag_col(lane);
    float b1 = INFINITY, b2 = INFINITY;
    int j1 = 0, j2 = 0;
    const long first = long(tg.y) * per_part, last = min(first + per_part, n_blocks);
    for (long block = first; block < last; block++) {
        device const T* wb = w + block * (DOT_KF * DOT_TN * 64);
        simdgroup_float8x8 acc[DOT_TN];
        for (int tn = 0; tn < DOT_TN; tn++) acc[tn] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (int kf = 0; kf < DOT_KF; kf++)
            for (int tn = 0; tn < DOT_TN; tn++) {
                simdgroup_matrix<T, 8, 8> b;
                simdgroup_load(b, wb + (kf * DOT_TN + tn) * 64, 8);
                simdgroup_multiply_accumulate(acc[tn], a[kf], b, acc[tn]);
            }
        const int node = int(block) * DOT_BN + col;
        for (int tn = 0; tn < DOT_TN; tn++) {
            float2 bias_tn = COLUMN ? float2(0.0f) : *(device const float2*)(bias + node + tn * 8);
            auto e = acc[tn].thread_elements();
            update<TOP2>(e[0] * unscale + bias_tn[0], node + tn * 8, b1, j1, b2, j2);
            update<TOP2>(e[1] * unscale + bias_tn[1], node + tn * 8 + 1, b1, j1, b2, j2);
        }
    }
    store_row<TOP2>(lane, out_node, out_score, n, row0 + sg * 8 + frag_row(lane), tg.y, b1, j1, b2, j2);
}

#define BMU_DOT(name, CENTRE, TOP2, T)                                                                               \
    template [[host_name(name)]] kernel void bmu_dot<CENTRE, TOP2, T>(                                               \
        device const float*, device const float*, device const T*, device const float*, device long*, device float*,  \
        constant long*, device const float*, uint2, ushort, ushort, ushort);
BMU_DOT("bmu_dot", false, false, float)
BMU_DOT("bmu_dot_centre", true, false, float)
BMU_DOT("bmu_dot_top2", false, true, float)
BMU_DOT("bmu_dot_centre_top2", true, true, float)
BMU_DOT("bmu_dot_half", false, false, half)
BMU_DOT("bmu_dot_centre_half", true, false, half)
#endif

// bmu_dot's search for many features or nodes: the samples come padded and centred, (rows, ST_KP) with the column of
// ones in feature D, the nodes as (ST_KP, nodes) with the bias in feature D (infinite for padded nodes). Per block of
// ST_BN nodes, each chunk of ST_BK features of the nodes goes to threadgroup memory once for the threadgroup's ST_SG
// simdgroups, which take their samples' fragments straight from device memory; the threadgroup memory stays small,
// so that many threadgroups share a core. In half, both come scaled as for bmu_dot, their products times unscale[0],
// with zeros in feature D and the bias in bias. p as for bmu_dot, with the nodes' row stride fifth
template <bool TOP2, typename T>
kernel void bmu_staged(
    device const T* x [[buffer(0)]],
    device const T* w [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device long* out_node [[buffer(3)]],
    device float* out_score [[buffer(4)]],
    constant long* p [[buffer(5)]],
    device const float* unscale [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort tid [[thread_index_in_threadgroup]])
{
    constexpr bool HALF = is_same<T, half>::value;
    const long n = p[0], ld = p[1], n_blocks = p[2], per_part = p[3], ld_w = p[4];
    const float product_scale = HALF ? unscale[0] : 1.0f;
    threadgroup T ws[ST_BK * ST_BN];
    const long row0 = long(tg.x) * ST_BM + sg * ST_TM * 8;
    device const T* xg = x + row0 * ld;
    const int col = frag_col(lane);
    float b1[ST_TM], b2[ST_TM];
    int j1[ST_TM], j2[ST_TM];
    for (int tm = 0; tm < ST_TM; tm++) {
        b1[tm] = b2[tm] = INFINITY;
        j1[tm] = j2[tm] = 0;
    }
    const long first = long(tg.y) * per_part, last = min(first + per_part, n_blocks);
    for (long block = first; block < last; block++) {
        simdgroup_float8x8 acc[ST_TM][ST_TN];
        for (int tm = 0; tm < ST_TM; tm++)
            for (int tn = 0; tn < ST_TN; tn++) acc[tm][tn] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (int k0 = 0; k0 < ST_KP; k0 += ST_BK) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = tid; i < ST_BK * ST_BN / 4; i += ST_SG * 32) {
                int k = i / (ST_BN / 4), c = i % (ST_BN / 4);
                ((threadgroup vec<T, 4>*)ws)[i] = ((device const vec<T, 4>*)(w + (k0 + k) * ld_w + block * ST_BN))[c];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int k8 = 0; k8 < ST_BK; k8 += 8) {
                simdgroup_matrix<T, 8, 8> b[ST_TN];
                for (int tn = 0; tn < ST_TN; tn++) simdgroup_load(b[tn], ws + k8 * ST_BN + tn * 8, ST_BN);
                for (int tm = 0; tm < ST_TM; tm++) {
                    simdgroup_matrix<T, 8, 8> a;
                    simdgroup_load(a, xg + tm * 8 * ld + k0 + k8, ld);
                    for (int tn = 0; tn < ST_TN; tn++)
                        simdgroup_multiply_accumulate(acc[tm][tn], a, b[tn], acc[tm][tn]);
                }
            }
        }
        const int node = int(block) * ST_BN + col;
        for (int tn = 0; tn < ST_TN; tn++) {
            float2 bias_tn = HALF ? *(device const float2*)(bias + node + tn * 8) : float2(0.0f);
            for (int tm = 0; tm < ST_TM; tm++) {
                auto e = acc[tm][tn].thread_elements();
                update<TOP2>(e[0] * product_scale + bias_tn[0], node + tn * 8, b1[tm], j1[tm], b2[tm], j2[tm]);
                update<TOP2>(e[1] * product_scale + bias_tn[1], node + tn * 8 + 1, b1[tm], j1[tm], b2[tm], j2[tm]);
            }
        }
    }
    for (int tm = 0; tm < ST_TM; tm++)
        store_row<TOP2>(lane, out_node, out_score, n, row0 + tm * 8 + frag_row(lane), tg.y, b1[tm], j1[tm], b2[tm],
                        j2[tm]);
}

#define BMU_STAGED(name, TOP2, T)                                                                                    \
    template [[host_name(name)]] kernel void bmu_staged<TOP2, T>(                                                    \
        device const T*, device const T*, device const float*, device long*, device float*, constant long*,           \
        device const float*, uint2, ushort, ushort, ushort);
BMU_STAGED("bmu_staged", false, float)
BMU_STAGED("bmu_staged_top2", true, float)
BMU_STAGED("bmu_staged_half", false, half)

// adds value to the 64-bit integer at word (its low word first) by two 32-bit atomic additions, as Apple GPUs have no
// 64-bit one: the low word's returns its old value, from which follows the carry into the high word, so that the sum
// is exact (mod 2^64) in any order
inline void add64(device atomic_uint* word, long value) {
    uint low = uint(ulong(value)), high = uint(ulong(value) >> 32);
    uint old = atomic_fetch_add_explicit(word, low, memory_order_relaxed);
    high += old + low < old;
    if (high) atomic_fetch_add_explicit(word + 1, high, memory_order_relaxed);
}

// fit's deterministic sums: adds each sample's features to its BMU's as fixed point, x * 2^shift (exact) rounded to an
// integer into sums and the remainder times 2^low_shift, rounded, into lows, int64 per node and feature, so that the
// sums do not depend on the order of the additions. A thread per sample and feature; zeros add nothing, and the
// remainder is 0 for all but the smallest values. p: samples, x's row stride, low_shift
kernel void fixed_sums(
    device const float* x [[buffer(0)]],
    device const long* bmus [[buffer(1)]],
    device const int* shift [[buffer(2)]],
    device atomic_uint* sums [[buffer(3)]],
    device atomic_uint* lows [[buffer(4)]],
    constant long* p [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    const long n = p[0], ld = p[1], s = i / D;
    const int k = i % D;
    if (s >= n) return;
    float v = x[s * ld + k];
    if (v == 0.0f) return;
    float scaled = ldexp(v, shift[k]), high = rint(scaled), rest = scaled - high;  // both exact
    long word = 2 * (bmus[s] * D + k);
    add64(sums + word, long(high));
    if (rest != 0.0f) add64(lows + word, long(rint(ldexp(rest, int(p[2])))));
}

#if L1
// the node j of least ||x - w_j||_1 per sample x. w holds the nodes in blocks of L1_BN as (block, D, L1_BN); padded
// nodes are infinite. p as for bmu_dot
template <bool TOP2>
kernel void bmu_l1(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device long* out_node [[buffer(2)]],
    device float* out_score [[buffer(3)]],
    constant long* p [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort tid [[thread_index_in_threadgroup]])
{
    const long n = p[0], ld = p[1], n_blocks = p[2], per_part = p[3];
    threadgroup float xs[D * L1_BM];  // the threadgroup's samples, transposed (features x samples)
    const long row0 = long(tg.x) * L1_BM;
    for (int i = tid; i < L1_BM * D; i += L1_THREADS) {
        int r = i / D, c = i % D;
        long s = row0 + r;
        xs[c * L1_BM + r] = s < n ? x[s * ld + c] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int t = tid % L1_TX, u = tid / L1_TX;  // the thread's nodes t * L1_RN + j of a block, samples u * L1_RM + i
    float b1[L1_RM], b2[L1_RM];
    int j1[L1_RM], j2[L1_RM];
    for (int i = 0; i < L1_RM; i++) {
        b1[i] = b2[i] = INFINITY;
        j1[i] = j2[i] = 0;
    }
    const long first = long(tg.y) * per_part, last = min(first + per_part, n_blocks);
    for (long block = first; block < last; block++) {
        device const float* wb = w + block * (D * L1_BN) + t * L1_RN;
        float acc[L1_RM][L1_RN] = {};
        for (int k = 0; k < D; k++) {
            float xv[L1_RM], wv[L1_RN];
            for (int i = 0; i < L1_RM; i++) xv[i] = xs[k * L1_BM + u * L1_RM + i];
            for (int j = 0; j < L1_RN; j++) wv[j] = wb[k * L1_BN + j];
            for (int i = 0; i < L1_RM; i++)
                for (int j = 0; j < L1_RN; j++) acc[i][j] += fabs(xv[i] - wv[j]);
        }
        const int node = int(block) * L1_BN + t * L1_RN;
        for (int i = 0; i < L1_RM; i++)
            for (int j = 0; j < L1_RN; j++) update<TOP2>(acc[i][j], node + j, b1[i], j1[i], b2[i], j2[i]);
    }
    for (int i = 0; i < L1_RM; i++) {
        for (ushort mask = 1; mask < L1_TX; mask <<= 1) merge<TOP2>(mask, b1[i], j1[i], b2[i], j2[i]);
        long row = row0 + u * L1_RM + i;
        if (t == 0 && row < n) store<TOP2>(out_node, out_score, n, row, tg.y, b1[i], j1[i], b2[i], j2[i]);
    }
}

#define BMU_L1(name, TOP2)                                                                                           \
    template [[host_name(name)]] kernel void bmu_l1<TOP2>(                                                           \
        device const float*, device const float*, device long*, device float*, constant long*, uint2, ushort, ushort);
BMU_L1("bmu_l1", false)
BMU_L1("bmu_l1_top2", true)
#endif
"""


def _fragments(n_features: int) -> int:
    """Fragments of 8 features of bmu_dot, with the bias column if there is room."""
    return -(-n_features // 8)


def _l1_samples(n_features: int) -> int:
    """Samples per thread of the manhattan kernel.

    Samples per thread of bmu_l1: 8, or fewer for its threadgroup's samples to fit in its threadgroup memory; 0
    beyond 512 features.
    """
    return min(8, _THREADGROUP_MEMORY // (4 * n_features * (_L1_THREADS // _L1_TX)))


def _use_staged(n_nodes: int, n_features: int) -> bool:
    """Whether the dot-product search takes bmu_staged rather than bmu_dot (see _REGISTER_FRAGMENTS)."""
    fragments = _fragments(n_features)
    return fragments > _REGISTER_FRAGMENTS or n_nodes * fragments * 32 > _REGISTER_NODE_BYTES


def supported(n_features: int, manhattan: bool) -> bool:
    """Whether the Metal kernels support this many features.

    Whether the kernels take this many features: any under euclidean, cosine and correlation, up to 512 under
    manhattan, whose threadgroup's samples must fit in its threadgroup memory.
    """
    return not manhattan or _l1_samples(n_features) >= 1


def sample_width(n_features: int) -> int:
    """Floats per sample of the largest copy of a batch that the search makes, bmu_staged's (see _staged_samples)."""
    return (n_features + _STAGED_BK) // _STAGED_BK * _STAGED_BK


@cache
def _library(n_features: int):
    """The kernels for this many features: bmu_staged, and bmu_dot and bmu_l1 if they take it."""
    switches = {"DOT": _fragments(n_features) <= _REGISTER_FRAGMENTS, "L1": supported(n_features, True)}
    constants = {
        "D": n_features,
        "DOT_TN": _DOT_TN,
        "DOT_SG": _DOT_SG,
        "ST_TM": _STAGED_TM,
        "ST_TN": _STAGED_TN,
        "ST_SG": _STAGED_SG,
        "ST_BK": _STAGED_BK,
        "L1_THREADS": _L1_THREADS,
        "L1_TX": _L1_TX,
        "L1_RN": _L1_RN,
        "L1_RM": max(1, _l1_samples(n_features)),
    }
    header = "#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\n"
    header += "".join(f"#define {name} {int(on)}\n" for name, on in switches.items())
    header += "".join(f"constant constexpr int {name} = {value};\n" for name, value in constants.items())
    return torch.mps.compile_shader(header + _SOURCE)


def _parts(n_samples: int, block_m: int, n_blocks: int, parts: int | None) -> tuple[int, int]:
    """(parts of the nodes, blocks per part): as few parts as give _MIN_THREADGROUPS threadgroups."""
    if parts is None:
        parts = min(n_blocks, -(-_MIN_THREADGROUPS // -(-n_samples // block_m)))
    per_part = -(-n_blocks // max(1, parts))
    return -(-n_blocks // per_part), per_part


def _padded_bias(nodes: torch.Tensor, bias: torch.Tensor | None, n_padded: int) -> torch.Tensor:
    """The bias of each node (0 without one), infinite for the padded nodes, which so never win."""
    padded = nodes.new_full((n_padded,), float("inf"))
    padded[: len(nodes)] = 0 if bias is None else bias
    return padded


# the magnitudes of samples that precision="float16" takes (else float32), so that all scales of half_scales stay
# within float32's normal exponents
HALF_RANGE = (2.0**-30, 2.0**30)


def in_half_range(magnitude: torch.Tensor) -> bool:
    """Whether the samples' largest |x| per feature lie within HALF_RANGE (or are 0)."""
    low, high = HALF_RANGE
    return bool(((magnitude == 0) | ((magnitude >= low) & (magnitude <= high))).all())


def _power_of_2(exponent: torch.Tensor) -> torch.Tensor:
    """2^exponent (integers within float32's normal exponents) as float32, exact, from its bits."""
    return ((exponent.int() + 127) << 23).view(torch.float32)


def _exponent(magnitude: torch.Tensor) -> torch.Tensor:
    """Integer upper bound on the base-2 exponent of each element.

    An integer e per element with magnitude < 2^e, on the device, as frexp is not on MPS; log2's rounding can only
    raise it by 1. Magnitudes are clamped to HALF_RANGE's low end and 2^34, beyond the largest the scales see (nodes
    -2 (w - c) of samples within 2^30 reach 2^32), so that every scaled value stays below 2^8 (2^8 after rounding to
    half); the products of half inputs are exact in float32 (on an M3 Max 256 x 256 and 512 x 256 come out exactly).
    """
    return torch.floor(torch.log2(magnitude.float().clamp(HALF_RANGE[0], 2.0**34))) + 1


def half_scales(nodes: torch.Tensor, bound: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-feature power-of-2 scales for float16 searches.

    For precision float16: powers of 2 per feature for the samples, whose |x - centre| is at most bound, and for the
    nodes, that put both below about 2^8 in half (65504 at most, 2^-14 the smallest normal) and every feature's
    products at 2^g times their value: (the samples' scales with 2^-g appended, the nodes' scales). Computed on the
    device, as synchronising in every epoch would stall fit on small maps.
    """
    exponent_x, exponent_w = _exponent(bound), _exponent(nodes.abs().amax(0))
    g = 16 - (exponent_x + exponent_w).amax()
    return torch.cat([_power_of_2(8 - exponent_x), _power_of_2(-g)[None]]), _power_of_2(g - 8 + exponent_x)


def _dot_nodes(
    nodes: torch.Tensor, bias: torch.Tensor | None, w_scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Lay out the nodes for bmu_dot.

    The nodes as bmu_dot loads them: (fragments, padded bias, blocks); with w_scale scaled, in half, without the
    bias column.
    """
    n_nodes, n_features = nodes.shape
    block, kf = _DOT_TN * 8, _fragments(n_features)
    n_blocks = -(-n_nodes // block)
    padded_bias = _padded_bias(nodes, bias, n_blocks * block)
    full = nodes.new_zeros(n_blocks * block, kf * 8)
    full[:n_nodes, :n_features] = nodes if w_scale is None else nodes * w_scale
    if n_features % 8 and w_scale is None:
        full[:, n_features] = padded_bias
    fragments = full.view(n_blocks, _DOT_TN, 8, kf, 8).permute(0, 3, 1, 4, 2).contiguous()
    return (fragments if w_scale is None else fragments.half()), padded_bias, n_blocks


def _staged_nodes(
    nodes: torch.Tensor, bias: torch.Tensor | None, w_scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Lay out the nodes for bmu_staged.

    The nodes as bmu_staged loads them, (features, nodes) with the bias in feature n_features: (nodes, padded bias,
    blocks); with w_scale scaled, in half, with zeros in feature n_features and the bias added after the products.
    """
    n_nodes, n_features = nodes.shape
    block = _STAGED_TN * 8
    n_blocks = -(-n_nodes // block)
    padded_bias = _padded_bias(nodes, bias, n_blocks * block)
    full = nodes.new_zeros(sample_width(n_features), n_blocks * block)
    if w_scale is None:
        full[:n_features, :n_nodes] = nodes.T
        full[n_features] = padded_bias
        return full, padded_bias, n_blocks
    full[:n_features, :n_nodes] = (nodes * w_scale).T
    return full.half(), padded_bias, n_blocks


def _staged_samples(x: torch.Tensor, centre: torch.Tensor | None, x_scale: torch.Tensor | None) -> torch.Tensor:
    """Lay out the samples for bmu_staged.

    The samples as bmu_staged loads them: x - centre with a column of ones for the bias, zeros beyond, and rows to a
    multiple of a threadgroup's; with x_scale scaled, in half, without the column of ones.
    """
    n, n_features = x.shape
    block = _STAGED_SG * _STAGED_TM * 8
    out = x.new_zeros(
        -(-n // block) * block, sample_width(n_features), dtype=torch.half if x_scale is not None else None
    )
    if x_scale is not None:
        out[:n, :n_features] = (x if centre is None else x - centre) * x_scale
    elif centre is None:
        out[:n, :n_features] = x
    else:  # without a copy of x - centre
        torch.sub(x, centre, out=out[:n, :n_features])
    if x_scale is None:
        out[:n, n_features] = 1
    return out


def _l1_nodes(nodes: torch.Tensor) -> tuple[torch.Tensor, int]:
    """The nodes as bmu_l1 loads them: (block, features, nodes), infinite beyond the nodes."""
    n_nodes, n_features = nodes.shape
    block = _L1_TX * _L1_RN
    n_blocks = -(-n_nodes // block)
    full = nodes.new_full((n_blocks * block, n_features), float("inf"))
    full[:n_nodes] = nodes
    return full.view(n_blocks, block, n_features).transpose(1, 2).contiguous(), n_blocks


def _in_parts(find: Callable[[torch.Tensor], torch.Tensor], samples: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """find on parts of at most this many samples, concatenated."""
    return lambda x: find(x) if len(x) <= samples else torch.cat([find(part) for part in x.split(samples)])


def _rows(x: torch.Tensor) -> torch.Tensor:
    """x with contiguous rows, as bmu_dot and bmu_l1 read it (any row stride)."""
    return x if x.stride(1) == 1 else x.contiguous()


def _bmus(
    x: torch.Tensor,
    n_samples: int,
    kernel,
    args: tuple,
    block_m: int,
    threads: int,
    n_blocks: int,
    top2: bool,
    parts: int | None,
    ld_w: int = 0,
    after: tuple = (),
) -> torch.Tensor:
    """Run a BMU kernel on the first n_samples rows of x.

    The BMUs of the first n_samples rows of x (as the kernel loads them) by the kernel, given its node arguments,
    samples and threads per threadgroup, blocks of nodes, for bmu_staged the nodes' row stride, and the kernel's
    arguments after the parameters.
    """
    if n_samples == 0:  # e.g. none of the samples has a BMU, which leaves no threadgroups to split the nodes over
        return torch.empty((0, 2) if top2 else (0,), dtype=torch.long, device=x.device)
    n_parts, per_part = _parts(n_samples, block_m, n_blocks, parts)
    rows = n_parts * (2 if top2 else 1)
    node = torch.empty(rows, n_samples, dtype=torch.long, device=x.device)
    score = torch.empty(rows, n_samples, dtype=torch.float32, device=x.device)
    params = [n_samples, x.stride(0), n_blocks, per_part, ld_w]
    grid = (-(-n_samples // block_m) * threads, n_parts)
    kernel(x, *args, node, score, params, *after, threads=grid, group_size=(threads, 1))
    if top2:
        return node.T if n_parts == 1 else node.gather(0, score.topk(2, 0, largest=False).indices).T
    return node[0] if n_parts == 1 else node.gather(0, score.argmin(0, keepdim=True))[0]


def search(
    nodes: torch.Tensor,
    bias: torch.Tensor | None,
    centre: torch.Tensor | None,
    manhattan: bool,
    top2: bool = False,
    *,
    bound: torch.Tensor | None = None,
    parts: int | None = None,
    staged: bool | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the BMU search of a Metal kernel variant.

    Returns a function of samples x, shape (n, n_features), giving each one's BMU (int64): the node j with the
    least (x - centre) . nodes_j + bias_j, or ||x - nodes_j||_1 if manhattan, the first on ties; with top2 the best
    two, shape (n, 2), in either order on ties, as torch's topk. With bound, the most |x - centre| per feature of the
    samples to come (within HALF_RANGE), dot products take half inputs, scaled by half_scales (precision float16, not
    with top2, which only topographic_error takes). parts sets the parts the nodes are split into (else chosen from the
    batch), whose results torch then reduces; staged whether bmu_staged searches dot products (else chosen from the
    sizes).
    """
    if bound is not None and (top2 or manhattan):
        raise ValueError("the half kernels take neither top2 nor manhattan")
    nodes = nodes.float().contiguous()
    n_nodes, n_features = nodes.shape
    library = _library(n_features)
    suffix = "_top2" if top2 else ""
    if manhattan:
        w, n_blocks = _l1_nodes(nodes)
        kernel, block_m = getattr(library, "bmu_l1" + suffix), _L1_THREADS // _L1_TX * _l1_samples(n_features)

        def find_l1(x: torch.Tensor) -> torch.Tensor:
            return _bmus(_rows(x), len(x), kernel, (w,), block_m, _L1_THREADS, n_blocks, top2, parts)

        return _in_parts(find_l1, _MAX_THREADS // _L1_THREADS * block_m)
    centre = None if centre is None else centre.float().contiguous()
    x_scale, w_scale = (None, None) if bound is None else half_scales(nodes, bound)
    suffix += "" if bound is None else "_half"
    if staged is None:
        staged = _use_staged(n_nodes, n_features)
    if staged:
        w, padded_bias, n_blocks = _staged_nodes(nodes, bias, w_scale)
        kernel, block_m, threads = getattr(library, "bmu_staged" + suffix), _STAGED_SG * _STAGED_TM * 8, _STAGED_SG * 32

        after = (padded_bias if x_scale is None else x_scale[-1:],)  # the products' scale, not read in float32

        def find_staged(x: torch.Tensor) -> torch.Tensor:
            samples = _staged_samples(x, centre, None if x_scale is None else x_scale[:-1])
            args = (w, padded_bias)
            return _bmus(samples, len(x), kernel, args, block_m, threads, n_blocks, top2, parts, w.shape[1], after)

        return _in_parts(find_staged, _MAX_THREADS // threads * block_m)
    w, padded_bias, n_blocks = _dot_nodes(nodes, bias, w_scale)
    kernel = getattr(library, "bmu_dot" + ("_centre" if centre is not None else "") + suffix)
    args = (padded_bias if centre is None else centre, w, padded_bias)  # the centre is not read without one
    after = (padded_bias if x_scale is None else x_scale,)  # nor the scale in float32

    def find(x: torch.Tensor) -> torch.Tensor:
        return _bmus(_rows(x), len(x), kernel, args, _DOT_SG * 8, _DOT_SG * 32, n_blocks, top2, parts, 0, after)

    return _in_parts(find, _MAX_THREADS // (_DOT_SG * 32) * _DOT_SG * 8)


def fixed_sums(
    x: torch.Tensor, bmus: torch.Tensor, shift: torch.Tensor, low_shift: int, sums: torch.Tensor, lows: torch.Tensor
) -> None:
    """Add samples to the fixed-point sums of their BMUs.

    Adds the samples x (n, n_features) to the fixed-point sums of their BMUs bmus (int64), in place: x * 2^shift
    (int32 per feature) rounded to sums, the remainders times 2^low_shift rounded to lows, both int64 (nodes,
    n_features); see the kernel. Subnormal values (below 2^-126) count as 0, as Apple GPUs flush them.
    """
    library, x = _library(x.shape[1]), _rows(x)
    rows = _MAX_THREADS // x.shape[1]
    for part, part_bmus in zip(x.split(rows), bmus.split(rows), strict=True):
        params = [len(part), part.stride(0), low_shift]
        library.fixed_sums(part, part_bmus, shift, sums, lows, params, threads=part.numel(), group_size=256)


def check(device: torch.device, n_features: int, manhattan: bool, bias: bool, top2: bool, half: bool = False) -> None:
    """Check the Metal kernels for these arguments on a small search.

    Raises unless the kernels for these arguments find the exact BMUs of a small search, with the nodes in one part
    and in several, for dot products by bmu_staged and, if it takes this many features, bmu_dot, with half the half
    variants. 250 samples and 250 nodes fill no tile, so that the partial tiles and padded nodes run too, and with bias
    the samples have an integer centre. Its values are -1, 0 or 1, for dot products times 2^-3 to 2^3 by feature (the
    nodes divided by it), which the half variants must undo, exact also in half and scaled by powers of 2, so that every
    product is -1, 0 or 1 and every sum an integer below 2^24, exact in float32 in any order, also in the reference on
    the CPU. Each node is there twice, in different parts of the nodes, so that ties must go to the first.
    """
    generator = torch.Generator().manual_seed(0)
    # an explicit generator and dtype keep it on the CPU in float32, which torch.set_default_device or
    # set_default_dtype would change otherwise
    x, nodes, offsets = torch.randint(-1, 2, (3, 250, n_features), generator=generator, dtype=torch.float32)
    nodes[125:] = nodes[:125]
    centre = offsets[1] if bias else None
    if not manhattan:  # features of different scales, 2^-3 to 2^3, and the nodes by the inverse: the same products
        scale = torch.ldexp(torch.ones(n_features), torch.arange(n_features) % 7 - 3)
        x, nodes, centre = x * scale, nodes / scale, None if centre is None else centre * scale
    if manhattan:
        scores = torch.cdist(x, nodes, p=1)
    else:
        scores = (x if centre is None else x - centre) @ nodes.T + (offsets[:, 0] if bias else 0)
    best2 = scores.topk(2, 1, largest=False).values
    b, c = (None, None) if centre is None else (offsets[:, 0].to(device), centre.to(device))  # a centre iff bias
    bound = None if not half else (x if centre is None else x - centre).abs().amax(0).to(device)
    kernels = [True, False] if _fragments(n_features) <= _REGISTER_FRAGMENTS else [True]  # bmu_staged, bmu_dot
    for staged, parts in product([None] if manhattan else kernels, (1, 4)):
        find = search(nodes.to(device), b, c, manhattan, top2, bound=bound, parts=parts, staged=staged)
        found = find(x.to(device)).cpu()
        if top2:
            exact = torch.equal(scores.gather(1, found).sort(1).values, best2) and bool(
                (found[:, 0] != found[:, 1]).all()
            )
        else:
            exact = torch.equal(found, scores.argmin(1))
        if not exact:
            raise RuntimeError("wrong BMUs in a self-check")
