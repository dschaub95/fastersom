# Reproducibility and precision

## Seeds

`random_state` seeds the initialisation, with NumPy's `RandomState`: `init="random"` (the default) draws the node
weights uniformly from $[-1, 1]^d$ and normalises them to unit length, and `init="sample"` picks random samples (ones
with a BMU under the metric). Training itself is not random: every epoch updates the map from all samples. With an
array as `init`, `random_state` has no effect.

## Deterministic fitting on GPUs

On the CPU, `fit` gives the same map in every run with the same data and parameters.

On CUDA and MPS, `fit` sums the samples per node with atomic additions, whose order varies from run to run. The sums
then differ in their last bits, and over several epochs this can flip the BMUs of near-tied samples, so that two runs
give equally good but not identical maps. `deterministic=True` makes them identical, for a fixed batch size:

```python
som = SOM(30, 30, random_state=0, deterministic=True)
```

- On CUDA, and on MPS with the fused Metal kernels, the samples are added as 64-bit fixed-point integers, whose sum
  does not depend on the order. On an A100 this makes `fit` up to 2x slower on small maps and 3-10% slower from
  100 x 100 nodes; on an M3 Max 1.4x slower at 10 x 10 nodes, 1.14x at 30 x 30 and no slower from 100 x 100.
- On MPS without the Metal kernels (`fused=False`, or torch < 2.6), the sums take a matrix product with a one-hot
  `nodes x batch` matrix, which sums in a fixed order (1.2-1.8x slower).

`deterministic=True` is also used when {func}`torch.use_deterministic_algorithms` is enabled.

Determinism holds for the same device, software versions and batch size. On GPUs the default batch size depends on
the free memory at the time of the fit, so pass an explicit `batch_size` for identical maps across runs. Maps fitted on different devices, or with
different batch sizes, still differ by float rounding, which can flip near-tied BMUs: they agree in quality (similar
quantization and topographic errors) rather than bit for bit.

## Precision

fastersom computes in float32 by default (`precision="float32"`) on every device: the data and the nodes are float32,
and so are the BMU search and the update. On A100, H100 and B200 GPUs, the fused CUDA kernel uses the tensor cores
with three TF32 products per float32 product (3xTF32), which keeps float32 accuracy (see [](devices.md)). The BMUs of
the fused kernels and of the torch path differ only on near-ties.

### `precision="float16"`

`precision="float16"` lets the BMU search of `fit` round samples and nodes to float16's 11 significant bits, summing
the products in float32. In fastersom 0.5.0 it applies only:

- on MPS with the fused Metal kernels,
- under the euclidean, cosine and correlation metrics,
- from 16 features, and
- when the samples and the initial weights are within $2^{-30}$ to $2^{30}$ in magnitude, as the values are scaled by
  powers of 2 per feature into float16's range.

Elsewhere, including on CUDA and the CPU, the search stays float32. `labels_`, `predict`, `transform` and the error
measures are always computed in float32.

The BMUs of near-tied samples then flip, each within the rounding error, more often on larger maps and depending on
the data: on clustered 0-1 pixels of 34 features about 1% of the samples at 10 x 10 nodes and 10-35% from 30 x 30
nodes. This gives an equally good map rather than the same one: in the tests of fastersom's authors the same
quantization error, but a topographic error up to 1.4x higher on synthetic clusters.

The option is meant for GPUs whose float16 matrix products are much faster than their float32 ones, such as Apple's
M5, on which it has not been tested. On an M3 Max, whose float16 products are no faster, `fit` is only 1.00-1.05x
faster with it.

```python
som = SOM(30, 30, device="mps", precision="float16").fit(X)  # X with at least 16 features
```
