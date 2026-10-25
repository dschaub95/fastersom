# Initialisation

The batch map of {class}`~fastersom.SOM` starts from the node weights given by `init`, and its result depends on them:
each epoch moves every node to a weighted mean of the samples nearest to it and its neighbours, so a different start
can end in a different map.

`"random"` (the default)
: Unit vectors drawn uniformly from $[-1, 1]^d$ with `random_state`. They do not depend on the data, which the wide
  initial neighbourhood then pulls the map towards. Data far from the unit sphere, such as intensities between 0 and
  255, start far from all nodes.

`"sample"`
: Randomly chosen samples, so that every node starts at a value that occurs in the data. Repeated values, such as the
  many identical background pixels of an image, are drawn as often as they occur.

`"pca"`
: Linear initialisation {cite:p}`Kohonen_2013`, defined as in the SOM Toolbox {cite:p}`Vesanto_2000`. The nodes lie on
  a regular grid in the plane of the samples' first two principal components: centred at the mean, with the longer
  side of the map along the first component, and each side spanning one standard deviation of its component to either
  side. Unit $(r, c)$ of an $R \times C$ map (with $R \ge C$) starts at

  $$
  w_{rc} = \mu + s_r \sqrt{\lambda_1}\, v_1 + t_c \sqrt{\lambda_2}\, v_2,
  \qquad s_r = \frac{2r}{R - 1} - 1, \quad t_c = \frac{2c}{C - 1} - 1,
  $$

  where $\mu$ is the mean, $\lambda_1 \ge \lambda_2$ the two largest eigenvalues of the covariance between the
  features and $v_1$, $v_2$ their eigenvectors (each with its largest entry positive). A side of one node stays at the
  mean. Under `metric="cosine"` and `"correlation"`, the components are those of the samples as the metric compares
  them: scaled to unit length and, for correlation, centred over the features. The map starts ordered, so that fewer
  epochs may suffice; whether that holds depends on the data.

An array
: Given weights, for example another map's `weights_` to continue training (see `sigma_initial`).

```python
from fastersom import SOM

som = SOM(30, 30, init="pca", random_state=0).fit(X)
```

The covariance is computed in two passes over the samples, each batch on the device and the sum in double precision
on the CPU, so that large inputs lose no precision.
