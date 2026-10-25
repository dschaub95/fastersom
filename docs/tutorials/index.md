# Tutorials

The tutorials are Jupyter notebooks; the buttons at the top of each page download them. Besides fastersom they use
[AnnData](https://anndata.readthedocs.io) {cite:p}`Virshup_2024` and [scanpy](https://scanpy.readthedocs.io)
{cite:p}`Wolf_2018` from the [scverse](https://scverse.org) ecosystem {cite:p}`Virshup_2023`:

```bash
pip install fastersom scanpy igraph matplotlib readfcs
```

They fit on the CPU, so that rerunning them on the same machine and software versions reproduces the stored outputs;
set `DEVICE = None` to use a GPU.

```{toctree}
:maxdepth: 1

cytometry
pixel_clustering
```
