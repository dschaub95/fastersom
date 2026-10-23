"""Downloads the spatiomic tutorial images and preprocesses them as the tutorial does up to its SOM step.

Mirrors cells 11-29 of https://spatiomic.org/latest/tutorials/full_example.html and writes the subsample the SOM is
fitted on, 480,000 x 34 pixels, to the given .npy path. Needs spatiomic, which memory.py installs into a temporary
environment for this step:

    uv run --with spatiomic==0.9.3 python benchmarks/data.py subsample_processed.npy
"""

import hashlib
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://data.complextissue.com/spatiomic/tutorial/data/"
IMAGES = ["245 g1", "245 g2", "247 g2", "247 g4", "248 g2", "248 g4"]
IMAGES += ["263 g2", "263 g4", "265 g2", "265 g3", "266 g4", "266 g5"]
SPATIOMIC = "spatiomic==0.9.3"
SHA256 = "0f6382d020d07787857a6017c7a25c3616764b3a539e2bbdde16112bae3bf611"


def download(name: str, path: str) -> None:
    # the server rejects urllib's default user agent
    request = urllib.request.Request(URL + name.replace(" ", "%20"), headers={"User-Agent": "fastersom-benchmarks"})
    with urllib.request.urlopen(request) as response, open(path, "wb") as file:
        shutil.copyfileobj(response, file)


def prepare(path: Path) -> None:
    import pandas as pd
    import spatiomic as so
    from skimage.transform import downscale_local_mean

    with tempfile.TemporaryDirectory() as raw:
        download("channel_names.csv", f"{raw}/channel_names.csv")
        channels_data = list(pd.read_csv(f"{raw}/channel_names.csv", header=0).T.to_numpy()[0])
        channels = sorted(channel for channel in channels_data if "DAPI" not in channel)
        subsamples = []
        for image in IMAGES:
            print(f"downloading {image}.tif", file=sys.stderr)
            download(f"{image}.tif", f"{raw}/image.tif")
            data = so.data.read().read_tiff(f"{raw}/image.tif", input_dimension_order="CYX")
            data = downscale_local_mean(data, factors=(4, 4, 1))
            data = so.data.subset(data, channel_names_data=channels_data, channel_names_subset=channels)
            subsamples.append(so.data.subsample().fit_transform(data, method="count", count=40000, seed=0))

    subsample = np.concatenate(subsamples, axis=0)
    min_values = np.percentile(subsample, 75.0, axis=0)
    max_values = np.percentile(subsample, 99.95, axis=0)
    clipper = so.process.clip(method="minmax", min_value=min_values, max_value=max_values, use_gpu=False)
    normalizer = so.process.normalize(min_value=0.0, max_value=1.0, use_gpu=False)
    np.save(path, normalizer.fit_transform(clipper.fit_transform(subsample)).astype(np.float32))

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        path.unlink()
        raise RuntimeError(f"{path} has sha256 {digest}, not {SHA256}; the data or its preprocessing changed")


if __name__ == "__main__":
    prepare(Path(sys.argv[1]))
