"""Fast batch self-organising maps in PyTorch."""

from importlib.metadata import version

from ._kmeans import KMeans
from ._som import SOM, BatchSOM

__all__ = ["SOM", "BatchSOM", "KMeans"]
__version__ = version("fastersom")
