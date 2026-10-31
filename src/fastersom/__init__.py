"""Fast batch self-organising maps in PyTorch."""

from importlib.metadata import version

from ._mst import SpanningTree, minimum_spanning_tree
from ._som import SOM, BatchSOM

__all__ = ["SOM", "BatchSOM", "SpanningTree", "minimum_spanning_tree"]
__version__ = version("fastersom")
