"""Fast batch self-organising maps in PyTorch."""

from importlib.metadata import version

from ._som import SOM, BatchSOM

__all__ = ["SOM", "BatchSOM"]
__version__ = version("fastersom")
