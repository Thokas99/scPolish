"""Publication-ready single-cell figures with minimal code."""

from importlib.metadata import PackageNotFoundError, version

from scpolish.pl import dimplot, featureplot, violinplot
from scpolish.plotting import embedding

try:
    __version__ = version("scpolish")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

__all__ = ["embedding", "dimplot", "featureplot", "violinplot", "__version__"]
