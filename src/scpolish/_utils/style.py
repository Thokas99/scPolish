"""Scoped, restrained defaults for scientific figures."""

import matplotlib as mpl
import numpy as np

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
}


def style_context():
    """Return a temporary Matplotlib rc context."""
    return mpl.rc_context(STYLE)


def auto_point_size(n_cells: int) -> float:
    """Return marker area in square points, scaled to the number of cells."""
    return float(np.clip(2.5 * np.sqrt(10_000 / max(n_cells, 1)), 0.8, 10.0))
