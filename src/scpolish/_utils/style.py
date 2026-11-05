"""Scoped, restrained defaults for scientific figures."""

import matplotlib as mpl

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
