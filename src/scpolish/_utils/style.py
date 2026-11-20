"""Scoped, restrained defaults for scientific figures."""

from numbers import Real

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


def validate_positive_number(value: Real, *, name: str) -> float:
    """Return a finite positive plotting value as a Python float."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number.")
    try:
        value = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number.") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return value


def validate_figsize(figsize: tuple[float, float] | None) -> tuple[float, float] | None:
    """Validate and normalize a Matplotlib figure size."""
    if figsize is None:
        return None
    try:
        width, height = figsize
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "figsize must contain two finite, positive dimensions."
        ) from exc
    return (
        validate_positive_number(width, name="figsize width"),
        validate_positive_number(height, name="figsize height"),
    )
