"""Shared plotting defaults and small layout helpers."""

from collections.abc import Iterable
from math import ceil
from numbers import Real
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection

from scpolish._utils.style import (
    auto_point_size,
    validate_figsize,
    validate_positive_number,
)

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10,
    "pdf.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": None,
}


def prepare(
    adata: AnnData, basis: str, *, obs_columns: Iterable[str] | None = None
) -> tuple[AnnData, str]:
    """Isolate plotting metadata without copying the expression matrix."""
    if not isinstance(basis, str):
        raise ValueError("basis must be a string.")
    basis = basis.removeprefix("X_")
    key = f"X_{basis}"
    if key not in adata.obsm:
        raise ValueError(
            f"Embedding '{key}' not found in adata.obsm. "
            f"Compute it first using sc.tl.{basis}()."
        )
    coords = np.asarray(adata.obsm[key])
    if adata.n_obs == 0 or coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("Plotting requires cells and at least two embedding columns.")
    if not np.isfinite(coords[:, :2]).all(axis=1).any():
        raise ValueError("Embedding has no finite coordinates.")
    obs = (
        adata.obs.copy()
        if obs_columns is None
        else adata.obs[
            list(dict.fromkeys(c for c in obs_columns if c in adata.obs))
        ].copy()
    )
    obj = AnnData(obs=obs)
    obj.obsm[key] = coords[:, :2].copy()
    return obj, basis


def panel_shape(n: int, ncols: int | None = None) -> tuple[int, int]:
    """Choose a compact publication grid, honoring explicit column counts."""
    if n < 1:
        raise ValueError("At least one panel is required.")
    if ncols is not None and (
        not isinstance(ncols, int) or isinstance(ncols, bool) or ncols < 1
    ):
        raise ValueError("ncols must be a positive integer.")
    cols = min(n, ncols or (1 if n == 1 else 3 if n == 3 or n >= 5 else 2))
    return ceil(n / cols), cols


def feature_names(features: str | Iterable[str]) -> list[str]:
    """Normalize and validate feature names shared by feature plots."""
    try:
        names = [features] if isinstance(features, str) else list(features)
    except TypeError as exc:
        raise ValueError("features must contain unique, non-empty names.") from exc
    if (
        not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("features must contain unique, non-empty names.")
    return names


def panels(n: int, ncols: int | None, figsize: tuple | None, ax: Axes | None):
    """Return figure and a two-dimensional array of embedding axes."""
    figsize = validate_figsize(figsize)
    rows, cols = panel_shape(n, ncols)
    if ax is not None:
        if n != 1:
            raise ValueError("ax is only supported for a single panel.")
        return ax.figure, np.array([[ax]])
    if figsize is None:
        figsize = (12, 8) if n == 1 else (4.5 * cols, 5.5 * rows)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=figsize,
        squeeze=False,
        layout="constrained",
    )
    for unused in axes.flat[n:]:
        unused.set_visible(False)
    return fig, axes


def finish(
    ax: Axes, basis: str, *, raster: bool = False, aspect: str = "equal"
) -> None:
    """Style embedding axes, leaving colorbars alone."""
    name = {"umap": "UMAP", "pca": "PC", "tsne": "tSNE"}.get(basis, basis.upper())
    ax.set(xlabel=f"{name} 1", ylabel=f"{name} 2", xticks=[], yticks=[], aspect=aspect)
    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for collection in ax.collections:
        if isinstance(collection, PathCollection):
            collection.set_rasterized(raster)


def export(fig, save: str | Path | None, dpi: int) -> None:
    if save is not None:
        dpi = validate_positive_number(dpi, name="dpi")
        fig.savefig(
            save,
            dpi=dpi,
            facecolor="white",
            format=Path(save).suffix.lstrip(".") or "pdf",
            bbox_inches=None,
        )


def point_size(n: int, size: float | None, point_scale: float) -> float:
    point_scale = validate_positive_number(point_scale, name="point_scale")
    result = auto_point_size(n) * point_scale if size is None else size
    return validate_positive_number(result, name="size")


def validate_alpha(alpha: float) -> None:
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be a finite number in [0, 1].")
