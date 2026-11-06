"""Shared plotting defaults and small layout helpers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection

from scpolish._utils.style import auto_point_size

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


def prepare(adata: AnnData, basis: str) -> tuple[AnnData, str]:
    """Isolate plotting metadata without copying the expression matrix."""
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
    obj = AnnData(obs=adata.obs.copy())
    obj.obsm[key] = coords[:, :2].copy()
    return obj, basis


def panels(n: int, ncols: int, figsize: tuple | None, ax: Axes | None):
    """Return figure and a two-dimensional array of embedding axes."""
    if ncols < 1:
        raise ValueError("ncols must be positive.")
    if ax is not None:
        if n != 1:
            raise ValueError("ax is only supported for a single panel.")
        return ax.figure, np.array([[ax]])
    cols = min(n, ncols)
    rows = int(np.ceil(n / cols))
    if figsize is None:
        figsize = (4.5 * cols, 5.5 * rows)
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
        fig.savefig(
            save,
            dpi=dpi,
            facecolor="white",
            format=Path(save).suffix.lstrip(".") or "pdf",
            bbox_inches=None,
        )


def point_size(n: int, size: float | None, point_scale: float) -> float:
    result = size if size is not None else auto_point_size(n) * point_scale
    if not np.isfinite(result) or result <= 0:
        raise ValueError("Point area must be finite and positive.")
    return result
