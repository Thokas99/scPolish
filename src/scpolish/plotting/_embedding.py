"""Scanpy-backed embedding plots."""

import matplotlib.pyplot as plt
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection

from scpolish._utils.style import auto_point_size, style_context


def embedding(
    adata: AnnData,
    basis: str = "umap",
    color: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    size: float | None = None,
    raster: bool = False,
) -> Axes:
    """Plot an AnnData embedding with Scanpy and return its Matplotlib axes."""
    if not isinstance(raster, bool):
        raise ValueError("raster must be a boolean.")
    basis_key = basis.removeprefix("X_")
    obsm_key = f"X_{basis_key}"
    if obsm_key not in adata.obsm:
        available = sorted(
            key.removeprefix("X_") for key in adata.obsm if key.startswith("X_")
        )
        raise ValueError(
            f"Embedding {basis!r} not found in adata.obsm. Available: {available}"
        )
    fill_panel = figsize is not None or ax is not None
    if ax is None:
        _, ax = plt.subplots(figsize=figsize or (6, 5))

    color_keys = [color]
    backups = {}
    for key in color_keys:
        if (
            isinstance(key, str)
            and key in adata.obs
            and adata.obs[key].dtype.name == "category"
        ):
            palette_key = f"{key}_colors"
            backups[palette_key] = (
                adata.uns[palette_key].copy() if palette_key in adata.uns else None
            )
    try:
        with style_context():
            import scanpy as sc

            sc.pl.embedding(
                adata,
                basis=basis_key,
                color=color,
                ax=ax,
                show=False,
                return_fig=False,
                size=size if size is not None else auto_point_size(adata.n_obs),
                title=title,
            )
    finally:
        for key, value in backups.items():
            if value is None:
                adata.uns.pop(key, None)
            else:
                adata.uns[key] = value
    for collection in ax.collections:
        if isinstance(collection, PathCollection):
            collection.set_rasterized(raster)
    ax.set_aspect("auto" if fill_panel else "equal")
    return ax
