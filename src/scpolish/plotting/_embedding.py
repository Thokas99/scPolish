"""Scanpy-backed embedding plots."""

import matplotlib.pyplot as plt
from anndata import AnnData
from matplotlib.axes import Axes

from scpolish._utils.style import style_context


def embedding(
    adata: AnnData,
    basis: str = "umap",
    color: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6, 5),
    title: str | None = None,
    size: float | None = None,
    raster: bool = True,
) -> Axes:
    """Plot an AnnData embedding with Scanpy and return its Matplotlib axes."""
    basis_key = basis.removeprefix("X_")
    obsm_key = f"X_{basis_key}"
    if obsm_key not in adata.obsm:
        available = sorted(
            key.removeprefix("X_") for key in adata.obsm if key.startswith("X_")
        )
        raise ValueError(
            f"Embedding {basis!r} not found in adata.obsm. Available: {available}"
        )
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

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
                size=size,
                title=title,
            )
    finally:
        for key, value in backups.items():
            if value is None:
                adata.uns.pop(key, None)
            else:
                adata.uns[key] = value
    if raster:
        for collection in ax.collections:
            collection.set_rasterized(True)
    return ax
