"""Continuous Scanpy embedding plots."""

from collections.abc import Sequence
from numbers import Real
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._styles import STYLE, export, finish, panels, point_size, prepare


def featureplot(
    adata: AnnData,
    features: str | Sequence[str],
    *,
    basis: str = "umap",
    layer: str | None = None,
    use_raw: bool | None = None,
    threshold: float | None = 0,
    cmap: str = "magma",
    figsize: tuple[float, float] | None = None,
    ncols: int = 2,
    size: float | None = None,
    point_scale: float = 1.0,
    alpha: float = 0.9,
    raster: bool = False,
    vmin=None,
    vmax=None,
    shared_scale: bool = False,
    title: str | None = None,
    ax: Axes | None = None,
    save: str | Path | None = None,
    dpi: int = 300,
) -> Axes | tuple[Figure, np.ndarray]:
    """Plot numeric obs or genes using Scanpy expression-source conventions.

    Return Axes for one feature, otherwise (Figure, 2D axes array), including
    hidden unused panels. Shared scales are opt-in for comparable quantities.
    Scanpy percentile strings and callables are accepted as vmin/vmax. Points
    stay vector unless ``raster=True``. Values at or below ``threshold`` are
    gray; pass ``None`` to color all values, including negatives.
    """
    features = [features] if isinstance(features, str) else list(features)
    if not features or len(set(features)) != len(features):
        raise ValueError("features must contain unique feature names.")
    if not isinstance(raster, bool):
        raise ValueError("raster must be a boolean.")
    if threshold is not None and (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not np.isfinite(threshold)
    ):
        raise ValueError("threshold must be a finite number or None.")
    obj, basis = prepare(adata, basis)
    raw = adata.raw is not None and layer is None if use_raw is None else use_raw
    if raw and layer is not None:
        raise ValueError("Cannot use layer and use_raw=True together.")
    values = sc.get.obs_df(adata, keys=features, layer=layer, use_raw=raw)
    for feature in features:
        if not pd.api.types.is_numeric_dtype(values[feature]):
            raise ValueError(f"Feature {feature!r} must be numeric.")
    if threshold is not None:
        values = values.where(values > threshold)
    for feature in features:
        obj.obs[feature] = values[feature].astype(float)
    if shared_scale:
        finite = values.to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            vmin = finite.min() if vmin is None else vmin
            vmax = finite.max() if vmax is None else vmax
    area = point_size(adata.n_obs, size, point_scale)
    fill_panel = figsize is not None or ax is not None
    with mpl.rc_context(STYLE):
        fig, axes = panels(len(features), ncols, figsize, ax)
        for axis, feature in zip(axes.flat, features, strict=False):
            sc.pl.embedding(
                obj,
                basis=basis,
                color=feature,
                use_raw=False,
                ax=axis,
                show=False,
                sort_order=True,
                size=area,
                alpha=alpha,
                cmap=cmap,
                colorbar_loc=None,
                vmin=vmin,
                vmax=vmax,
                na_color="#BDBDBD",
                title=title if len(features) == 1 else feature,
            )
            mappable = next(
                (
                    collection
                    for collection in axis.collections
                    if collection.get_array() is not None
                ),
                None,
            )
            if mappable is not None and np.ma.count(mappable.get_array()):
                colorbar = fig.colorbar(
                    mappable,
                    ax=axis,
                    fraction=0.045,
                    pad=0.025,
                    shrink=0.6,
                    aspect=30,
                )
                colorbar.ax.tick_params(labelsize=8, length=2, width=0.5)
                colorbar.outline.set_linewidth(0.5)
            finish(
                axis,
                basis,
                raster=raster,
                aspect="auto" if fill_panel else "equal",
            )
        if title and len(features) > 1:
            fig.suptitle(title)
        export(fig, save, dpi)
    return axes.flat[0] if len(features) == 1 else (fig, axes)
