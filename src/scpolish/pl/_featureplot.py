"""Continuous Scanpy embedding plots."""

from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from ._styles import STYLE, export, finish, panels, point_size, prepare


def _score_limit(limit, values, default):
    if limit is None:
        return default(values)
    if callable(limit):
        return limit(values)
    if isinstance(limit, str) and limit.startswith("p"):
        return np.percentile(values, float(limit[1:]))
    return limit


def featureplot(
    adata: AnnData,
    features: str | Sequence[str],
    *,
    basis: str = "umap",
    layer: str | None = None,
    use_raw: bool | None = None,
    style: str = "scanpy",
    threshold: float | None = None,
    cmap: str | None = None,
    figsize: tuple[float, float] | None = None,
    ncols: int | None = None,
    size: float | None = None,
    point_scale: float = 1.0,
    alpha: float = 0.9,
    raster: bool | None = None,
    vmin=None,
    vmax=None,
    shared_scale: bool = False,
    title: str | Mapping[str, str] | None = None,
    ax: Axes | None = None,
    save: str | Path | None = None,
    dpi: int = 300,
) -> Axes | tuple[Figure, np.ndarray]:
    """Plot numeric obs or genes using Scanpy expression-source conventions.

    Return Axes for one feature, otherwise (Figure, 2D axes array), including
    hidden unused panels. Shared scales are opt-in for comparable quantities.
    Gene panels accept Scanpy percentile strings and callables as vmin/vmax.
    ``style="scanpy"`` preserves Scanpy plotting by default. ``style="overlay"``
    draws full-embedding score overlays with missing cells in gray. Finite values,
    including negatives, are colored; an explicit threshold renders lower values
    gray. Set ``shared_scale=True`` only for comparable feature values.
    """
    features = [features] if isinstance(features, str) else list(features)
    if not features or len(set(features)) != len(features):
        raise ValueError("features must contain unique feature names.")
    if style not in ("scanpy", "overlay"):
        raise ValueError("style must be 'scanpy' or 'overlay'.")
    if isinstance(title, Mapping) and any(
        feature not in features or not isinstance(value, str)
        for feature, value in title.items()
    ):
        raise ValueError("title mapping keys must be requested features and strings.")
    if raster is not None and not isinstance(raster, bool):
        raise ValueError("raster must be a boolean or None.")
    raster = adata.n_obs >= 1_000 if raster is None else raster
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
    coords = np.asarray(obj.obsm[f"X_{basis}"])[:, :2]
    finite_xy = np.isfinite(coords).all(axis=1)
    limits = [
        (coords[finite_xy, i].min(), coords[finite_xy, i].max()) for i in range(2)
    ]
    area = point_size(adata.n_obs, size, point_scale)
    with mpl.rc_context(STYLE | {"axes.titlesize": 11, "axes.titleweight": "normal"}):
        fig, axes = panels(
            len(features),
            ncols,
            figsize or ((12, 8) if len(features) == 1 else None),
            ax,
        )
        for axis, feature in zip(axes.flat, features, strict=False):
            panel_title = (
                title.get(feature, feature)
                if isinstance(title, Mapping)
                else title
                if len(features) == 1 and title
                else feature
            )
            if style == "overlay":
                score = values[feature].to_numpy(dtype=float)
                scored = np.isfinite(score) & finite_xy
                axis.scatter(
                    *coords[finite_xy & ~scored, :2].T,
                    c="#DDDDDD",
                    s=1,
                    rasterized=raster,
                )
                if scored.any():
                    scale_values = finite if shared_scale else score[scored]
                    limits_min = _score_limit(vmin, scale_values, np.min)
                    limits_max = _score_limit(vmax, scale_values, np.max)
                    ranked = np.flatnonzero(scored)[
                        np.argsort(score[scored], kind="stable")
                    ]
                    points = axis.scatter(
                        *coords[ranked, :2].T,
                        c=score[ranked],
                        s=2,
                        cmap="viridis",
                        alpha=alpha,
                        vmin=limits_min,
                        vmax=limits_max,
                        rasterized=raster,
                        zorder=2,
                    )
                    colorbar = fig.colorbar(
                        points, ax=axis, fraction=0.045, pad=0.025, shrink=0.6
                    )
                    colorbar.set_label(feature, fontsize=8, labelpad=4)
                    colorbar.locator = MaxNLocator(nbins=4)
                    colorbar.update_ticks()
                    colorbar.ax.tick_params(labelsize=7, length=2, width=0.5)
                    colorbar.outline.set_linewidth(0.5)
                else:
                    axis.text(
                        0.5,
                        0.5,
                        "No scored observations",
                        transform=axis.transAxes,
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="#555555",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
                    )
                finish(axis, basis, raster=raster, aspect="equal")
                unscored = int((finite_xy & ~scored).sum())
                if unscored:
                    reason = (
                        "unscored" if threshold is None else "unscored/below cutoff"
                    )
                    axis.text(
                        0,
                        1.01,
                        f"Gray: {unscored:,} {reason}",
                        transform=axis.transAxes,
                        fontsize=7,
                        color="#555555",
                    )
                axis.set_title(panel_title)
                continue
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
                cmap=cmap or "viridis",
                colorbar_loc=None,
                vmin=vmin,
                vmax=vmax,
                na_color="#BDBDBD",
                title=panel_title,
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
                colorbar.set_label(feature, fontsize=8, labelpad=4)
                colorbar.outline.set_linewidth(0.5)
            finish(
                axis,
                basis,
                raster=raster,
                aspect="equal",
            )
            for setter, (lo, hi) in zip(
                (axis.set_xlim, axis.set_ylim), limits, strict=True
            ):
                pad = (hi - lo) * 0.05 or 0.5
                setter(lo - pad, hi + pad)
        if isinstance(title, str) and title and len(features) > 1:
            fig.suptitle(title)
        export(fig, save, dpi)
    return axes.flat[0] if len(features) == 1 else (fig, axes)
