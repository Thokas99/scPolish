"""Grouped violin plots for genes and numeric observation annotations."""

from collections.abc import Mapping, Sequence
from math import ceil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from adjustText import adjust_text
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from scpolish._utils.colors import categorical_categories, resolve_palette

from ._styles import STYLE, export


def _plot_axes(
    n: int,
    *,
    mode: str,
    orientation: str,
    ncols: int | None,
    figsize: tuple[float, float] | None,
    ax: Axes | None,
) -> tuple[Figure, np.ndarray]:
    if ax is not None:
        if n != 1:
            raise ValueError("ax is only supported for a single feature.")
        return ax.figure, np.array([[ax]])

    if mode == "stacked":
        rows, cols = n, 1
        size = figsize or (5.5, max(4.5, n * 1.1))
    else:
        cols = min(n, ncols or (3 if n in (6, 9) else 2))
        rows = ceil(n / cols)
        size = figsize or ((12, 8) if n == 1 else (5.5 * cols, 4.5 * rows))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=size,
        squeeze=False,
        layout="constrained",
        sharex=orientation == "vertical",
        sharey=orientation == "horizontal",
    )
    for unused in axes.flat[n:]:
        unused.set_visible(False)
    return fig, axes


def _style_axis(ax: Axes, *, orientation: str, rotate: bool) -> None:
    ax.set_facecolor("white")
    ax.grid(
        axis="y" if orientation == "vertical" else "x",
        color="#E6E6E6",
        linewidth=0.45,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if orientation == "vertical" and rotate:
        ax.tick_params(axis="x", labelrotation=45)


def _missing_label(categories: Sequence) -> str:
    label = "NA"
    while label in categories:
        label += " (missing)"
    return label


def violinplot(
    adata: AnnData,
    features: str | Sequence[str],
    groupby: str,
    *,
    mode: str = "standard",
    split_by: str | None = None,
    layer: str | None = None,
    use_raw: bool | None = None,
    palette: str | Sequence | Mapping | None = "muted",
    order: Sequence | None = None,
    inner: str | None = None,
    points: bool = False,
    orientation: str = "vertical",
    ncols: int | None = None,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    raster: bool = False,
    save: str | Path | None = None,
    dpi: int = 300,
    labelby: str | None = None,
    paired_by: str | None = None,
    show_counts: bool = False,
) -> Axes | tuple[Figure, np.ndarray]:
    """Plot grouped gene or numeric observation distributions as violins.

    One feature returns an :class:`~matplotlib.axes.Axes`; multiple features
    return ``(Figure, 2D axes array)``. Stacked mode uses one shared-category
    row per feature. Missing group and split levels are shown as ``NA``; missing
    split levels use gray. Missing feature values are omitted. Point overlays
    stay vector unless ``raster=True``.
    """
    features = [features] if isinstance(features, str) else list(features)
    if not features or any(not isinstance(feature, str) for feature in features):
        raise ValueError("features must contain one or more feature names.")
    if len(set(features)) != len(features):
        raise ValueError("features must contain unique names.")
    if mode not in ("standard", "stacked"):
        raise ValueError("mode must be 'standard' or 'stacked'.")
    if orientation not in ("vertical", "horizontal"):
        raise ValueError("orientation must be 'vertical' or 'horizontal'.")
    if ncols is not None and (
        not isinstance(ncols, int) or isinstance(ncols, bool) or ncols < 1
    ):
        raise ValueError("ncols must be a positive integer.")
    if not isinstance(points, bool):
        raise ValueError("points must be a boolean.")
    if not isinstance(raster, bool):
        raise ValueError("raster must be a boolean.")
    if inner not in ("box", "quart", "stick", "point", None):
        raise ValueError("inner must be 'box', 'quart', 'stick', 'point', or None.")
    if groupby not in adata.obs:
        raise ValueError(f"groupby {groupby!r} not found in adata.obs.")
    if split_by is not None and split_by not in adata.obs:
        raise ValueError(f"split annotation {split_by!r} not found in adata.obs.")
    if labelby is not None and labelby not in adata.obs:
        raise ValueError(f"label annotation {labelby!r} not found in adata.obs.")
    if paired_by is not None and paired_by not in adata.obs:
        raise ValueError(f"paired annotation {paired_by!r} not found in adata.obs.")
    if split_by == groupby:
        raise ValueError("split_by must differ from groupby.")
    if layer is not None and use_raw is True:
        raise ValueError("Cannot use layer and use_raw=True together.")
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"Layer {layer!r} not found in adata.layers.")
    raw = adata.raw is not None and layer is None if use_raw is None else use_raw
    if raw and adata.raw is None:
        raise ValueError("use_raw=True requires adata.raw.")

    categories = categorical_categories(adata.obs[groupby])
    group_missing = adata.obs[groupby].isna().any()
    if order is None:
        group_order = categories
    else:
        group_order = list(order)
        if not group_order or len(set(group_order)) != len(group_order):
            raise ValueError("order must contain unique, known categories.")
        if any(category not in categories for category in group_order):
            raise ValueError("order contains an unknown group category.")
    if not group_order and not group_missing:
        raise ValueError("groupby must contain at least one non-missing category.")
    group_missing_label = _missing_label(categories) if group_missing else None
    plot_group_order = group_order + ([group_missing_label] if group_missing else [])
    if ax is not None and len(features) != 1:
        raise ValueError("ax is only supported for a single feature.")
    if figsize is not None and (
        len(figsize) != 2 or not np.isfinite(figsize).all() or min(figsize) <= 0
    ):
        raise ValueError("figsize must contain two finite, positive dimensions.")

    try:
        if layer is None:
            values = sc.get.obs_df(adata, keys=features, use_raw=raw)
        else:
            values = sc.get.obs_df(adata, keys=features, layer=layer, use_raw=False)
    except KeyError as exc:
        raise ValueError(f"feature lookup failed: {exc}") from exc
    for feature in features:
        if feature not in values:
            raise ValueError(f"Feature {feature!r} not found.")
        if not pd.api.types.is_numeric_dtype(values[feature]):
            raise ValueError(f"Feature {feature!r} must be numeric.")
    if not np.isfinite(values.to_numpy(dtype=float)).any():
        raise ValueError("Features contain no finite values to plot.")

    columns = list(
        dict.fromkeys(
            [groupby] + [c for c in (split_by, labelby, paired_by) if c is not None]
        )
    )
    frame = adata.obs[columns].copy()
    group_values = frame[groupby].astype(object)
    selected = group_values.isna() | group_values.isin(group_order)
    frame = frame.loc[selected].copy()
    group_values = group_values.loc[selected]
    if group_missing:
        group_values = group_values.where(group_values.notna(), group_missing_label)
    frame[groupby] = pd.Categorical(
        group_values, categories=plot_group_order, ordered=True
    )
    for feature in features:
        frame[feature] = values.loc[frame.index, feature].to_numpy(
            dtype=float, copy=True
        )

    if split_by is None:
        split_categories = []
        plot_split_categories = []
        split_colors = []
    else:
        split_values = adata.obs[split_by]
        split_categories = categorical_categories(split_values)
        split_missing = split_values.isna().any()
        if not split_categories and not split_missing:
            raise ValueError("split_by must contain at least one non-missing category.")
        split_missing_label = (
            _missing_label(split_categories) if split_missing else None
        )
        plot_split_categories = split_categories + (
            [split_missing_label] if split_missing else []
        )
        split_values = frame[split_by].astype(object)
        if split_missing:
            split_values = split_values.where(split_values.notna(), split_missing_label)
        frame[split_by] = pd.Categorical(
            split_values, categories=plot_split_categories, ordered=True
        )
        split_color_map = resolve_palette(
            split_categories, palette, adata=adata, key=split_by
        )
        if split_missing:
            split_color_map[split_missing_label] = "#BDBDBD"
        split_colors = [split_color_map[category] for category in plot_split_categories]

    if split_by is None:
        ordered_palette = isinstance(palette, Sequence) and not isinstance(
            palette, (str, bytes)
        )
        palette_categories = group_order if ordered_palette else categories
        group_color_map = resolve_palette(
            palette_categories, palette, adata=adata, key=groupby
        )
        if group_missing:
            group_color_map[group_missing_label] = "#BDBDBD"
        color_map = {
            category: group_color_map[category] for category in plot_group_order
        }
        hue = groupby
        hue_order = plot_group_order
    else:
        color_map = dict(zip(plot_split_categories, split_colors, strict=True))
        hue = split_by
        hue_order = plot_split_categories
    split = split_by is not None and len(plot_split_categories) == 2
    dodge = split_by is not None and len(plot_split_categories) > 2
    rotate = (
        len(plot_group_order) > 7
        or max(map(len, map(str, plot_group_order)), default=0) > 12
    )
    orient = "x" if orientation == "vertical" else "y"

    with (
        mpl.rc_context(STYLE),
        sns.axes_style("whitegrid"),
        sns.plotting_context("paper", font_scale=0.85),
    ):
        fig, axes = _plot_axes(
            len(features),
            mode=mode,
            orientation=orientation,
            ncols=ncols,
            figsize=figsize,
            ax=ax,
        )
        visible_axes = list(axes.flat[: len(features)])
        for axis, feature in zip(visible_axes, features, strict=True):
            x, y = (
                (groupby, feature) if orientation == "vertical" else (feature, groupby)
            )
            sns.violinplot(
                data=frame,
                x=x,
                y=y,
                order=plot_group_order,
                hue=hue,
                hue_order=hue_order,
                palette=color_map,
                orient=orient,
                split=split,
                dodge=dodge,
                inner=inner,
                cut=0,
                linewidth=0.5,
                saturation=1,
                legend=False,
                ax=axis,
            )
            sns.boxplot(
                data=frame,
                x=x,
                y=y,
                order=plot_group_order,
                hue=hue,
                hue_order=hue_order,
                palette={k: "white" for k in hue_order},
                orient=orient,
                width=0.22,
                fliersize=0,
                linewidth=0.5,
                dodge=dodge,
                showcaps=True,
                boxprops={"zorder": 3},
                medianprops={"color": "#333333", "linewidth": 0.8, "zorder": 4},
                whiskerprops={"linewidth": 0.5},
                capprops={"linewidth": 0.5},
                legend=False,
                ax=axis,
            )
            if paired_by:
                paired = frame.dropna(subset=[paired_by, feature]).groupby(
                    paired_by, observed=True
                )
                for _, observation in paired:
                    levels = (
                        observation.groupby(groupby, observed=True)[feature]
                        .median()
                        .reindex(plot_group_order)
                    )
                    if levels.notna().sum() > 1:
                        positions = np.arange(len(plot_group_order))
                        values = levels.to_numpy(dtype=float)
                        axis.plot(
                            *(
                                (positions, values)
                                if orientation == "vertical"
                                else (values, positions)
                            ),
                            color="#777777",
                            linewidth=0.45,
                            alpha=0.5,
                            zorder=0,
                        )
            if points:
                sns.stripplot(
                    data=frame,
                    x=x,
                    y=y,
                    order=plot_group_order,
                    hue=split_by,
                    hue_order=hue_order if split_by is not None else None,
                    palette=color_map if split_by is not None else None,
                    orient=orient,
                    dodge=split_by is not None,
                    jitter=0.08,
                    size=np.sqrt(18),
                    alpha=0.45,
                    color="#333333" if split_by is None else None,
                    legend=False,
                    ax=axis,
                )
                for collection in axis.collections:
                    if isinstance(collection, mpl.collections.PathCollection):
                        collection.set_rasterized(raster)
            if labelby:
                texts = []
                for category, subset in frame.groupby(groupby, observed=True):
                    subset = subset.dropna(subset=[feature, labelby])
                    if len(subset):
                        if orientation == "vertical":
                            xval = np.full(
                                len(subset), plot_group_order.index(category)
                            )
                            yval = subset[feature].to_numpy()
                        else:
                            xval = subset[feature].to_numpy()
                            yval = np.full(
                                len(subset), plot_group_order.index(category)
                            )
                        for xx, yy, label in zip(
                            np.atleast_1d(xval),
                            np.atleast_1d(yval),
                            subset[labelby],
                            strict=True,
                        ):
                            texts.append(
                                axis.text(
                                    xx, yy, str(label), fontsize=6, color="#333333"
                                )
                            )
                if texts:
                    adjust_text(texts, ax=axis)
            if show_counts and orientation == "vertical":
                counts = frame.groupby(groupby, observed=False)[feature].count()
                axis.set_xticks(
                    np.arange(len(plot_group_order)),
                    labels=[f"{c}\n(n={counts.get(c, 0)})" for c in plot_group_order],
                )
            axis.set_xlabel(
                groupby.replace("_", " ") if orientation == "vertical" else feature
            )
            axis.set_ylabel(
                feature if orientation == "vertical" else groupby.replace("_", " ")
            )
            _style_axis(axis, orientation=orientation, rotate=rotate)

        if mode == "stacked":
            if orientation == "vertical":
                for axis in visible_axes[:-1]:
                    axis.tick_params(axis="x", labelbottom=False)
                visible_axes[-1].set_xlabel(groupby.replace("_", " "))
            else:
                for axis in visible_axes[1:]:
                    axis.tick_params(axis="y", labelleft=False)
        elif len(features) > 1 and orientation == "vertical":
            for row in range(axes.shape[0] - 1):
                for axis in axes[row, :]:
                    axis.tick_params(axis="x", labelbottom=False)

        if split_by is not None:
            handles = [
                Patch(facecolor=color, edgecolor="0.3", label=str(category))
                for category, color in zip(
                    plot_split_categories, split_colors, strict=True
                )
            ]
            fig.legend(
                handles=handles,
                title=split_by.replace("_", " "),
                loc="outside right upper",
                frameon=False,
            )
        export(fig, save, dpi)

    return axes.flat[0] if len(features) == 1 else (fig, axes)
