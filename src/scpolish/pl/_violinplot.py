"""Grouped violin plots for genes and numeric observation annotations."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from scpolish._utils.colors import categorical_categories, resolve_palette
from scpolish._utils.scanpy import obs_df
from scpolish._utils.style import validate_figsize

from ._styles import STYLE, export, feature_names, panel_shape


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
        size = figsize or ((12, 8) if n == 1 else (5.5, max(4.5, n * 1.1)))
    else:
        rows, cols = panel_shape(n, ncols)
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
    use: object | None = None,
    palette: str | Sequence | Mapping | None = "muted",
    order: Sequence | None = None,
    inner: str | None = None,
    points: bool = False,
    orientation: str = "vertical",
    ncols: int | None = None,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    raster: bool | None = None,
    save: str | Path | None = None,
    dpi: int = 300,
    labelby: str | None = None,
    paired_by: str | None = None,
    show_counts: bool = False,
    pairs: Sequence[tuple] | None = None,
    pvalues: Sequence[float] | Mapping[str, Sequence[float]] | None = None,
) -> Axes | tuple[Figure, np.ndarray]:
    """Plot grouped gene or numeric observation distributions as violins.

    One feature returns an :class:`~matplotlib.axes.Axes`; multiple features
    return ``(Figure, 2D axes array)``. Stacked mode uses one shared-category
    row per feature. Missing group and split levels are shown as ``NA``; missing
    split levels use gray. Missing feature values are omitted. Optional
    ``pairs`` and supplied ``pvalues`` add statannotations brackets without
    calculating tests. Multi-feature plots need p-values keyed by feature.
    Point overlays stay vector unless ``raster=True``. ``use`` accepts a
    Scanpy 1.13 AnnData accessor and cannot be combined with ``layer`` or
    ``use_raw``.
    """
    features = feature_names(features)
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
    if not isinstance(show_counts, bool):
        raise ValueError("show_counts must be a boolean.")
    if (pairs is None) != (pvalues is None):
        raise ValueError("pairs and pvalues must be provided together.")
    if raster is not None and not isinstance(raster, bool):
        raise ValueError("raster must be a boolean or None.")
    raster = adata.n_obs >= 1_000 if raster is None else raster
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
    if use is not None and (layer is not None or use_raw is not None):
        raise ValueError("use cannot be combined with layer or use_raw.")
    if layer is not None and use_raw is True:
        raise ValueError("Cannot use layer and use_raw=True together.")
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"Layer {layer!r} not found in adata.layers.")
    if use is not None:
        raw = False
    elif use_raw is None:
        raw = adata.raw is not None and layer is None
    else:
        raw = use_raw
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
    figsize = validate_figsize(figsize)

    try:
        values = obs_df(adata, keys=features, layer=layer, use_raw=raw, use=use)
    except KeyError as exc:
        raise ValueError(f"feature lookup failed: {exc}") from exc
    for feature in features:
        if feature not in values:
            raise ValueError(f"Feature {feature!r} not found.")
        if not pd.api.types.is_numeric_dtype(
            values[feature]
        ) or pd.api.types.is_complex_dtype(values[feature]):
            raise ValueError(f"Feature {feature!r} must be real and numeric.")
    if not np.isfinite(values.to_numpy(dtype=float)).any():
        raise ValueError("Features contain no finite values to plot.")

    if pairs is not None:
        pairs = list(pairs)
        if not pairs:
            raise ValueError("pairs must contain at least one comparison.")
        if isinstance(pvalues, Mapping):
            if set(pvalues) != set(features):
                raise ValueError("pvalues mapping keys must match features.")
            raw_pvalues = pvalues
        elif len(features) == 1:
            raw_pvalues = {features[0]: pvalues}
        else:
            raise ValueError("Multiple features require pvalues keyed by feature.")
        feature_pvalues = {}
        for feature, feature_values in raw_pvalues.items():
            try:
                pvalue_array = np.asarray(feature_values, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"pvalues for {feature!r} must be numeric.") from exc
            if (
                pvalue_array.ndim != 1
                or len(pvalue_array) != len(pairs)
                or not np.isfinite(pvalue_array).all()
                or np.any((pvalue_array < 0) | (pvalue_array > 1))
            ):
                raise ValueError(
                    f"pvalues for {feature!r} must match pairs and lie in [0, 1]."
                )
            feature_pvalues[feature] = pvalue_array

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
    if (
        paired_by is not None
        and frame.dropna(subset=[paired_by]).duplicated([paired_by, groupby]).any()
    ):
        raise ValueError(
            "paired_by requires one observation per group; no aggregation is performed."
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
                linewidth=0.75,
                linecolor="#222222",
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
                dodge=dodge or split,
                showcaps=True,
                boxprops={"zorder": 3, "edgecolor": "#222222"},
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
                    levels = observation.set_index(groupby)[feature].reindex(
                        plot_group_order
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
                    palette=(
                        {category: "#333333" for category in hue_order}
                        if split_by is not None
                        else None
                    ),
                    orient=orient,
                    dodge=split_by is not None,
                    jitter=0.08,
                    size=np.sqrt(18),
                    alpha=0.7,
                    color="#333333" if split_by is None else None,
                    legend=False,
                    ax=axis,
                )
                for collection in axis.collections:
                    if isinstance(collection, mpl.collections.PathCollection):
                        collection.set_rasterized(raster)
            if labelby:
                texts = []
                targets = []
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
                            targets.append((xx, yy))
                if texts:
                    point_offsets = [
                        np.asarray(collection.get_offsets())
                        for collection in axis.collections
                        if isinstance(collection, mpl.collections.PathCollection)
                        and len(collection.get_offsets())
                    ]
                    avoid = (
                        np.concatenate(point_offsets)
                        if point_offsets
                        else np.asarray(targets)
                    )
                    targets = np.asarray(targets)
                    adjust_text(
                        texts,
                        x=avoid[:, 0],
                        y=avoid[:, 1],
                        objects=[
                            axis.transData.inverted().transform_bbox(
                                patch.get_window_extent().expanded(1.15, 1.12)
                            )
                            for patch in axis.patches
                        ],
                        target_x=targets[:, 0],
                        target_y=targets[:, 1],
                        ax=axis,
                        iter_lim=300,
                        min_arrow_len=8,
                        arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.4},
                    )
            if pairs is not None:
                try:
                    from statannotations.Annotator import Annotator
                except ImportError as exc:
                    raise ImportError(
                        "Install scpolish[annotations] to use pair annotations."
                    ) from exc
                Annotator(
                    axis,
                    pairs,
                    plot="violinplot",
                    data=frame,
                    x=x,
                    y=y,
                    hue=split_by,
                    order=plot_group_order,
                    hue_order=hue_order if split_by is not None else None,
                    orient=orient,
                    verbose=0,
                ).configure(loc="outside", verbose=0).set_pvalues_and_annotate(
                    feature_pvalues[feature]
                )
            if show_counts:
                counts = frame.groupby(groupby, observed=False)[feature].count()
                tick_setter = (
                    axis.set_xticks if orientation == "vertical" else axis.set_yticks
                )
                tick_setter(
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
