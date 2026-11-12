"""Categorical Scanpy embedding plots with observed-cell label anchors."""

import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import scanpy as sc
from adjustText import adjust_text
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from scpolish._utils.colors import categorical_categories, resolve_palette

from ._styles import STYLE, export, finish, panel_shape, panels, point_size, prepare


def _label_text_color(color: str) -> str:
    rgb = np.asarray(to_rgb(color))
    luminance = np.dot(
        np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4),
        [0.2126, 0.7152, 0.0722],
    )
    return "#222222" if luminance > 0.45 else "white"


def dimplot(
    adata: AnnData,
    groupby: str,
    *,
    basis: str = "umap",
    labels: str | bool = "auto",
    palette: str | Sequence | Mapping | None = "muted",
    figsize: tuple[float, float] | None = None,
    size: float | None = None,
    point_scale: float = 1.0,
    alpha: float = 1.0,
    raster: bool | None = None,
    title: str | None = None,
    split_by: str | None = None,
    ncols: int | None = None,
    legend: bool | None = None,
    ax: Axes | None = None,
    save: str | Path | None = None,
    dpi: int = 300,
) -> Axes | tuple[Figure, np.ndarray]:
    """Plot categories, preserving AnnData with shared categorical palettes.

    Return Axes for one panel, otherwise (Figure, 2D axes array), including
    hidden unused panels. Split panels share limits, colors and marker areas;
    labels use observed cells nearest category medians. ``legend=True`` adds
    category names and counts; dense point clouds rasterize by default.
    """
    if labels not in ("auto", "direct", "numbered", False):
        raise ValueError("labels must be 'auto', 'direct', 'numbered', or False.")
    if raster is not None and not isinstance(raster, bool):
        raise ValueError("raster must be a boolean or None.")
    if legend is not None and not isinstance(legend, bool):
        raise ValueError("legend must be a boolean or None.")
    raster = adata.n_obs >= 1_000 if raster is None else raster
    obj, basis = prepare(adata, basis)
    if groupby not in obj.obs:
        raise ValueError(f"Annotation {groupby!r} not found in adata.obs.")
    series = obj.obs[groupby]
    categories = categorical_categories(series)
    if not categories:
        raise ValueError(f"Annotation {groupby!r} has no observed categories.")
    obj.obs[groupby] = pd.Categorical(
        series.astype(object),
        categories=categories,
        ordered=(
            series.cat.ordered
            if isinstance(series.dtype, pd.CategoricalDtype)
            else False
        ),
    )
    mode = (
        ("direct" if len(categories) <= 20 else "numbered")
        if labels == "auto"
        else labels
    )
    show_legend = (mode == "numbered" or mode is False) if legend is None else legend
    if split_by is not None and split_by not in obj.obs:
        raise ValueError(f"Split annotation {split_by!r} not found in adata.obs.")
    splits = [None] if split_by is None else list(pd.unique(obj.obs[split_by]))
    area = point_size(adata.n_obs, size, point_scale)
    coords = obj.obsm[f"X_{basis}"]
    finite = np.isfinite(coords).all(axis=1)
    limits = [(coords[finite, i].min(), coords[finite, i].max()) for i in range(2)]
    with mpl.rc_context(STYLE):
        colors = resolve_palette(categories, palette, adata=adata, key=groupby)
        legend_entries = []
        for i, category in enumerate(categories):
            count = int((obj.obs[groupby] == category).sum())
            legend_entries.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    color=colors[category],
                    label=f"{i + 1}  {category} ({count:,})",
                )
            )
        numbered = mode == "numbered"
        canvas = figsize
        if canvas is None and ax is None:
            rows, cols = panel_shape(len(splits), ncols)
            canvas = (12, 8) if len(splits) == 1 else (4.5 * cols, 5.5 * rows)
            if show_legend:
                legend_chars = max(
                    (len(handle.get_label()) for handle in legend_entries), default=0
                )
                canvas = (
                    canvas[0] + max(0, legend_chars - 32) * 0.06,
                    max(canvas[1], len(categories) * 0.22 + 1),
                )
        fig, axes = panels(len(splits), ncols, canvas, ax)
        for axis, split in zip(axes.flat, splits, strict=False):
            mask = (
                np.ones(obj.n_obs, dtype=bool)
                if split_by is None
                else (
                    obj.obs[split_by].isna().to_numpy()
                    if pd.isna(split)
                    else (obj.obs[split_by] == split).to_numpy()
                )
            )
            panel = obj[mask].copy()
            # Subsetting removes unused categories; restore the full color mapping.
            panel.obs[groupby] = pd.Categorical(
                panel.obs[groupby], categories=categories
            )
            sc.pl.embedding(
                panel,
                basis=basis,
                color=groupby,
                palette=[colors[c] for c in categories],
                ax=axis,
                show=False,
                size=area,
                alpha=alpha,
                na_color="#BDBDBD",
                legend_loc=None,
                title=title if split_by is None else str(split),
            )
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
            texts, anchors = [], []
            if mode:
                for i, category in enumerate(categories):
                    points = coords[
                        mask & finite & (obj.obs[groupby] == category).to_numpy()
                    ]
                    if not len(points):
                        continue
                    median = np.median(points, axis=0)
                    anchor = points[np.argmin(np.sum((points - median) ** 2, axis=1))]
                    label = (
                        str(i + 1)
                        if numbered
                        else textwrap.fill(str(category), 24, break_long_words=False)
                    )
                    texts.append(
                        axis.text(
                            *anchor,
                            label,
                            ha="center",
                            va="center",
                            fontsize=9,
                            fontweight="semibold",
                            color=_label_text_color(colors[category]),
                            path_effects=[
                                pe.withStroke(
                                    linewidth=1.2,
                                    foreground=(
                                        "white"
                                        if _label_text_color(colors[category])
                                        == "#222222"
                                        else "black"
                                    ),
                                )
                            ],
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor=colors[category],
                                edgecolor="black",
                                linewidth=0.5,
                                alpha=0.95,
                            ),
                        )
                    )
                    anchors.append(anchor)
                if texts:
                    anchors = np.asarray(anchors)
                    adjust_text(
                        texts,
                        target_x=anchors[:, 0],
                        target_y=anchors[:, 1],
                        ax=axis,
                        expand=(1.15, 1.3),
                        iter_lim=300,
                        min_arrow_len=12,
                        arrowprops=dict(arrowstyle="-", color="0.45", lw=0.45),
                    )
        if show_legend:
            for i, (handle, category) in enumerate(
                zip(legend_entries, categories, strict=True)
            ):
                prefix = f"{i + 1}  " if numbered else ""
                count = int((obj.obs[groupby] == category).sum())
                handle.set_label(f"{prefix}{category} ({count:,})")
            if obj.obs[groupby].isna().any():
                legend_entries.append(
                    Line2D(
                        [],
                        [],
                        marker="o",
                        linestyle="",
                        color="#BDBDBD",
                        label=f"NA ({obj.obs[groupby].isna().sum():,})",
                    )
                )
            legend_args = {"handles": legend_entries, "frameon": False, "ncols": 1}
            if ax is None:
                fig.legend(loc="outside right center", **legend_args)
            else:
                axes.flat[len(splits) - 1].legend(
                    loc="center left", bbox_to_anchor=(1.02, 0.5), **legend_args
                )
        if title and split_by is not None:
            fig.suptitle(title)
        export(fig, save, dpi)
    return axes.flat[0] if len(splits) == 1 else (fig, axes)
