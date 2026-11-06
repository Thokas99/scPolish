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
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from scpolish._utils.colors import categorical_categories, resolve_palette

from ._styles import STYLE, export, finish, panels, point_size, prepare


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
    raster: bool = False,
    title: str | None = None,
    split_by: str | None = None,
    ax: Axes | None = None,
    save: str | Path | None = None,
    dpi: int = 300,
) -> Axes | tuple[Figure, np.ndarray]:
    """Plot categories, preserving AnnData with shared categorical palettes.

    Return Axes for one panel, otherwise (Figure, 2D axes array), including
    hidden unused panels. Split panels share limits, colors and marker areas;
    labels use observed cells nearest category medians. Repulsion runs only
    for unsplit plots. Points stay vector unless ``raster=True``. Missing
    split values receive their own panel.
    """
    if labels not in ("auto", "direct", "numbered", False):
        raise ValueError("labels must be 'auto', 'direct', 'numbered', or False.")
    if not isinstance(raster, bool):
        raise ValueError("raster must be a boolean.")
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
    if split_by is not None and split_by not in obj.obs:
        raise ValueError(f"Split annotation {split_by!r} not found in adata.obs.")
    splits = [None] if split_by is None else list(pd.unique(obj.obs[split_by]))
    area = point_size(adata.n_obs, size, point_scale)
    fill_panel = figsize is not None or ax is not None
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
        width = max(
            3.0, max((len(h.get_label()) for h in legend_entries), default=0) * 0.075
        )
        canvas = figsize
        if canvas is None and ax is None:
            cols = min(len(splits), 3)
            rows = int(np.ceil(len(splits) / cols))
            canvas = (4.5 * cols, 5.5 * rows)
            if numbered or mode is False:
                canvas = (
                    canvas[0] + width,
                    max(canvas[1], len(categories) * 0.22 + 1),
                )
        fig, axes = panels(len(splits), min(3, len(splits)), canvas, ax)
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
                aspect="auto" if fill_panel else "equal",
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
                            color="white",
                            path_effects=[
                                pe.withStroke(linewidth=1.2, foreground="black")
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
                if texts and split_by is None:
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
        if numbered or mode is False:
            if not numbered:
                for handle, category in zip(legend_entries, categories, strict=True):
                    handle.set_label(str(category))
            if obj.obs[groupby].isna().any():
                legend_entries.append(
                    Line2D(
                        [], [], marker="o", linestyle="", color="#BDBDBD", label="NA"
                    )
                )
            axes.flat[len(splits) - 1].legend(
                handles=legend_entries,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                ncols=1,
            )
        if title and split_by is not None:
            fig.suptitle(title)
        export(fig, save, dpi)
    return axes.flat[0] if len(splits) == 1 else (fig, axes)
