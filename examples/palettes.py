"""Validate categorical palettes on PBMC3k and synthetic category counts."""

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

import scpolish as scp
from scpolish._utils.colors import resolve_palette


def _save(fig, output: Path, name: str) -> None:
    for extension in ("pdf", "png"):
        fig.savefig(output / f"{name}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    output = Path(__file__).resolve().parent / "output" / "palettes"
    output.mkdir(parents=True, exist_ok=True)
    adata = sc.datasets.pbmc3k_processed()
    groupby = "louvain"
    features = ["MS4A1"]
    missing = [feature for feature in features if feature not in adata.var_names]
    if groupby not in adata.obs or missing:
        raise ValueError(f"PBMC3k is missing {groupby!r} or genes: {missing}")

    print(f"PBMC3k observations: {list(adata.obs.columns)}")
    print(
        f"Grouping categories in {groupby}: {list(adata.obs[groupby].cat.categories)}"
    )
    print(f"Available marker genes used: {features}")

    counts = [8, 18, 30, 50]
    for count in counts:
        categories = [f"Type {index + 1:02d}" for index in range(count)]
        adata.obs[f"synthetic_{count}"] = pd.Categorical(
            np.resize(categories, adata.n_obs), categories=categories, ordered=True
        )

    _palette_comparison(output, counts)
    for preset in ("muted", "pastel"):
        _pbmc_comparison(adata, output, groupby, features, preset)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")
        scp.dimplot(
            adata,
            groupby,
            labels="direct",
            palette=["#4E79A7"],
            title="Invalid list falls back to muted Glasbey",
            ax=ax,
        )
        _save(fig, output, "palette_invalid_fallback")
    if not any(issubclass(item.category, UserWarning) for item in captured):
        raise AssertionError("Invalid palette did not emit the documented warning.")
    print(f"Saved palette comparisons as PDF and PNG in {output}")


def _palette_comparison(output: Path, counts: list[int]) -> None:
    rows = [(preset, count) for preset in ("muted", "pastel") for count in counts]
    with mpl.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        fig, axes = plt.subplots(len(rows), 1, figsize=(14, 7), squeeze=False)
        fig.subplots_adjust(left=0.15, right=0.99, top=0.98, bottom=0.04, hspace=0.55)
        for axis, (preset, count) in zip(axes[:, 0], rows, strict=True):
            categories = [f"category_{index}" for index in range(count)]
            colors = list(resolve_palette(categories, preset).values())
            axis.bar(range(count), np.ones(count), color=colors, width=1, linewidth=0)
            axis.set(xlim=(-0.5, count - 0.5), ylim=(0, 1), yticks=[], xticks=[])
            axis.set_ylabel(
                f"{preset} · N={count}", rotation=0, ha="right", va="center"
            )
            for spine in axis.spines.values():
                spine.set_visible(False)
        _save(fig, output, "palette_sizes")


def _pbmc_comparison(
    adata, output: Path, groupby: str, features: list[str], preset: str
) -> None:
    with mpl.rc_context({"font.family": "DejaVu Sans"}):
        fig, (embedding_ax, violin_ax) = plt.subplots(
            1, 2, figsize=(12, 5.5), layout="constrained"
        )
        scp.dimplot(
            adata,
            groupby,
            labels="direct",
            palette=preset,
            title=f"PBMC3k clusters · {preset}",
            ax=embedding_ax,
        )
        scp.violinplot(
            adata,
            features[0],
            groupby,
            palette=preset,
            ax=violin_ax,
        )
        violin_ax.set_title(f"{features[0]} · {preset}")
        _save(fig, output, f"pbmc_{preset}")


if __name__ == "__main__":
    main()
