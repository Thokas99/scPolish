import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PolyCollection

import scpolish as scp
from scpolish._utils.colors import (
    _glasbey_palette,
    categorical_categories,
    resolve_palette,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("preset", ["muted", "pastel"])
@pytest.mark.parametrize("count", [8, 18, 30, 50])
def test_glasbey_returns_exact_distinct_cached_colors(preset, count):
    categories = [f"category_{index}" for index in range(count)]
    first = _glasbey_palette(preset, count)
    assert _glasbey_palette(preset, count) is first
    colors = resolve_palette(categories, preset)
    assert len(colors) == count
    assert len(set(colors.values())) == count
    assert all(color.startswith("#") and len(color) == 7 for color in colors.values())
    assert colors == dict(zip(categories, first, strict=True))


def test_categories_keep_categorical_order_and_drop_unused_and_missing(adata):
    adata.obs["cell_type"] = pd.Categorical(
        ["A", "B", None] * 10,
        categories=["B", "unused", "A"],
        ordered=True,
    )
    assert categorical_categories(adata.obs["cell_type"]) == ["B", "A"]
    ax = scp.dimplot(adata, "cell_type", labels=False)
    legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert legend_labels == ["B", "A", "NA"]


def test_violin_missing_group_uses_gray_and_is_not_generated(adata):
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(object)
    adata.obs.loc[adata.obs.index[:6], "cell_type"] = None
    ax = scp.violinplot(adata, "gene0", "cell_type")
    assert [text.get_text() for text in ax.get_xticklabels()] == ["A", "B", "C", "NA"]
    colors = {
        text.get_text(): mpl.colors.to_hex(collection.get_facecolor()[0])
        for text, collection in zip(
            ax.get_xticklabels(),
            [c for c in ax.collections if isinstance(c, PolyCollection)],
            strict=True,
        )
    }
    assert colors["NA"] == "#bdbdbd"


def test_split_violin_missing_split_color_is_gray(adata):
    adata.obs["sample"] = ["one", "two"] * 15
    adata.obs.loc[adata.obs.index[0], "sample"] = None
    ax = scp.violinplot(adata, "gene0", "cell_type", split_by="sample")
    legend = {
        text.get_text(): handle
        for text, handle in zip(
            ax.figure.legends[0].get_texts(),
            ax.figure.legends[0].legend_handles,
            strict=True,
        )
    }
    assert mpl.colors.to_hex(legend["NA"].get_facecolor()) == "#bdbdbd"


def test_non_categorical_category_order_is_deterministic(adata):
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(object)
    assert categorical_categories(adata.obs["cell_type"]) == ["A", "B", "C"]


def test_ordered_category_colors_match_across_plot_types(adata):
    adata.obs["cell_type"] = pd.Categorical(
        adata.obs["cell_type"], categories=["C", "A", "B"], ordered=True
    )
    dim_ax = scp.dimplot(adata, "cell_type", labels=False, palette="muted")
    point_colors = np.concatenate(
        [collection.get_facecolors() for collection in dim_ax.collections]
    )
    labels = adata.obs["cell_type"].to_numpy()
    categories = categorical_categories(adata.obs["cell_type"])
    expected = {
        category: mpl.colors.to_hex(
            np.unique(point_colors[labels == category], axis=0)[0]
        )
        for category in categories
    }
    violin_ax = scp.violinplot(adata, "gene0", "cell_type", palette="muted")
    actual = [
        mpl.colors.to_hex(collection.get_facecolor()[0])
        for collection in violin_ax.collections
        if isinstance(collection, PolyCollection)
    ]
    assert actual == [expected[category] for category in categories]


def test_default_and_pastel_colors_match_between_plots(adata):
    for preset in ("muted", "pastel"):
        dim_ax = scp.dimplot(adata, "cell_type", labels=False, palette=preset)
        point_colors = np.concatenate(
            [collection.get_facecolors() for collection in dim_ax.collections]
        )
        labels = adata.obs["cell_type"].to_numpy()
        categories = categorical_categories(adata.obs["cell_type"])
        expected = {
            category: mpl.colors.to_hex(
                np.unique(point_colors[labels == category], axis=0)[0]
            )
            for category in categories
        }
        violin_ax = scp.violinplot(adata, "gene0", "cell_type", palette=preset)
        actual = [
            mpl.colors.to_hex(collection.get_facecolor()[0])
            for collection in violin_ax.collections
            if isinstance(collection, PolyCollection)
        ]
        assert actual == [expected[category] for category in categories]


def test_custom_mapping_is_category_keyed_across_plot_order(adata):
    palette = {"C": "green", "A": "red", "B": "blue"}
    ax = scp.dimplot(adata, "cell_type", labels=False, palette=palette)
    facecolors = np.concatenate(
        [collection.get_facecolors() for collection in ax.collections]
    )
    labels = adata.obs["cell_type"].to_numpy()
    expected = {
        "A": mpl.colors.to_hex("red"),
        "B": mpl.colors.to_hex("blue"),
        "C": mpl.colors.to_hex("green"),
    }
    for category in categorical_categories(adata.obs["cell_type"]):
        assert (
            mpl.colors.to_hex(np.unique(facecolors[labels == category], axis=0)[0])
            == expected[category]
        )
    violin_ax = scp.violinplot(
        adata, "gene0", "cell_type", order=["C", "A", "B"], palette=palette
    )
    actual = [
        mpl.colors.to_hex(collection.get_facecolor()[0])
        for collection in violin_ax.collections
        if isinstance(collection, PolyCollection)
    ]
    assert actual == [expected[category] for category in ["C", "A", "B"]]


def test_custom_list_follows_resolved_category_order(adata):
    palette = ["#112233", "#445566", "#778899"]
    mapping = resolve_palette(["C", "A", "B"], palette)
    assert mapping == {"C": "#112233", "A": "#445566", "B": "#778899"}


@pytest.mark.parametrize(
    "palette, message",
    [
        (["red"], "contains 1 colors"),
        ({"A": "red", "unknown": "blue"}, "unknown categories"),
        (["not-a-color", "blue", "green"], "invalid color"),
        (["red", "#ff0000", "green"], "duplicate colors"),
        (17, "Unsupported custom palette type"),
    ],
)
def test_invalid_custom_palettes_warn_and_fall_back(adata, palette, message):
    categories = ["A", "B", "C"]
    expected = resolve_palette(categories, "muted")
    with pytest.warns(UserWarning, match=message):
        mapping = resolve_palette(categories, palette)
    assert mapping == expected


def test_invalid_custom_palette_continues_plotting(adata):
    with pytest.warns(UserWarning, match="contains 1 colors"):
        ax = scp.dimplot(adata, "cell_type", labels=False, palette=["red"])
    assert ax.collections
    with pytest.warns(UserWarning, match="contains 1 colors"):
        ax = scp.violinplot(adata, "gene0", "cell_type", palette=["red"])
    assert any(isinstance(collection, PolyCollection) for collection in ax.collections)


def test_scanpy_palette_reuses_stored_and_falls_back_when_invalid(adata):
    categories = categorical_categories(adata.obs["cell_type"])
    adata.uns["cell_type_colors"] = ["#112233", "#445566", "#778899"]
    before = adata.uns["cell_type_colors"].copy()
    assert resolve_palette(categories, "scanpy", adata=adata, key="cell_type") == dict(
        zip(categories, before, strict=True)
    )
    assert adata.uns["cell_type_colors"] == before

    adata.uns["cell_type_colors"] = ["bad-color", "#445566", "#778899"]
    colors = resolve_palette(categories, "scanpy", adata=adata, key="cell_type")
    assert len(colors) == len(categories)
    assert all(mpl.colors.is_color_like(color) for color in colors.values())
    assert adata.uns["cell_type_colors"][0] == "bad-color"


def test_unknown_named_palette_is_a_configuration_error(adata):
    with pytest.raises(ValueError, match="Unknown palette preset"):
        scp.dimplot(adata, "cell_type", palette="set3")


def test_split_dimplot_uses_full_category_palette(adata):
    adata.obs["sample"] = ["one", "two"] * 15
    palette = resolve_palette(["A", "B", "C"], "pastel")
    _, axes = scp.dimplot(
        adata, "cell_type", split_by="sample", labels=False, palette="pastel"
    )
    for ax in axes.flat:
        ax.figure.canvas.draw()
        assert all(
            mpl.colors.to_hex(color) in set(palette.values())
            for color in ax.collections[0].get_facecolors()
        )


def test_split_violin_uses_one_palette_across_feature_panels(adata):
    adata.obs["sample"] = pd.Categorical(
        ["one", "two"] * 15, categories=["two", "unused", "one"], ordered=True
    )
    categories = categorical_categories(adata.obs["sample"])
    expected = set(resolve_palette(categories, "pastel").values())
    _, axes = scp.violinplot(
        adata,
        ["gene0", "gene1"],
        "cell_type",
        split_by="sample",
        palette="pastel",
    )
    for ax in axes.flat[:2]:
        actual = {
            mpl.colors.to_hex(collection.get_facecolor()[0])
            for collection in ax.collections
            if isinstance(collection, PolyCollection)
        }
        assert actual == expected
