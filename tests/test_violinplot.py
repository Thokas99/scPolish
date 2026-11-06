import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.figure import Figure
from scipy import sparse

import scpolish as scp
import scpolish.pl as pl


def _result_axes(result):
    if isinstance(result, Axes):
        return [result]
    fig, axes = result
    assert isinstance(fig, Figure)
    return list(axes.flat)


def _violin_values(ax, orientation="vertical"):
    dimension = 0 if orientation == "horizontal" else 1
    return np.concatenate(
        [
            path.vertices[:, dimension]
            for collection in ax.collections
            if isinstance(collection, PolyCollection)
            for path in collection.get_paths()
        ]
    )


def test_exports_and_single_feature(adata):
    assert scp.violinplot is pl.violinplot
    assert isinstance(scp.violinplot(adata, "gene0", "cell_type"), Axes)


def test_paired_lines_preserve_missing_category_gaps(adata):
    adata.obs["donor"] = "demo"
    adata.obs.loc[adata.obs["cell_type"] == "B", "score"] = np.nan
    ax = scp.violinplot(adata, "score", "cell_type", paired_by="donor")
    trajectory = next(line for line in ax.lines if line.get_linewidth() == 0.45)
    assert np.isnan(trajectory.get_ydata()[1])


def test_multiple_features_form_requested_grid(adata):
    fig, axes = scp.violinplot(adata, ["gene0", "gene1", "gene2"], "cell_type", ncols=2)
    assert isinstance(fig, Figure)
    assert axes.shape == (2, 2)
    assert not axes[1, 1].get_visible()
    assert tuple(fig.get_size_inches()) == (11, 9)


@pytest.mark.parametrize(
    "count,shape", [(2, (1, 2)), (4, (2, 2)), (6, (2, 3)), (9, (3, 3))]
)
def test_publication_layout_defaults(adata, count, shape):
    for i in range(count):
        adata.obs[f"layout_{i}"] = np.arange(adata.n_obs, dtype=float)
    result = scp.violinplot(adata, [f"layout_{i}" for i in range(count)], "cell_type")
    assert result[1].shape == shape


def test_stacked_features_share_categories_and_return_axes(adata):
    fig, axes = scp.violinplot(
        adata,
        ["gene0", "gene1", "gene2"],
        "cell_type",
        mode="stacked",
    )
    assert axes.shape == (3, 1)
    assert all(not ax.get_xticklabels() for ax in axes[:-1, 0])
    assert [t.get_text() for t in axes[-1, 0].get_xticklabels()] == ["A", "B", "C"]
    assert isinstance(fig, Figure)


def test_category_order_and_palette_match_dimplot(adata):
    colors = ["#112233", "#445566", "#778899"]
    adata.uns["cell_type_colors"] = colors
    ax = scp.violinplot(
        adata, "gene0", "cell_type", order=["C", "A", "B"], palette="scanpy"
    )
    assert [t.get_text() for t in ax.get_xticklabels()] == ["C", "A", "B"]
    facecolors = [
        mpl.colors.to_hex(c.get_facecolor()[0])
        for c in ax.collections
        if isinstance(c, PolyCollection)
    ]
    assert facecolors == [colors[2], colors[0], colors[1]]


def test_custom_palette_matches_reordered_rendering(adata):
    palette = {"C": "#778899", "A": "#112233", "B": "#445566"}
    ax = scp.violinplot(
        adata, "gene0", "cell_type", order=["C", "A", "B"], palette=palette
    )
    facecolors = [
        mpl.colors.to_hex(collection.get_facecolor()[0])
        for collection in ax.collections
        if isinstance(collection, PolyCollection)
    ]
    assert facecolors == [palette[key] for key in ["C", "A", "B"]]


def test_default_palette_matches_dimplot(adata):
    dimplot_ax = scp.dimplot(adata, "cell_type", labels=False)
    point_colors = np.concatenate(
        [collection.get_facecolors() for collection in dimplot_ax.collections]
    )
    labels = adata.obs["cell_type"].to_numpy()
    expected = [
        mpl.colors.to_hex(np.unique(point_colors[labels == category], axis=0)[0])
        for category in adata.obs["cell_type"].cat.categories
    ]
    ax = scp.violinplot(adata, "gene0", "cell_type")
    actual = [
        mpl.colors.to_hex(collection.get_facecolor()[0])
        for collection in ax.collections
        if isinstance(collection, PolyCollection)
    ]
    assert actual == expected


def test_split_two_categories_are_labeled(adata):
    adata.obs["sample"] = pd.Categorical(["one", "two"] * 15)
    ax = scp.violinplot(adata, "gene0", "cell_type", split_by="sample")
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B", "C"]
    assert {t.get_text() for t in ax.figure.legends[0].get_texts()} == {"one", "two"}


def test_split_with_more_than_two_categories_is_dodged(adata):
    adata.obs["sample"] = np.repeat(["one", "two", "three"], 10)
    ax = scp.violinplot(adata, "gene0", "cell_type", split_by="sample")
    assert {t.get_text() for t in ax.figure.legends[0].get_texts()} == {
        "one",
        "two",
        "three",
    }
    assert (
        sum(
            len(collection.get_paths())
            for collection in ax.collections
            if isinstance(collection, PolyCollection)
        )
        == 9
    )


def test_horizontal_orientation_and_numeric_metadata(adata):
    ax = scp.violinplot(adata, "score", "cell_type", orientation="horizontal")
    assert [t.get_text() for t in ax.get_yticklabels()] == ["A", "B", "C"]
    assert ax.get_xlabel() == "score"
    assert ax.get_xlim()[0] <= 0
    assert ax.get_xlim()[1] >= 1


def test_layer_raw_and_sparse_expression_sources(adata):
    adata.layers["other"] = adata.X + 50
    ax = scp.violinplot(adata, "gene0", "cell_type", layer="other")
    assert _violin_values(ax).min() > 40

    adata.raw = adata.copy()
    adata.X = adata.X + 100
    ax = scp.violinplot(adata, "gene0", "cell_type")
    assert _violin_values(ax).max() < 10

    adata.X = sparse.csr_matrix(adata.X)
    ax = scp.violinplot(adata, "gene0", "cell_type", use_raw=False)
    assert _violin_values(ax).max() > 90

    adata.layers["sparse"] = adata.X + sparse.csr_matrix(np.full(adata.shape, 50))
    adata.raw = adata.copy()
    original_shape = adata.shape
    toarray = sparse.csr_matrix.toarray

    def guard_full_matrix(self, *args, **kwargs):
        if self.shape == original_shape:
            raise AssertionError("plotting densified the full expression matrix")
        return toarray(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sparse.csr_matrix, "toarray", guard_full_matrix)
        assert (
            _violin_values(
                scp.violinplot(adata, "gene0", "cell_type", layer="sparse")
            ).min()
            > 40
        )
        assert (
            _violin_values(
                scp.violinplot(adata, "gene0", "cell_type", use_raw=True)
            ).max()
            > 90
        )


def test_missing_values_and_input_are_preserved(adata):
    adata.layers["counts"] = adata.X.copy()
    adata.raw = adata.copy()
    adata.obs["sample"] = pd.Categorical(["one", "two"] * 15)
    adata.obs.iloc[0, adata.obs.columns.get_loc("cell_type")] = None
    adata.obs.iloc[1, adata.obs.columns.get_loc("sample")] = None
    adata.obs.loc[adata.obs.index[2], "score"] = np.nan
    adata.uns["cell_type_colors"] = ["red", "blue", "green"]
    before = copy.deepcopy(adata)
    rc = mpl.rcParams.copy()

    ax = scp.violinplot(adata, "score", "cell_type", split_by="sample", points=True)
    plotted_points = sum(
        len(collection.get_offsets())
        for collection in ax.collections
        if isinstance(collection, PathCollection)
    )
    assert plotted_points == adata.n_obs - 1

    pd.testing.assert_frame_equal(adata.obs, before.obs)
    np.testing.assert_array_equal(adata.X, before.X)
    np.testing.assert_array_equal(adata.layers["counts"], before.layers["counts"])
    np.testing.assert_array_equal(adata.raw.X, before.raw.X)
    assert adata.uns == before.uns
    assert mpl.rcParams == rc


def test_categorical_order_is_preserved(adata):
    adata.obs["cell_type"] = pd.Categorical(
        adata.obs["cell_type"], categories=["C", "unused", "A", "B"], ordered=True
    )
    ax = scp.violinplot(adata, "gene0", "cell_type")
    assert [t.get_text() for t in ax.get_xticklabels()] == ["C", "A", "B"]


def test_points_are_optional_and_rasterization_is_opt_in(adata):
    ax = scp.violinplot(adata, "gene0", "cell_type")
    assert not any(isinstance(c, PathCollection) for c in ax.collections)

    ax = scp.violinplot(adata, "gene0", "cell_type", points=True)
    point_collections = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert point_collections
    assert not any(c.get_rasterized() for c in point_collections)

    ax = scp.violinplot(adata, "gene0", "cell_type", points=True, raster=True)
    point_collections = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert all(c.get_rasterized() for c in point_collections)


def test_pdf_export_and_external_axes(adata, tmp_path):
    plt.close("all")
    path = tmp_path / "violin.pdf"
    _, ax = plt.subplots()
    assert scp.violinplot(adata, "gene0", "cell_type", ax=ax, save=path) is ax
    assert path.read_bytes().startswith(b"%PDF")


def test_source_conflict_and_invalid_arguments(adata):
    adata.layers["other"] = adata.X.copy()
    adata.raw = adata.copy()
    with pytest.raises(ValueError, match="layer"):
        scp.violinplot(adata, "gene0", "cell_type", layer="other", use_raw=True)
    with pytest.raises(ValueError, match="mode"):
        scp.violinplot(adata, "gene0", "cell_type", mode="other")
    with pytest.raises(ValueError, match="orientation"):
        scp.violinplot(adata, "gene0", "cell_type", orientation="diagonal")
    with pytest.raises(ValueError, match="feature"):
        scp.violinplot(adata, [], "cell_type")
    with pytest.raises(ValueError, match="unique"):
        scp.violinplot(adata, ["gene0", "gene0"], "cell_type")
    with pytest.raises(ValueError, match="groupby"):
        scp.violinplot(adata, "gene0", "absent")
    with pytest.raises(ValueError, match="split"):
        scp.violinplot(adata, "gene0", "cell_type", split_by="absent")
    with pytest.raises(ValueError, match="feature"):
        scp.violinplot(adata, "missing_gene", "cell_type")
    with pytest.raises(ValueError, match="order"):
        scp.violinplot(adata, "gene0", "cell_type", order=["unknown"])


def test_external_axes_only_support_a_single_feature(adata):
    plt.close("all")
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="single"):
        scp.violinplot(adata, ["gene0", "gene1"], "cell_type", ax=ax)
