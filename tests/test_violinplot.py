import copy
import sys
from types import ModuleType

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.transforms import BboxBase
from scipy import sparse

import scpolish as scp
import scpolish.pl as pl
import scpolish.pl._violinplot as violin_module


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
    paired = adata[[0, 1, 2]].copy()
    paired.obs["donor"] = "demo"
    paired.obs.loc[paired.obs["cell_type"] == "B", "score"] = np.nan
    ax = scp.violinplot(paired, "score", "cell_type", paired_by="donor")
    trajectory = next(line for line in ax.lines if line.get_linewidth() == 0.45)
    assert np.isnan(trajectory.get_ydata()[1])


def test_multiple_features_form_requested_grid(adata):
    fig, axes = scp.violinplot(adata, ["gene0", "gene1", "gene2"], "cell_type", ncols=2)
    assert isinstance(fig, Figure)
    assert axes.shape == (2, 2)
    assert not axes[1, 1].get_visible()
    assert tuple(fig.get_size_inches()) == (11, 9)


@pytest.mark.parametrize(
    "count,shape",
    [
        (1, (1, 1)),
        (2, (1, 2)),
        (3, (1, 3)),
        (4, (2, 2)),
        (5, (2, 3)),
        (6, (2, 3)),
        (7, (3, 3)),
        (9, (3, 3)),
    ],
)
def test_publication_layout_defaults(adata, count, shape):
    for i in range(count):
        adata.obs[f"layout_{i}"] = np.arange(adata.n_obs, dtype=float)
    result = scp.violinplot(adata, [f"layout_{i}" for i in range(count)], "cell_type")
    if count == 1:
        assert isinstance(result, Axes)
    else:
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
    assert tuple(fig.get_size_inches()) == (5.5, 4.5)


def test_single_stacked_feature_uses_publication_canvas(adata):
    ax = scp.violinplot(adata, "gene0", "cell_type", mode="stacked")
    assert tuple(ax.figure.get_size_inches()) == (12, 8)


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


def test_violins_have_dark_outlines_and_narrow_white_boxes(adata):
    ax = scp.violinplot(adata, "gene0", "cell_type")
    violins = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(violins) == 3
    assert all(
        np.allclose(c.get_edgecolors()[0][:3], mpl.colors.to_rgb("#222222"))
        for c in violins
    )
    assert all(0.5 <= c.get_linewidths()[0] <= 1 for c in violins)

    boxes = [p for p in ax.patches if isinstance(p, PathPatch)]
    assert len(boxes) == 3
    assert all(
        np.ptp(box.get_path().vertices[:, 0]) == pytest.approx(0.22) for box in boxes
    )
    assert all(np.allclose(box.get_facecolor()[:3], (1, 1, 1)) for box in boxes)


def test_split_box_overlays_align_with_each_hue(adata):
    adata.obs["sample"] = pd.Categorical(["one", "two"] * 15)
    ax = scp.violinplot(adata, "gene0", "cell_type", split_by="sample")
    boxes = [p for p in ax.patches if isinstance(p, PathPatch)]
    centers = np.array(
        [
            np.mean(
                [
                    box.get_path().vertices[:, 0].min(),
                    box.get_path().vertices[:, 0].max(),
                ]
            )
            for box in boxes
        ]
    )
    for group in range(3):
        group_centers = centers[np.abs(centers - group) < 0.5]
        assert len(group_centers) == 2
        assert group_centers.min() < group < group_centers.max()


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
    assert all(
        np.allclose(c.get_facecolors()[:, :3], mpl.colors.to_rgb("#333333"))
        for c in point_collections
    )
    assert all(np.allclose(c.get_sizes(), 18) for c in point_collections)


def test_label_adjustment_avoids_plotted_points_and_boxes(monkeypatch, adata):
    adata = adata[:12].copy()
    adata.obs["condition"] = pd.Categorical(["Control", "Treatment"] * 6)
    adata.obs["donor"] = np.repeat([f"donor{i}" for i in range(6)], 2)
    adata.obs["demo_value"] = np.repeat([-1.1, -0.2, 0.4, 1.0, 1.4, 2.3], 2) + np.tile(
        [0, 0.6], 6
    )
    real_adjust_text = violin_module.adjust_text
    received = {}

    def capture_adjustment(texts, **kwargs):
        received.update(kwargs)
        return real_adjust_text(texts, **kwargs)

    monkeypatch.setattr(violin_module, "adjust_text", capture_adjustment)
    ax = scp.violinplot(
        adata,
        "demo_value",
        "condition",
        points=True,
        labelby="donor",
        paired_by="donor",
    )
    plotted_points = np.concatenate(
        [
            np.asarray(collection.get_offsets())
            for collection in ax.collections
            if isinstance(collection, PathCollection)
        ]
    )
    np.testing.assert_allclose(
        np.column_stack((received["x"], received["y"])), plotted_points
    )
    assert all(isinstance(item, BboxBase) for item in received["objects"])
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    labels = [text for text in ax.texts if text.get_text().startswith("donor")]
    boxes = [patch for patch in ax.patches if isinstance(patch, PathPatch)]
    assert len(labels) == 12
    assert all(
        not label.get_window_extent(renderer).overlaps(box.get_window_extent(renderer))
        for label in labels
        for box in boxes
    )


def test_optional_pair_annotations_use_supplied_pvalues(monkeypatch, adata):
    calls = []

    class FakeAnnotator:
        def __init__(self, ax, pairs, **kwargs):
            calls.append({"ax": ax, "pairs": pairs, "kwargs": kwargs})

        def configure(self, **kwargs):
            calls[-1]["configuration"] = kwargs
            return self

        def set_pvalues_and_annotate(self, pvalues):
            calls[-1]["pvalues"] = pvalues
            return self

    package = ModuleType("statannotations")
    annotator_module = ModuleType("statannotations.Annotator")
    annotator_module.Annotator = FakeAnnotator
    monkeypatch.setitem(sys.modules, "statannotations", package)
    monkeypatch.setitem(sys.modules, "statannotations.Annotator", annotator_module)
    pairs = [("A", "B")]
    scp.violinplot(
        adata,
        ["gene0", "gene1"],
        "cell_type",
        pairs=pairs,
        pvalues={"gene0": [0.03], "gene1": [0.08]},
    )
    assert [call["pvalues"].tolist() for call in calls] == [[0.03], [0.08]]
    assert all(call["pairs"] == pairs for call in calls)
    assert all(call["kwargs"]["plot"] == "violinplot" for call in calls)
    assert all(call["kwargs"]["orient"] == "x" for call in calls)
    assert all(
        call["configuration"] == {"loc": "outside", "verbose": 0} for call in calls
    )


def test_pair_annotations_require_matching_pvalues(adata):
    with pytest.raises(ValueError, match="provided together"):
        scp.violinplot(adata, "gene0", "cell_type", pairs=[("A", "B")])
    with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
        scp.violinplot(
            adata,
            "gene0",
            "cell_type",
            pairs=[("A", "B")],
            pvalues=[1.1],
        )


def test_pairing_does_not_aggregate_repeated_cells(adata):
    adata.obs["donor"] = "shared"
    with pytest.raises(ValueError, match="no aggregation"):
        scp.violinplot(adata, "score", "cell_type", paired_by="donor")


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
