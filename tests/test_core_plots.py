import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure

import scpolish as scp
import scpolish.pl as pl
from scpolish._utils.colors import categorical_colors, resolve_palette
from scpolish._utils.style import auto_point_size


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_exports():
    assert scp.dimplot is pl.dimplot
    assert scp.featureplot is pl.featureplot


def test_categorical_color_helper_resolves_palette_in_category_order(adata):
    categories = ["C", "A", "B"]
    colors = ["#aa0000", "#00aa00", "#0000aa"]
    assert categorical_colors(adata, "cell_type", categories, colors) == tuple(colors)
    mapping = {"C": "blue", "A": "red", "B": "green"}
    assert categorical_colors(adata, "cell_type", categories, mapping) == tuple(
        mpl.colors.to_hex(mapping[key]) for key in categories
    )


def test_default_categorical_colors_match_dimplot(adata):
    categories = list(adata.obs["cell_type"].cat.categories)
    ax = scp.dimplot(adata, "cell_type", labels=False)
    facecolors = np.concatenate(
        [collection.get_facecolors() for collection in ax.collections]
    )
    labels = adata.obs["cell_type"].to_numpy()
    expected = [
        mpl.colors.to_hex(np.unique(facecolors[labels == category], axis=0)[0])
        for category in categories
    ]
    colors = resolve_palette(categories, "muted", adata=adata, key="cell_type")
    assert [colors[category] for category in categories] == expected


def test_stored_palette_tracks_categories_when_reordered(adata):
    stored = ["#112233", "#445566", "#778899"]
    adata.uns["cell_type_colors"] = stored
    colors = categorical_colors(adata, "cell_type", ["C", "A", "B"])
    assert colors == (stored[2], stored[0], stored[1])


def test_string_group_palette_tracks_sorted_categories_when_reordered(adata):
    stored = ["#112233", "#445566", "#778899"]
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(object)
    adata.uns["cell_type_colors"] = stored
    colors = categorical_colors(adata, "cell_type", ["C", "A", "B"])
    assert colors == (stored[2], stored[0], stored[1])


@pytest.mark.parametrize("mode", ["auto", "direct", "numbered", False])
@pytest.mark.parametrize("raster", [False, True])
def test_dimplot(adata, mode, raster, tmp_path):
    adata.uns["cell_type_colors"] = ["#112233", "#445566", "#778899"]
    before = adata.copy()
    rc = mpl.rcParams.copy()
    path = tmp_path / "dim.pdf"
    ax = scp.dimplot(
        adata,
        "cell_type",
        labels=mode,
        palette="scanpy",
        raster=raster,
        save=path,
    )
    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "UMAP 1"
    assert len(ax.texts) == (0 if mode is False else 3)
    assert path.read_bytes().startswith(b"%PDF")
    assert b"/FontFile2" in path.read_bytes()
    assert all(
        c.get_rasterized() is raster
        for c in ax.collections
        if isinstance(c, PathCollection)
    )
    assert adata.uns == before.uns
    pd.testing.assert_frame_equal(adata.obs, before.obs)
    np.testing.assert_array_equal(adata.X, before.X)
    np.testing.assert_array_equal(adata.obsm["X_umap"], before.obsm["X_umap"])
    assert mpl.rcParams == rc
    for text, color in zip(ax.texts, adata.uns["cell_type_colors"], strict=False):
        np.testing.assert_allclose(
            text.get_bbox_patch().get_facecolor()[:3], mpl.colors.to_rgb(color)
        )


def test_many_categories(adata):
    adata.obs["many"] = pd.Categorical([f"type{i}" for i in range(30)])
    ax = scp.dimplot(adata, "many")
    assert len(ax.texts) == 30
    assert ax.get_legend().get_texts()[0].get_text().startswith("1  ")
    assert ax.figure.get_figwidth() > 5.5


def test_split(adata):
    adata.obs["sample"] = ["one", "two"] * 15
    adata.uns["cell_type_colors"] = ["red", "blue", "green"]
    fig, axes = scp.dimplot(adata, "cell_type", split_by="sample", palette="scanpy")
    assert isinstance(fig, Figure)
    assert axes.shape == (1, 2)
    assert axes[0, 0].get_xlim() == axes[0, 1].get_xlim()
    for ax in axes.flat:
        assert len(ax.collections[0].get_offsets()) == 15
        assert set(map(tuple, ax.collections[0].get_facecolors()[:, :3])) == {
            mpl.colors.to_rgb(c) for c in ["red", "blue", "green"]
        }


@pytest.mark.parametrize("feature", ["score", "gene0"])
@pytest.mark.parametrize("raster", [False, True])
def test_feature(adata, feature, raster):
    ax = scp.featureplot(adata, feature, raster=raster)
    assert isinstance(ax, Axes)
    if feature == "score":
        assert all(c.get_rasterized() for c in ax.collections)
    else:
        assert all(c.get_rasterized() is raster for c in ax.collections)
    assert "cell_type_colors" not in adata.uns


def test_featureplot_default_cmap_and_compact_colorbar(adata):
    ax = scp.featureplot(adata, "gene0")
    assert ax.collections[0].get_cmap().name == "magma"
    colorbar_ax = next(axis for axis in ax.figure.axes if axis is not ax)
    assert colorbar_ax.get_position().height < ax.get_position().height
    assert colorbar_ax.get_position().width < ax.get_position().width * 0.1


def test_featureplot_threshold_and_none(adata):
    adata.obs["score"] = [-2, -1, 0, 1, 2] * 6
    original = adata.obs["score"].copy()
    ax = scp.featureplot(adata, "score", threshold=0)
    assert len(ax.collections[0].get_offsets()) == 18
    assert len(ax.collections[1].get_offsets()) == 12
    pd.testing.assert_series_equal(adata.obs["score"], original)

    ax = scp.featureplot(adata, "score", threshold=10)
    assert len(ax.collections[0].get_offsets()) == 30
    assert len(ax.collections) == 1
    assert len(ax.figure.axes) == 1

    ax = scp.featureplot(adata, "score", threshold=1)
    assert len(ax.collections[0].get_offsets()) == 24

    ax = scp.featureplot(adata, "score", threshold=None)
    values = np.ma.asarray(
        next(c for c in ax.collections if c.get_array() is not None).get_array()
    )
    assert np.ma.getmaskarray(values).sum() == 0
    assert values.min() == -2


@pytest.mark.parametrize("threshold", [True, float("inf"), "0"])
def test_featureplot_rejects_invalid_threshold(adata, threshold):
    with pytest.raises(ValueError, match="threshold"):
        scp.featureplot(adata, "score", threshold=threshold)


def test_point_area_adapts_to_cell_count(adata):
    assert auto_point_size(adata.n_obs) > auto_point_size(30_000)
    ax = scp.featureplot(adata, "gene0")
    np.testing.assert_allclose(
        ax.collections[0].get_sizes(), auto_point_size(adata.n_obs)
    )


def test_grid_and_shared(adata):
    fig, axes = scp.featureplot(adata, ["score", "gene0", "gene1"], shared_scale=True)
    assert axes.shape == (2, 2)
    assert not axes[1, 1].get_visible()
    norms = [
        (
            next(c for c in a.collections if c.get_array() is not None).norm.vmin,
            next(c for c in a.collections if c.get_array() is not None).norm.vmax,
        )
        for a in axes.flat[:3]
    ]
    assert len(set(norms)) == 1
    assert tuple(fig.get_size_inches()) == (9, 11)


def test_score_panels_use_adaptive_layout_and_preserve_missing(adata):
    adata.obs["score_b"] = np.linspace(0, 1, adata.n_obs)
    adata.obs.loc[adata.obs.index[::4], "score"] = np.nan
    original = adata.obs["score"].copy()
    _, axes = scp.featureplot(adata, ["score", "score_b"])
    assert axes.shape == (1, 2)
    assert [c.get_offsets().shape[0] for c in axes[0, 0].collections] == [8, 22]
    bars = [a for a in axes[0, 0].figure.axes if a.get_label() == "<colorbar>"]
    assert len(bars) == 2
    pd.testing.assert_series_equal(adata.obs["score"], original)


def test_score_defaults_retain_negatives_and_honor_overrides(adata):
    adata.obs["score"] = np.linspace(-2, 2, adata.n_obs)
    ax = scp.featureplot(adata, "score", cmap="cividis", alpha=0.6, size=7)
    foreground = ax.collections[1]
    assert foreground.get_array().min() == -2
    assert foreground.get_cmap().name == "cividis"
    assert foreground.get_alpha() == 0.6
    np.testing.assert_allclose(foreground.get_sizes(), 7)


def test_score_percentile_limits_and_empty_panel(adata):
    ax = scp.featureplot(adata, "score", vmin="p10", vmax=lambda x: np.max(x))
    assert ax.collections[1].norm.vmin == pytest.approx(0.1)
    assert ax.collections[1].norm.vmax == 1
    adata.obs["score"] = np.nan
    ax = scp.featureplot(adata, "score")
    assert any(t.get_text() == "No scored observations" for t in ax.texts)
    assert len(ax.figure.axes) == 1


def test_single_panel_default_size_is_compact(adata):
    ax = scp.dimplot(adata, "cell_type", labels=False)
    assert tuple(ax.figure.get_size_inches()) == (7.5, 5.5)
    ax = scp.featureplot(adata, "gene0")
    assert tuple(ax.figure.get_size_inches()) == (12, 8)
    ax = scp.dimplot(adata, "cell_type", labels="numbered", figsize=(6, 4))
    assert tuple(ax.figure.get_size_inches()) == (6, 4)


def test_explicit_canvas_fills_axes_and_keeps_pdf_dimensions(
    adata, monkeypatch, tmp_path
):
    calls = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **kw: calls.append(kw))
    ax = scp.featureplot(adata, "gene0", figsize=(12, 8), save=tmp_path / "wide.pdf")
    ax.figure.canvas.draw()
    assert ax.get_aspect() == 1
    assert tuple(ax.figure.get_size_inches()) == (12, 8)
    assert calls[0]["bbox_inches"] is None


def test_dimplot_explicit_canvas_fills_panel(adata):
    ax = scp.dimplot(adata, "cell_type", labels=False, figsize=(12, 8))
    ax.figure.canvas.draw()
    assert ax.get_aspect() == "auto"
    assert ax.get_position().width > 0.75


@pytest.mark.parametrize("basis", ["pca", "tsne", "X_umap"])
@pytest.mark.parametrize(
    "function,args", [(scp.dimplot, ("cell_type",)), (scp.featureplot, ("score",))]
)
def test_basis_and_axes(adata, basis, function, args):
    adata.obsm["X_" + basis.removeprefix("X_")] = adata.obsm["X_umap"].copy()
    _, ax = plt.subplots()
    figures = plt.get_fignums()
    assert function(adata, *args, basis=basis, ax=ax) is ax
    assert figures == plt.get_fignums()


def test_sources(adata):
    adata.raw = adata.copy()
    adata.layers["other"] = adata.X + 50
    ax = scp.featureplot(adata, "gene0", layer="other")
    np.testing.assert_allclose(
        np.sort(ax.collections[0].get_array()), np.sort(adata.layers["other"][:, 0])
    )
    adata.X[:] = 100
    ax = scp.featureplot(adata, "gene0", threshold=None)
    np.testing.assert_allclose(
        np.sort(ax.collections[0].get_array()), np.sort(adata.raw.X[:, 0])
    )
    with pytest.raises(ValueError, match="layer"):
        scp.featureplot(adata, "gene0", layer="other", use_raw=True)


def test_missing_and_strings(adata):
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(object)
    adata.obs.iloc[0, 0] = None
    adata.obs.loc[adata.obs.index[0], "score"] = np.nan
    before = copy.deepcopy(adata.obs)
    scp.dimplot(adata, "cell_type")
    scp.featureplot(adata, "score")
    pd.testing.assert_frame_equal(adata.obs, before)


def test_invalid(adata):
    for function, arg in [(scp.dimplot, "cell_type"), (scp.featureplot, "score")]:
        with pytest.raises(ValueError, match="X_absent"):
            function(adata, arg, basis="absent")
        with pytest.raises(ValueError):
            function(adata[:0], arg)
        with pytest.raises(ValueError):
            function(adata, arg, size=-1)
    with pytest.raises(ValueError):
        scp.featureplot(adata, [])
    with pytest.raises(ValueError, match="numeric"):
        scp.featureplot(adata, "cell_type")
    with pytest.raises(ValueError):
        scp.dimplot(adata, "cell_type", labels="wrong")


def test_export_dpi(adata, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **kw: calls.append(kw))
    scp.featureplot(adata, "score", save=tmp_path / "feature.pdf")
    assert calls[0]["dpi"] == 300
    assert calls[0]["bbox_inches"] is None
